"""Run this script with 'torchrun'."""

import logging
import os
import re
import socket
import sys
import time
from datetime import datetime
from os.path import join
from pathlib import Path
import json

import torch
import wandb
from beaker import Beaker
from omegaconf import OmegaConf, open_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from olmo.exceptions import OLMoCliError, OLMoConfigurationError
from olmo.io import file_exists, write_file
from olmo.train.checkpointer import Checkpointer, load_model_state_unsharded, load_model_state_hf, is_unsharded_checkpoint, is_hf_checkpoint
from olmo.torch_util import (
    barrier,
    get_global_rank,
    get_local_rank,
    get_world_size,
    peak_gpu_memory,
    seed_all,
    freeze_module,
)
from olmo.train.trainer import Trainer, BeakerLogger
from olmo.train.trainer_config import TrainConfig, RuntimeData
from olmo.util import (
    clean_opt,
    log_extra_field,
    prepare_torchrun_environment,
)

from olmo.train.trainer_config import (
    SpeedMonitorConfig, CheckpointType,
    TrainConfig, BatchDivisor,
)

from olmo.train.checkpointer import Checkpointer, save_unsharded

log = logging.getLogger("train")


def get_trainable_params(model):
    """
    Calculate the size of a PyTorch model in bytes.
    """
    param_size = 0
    trainable_param_size = 0
    param_num = 0
    trainable_para_num = 0
    for param in model.parameters():
        param_num += param.nelement() 
        param_size += param.nelement() * param.element_size()
        trainable_para_num += param.nelement() if param.requires_grad else 0
        trainable_param_size += param.nelement() * param.element_size() if param.requires_grad else 0

    log.info(f'Number of trainable parameters: {trainable_para_num:,d}')

def run_trainer(cfg: TrainConfig) -> None:
    # TODO: this will decrease training throughput, find a workaround
    if cfg.model.lora_enable:
        os.environ["TORCHDYNAMO_DISABLE"] = "1"

    if cfg.run_name is None:
        log_extra_field("run_name", cfg.run_name)

    # Additional environment setup
    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    device = torch.device("cuda")
    seed_all(cfg.seed)
    barrier()

    # Display the configuration.
    if get_global_rank() == 0:
        log.info("Configuration:")
        log.info(cfg)

    # Figure out what checkpoint we are starting from, if any
    start_from = None
    reset_opt, reset_train = False, False
    is_resuming = False
    if cfg.allow_resume:
        # Check if there is a checkpoint for us to resume from in our save folder, in which
        # case we ignore `cfg.load_from` and use it
        try:
            lastest_checkpoint = Checkpointer.latest_checkpoint(cfg.save_folder)
        except FileNotFoundError:
            lastest_checkpoint = None
        if lastest_checkpoint:
            log.info(f"Resuming from {lastest_checkpoint}")
            if get_global_rank() == 0:
                saved_config: TrainConfig = TrainConfig.load(join(cfg.save_folder, "config.yaml"))
                if saved_config.model != cfg.model:
                    log.warning("Model config does not match the one resuming from")
                if saved_config.optimizer != cfg.optimizer:
                    log.warning("Optimizer config does not match the one resuming from")
                if saved_config.data != cfg.data:
                    log.warning("Data config does not match the one resuming from")
            start_from = str(lastest_checkpoint)
            reset_opt, reset_train = False, False
            is_resuming = True
        else:
            log.info("Not resuming since no latest checkpoint found")

    if start_from is None and cfg.load_path:
        start_from = cfg.load_path
        reset_train, reset_opt = cfg.reset_trainer_state, cfg.reset_optimizer_state
    elif start_from is None and cfg.initial_model_checkpoint is not None:
        start_from = cfg.initial_model_checkpoint
        reset_train, reset_opt = True, True
    start_from_unsharded = start_from and is_unsharded_checkpoint(start_from)
    if start_from_unsharded:
        assert reset_opt and reset_train, "Unshared checkpoints do not support optim/train state loading"
    
    start_from_hf = start_from and is_hf_checkpoint(start_from)
    if start_from_hf:
        assert reset_opt and reset_train, "Huggingface checkpoints do not support optim/train state loading"

    # Fail fast if we would be overwriting another save directory
    if not cfg.dry_run and not is_resuming and not cfg.save_overwrite:
        save_path = join(cfg.save_folder, "config.yaml")
        if file_exists(save_path):
            raise OLMoConfigurationError(f"{save_path} already exists, use --save_overwrite to overwrite")

    barrier()

    # Init the model
    model_cfg = cfg.model
    with torch.device("meta"):
        olmo_model = model_cfg.build_model()


    if model_cfg.norm_stats_path is not None:
        
        p = Path(os.path.expandvars(os.path.expanduser(model_cfg.norm_stats_path)))
        if not p.exists():
            raise FileNotFoundError(f"norm_stats_path not found: {p}")
        norm_stats = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(norm_stats, dict):
            raise TypeError(f"norm_stats JSON root must be an object/dict, got {type(norm_stats).__name__}")

        model_cfg.norm_stats = OmegaConf.create(norm_stats)


    # Freeze parameters depending on what we are tuning
    if not cfg.ft_connector:
        log.info(f"Freezing connector")
        for param in olmo_model.get_connector_parameters():
            param.requires_grad = False
    if not cfg.ft_vit:
        log.info(f"Freezing vision backbone")
        for param in olmo_model.get_vit_parameters():
            param.requires_grad = False
    if not cfg.ft_llm:
        log.info(f"Freezing LLM")
        for param in olmo_model.get_llm_parameters():
            param.requires_grad = False
    elif cfg.ft_embedding != "all":
        freeze_wte, freeze_out, freeze_ln_f = True, True, True
        if cfg.ft_embedding == "ln_f":
            freeze_ln_f = False
        elif cfg.ft_embedding == "lm_head":
            freeze_ln_f = False
            freeze_out = False
        elif cfg.ft_embedding == "wte":
            freeze_wte = False
        else:
            raise NotImplementedError(cfg.fsdp)
        if freeze_ln_f:
            log.info(f"Freezing LLM: ln_f")
            freeze_module(olmo_model.transformer.ln_f)
        if freeze_out and hasattr(olmo_model.transformer, "ff_out"):
            log.info(f"Freezing LLM: ff_out")
            freeze_module(olmo_model.transformer.ff_out)
        if freeze_wte:
            log.info(f"Freezing LLM: wte")
            olmo_model.transformer.wte.embedding.requires_grad = False

    # Do some other model setup
    if cfg.activation_checkpointing:
        olmo_model.apply_activation_checkpointing()
    # Stops the compiler get confused due to cache modifications
    olmo_model.warmup_cache(device)
    if cfg.compile:
        olmo_model.apply_compile(**cfg.compile.compile_args())

    # Shard the model, and initialize if we are not loading a checkpoint
    if cfg.fsdp and not cfg.fsdp.fsdp2:
        log.info("Wrapping model with FSDP...")
        if start_from is None:
            # Just run our `reset_with_pretrained_weights` on rank0 and broadcast so we
            # don't have to port all the init logic to a FSDP param_init_fn function
            if get_global_rank() == 0:
                # Load on CPU in case model doesn't fit on a single GPU
                olmo_model = olmo_model.to_empty(device="cpu")
                olmo_model.reset_with_pretrained_weights()
            sync_module_states = True
        else:
            sync_module_states = False

        # meta-device parameters can just become empty since we are either broadcasting from rank0
        # or going to load a checkpoint anyway
        def dummy_init_fn(module: torch.nn.Module) -> None:
            module.to_empty(device=device, recurse=False)

        fsdp_model = FSDP(
            olmo_model,
            **cfg.fsdp.get_fsd_args(cfg.autocast_precision),
            param_init_fn=dummy_init_fn,
            auto_wrap_policy=olmo_model.get_fsdp_wrap_policy(cfg.fsdp.wrapping_strategy),
            device_id=get_local_rank(),
            sync_module_states=sync_module_states,
        )

    elif cfg.fsdp.fsdp2:
        log.info("Wrapping model with FSDP2...")
        olmo_model.apply_fsdp2(**cfg.fsdp.get_fsd2_args(cfg.autocast_precision))
        olmo_model.to_empty(device=device)
        if start_from is None:
            olmo_model.reset_with_pretrained_weights()
        fsdp_model = olmo_model
    else:
        raise NotImplementedError()

    torch.cuda.empty_cache()

    log.info(f"Total number of parameters: {olmo_model.num_params():,d}")
    log.info(f"Number of non-embedding parameters: {olmo_model.num_params(include_embedding=False):,d}")
    get_trainable_params(olmo_model)
    if olmo_model.config.llm.block_type == "moe":
        log.info(f"Number of active parameters: {olmo_model.num_params(include_inactive_params=False):,d}")
    log.info(f"Peak GPU Memory (MB) after FSDP: {int(peak_gpu_memory() or 0)}")
    log.info("Model:")
    log.info(fsdp_model)

    # Construct optimizer/scheduler/checkpointer
    optim = cfg.optimizer.build_optimizer(cfg.max_grad_norm, cfg.max_grad_norm_ratio, fsdp_model)
    scheduler = cfg.scheduler.build()
    checkpointer = cfg.checkpointer_config.build(cfg.save_overwrite)

    # Construct data loader and evaluators
    train_loader = cfg.data.build_train_dataloader(cfg.model, cfg.global_train_batch_size, device)


    if isinstance(cfg.max_duration, str):
        if cfg.max_duration.endswith("ep"):
            train_num_data = train_loader.dataset.total_size
            cfg.max_duration = int(float(cfg.max_duration[:-2].strip()) * train_num_data // cfg.global_train_batch_size)
        else:
            cfg.max_duration = int(cfg.max_duration.strip())
        
        cfg.stop_at = cfg.max_duration

    log.info(f"Training for {cfg.max_duration} steps")

    if cfg.save_every_n_epoch:
        cfg.save_interval_epoch = int(train_num_data // cfg.global_train_batch_size)
        

    if cfg.eval_interval > 0 or cfg.eval_on_load:
        evaluators = [v.build_dataset_evaluator(cfg.model, device) for v in cfg.evaluators]
    else:
        evaluators = None
    if cfg.inf_eval_interval > 0 or cfg.eval_on_load:
        inf_evaluators = [v.build_dataset_evaluator(cfg.model, None, device) for v in cfg.inf_evaluators]
    else:
        inf_evaluators = None

    # Maybe build the BeakerLogger
    if "BEAKER_EXPERIMENT_ID" in os.environ and "BEAKER_TOKEN" in os.environ:
        if get_global_rank() == 0:
            experiment_id = os.environ["BEAKER_EXPERIMENT_ID"]
            client = Beaker.from_env()
            experiment = client.experiment.get(experiment_id)
            beaker_logger = BeakerLogger(client, experiment, cfg.beaker_log_interval, experiment.description)
            beaker_logger.log_init()
        else:
            beaker_logger = None
    else:
        if cfg.beaker_log_interval > 0 and "BEAKER_EXPERIMENT_ID" in os.environ:
            logging.info(f"Beaker log interval set to {cfg.beaker_log_interval}, but beaker "
                         f"token is missing, so beaker logging will turned off")
        beaker_logger = None

    # Maybe start W&B run.
    if cfg.wandb is not None and (get_global_rank() == 0 or not cfg.wandb.rank_zero_only):
        wandb_dir = Path(cfg.save_folder) / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb_cfg = cfg.asdict(exclude=["wandb"])

        if "BEAKER_EXPERIMENT_ID" in os.environ:
            wandb_cfg["beaker_experiment_id"] = os.environ["BEAKER_EXPERIMENT_ID"]
            if beaker_logger is not None:
                wandb_cfg["beaker_url"] = beaker_logger.beaker.experiment.url(beaker_logger.experiment)
        if is_resuming:
            wandb_cfg["resuming_from"] = start_from
        if is_resuming and cfg.wandb.allow_resume:
            resume_run_id = saved_config.runtime_data.wandb_id
            resume_step = int(re.match(r".*step([0-9]+).*", lastest_checkpoint).group(1))
            resume_from = f"{resume_run_id}?_step={resume_step}"
            run_id = resume_run_id
        else:
            run_id = None
            resume_from = None

        wandb.init(
            dir=str(wandb_dir),
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            name=cfg.wandb.name,
            tags=cfg.wandb.tags,
            config=wandb_cfg,
            id=run_id,
            resume_from=resume_from
        )
        wandb_url = wandb.run.get_url()
        if beaker_logger is not None:
            beaker_logger.add_wandb(wandb_url)  # add wandb url to beaker description

    # Fill in some runtime data so it will be recorded when we save the config
    cfg.runtime_data = RuntimeData(
        hostname=socket.gethostname(),
        date=datetime.now().strftime("%m/%d/%Y, %H:%M"),
        world_size=get_world_size(),
        beaker_experiment_id=os.environ.get("BEAKER_EXPERIMENT_ID"),
        beaker_experiment_url=(None if beaker_logger is None else
                               beaker_logger.beaker.experiment.url(beaker_logger.experiment)),
        wandb_url=wandb.run.get_url() if wandb.run else None,
        wandb_id=wandb.run.id if wandb.run else None,
        args=" ".join(sys.argv),
        resuming_from=start_from if is_resuming else None
    )

    # Save the config in a top-level file, note if we are resuming
    # the current config will still be saved next to new checkpoints
    if not cfg.dry_run and not is_resuming:
        if get_global_rank() == 0:
            write_file(cfg.save_folder, "config.yaml",
                       OmegaConf.to_yaml(cfg, resolve=True), cfg.save_overwrite)
    barrier()

    with Trainer(
        cfg=cfg,
        epoch=cfg.epoch,
        model=olmo_model,
        fsdp_model=fsdp_model,
        checkpointer=checkpointer,
        optim=optim,
        scheduler=scheduler,
        train_loader=train_loader,
        device=device,
        evaluators=evaluators,
        inference_evaluators=inf_evaluators,
        beaker_logger=beaker_logger,
    ) as trainer:

        if start_from:
            # Load the starting checkpoint if there is one
            t0 = time.perf_counter()
            if start_from_unsharded:
                log.info(f"Loading unshared model from {start_from}")
                load_model_state_unsharded(start_from, fsdp_model)

            if start_from_hf:
                log.info(f"Loading huggingface model from {start_from}")
                load_model_state_hf(start_from, fsdp_model)

            if model_cfg.lora_enable:
                from peft import LoraConfig, get_peft_model

                # same lora config as openvla
                lora_config = LoraConfig(
                    r=model_cfg.lora_rank,
                    lora_alpha=min(model_cfg.lora_rank, model_cfg.lora_alpha),
                    # target_modules=find_all_linear_names(fsdp_model, model_cfg),
                    target_modules="all-linear",
                    lora_dropout=model_cfg.lora_dropout,
                    bias=model_cfg.lora_bias,
                    init_lora_weights="gaussian",
                )

                log.info("Adding LoRA adapters...")
                trainer.fsdp_model = get_peft_model(fsdp_model, lora_config)


                def _sync_grad(grad):
                    torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.SUM)
                    grad /= torch.distributed.get_world_size()
                    return grad

                for n, p in trainer.fsdp_model.named_parameters():
                    if p.requires_grad and "lora_" in n:
                        p.register_hook(_sync_grad)


                log.info(f"Total number of parameters: {olmo_model.num_params():,d}")
                log.info(f"Number of non-embedding parameters: {olmo_model.num_params(include_embedding=False):,d}")
                get_trainable_params(olmo_model)
                if olmo_model.config.llm.block_type == "moe":
                    log.info(f"Number of active parameters: {olmo_model.num_params(include_inactive_params=False):,d}")
                log.info(f"Peak GPU Memory (MB) after LoRA: {int(peak_gpu_memory() or 0)}")
                log.info("LoRA Model:")
                log.info(trainer.fsdp_model)

                log.info("Updating optimizer to include LoRA parameters...")
                trainer.optim = cfg.optimizer.build_optimizer(cfg.max_grad_norm, cfg.max_grad_norm_ratio, trainer.fsdp_model)

            if not start_from_unsharded and not start_from_hf:
                if reset_train and reset_opt:
                    log.info(f"Loading model from {start_from}")
                elif not reset_opt and not reset_train:
                    log.info(f"Resuming from checkpoint {start_from}")
                else:
                    log.info(f"Restoring checkpoint {start_from}, but resetting "
                             f"{'Trainer' if reset_train else 'Optimizer'}")
                trainer.restore_checkpoint(
                    start_from,
                    load_optimizer_state=not reset_opt,
                    load_trainer_state=not reset_train,
                )
            log.info(f"Checkpoint successfully loaded in {time.perf_counter()-t0:0.1f} seconds")

        barrier()

        for name, param in fsdp_model.named_parameters():
            if not torch.all(torch.isfinite(param)):
                raise ValueError(name)

        # Ready to start training
        if not cfg.dry_run:
            log.info("Starting training...")
            trainer.fit()
            log.info("Training complete")
        else:
            log.info("Dry run complete")


if __name__ == "__main__":
    prepare_torchrun_environment()

    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OLMoCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])
    run_trainer(cfg)
