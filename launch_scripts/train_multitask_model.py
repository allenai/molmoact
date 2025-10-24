import argparse
import logging
from os.path import join, exists
from typing import cast, List

import omegaconf
from omegaconf import OmegaConf

from launch_scripts.utils import get_evaluation, DEBUG_MODEL
from olmo.train.optim import OptimizerType, OptimizerConfig, SchedulerConfig, SchedulerType
from olmo.train.trainer_config import (
    WandbConfig, BatchDivisor, SpeedMonitorConfig,
    FSDPConfig, FSDPPrecision, CompilerConfig, TrainConfig
)
from olmo.models.model import FSDPWrapStrategy
from olmo.models.molmo.molmo import MolmoConfig
from olmo.data.data_loader import DataLoaderConfig, RootSizeMixture
from olmo.torch_util import get_world_size
from olmo.util import clean_opt, prepare_torchrun_environment, select_checkpoint
from scripts.train import run_trainer

from olmo.tokenizer import DEPTH_TOKENS

import torch.distributed as dist
import torch.multiprocessing as mp

import os
from olmo.eval.loss_evaluator import LossDatasetEvaluatorConfig

import tempfile
import requests
from huggingface_hub import hf_hub_url
from huggingface_hub.utils import build_hf_headers

log = logging.getLogger("train")


AUX_EXCEPT_DOCS = [
    # Supervised datasets we want eval on
    "coco_2014_vqa_multi",
    "text_vqa",
    "okvqa",
    "chart_qa_weighted",
    "doc_qa",
    "info_qa",
    "ai2_diagram_v2_mix_transparent",
    "a_okvqa_mc",
    "a_okvqa_da",
    "android_control",

    # Some other datasets we might want to eval on
    "science_qa_img",
    "tabwmp_da",
    "st_qa",
    "tally_qa",

    # ("pixmo_clocks", 250000),  # Downsample since it is huge

    # Other synthetic data, also downsampled since they are huge
    ("dv_qa", 10000),
    ("figure_qa", 10000),
    ("plot_qa", 20000),
]


AUX = AUX_EXCEPT_DOCS + [
    "pixmo_docs_charts",
    "pixmo_docs_tables",
    "pixmo_docs_other",
    "pixmo_docs_diagrams",
]


AUX_COSYN_V1 = AUX_EXCEPT_DOCS + [
    "cosyn_chart_exp",
    "cosyn_chemical_exp",
    # "cosyn_circuit_exp", # quality not good
    "cosyn_diagram_exp",
    "cosyn_document",
    # "cosyn_graphic_exp", # quality not good
    "cosyn_math_exp",
    "cosyn_music_exp",
    # "cosyn_nutrition_exp", # zero-shot evaluation dataset
    "cosyn_table_exp",
]


def _stream_yaml_from_hub(repo_id: str, filename: str, revision: str | None = None, token: str | None = None) -> str:
    """
    Returns a temporary file path containing the streamed YAML.
    For private repos, pass a token or set HUGGINGFACE_HUB_TOKEN env var and use build_hf_headers(None).
    """
    url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision, repo_type="model")
    headers = build_hf_headers(token=token)  # includes auth if provided / available
    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code == 404:
        raise FileNotFoundError(filename)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile("wb", suffix=f"-{filename}", delete=False) as tmp:
        tmp.write(r.content)
        return tmp.name


def get_training_mixture(submixture):
    resolved_weights = {}
    for task_name in submixture:
        mix = {}
        if isinstance(task_name, tuple):
            task_name, size = task_name
        else:
            size = None
        resolved_weights[task_name] = size
    return resolved_weights


if __name__ == "__main__":
    prepare_torchrun_environment()

    parser = argparse.ArgumentParser(prog="Train a multitask model")
    parser.add_argument("mixture", help="Name of datset mixture to train on")
    parser.add_argument("checkpoint", help="Path to checkpoint to start from")
    parser.add_argument("--seq_len", default=2304, type=int)
    parser.add_argument("--inf_seq_len", default=1792, type=int)
    parser.add_argument("--duration", default="30000", type=str)
    parser.add_argument("--max_inf_examples", default=2048, type=int)
    parser.add_argument("--global_batch_size", default=256, type=int)
    parser.add_argument("--lr_connector", default=5e-6, type=float)
    parser.add_argument("--lr_vit", default=5e-6, type=float)
    parser.add_argument("--lr_llm", default=1e-5, type=float)
    parser.add_argument("--lr_scheduler", default="cosine_with_warmup", type=str)
    parser.add_argument("--device_eval_batch_size", default=4, type=int)
    parser.add_argument("--device_inf_batch_size", default=4, type=int)
    parser.add_argument("--device_train_batch_size", default=4, type=int)
    parser.add_argument("--include_image", action="store_true",
                        help="Include image in the evaluation outputs")
    parser.add_argument("--turn_off_inference", action="store_true",
                        help="Turn off inference during training")
    parser.add_argument("--max_crops", default=8, type=int)
    parser.add_argument("--max_multi_image_crops", default=8, type=int)
    parser.add_argument("--image_pooling_h", default=None, type=int)
    parser.add_argument("--image_pooling_w", default=None, type=int)
    parser.add_argument("--max_images", default=1, type=int)
    parser.add_argument("--depth_tokens", action='store_true')
    parser.add_argument("--lora_enable", action='store_true')
    parser.add_argument("--lora_rank", default=64, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_bias", default="none", type=str)
    parser.add_argument("--norm_stats_path", default=None, type=str)
    parser.add_argument("--img_aug", action='store_true')
    parser.add_argument("--pin_memory", action='store_true')
    parser.add_argument("--ft_embedding", default="lm_head", type=str)
    parser.add_argument("--warmup", default=200, type=int)
    parser.add_argument("--save_interval", default=2000, type=int)
    parser.add_argument("--log_interval", default=20, type=int)
    parser.add_argument("--save_num_checkpoints_to_keep", default=1, type=int)
    parser.add_argument("--save_intermediate_unsharded_checkpoint", action='store_true')
    parser.add_argument("--save_final_unsharded_checkpoint", action='store_true')
    parser.add_argument("--save_every_n_epoch", default=None, type=float)

    
    

    args, other_args = parser.parse_known_args()

    eval_tasks = []
    eval_tasks_loss = []
    if args.mixture.startswith("single"):
        task_name = args.mixture.split("_", 1)[1]
        eval_tasks = [task_name,]
        tasks = [["eval", eval_tasks, 1.0]]
    elif args.mixture == "android":
        eval_tasks = ["android_control_ll"]
        tasks = [["eval", ["android_control"], 1.0]]
    elif args.mixture in ["small1", "debug"]:
        eval_tasks = ["chart_qa", "doc_qa"]
        tasks = [["aux", ["chart_qa", "doc_qa"], 1.0]]
    elif args.mixture in ["pointing"]:
        eval_tasks = ["pointing_eval:test"]
        tasks = [["pointing", [
            "pixmo_points",
            "pixmo_count",
            "pixmo_points_high_freq",
            "pixmo_points_counting",
            "pixmo_points_high_freq_counting",
            "pixmo_count_counting",
        ], 1.0]]

    elif args.mixture == "small2":
        eval_tasks = ["chart_qa", "doc_qa", "info_qa"]
        tasks = [["aux", [("chart_qa", 4*4),
                          ("doc_qa", 2*2), ("info_qa", 1)], 1.0]]
    elif args.mixture in ["3.2-synthetic"]:
        aux = list(AUX)
        eval_tasks = [
            "chart_qa",
            "info_qa",
            "doc_qa",
            "ai2_diagram_v2_mix_transparent",
            "coco_2014_vqa_multi",
            "pixmo_clocks",
            "android_control_ll",
            "pointing_eval:test",
        ]
        tasks = [
            ["demo", [
                "pixmo_ask_model_anything",
                ("pixmo_cap", 50000),
                "pixmo_cap_qa_as_user_qa",
                "pixmo_pointing_explanations"
            ], 0.15],
            ["aux", aux, 0.50],
            ["pointing", [
                "pixmo_points_train",
                "pixmo_count_train",
                "pixmo_points_high_freq_train",
            ], 0.35]
        ]
    elif args.mixture in ["molmoact-pretrain"]:
        aux = list(AUX_EXCEPT_DOCS)
        tasks = [
            # multimodal web data
            ["demo", [
                "pixmo_ask_model_anything",
                ("pixmo_cap", 50000),
                "pixmo_cap_qa",
                "pixmo_pointing_explanations",
            ], 0.05 * 0.15],
            ["vqa", aux, 0.05 * 0.4],
            ["pointing", [
                "pixmo_points_train",
                "pixmo_count_train",
                "pixmo_points_high_freq_train",
            ], 0.05 * 0.35],
            ["bbox", [
                "lvis",
            ], 0.05 * 0.1],

            # auxiliary depth/trace data
            ["auxiliary", [
                "auxiliary_depth_data",
                "auxiliary_trace_data",
            ], 0.15],

            # action reasoning data + trajectory-conditioned action data
            ["bc_z", [
                "bc_z",
            ], 0.15],
            ["bridge", [
                "bridge_data_v2",
            ], 0.25],
            ["rt_1", [
                "rt_1",
            ], 0.40],
        ]
    elif args.mixture in ["molmoact-midtrain"]:
        # this will be uniform sampling
        tasks = [
            ["molmoact_dataset_home_primary", [
                "molmoact_dataset_home_primary",
            ], 1.0],
            ["molmoact_dataset_home_secondary", [
                "molmoact_dataset_home_secondary",
            ], 1.0],
            ["molmoact_dataset_tabletop_primary", [
                "molmoact_dataset_tabletop_primary",
            ], 1.0],
            ["molmoact_dataset_tabletop_secondary", [
                "molmoact_dataset_tabletop_secondary",
            ], 1.0],
        ]
    elif args.mixture in ["libero-spatial"]:
        # this will be uniform sampling
        tasks = [
            ["libero_spatial", [
                "libero_spatial",
            ], 1.0],
        ]
    elif args.mixture in ["libero-object"]:
        # this will be uniform sampling
        tasks = [
            ["libero_object", [
                "libero_object",
            ], 1.0],
        ]
    elif args.mixture in ["libero-goal"]:
        # this will be uniform sampling
        tasks = [
            ["libero_goal", [
                "libero_goal",
            ], 1.0],
        ]
    elif args.mixture in ["libero-long"]:
        # this will be uniform sampling
        tasks = [
            ["libero_long", [
                "libero_long",
            ], 1.0],
        ]
    elif args.mixture in ["robot-finetune"]:
        tasks = [
            ["finetune", [
                "finetune:/path/to/processed_dataset",
            ], 1.0],
        ]
    else:
        raise NotImplementedError(args.mixture)

    debug = args.checkpoint in ["debug", "debug2"]
    if debug:
        checkpoint = None
        model_cfg = DEBUG_MODEL
        if args.checkpoint == "debug2":
            model_cfg.max_crops = 12
            model_cfg.crop_mode = "overlap-and-resize-c2"
            model_cfg.tokenizer.identifier = "mm:hf-Qwen/Qwen2-7B"
            model_cfg.embedding_size = 152064
            model_cfg.vocab_size = 152064
            model_cfg.pad_tokenizer = True
        global_batch_size = 8
        model_init = None
        inf_eval_interval = 20
        eval_interval = 20
        log_interval = args.log_interval
        eval_examples = 16
        max_inf_examples = 16
        duration = 1000
        eval_subset_batches = 4
    else:
        eval_examples = 2048
        max_inf_examples = args.max_inf_examples
        log_interval = args.log_interval
        global_batch_size = args.global_batch_size
        inf_eval_interval = 2000
        eval_interval = 2000
        duration = args.duration
        checkpoint, is_hf_remote = select_checkpoint(args.checkpoint)
        if is_hf_remote:
            p = _stream_yaml_from_hub(checkpoint, "model.yaml")
            model_cfg = MolmoConfig.load(p)
        if exists(join(checkpoint, "model.yaml")):
            model_cfg = MolmoConfig.load(join(checkpoint, "model.yaml"))
        elif exists(join(checkpoint, "config.yaml")):
            model_cfg = MolmoConfig.load(join(checkpoint, "config.yaml"), key="model")

        eval_subset_batches = eval_examples//(args.device_eval_batch_size*get_world_size())
        logging.info(f"Setting eval subset batches to {eval_subset_batches}")
        assert eval_subset_batches > 0

    # Fine-tuning settings
    model_cfg.llm.residual_dropout = 0.1
    model_cfg.llm.response_residual_dropout = 0.0
    model_cfg.data_formatter.prompt_templates = "uber_model"
    model_cfg.data_formatter.message_format = "role"
    model_cfg.data_formatter.system_prompt = "demo_or_style"
    model_cfg.mm_preprocessor.loss_token_weighting = "root_subsegments"

    # Additional tokens
    model_cfg.llm.tokenizer.depth_tokens = args.depth_tokens

    # LoRA settings
    model_cfg.lora_enable = args.lora_enable
    model_cfg.lora_rank = args.lora_rank
    model_cfg.lora_alpha = args.lora_alpha
    model_cfg.lora_dropout = args.lora_dropout
    model_cfg.lora_bias = args.lora_bias

    # Path to dataset statistics for normalization
    model_cfg.norm_stats_path = args.norm_stats_path

    # Overriding model config
    model_cfg.mm_preprocessor.max_crops = args.max_crops or model_cfg.mm_preprocessor.max_crops
    model_cfg.mm_preprocessor.pooling_w = args.image_pooling_w or model_cfg.mm_preprocessor.pooling_w
    model_cfg.mm_preprocessor.pooling_h = args.image_pooling_h or model_cfg.mm_preprocessor.pooling_h

    # Multi-image settings
    model_cfg.mm_preprocessor.max_images = args.max_images or model_cfg.mm_preprocessor.max_images
    model_cfg.mm_preprocessor.max_multi_image_crops = args.max_multi_image_crops or model_cfg.mm_preprocessor.max_multi_image_crops

    # Image augmentation
    model_cfg.mm_preprocessor.img_aug = args.img_aug

    if model_cfg.llm.max_sequence_length < args.seq_len:
        model_cfg.llm.max_sequence_length = args.seq_len

    root_size_mixture: List[RootSizeMixture] = []
    for name, submixture, rate in tasks:
        submixture = get_training_mixture(submixture)
        root_size_mixture.append(RootSizeMixture(rate, submixture))

    num_workers = 0
    evaluations = []
    if not args.turn_off_inference:
        for task in eval_tasks:
            evaluation = get_evaluation(
                task,
                args.inf_seq_len,
                device_batch_size=args.device_inf_batch_size,
                max_examples=max_inf_examples,
                num_workers=num_workers,
                include_image=args.include_image,
            )
            evaluation.data.persistent_workers = True
            evaluations.append(evaluation)

    evaluations_loss = []
    if not args.turn_off_inference:
        for task in eval_tasks_loss:
            evaluation_loss = LossDatasetEvaluatorConfig(
                label=os.path.splitext(os.path.basename(task))[0],
                max_examples=eval_examples,
                device_batch_size=args.device_eval_batch_size,
                console_log_interval="${console_log_interval}",
                data=DataLoaderConfig(
                    seed="${seed}",
                    dataset=task,
                    shuffle=False,
                    split="validation",
                    drop_last=True,
                    sequence_length=args.seq_len,
                    num_workers=2,
                    pin_memory=True,
                    persistent_workers=True,
                ),
            )
            evaluations_loss.append(evaluation_loss)
    
    cfg = TrainConfig(
        run_name="multitask_train",
        save_folder="debug_run" if debug else omegaconf.MISSING,
        seed=6198,
        dry_run=False,
        wandb=None if debug else WandbConfig(
            name="${run_name}",
            project="${oc.env:WANDB_PROJECT}",
            group=None,
            entity="${oc.env:WANDB_ENTITY}",
            log_interval=log_interval
        ),
        compile=CompilerConfig(mode="default", dynamic=False),
        fused_loss=False,
        allow_resume=True,
        model=model_cfg,
        save_overwrite=debug,
        data=DataLoaderConfig(
            root_size_mixture=root_size_mixture,
            shuffle=True,
            split="train",
            drop_last=True,
            sequence_length=args.seq_len,
            num_workers=num_workers,
            pad="to_max",
            pin_memory=args.pin_memory, # set false to avoid OOM for large dataset
            seed=50189,
        ),
        ft_connector=True,
        ft_llm=True,
        ft_vit=True,
        ft_embedding=args.ft_embedding,
        optimizer=OptimizerConfig(
            name=OptimizerType.adamw,
            connector_learning_rate=args.lr_connector,
            vit_learning_rate=args.lr_vit,
            llm_learning_rate=args.lr_llm,
            connector_weight_decay=0.0,
            vit_weight_decay=0.0,
            llm_weight_decay=0.0,
            connector_betas=[0.9, 0.95],
            vit_betas=[0.9, 0.95],
            llm_betas=[0.9, 0.95],
            connector_eps=1e-6,
            vit_eps=1e-6,
            llm_eps=1e-6,
        ),
        scheduler=SchedulerConfig(
            name=args.lr_scheduler,
            connector_t_warmup=args.warmup,
            vit_t_warmup=args.warmup,
            llm_t_warmup=args.warmup,
            alpha_f=0.1,
            warmup_min_lr=0.0
        ),
        fsdp=FSDPConfig(
            use_orig_params=True,
            wrapping_strategy=FSDPWrapStrategy.by_block_and_size,
            precision=FSDPPrecision.float
        ),
        load_path=None,
        initial_model_checkpoint=checkpoint,
        save_interval=args.save_interval, # 2000 or 1 for debug
        save_num_checkpoints_to_keep=args.save_num_checkpoints_to_keep,
        global_train_batch_size=global_batch_size,
        device_train_microbatch_size=args.device_train_batch_size,
        time_limit=None,
        max_duration=duration,
        stop_at="${max_duration}",
        max_grad_norm=1,
        batch_divisor=BatchDivisor.global_batch,
        precision="amp_bf16",
        console_log_interval=log_interval, # log_interval or 1 for debug
        compile_loss=True,
        speed_monitor=SpeedMonitorConfig(window_size=20),
        softmax_auxiliary_loss=True,
        softmax_auxiliary_loss_scale=1e-4,
        eval_interval=eval_interval,
        inf_eval_interval=inf_eval_interval,
        inf_evaluators=evaluations,
        save_intermediate_unsharded_checkpoint=args.save_intermediate_unsharded_checkpoint,
        save_final_unsharded_checkpoint=args.save_final_unsharded_checkpoint,
        save_every_n_epoch=args.save_every_n_epoch,
        save_interval_epoch=0,
        evaluators=evaluations_loss
    )

    conf = OmegaConf.create(cfg)
    conf.merge_with_dotlist([clean_opt(arg) for arg in other_args])
    cfg = cast(TrainConfig, OmegaConf.to_object(conf))
    run_trainer(cfg)
