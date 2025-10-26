import argparse
import os
import shutil
import logging
import json
import gc
from typing import Dict, Any, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, AutoProcessor, GenerationConfig
from transformers import Qwen2Tokenizer, Qwen2TokenizerFast, GPT2TokenizerFast

from olmo.models.molmo.molmo import Molmo, MolmoConfig as ModelConfig
from olmo.train.checkpointer import load_model_state
from olmo.util import (
    prepare_cli_environment,
    resource_path
)

from .configuration_molmoact import MolmoActConfig, MolmoActVitConfig, MolmoActAdapterConfig, MolmoActLlmConfig
from .modeling_molmoact import MolmoActForCausalLM, MolmoActForActionReasoning
from .processing_molmoact import MolmoActProcessor
from .image_processing_molmoact import MolmoActImageProcessor

from omegaconf import OmegaConf
from pathlib import Path


logger = logging.getLogger(__name__)


demo_chat_template = (
    "{% for message in messages %}"
    "{%- if (loop.index % 2 == 1 and message['role'].lower() != 'user') or (loop.index % 2 == 0 and message['role'].lower() != 'assistant') -%}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{%- endif -%}"
    "{% if message['content'] is string %}"
    "{{ message['content'] }}"
    "{% else %}"
    "{% for content in message['content'] %}"
    "{% if content['type'] == 'text' %}"
    "{{ content['text'] }}"
    "{%- if not loop.last -%}"
    "{{ ' ' }}"
    "{%- endif -%}"
    "{% endif %}"
    "{% endfor %}"
    "{% endif %}"
    "{%- if not loop.last -%}"
    "{{ ' ' }}"
    "{%- endif -%}"
    "{% endfor %}"
)


demo_role_chat_template = (
    "{% for message in messages %}"
    "{%- if (loop.index % 2 == 1 and message['role'].lower() != 'user') or (loop.index % 2 == 0 and message['role'].lower() != 'assistant') -%}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{%- endif -%}"
    "{{ message['role'].capitalize() + ': ' }}"
    "{% if message['content'] is string %}"
    "{{ message['content'] }}"
    "{% else %}"
    "{% for content in message['content'] %}"
    "{% if content['type'] == 'text' %}"
    "{{ content['text'] }}"
    "{%- if not loop.last -%}"
    "{{ ' ' }}"
    "{%- endif -%}"
    "{% endif %}"
    "{% endfor %}"
    "{% endif %}"
    "{%- if not loop.last -%}"
    "{{ ' ' }}"
    "{%- endif -%}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ ' Assistant:' }}"
    "{% endif %}"
)


def flatten_dict(xs, sep=None):
    def _key(path):
        if sep is None:
            return path
        return sep.join(path)

    def _flatten(xs, prefix):
        if not isinstance(xs, dict):
            return {_key(prefix): xs}
        result = {}
        is_empty = True
        for key, value in xs.items():
            is_empty = False
            path = prefix + (key,)
            result.update(_flatten(value, path))
        return result
    return _flatten(xs, ())


def unflatten_dict(xs, sep=None):
    assert isinstance(xs, dict), f'input is not a dict; it is a {type(xs)}'
    result = {}
    for path, value in xs.items():
        if sep is not None:
            path = path.split(sep)
        cursor = result
        for key in path[:-1]:
            if key not in cursor:
                cursor[key] = {}
            cursor = cursor[key]
        cursor[path[-1]] = value
    return result


def convert_config(model_config: ModelConfig) -> MolmoActConfig:
    """Convert config to HF-compatible config"""
    vision_backbone_cfg = model_config.vision_backbone
    vit_config = vision_backbone_cfg.vit
    llm_config = model_config.llm
    vit_image_model_type = vit_config.image_model_type

    molmoact_vit_config = MolmoActVitConfig(
        hidden_size=vit_config.image_emb_dim,
        intermediate_size=vit_config.image_mlp_dim,
        num_hidden_layers=vit_config.image_num_layers,
        num_attention_heads=vit_config.image_num_heads,
        num_key_value_heads=vit_config.image_num_key_value_heads,
        head_dim=vit_config.image_head_dim,
        hidden_act=vit_config.image_mlp_activations,
        layer_norm_eps=vit_config.image_norm_eps,
        image_default_input_size=vit_config.image_default_input_size,
        image_patch_size=vit_config.image_patch_size,
        image_num_pos=vit_config.image_num_pos,
        attention_dropout=0.0,
        residual_dropout=0.0,
        initializer_range=vit_config.initializer_range,
        float32_attention=vit_config.float32_attention,
        use_cls_token=True if vit_image_model_type=="openai" else False,
        patch_bias=False if vit_image_model_type=="openai" else True,
        pre_layernorm=True if vit_image_model_type=="openai" else False,
    )
    adapter_hidden_act = "silu" if llm_config.activation_type == "swiglu" else llm_config.activation_type
    adapter_intermediate_size = (
        llm_config.mlp_hidden_size if llm_config.mlp_hidden_size is not None
        else llm_config.mlp_ratio * llm_config.d_model
    ) // 2
    molmoact_adapter_config = MolmoActAdapterConfig(
        vit_layers=vision_backbone_cfg.vit_layers,
        hidden_size=vit_config.image_emb_dim,
        num_attention_heads=vit_config.image_num_heads,
        num_key_value_heads=vit_config.image_num_key_value_heads,
        head_dim=vit_config.image_head_dim,
        float32_attention=vit_config.float32_attention,
        attention_dropout=0.0,
        residual_dropout=0.0,
        hidden_act=adapter_hidden_act,
        intermediate_size=adapter_intermediate_size,
        text_hidden_size=llm_config.d_model,
        image_feature_dropout=vision_backbone_cfg.image_feature_dropout,
        initializer_range=llm_config.initializer_range,
        # pooling_mode="2x2_attention" if vit_image_model_type=="openai" else "indices",
        image_padding_embed=vision_backbone_cfg.image_padding_embed,
    )
    llm_head_dim = llm_config.d_model // llm_config.n_heads if llm_config.head_dim is None else llm_config.head_dim
    llm_intermediate_size = (
        llm_config.mlp_hidden_size if llm_config.mlp_hidden_size is not None
        else llm_config.mlp_ratio * llm_config.d_model
    ) // 2
    llm_hidden_act = "silu" if llm_config.activation_type == "swiglu" else llm_config.activation_type
    rope_scaling: Optional[Dict[str, Any]] = None
    if all(
        v is not None for v in
        [
            llm_config.rope_factor,
            llm_config.rope_low_freq_factor,
            llm_config.rope_high_freq_factor,
            llm_config.rope_original_max_position_embeddings
        ]
    ):
        rope_scaling = dict(
            rope_type=llm_config.rope_type,
            factor=llm_config.rope_factor,
            low_freq_factor=llm_config.rope_low_freq_factor,
            high_freq_factor=llm_config.rope_high_freq_factor,
            original_max_position_embeddings=llm_config.rope_original_max_position_embeddings,
        )
        
    molmoact_llm_config = MolmoActLlmConfig(
        hidden_size=llm_config.d_model,
        num_attention_heads=llm_config.n_heads,
        num_key_value_heads=llm_config.effective_n_kv_heads,
        head_dim=llm_head_dim,
        vocab_size=llm_config.embedding_size or llm_config.vocab_size,
        additional_vocab_size=llm_config.additional_vocab_size,
        qkv_bias=llm_config.qkv_bias,
        num_hidden_layers=llm_config.n_layers,
        intermediate_size=llm_intermediate_size,
        hidden_act=llm_hidden_act,
        embedding_dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
        max_position_embeddings=llm_config.max_position_embeddings or llm_config.max_sequence_length,
        rope_theta=llm_config.rope_theta,
        rope_scaling=rope_scaling,
        use_qk_norm=llm_config.attention_layer_norm,
        qk_norm_type=llm_config.attention_layer_norm_type,
        layer_norm_eps=llm_config.layer_norm_eps,
        norm_after=llm_config.norm_after,
        initializer_range=llm_config.initializer_range,
    )

    tokenizer = model_config.build_tokenizer()
    image_patch_id = tokenizer.image_patch_token_id

    molmoact_config = MolmoActConfig(
        vit_config=molmoact_vit_config,
        adapter_config=molmoact_adapter_config,
        llm_config=molmoact_llm_config,
        use_cache=True,
        tie_word_embeddings=llm_config.weight_tying,
        image_patch_id=image_patch_id,
        initializer_range=llm_config.initializer_range,
        n_action_bins=model_config.n_action_bins,
        norm_stats=model_config.norm_stats,
    )
    return molmoact_config


def convert_lm_head_and_prefix(state_dict: dict[str, Any], base_model_prefix: str, weight_tying: bool) -> dict[str, Any]:
    new_state_dict = {}
    for key, val in state_dict.items():
        if key == "transformer.ff_out.weight":
            new_key = "lm_head.weight"
        else:
            new_key = f"{base_model_prefix}.{key}"
        new_state_dict[new_key] = val
    
    if weight_tying:
        new_state_dict["lm_head.weight"] = state_dict["transformer.wte.embedding"]
    
    return new_state_dict


def convert_molmoact(state_dict: dict[str, Any], config: Union[MolmoActLlmConfig, MolmoActConfig], weight_tying: bool, text_only: bool) -> dict[str, Any]:
    if text_only:
        base_model_prefix = MolmoActForCausalLM.base_model_prefix
        state_dict = convert_lm_head_and_prefix(state_dict, base_model_prefix, weight_tying)
        new_state_dict = {}
        for key, val in state_dict.items():
            if 'vision_backbone' in key:
                continue
            key = key.replace("transformer.", "")
            new_state_dict[key] = val
        model_prefix = base_model_prefix
    else:
        base_model_prefix = MolmoActForActionReasoning.base_model_prefix
        new_state_dict = convert_lm_head_and_prefix(state_dict, base_model_prefix, weight_tying)
        model_prefix = f"{base_model_prefix}.transformer"
    qkv_bias = config.qkv_bias if isinstance(config, MolmoActLlmConfig) else config.llm_config.qkv_bias
    use_qk_norm = config.use_qk_norm if isinstance(config, MolmoActLlmConfig) else config.llm_config.use_qk_norm
    for layer_i in range(config.num_hidden_layers):
        prefix = f"{model_prefix}.blocks.{layer_i}"

        move_to_attn = ["att_proj.weight", "attn_out.weight"]
        if qkv_bias:
            move_to_attn.append("att_proj.bias")
        if use_qk_norm:
            move_to_attn += ["q_norm.weight", "k_norm.weight"]
        
        for k in move_to_attn:
            assert f"{prefix}.self_attn.{k}" not in new_state_dict
            new_state_dict[f"{prefix}.self_attn.{k}"] = new_state_dict.pop(f"{prefix}.{k}")
        
        move_to_mlp = ["ff_proj.weight", "ff_out.weight"]
        for k in move_to_mlp:
            assert f"{prefix}.mlp.{k}" not in new_state_dict
            new_state_dict[f"{prefix}.mlp.{k}"] = new_state_dict.pop(f"{prefix}.{k}")
    
    return new_state_dict


def convert_text_only_model(
    checkpoint_dir: str,
    model_config: ModelConfig,
    hf_config: MolmoActLlmConfig,
    precision: str,
) -> MolmoActForCausalLM:
    """Convert text only model to HF-compatible model"""
    with torch.device("meta"):
        model: Molmo = model_config.build_model()
        hf_model = MolmoActForCausalLM(hf_config)
    model.to_empty(device=torch.device("cpu"))
    hf_model.to_empty(device=torch.device("cpu"))

    load_model_state(checkpoint_dir, model)
    model.eval()
    if precision == "bf16":
        model = model.to(torch.bfloat16)
    elif precision == "fp32":
        model = model.to(torch.float32)
    else:
        raise ValueError(f"Invalid precision: {precision}")
    state_dict = model.state_dict()

    new_state_dict = convert_molmoact(state_dict, hf_config, model_config.llm.weight_tying, text_only=True)
    hf_model.eval()
    if precision == "bf16":
        hf_model = hf_model.to(torch.bfloat16)
    elif precision == "fp32":
        hf_model = hf_model.to(torch.float32)
    else:
        raise ValueError(f"Invalid precision: {precision}")
    hf_model.load_state_dict(new_state_dict)
    return hf_model


def convert_model(
    checkpoint_dir: str,
    model_config: ModelConfig,
    hf_config: MolmoActConfig,
    precision: str,
) -> MolmoActForActionReasoning:
    """Convert model to HF-compatible model"""
    with torch.device("meta"):
        model: Molmo = model_config.build_model()
        hf_model = MolmoActForActionReasoning(hf_config)
    model.to_empty(device=torch.device("cpu"))
    hf_model.to_empty(device=torch.device("cpu"))

    load_model_state(checkpoint_dir, model)
    model.eval()
    if precision == "bf16":
        model = model.to(torch.bfloat16)
    elif precision == "fp32":
        model = model.to(torch.float32)
    else:
        raise ValueError(f"Invalid precision: {precision}")
    state_dict = model.state_dict()

    new_state_dict = convert_molmoact(state_dict, hf_config, model_config.llm.weight_tying, text_only=False)
    hf_model.eval()
    if precision == "bf16":
        hf_model = hf_model.to(torch.bfloat16)
    elif precision == "fp32":
        hf_model = hf_model.to(torch.float32)
    else:
        raise ValueError(f"Invalid precision: {precision}")
    hf_model.load_state_dict(new_state_dict)
    return hf_model


def save(
    checkpoint_dir: str,
    output_dir: str,
    style: str,
    text_only: bool,
    override_tokenizer: bool,
    chat_template: str,
    precision: str,
    norm_stats_path: str,
) -> None:
    logger.info(f"Loading model config from {checkpoint_dir}")
    config_path = resource_path(checkpoint_dir, "config.yaml")
    model_config: ModelConfig = ModelConfig.load(config_path, key="model", validate_paths=False)

    if norm_stats_path is not None:
        p = Path(os.path.expandvars(os.path.expanduser(norm_stats_path)))
        if not p.exists():
            raise FileNotFoundError(f"norm_stats_path not found: {p}")
        norm_stats = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(norm_stats, dict):
            raise TypeError(f"norm_stats JSON root must be an object/dict, got {type(norm_stats).__name__}")
        model_config.norm_stats = norm_stats

    hf_config = convert_config(model_config)
    if text_only:
        logger.info(f"Save HF-compatible text only model config and checkpoint to {output_dir}")
        hf_model = convert_text_only_model(checkpoint_dir, model_config, hf_config.llm_config, precision)
    else:
        logger.info(f"Save HF-compatible model config and checkpoint to {output_dir}")
        hf_model = convert_model(checkpoint_dir, model_config, hf_config, precision)

    hf_model.save_pretrained(output_dir)

    gc.collect()

    model_yaml = os.path.join(output_dir, "model.yaml")
    if not os.path.exists(model_yaml):
        logger.warning(f"Save model training config to {model_yaml}")
        OmegaConf.save(model_config, model_yaml, resolve=True)

    model_file = os.path.join(output_dir, "modeling_molmoact.py")
    if not os.path.exists(model_file):
        logger.warning(f"Copy model file to {model_file} manually")
        shutil.copyfile(
            "olmo/hf_model/molmoact/modeling_molmoact.py",
            model_file,
        )

    with open(os.path.join(output_dir, "config.json")) as f:
        config = json.load(f)
    
    auto_map = config.get("auto_map", None)
    if auto_map is None:
        auto_map = {}
    if text_only and "AutoModelForCausalLM" not in auto_map:
        logger.warning("Add AutoModelForCausalLM to auto_map")
        auto_map["AutoModelForCausalLM"] = "modeling_molmoact.MolmoActForCausalLM"
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
    elif "AutoModelForImageTextToText" not in auto_map:
        logger.warning("Add AutoModelForImageTextToText to auto_map")
        auto_map["AutoModelForImageTextToText"] = "modeling_molmoact.MolmoActForActionReasoning"
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    tokenizer = model_config.build_tokenizer().tokenizer
    if not tokenizer.bos_token:
        tokenizer.bos_token = tokenizer.eos_token
        tokenizer.bos_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    data_formatter = model_config.data_formatter
    is_captioner = (
        data_formatter.message_format == "none" and
        data_formatter.system_prompt == "style_and_length" and
        data_formatter.prompt_templates == "none"
    )
    if chat_template == "original":
        assert text_only, "original chat template is only supported for text only model"
        tokenizer_chat_template = tokenizer.chat_template
    elif chat_template == "long_caption":
        assert is_captioner, "long_caption chat template is only supported for captioner"
        assert style == "long_caption", "long_caption chat template is only supported for long_caption style"
        default_inference_len = data_formatter.default_inference_len
        tokenizer_chat_template = f"long_caption {default_inference_len}:"
    elif chat_template == "demo":
        assert style == "demo", "demo chat template is only supported for demo style"
        tokenizer_chat_template = demo_chat_template
        data_formatter.prompt_templates = "uber_model"
        data_formatter.system_prompt = "none"
        data_formatter.message_format = "none"
    elif chat_template == "demo_role":
        tokenizer_chat_template = demo_role_chat_template
        data_formatter.prompt_templates = "uber_model"
        data_formatter.system_prompt = "demo_or_style"
        data_formatter.message_format = "role"
    else:
        raise ValueError(f"Invalid chat template: {chat_template}")
    tokenizer.chat_template = tokenizer_chat_template

    if text_only:
        logger.info(f"Save tokenizer to {output_dir}")
        tokenizer.save_pretrained(output_dir)
        processor = None
    else:
        logger.info(f"Save tokenizer and processor to {output_dir}")

        mm_cfg = model_config.mm_preprocessor
        vit_cfg = model_config.vision_backbone.vit

        processor = MolmoActProcessor(
            MolmoActImageProcessor(
                crop_mode=mm_cfg.crop_mode,
                resize_mode=vit_cfg.resize_mode,
                normalize_mode=vit_cfg.normalize,
                max_crops=mm_cfg.max_crops,
                max_multi_image_crops=mm_cfg.max_multi_image_crops,
                overlap_margins=mm_cfg.overlap_margins,
                base_image_input_size=vit_cfg.image_default_input_size,
                pad_value=vit_cfg.pad_value,
                image_patch_size=vit_cfg.image_patch_size,
                image_pooling_w=mm_cfg.pooling_w,
                image_pooling_h=mm_cfg.pooling_h,
                do_convert_rgb=True,
                do_pad=True,
            ),
            tokenizer,
            chat_template=tokenizer_chat_template,
            prompt_templates=data_formatter.prompt_templates,
            message_format=data_formatter.message_format,
            system_prompt=data_formatter.system_prompt,
            style=style,
            always_start_with_space=data_formatter.always_start_with_space,
            default_inference_len=data_formatter.default_inference_len,
            use_col_tokens=mm_cfg.use_col_tokens,
            image_padding_mask=model_config.vision_backbone.image_padding_embed is not None,
        )
        processor.save_pretrained(output_dir)

    if isinstance(tokenizer, Qwen2TokenizerFast) and override_tokenizer:
        logger.info("Override the Qwen2Tokenizer to enable adding BOS token to the beginning of the sequence")
        from .custom_tokenizer import Qwen2TokenizerFastWithBOS
        tokenizer = Qwen2TokenizerFastWithBOS.from_pretrained(output_dir)
        tokenizer.init_kwargs["tokenizer_class"] = "Qwen2TokenizerFastWithBOS"
        tokenizer.init_kwargs["auto_map"] = {
            "AutoTokenizer": [
                "custom_tokenizer.Qwen2TokenizerWithBOS", 
                "custom_tokenizer.Qwen2TokenizerFastWithBOS"
            ]
        }
        tokenizer.save_pretrained(output_dir)
        custom_tokenizer_file = os.path.join(output_dir, "custom_tokenizer.py")
        if not os.path.exists(custom_tokenizer_file):
            logger.warning(f"Copying custom tokenizer file to {custom_tokenizer_file} manually")
            shutil.copyfile(
                "olmo/hf_model/molmoact/custom_tokenizer.py",
                custom_tokenizer_file,
            )
    
    elif isinstance(tokenizer, GPT2TokenizerFast) and override_tokenizer:
        logger.info("Override the GPT2Tokenizer to enable adding BOS token to the beginning of the sequence")
        from .custom_tokenizer import GPT2TokenizerFastWithBOS
        tokenizer = GPT2TokenizerFastWithBOS.from_pretrained(output_dir)
        tokenizer.init_kwargs["tokenizer_class"] = "GPT2Tokenizer"
        tokenizer.init_kwargs["auto_map"] = {
            "AutoTokenizer": [
                "GPT2Tokenizer", 
                "custom_tokenizer.GPT2TokenizerFastWithBOS"
            ]
        }
        tokenizer.save_pretrained(output_dir)
        custom_tokenizer_file = os.path.join(output_dir, "custom_tokenizer.py")
        if not os.path.exists(custom_tokenizer_file):
            logger.warning(f"Copying custom tokenizer file to {custom_tokenizer_file} manually")
            shutil.copyfile(
                "olmo/hf_model/molmoact/custom_tokenizer.py",
                custom_tokenizer_file,
            )


    logger.info(f"Save generation config to {output_dir}")
    generation_config = GenerationConfig(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    generation_config.save_pretrained(output_dir)

    del hf_model, processor, tokenizer, generation_config
    gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description="Convert Molmo checkpoint to HuggingFace format."
    )
    parser.add_argument("--checkpoint_dir", help="Location of Molmo checkpoint.")
    parser.add_argument("--output_dir", help="Location to save the converted checkpoint.")
    parser.add_argument(
        "--style",
        type=str,
        help="task style to use for the model",
    )
    parser.add_argument(
        "--text_only",
        action="store_true",
        help="Only convert the text model",
    )
    parser.add_argument(
        "--override_tokenizer",
        action="store_true",
        help="Override the tokenizer with the custom tokenizer",
    )
    parser.add_argument(
        "--chat_template",
        choices=["demo_role", "demo", "long_caption", "original"],
        default="demo_role",
        help="Chat template to use for the model",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Precision to use for the model",
    )
    parser.add_argument(
        "--norm_stats_path",
        default=None,
        help="Path to the json file with dataset statistics",
    )
    args = parser.parse_args()

    prepare_cli_environment()
    save(
        args.checkpoint_dir,
        args.output_dir,
        args.style,
        args.text_only,
        args.override_tokenizer,
        args.chat_template,
        args.precision,
        args.norm_stats_path
    )


if __name__ == "__main__":
    main()