import argparse
import os
import logging

import torch

from olmo.models.molmo.molmo import Molmo, MolmoConfig
from olmo.train.checkpointer import load_model_state, save_unsharded
from olmo.util import (
    prepare_cli_environment,
    resource_path
)

from olmo.train.trainer_config import TrainConfig

from peft import PeftModel


logger = logging.getLogger(__name__)


def convert_checkpoint(checkpoint_dir: str, lora_dir: str, output_dir: str) -> None:
    config_dir = lora_dir[:-5]
    logger.info(f"Loading model config from {config_dir}")
    config_path = resource_path(config_dir, "config.yaml")
    config: TrainConfig = TrainConfig.load(config_path)
    # model_config: MolmoConfig = MolmoConfig.load(config_path, key="model", validate_paths=False)
    model_config = config.model


    logger.info(f"Loading model checkpoint from {checkpoint_dir}")
    with torch.device("meta"):
        model: Molmo = model_config.build_model()
    model.to_empty(device=torch.device("cpu"))
    load_model_state(checkpoint_dir, model)

    logger.info(f"Merging...")
    lora_model = PeftModel.from_pretrained(model, lora_dir)
    # print(lora_model)
    # print(lora_model.base_model.model.transformer.blocks[0].attn_out.lora_B['default'].weight)
    merged = lora_model.merge_and_unload()

    logger.info(f"Saving model config and merged checkpoint to {output_dir}")
    save_unsharded(output_dir, model, None, config, True)

    logger.info(f"Completed")


def main():
    parser = argparse.ArgumentParser(
        description="Adds a config.json to the checkpoint directory, creates pytorch_model.bin, and save the toeknizer,"
        "making it easier to load weights as HF models."
    )
    parser.add_argument(
        "--base_dir",
        help="Location of Molmo checkpoint.",
    )
    parser.add_argument(
        "--lora_dir",
        help="Location of Molmo checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to save the converted checkpoint.",
    )
    args = parser.parse_args()
    prepare_cli_environment()
    convert_checkpoint(args.base_dir, args.lora_dir, args.output_dir)


if __name__ == "__main__":
    main()