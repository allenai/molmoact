import argparse
from PIL import Image
from pathlib import Path

import torch
from vllm import LLM, ModelRegistry
from vllm.model_executor.models.registry import _MULTIMODAL_MODELS
from vllm.sampling_params import SamplingParams
from olmo.vllm.molmoact.molmoact import MolmoActForActionReasoning, MolmoActParser
ModelRegistry.register_model("MolmoActForActionReasoning", MolmoActForActionReasoning)
_MULTIMODAL_MODELS["MolmoActForActionReasoning"] = ("molmoact", "MolmoActForActionReasoning")

from transformers import AutoProcessor


def parse_args():
    ap = argparse.ArgumentParser(description="Run MolmoAct with custom checkpoint, images, and instruction.")
    ap.add_argument("--checkpoint_dir", required=True,
                    help="HuggingFace repo id or local path to the checkpoint directory.")
    ap.add_argument("--images", nargs="+", required=True,
                    help="List of image paths (space-separated). Use at least one; multiple views allowed.")
    ap.add_argument("--instruction", required=True,
                    help="Task instruction string, e.g., 'close the box'.")
    ap.add_argument("--unnorm_key", required=True,
                    help="Key for action unnormalization, e.g., 'molmoact'.")
    return ap.parse_args()


def apply_chat_template(processor: AutoProcessor, text: str):
    messages = [
        {
            "role": "user",
            "content": [dict(type="text", text=text)]
        }
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    return prompt


def main():
    args = parse_args()
    ckpt = args.checkpoint_dir

    # load the processor
    processor = AutoProcessor.from_pretrained(
        ckpt,
        trust_remote_code=True,
        torch_dtype="bfloat16",
        device_map="auto",
        padding_side="left",
    )

    sampling_params = SamplingParams(
        max_tokens=256,
        temperature=0
    )

    # define an extra parser to parse depth/trace/action
    # as vLLM doesnt allow adding this to model class
    parser = MolmoActParser.from_pretrained(ckpt)

    # load the model
    llm = LLM(
        model=ckpt,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.95,
        dtype="bfloat16",
    )

    # task instruction
    instruction = args.instruction

    # strictly follow this reasoning prompt
    # modify anything if there is a change in your training prompt
    prompt = (
        f"The task is {instruction}. "
        "What is the action that the robot should take. "
        f"To figure out the action that the robot should take to {instruction}, "
        "let's think through it step by step. "
        "First, what is the depth map for the first image? "
        "Second, what is the trajectory of the end effector in the first image? "
        "Based on the depth map of the first image and the trajectory of the end effector in the first image, "
        "along with other images from different camera views as additional information, "
        "what is the action that the robot should take?"
    )

    # load images from paths, sequence matters
    img_paths = [Path(p) for p in args.images]
    imgs = [Image.open(p).convert("RGB") for p in img_paths]


    inputs = [
        {
            "prompt": apply_chat_template(processor, prompt),
            "multi_modal_data": {
                "image": [imgs]
            },
        },
    ]

    outputs = llm.generate(inputs, sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text

    # print the generated text
    print(f"generated text: {generated_text}")

    # parse out all depth perception tokens
    depth = parser.parse_depth(generated_text)
    print(f"generated depth perception tokens: {depth}")

    # parse out all visual reasoning traces
    trace = parser.parse_trace(generated_text)
    print(f"generated visual reasoning trace: {trace}")

    # parse out all actions, unnormalizing with key of "molmoact"
    action = parser.parse_action(generated_text, unnorm_key=args.unnorm_key)
    print(f"generated action: {action}")


if __name__ == "__main__":
    main()