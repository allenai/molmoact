import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


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

    # load the model
    model = AutoModelForImageTextToText.from_pretrained(
        ckpt,
        trust_remote_code=True,
        torch_dtype="bfloat16",
        device_map="auto",
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

    # apply chat template
    text = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [dict(type="text", text=prompt)]
            }
        ], 
        tokenize=False, 
        add_generation_prompt=True,
    )

    # load images from paths, sequence matters
    img_paths = [Path(p) for p in args.images]
    imgs = [Image.open(p).convert("RGB") for p in img_paths]

    # process the image and text
    inputs = processor(
        images=[imgs],
        text=text,
        padding=True,
        return_tensors="pt",
    )

    # move inputs to the correct device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # generate output
    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            generated_ids = model.generate(**inputs, max_new_tokens=512)

    # only get generated tokens; decode them to text
    generated_tokens = generated_ids[:, inputs['input_ids'].size(1):]
    generated_text = processor.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # print the generated text
    print(f"generated text: {generated_text}")

    # parse out all depth perception tokens
    depth = model.parse_depth(generated_text)
    print(f"generated depth perception tokens: {depth}")

    # parse out all visual reasoning traces
    trace = model.parse_trace(generated_text)
    print(f"generated visual reasoning trace: {trace}")

    # parse out all actions, unnormalizing with the provided key
    action = model.parse_action(generated_text, unnorm_key=args.unnorm_key)
    print(f"generated action: {action}")


if __name__ == "__main__":
    main()