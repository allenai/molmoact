import argparse
from collections import defaultdict
import numpy as np
import cv2
from PIL import Image
import torch
from experiments.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    save_rollout_video,
)
from experiments.libero.robot_utils import (
    DATE_TIME,
    get_image_resize_size,
    set_seed_everywhere,
)
import tqdm
import ast
from libero.libero import benchmark
from transformers import AutoProcessor, AutoModelForImageTextToText 
import math


def crop_and_resize_pil(img: Image.Image, crop_scale: float) -> Image.Image:
    """
    Center‐crop a PIL image to crop_scale of its area,
    then resize back to the ORIGINAL image size.
    """
    w, h = img.size
    # sqrt(crop_scale) to get relative side length
    rel = math.sqrt(crop_scale)
    cw, ch = int(w * rel), int(h * rel)
    left = (w - cw) // 2
    top  = (h - ch) // 2
    cropped = img.crop((left, top, left + cw, top + ch))
    # resize back to the original dimensions (w, h)
    return cropped.resize((w, h), Image.BILINEAR)


def center_crop_image(img: Image.Image) -> Image.Image:
    # fixed 0.9 area scale
    return crop_and_resize_pil(img, 0.9)


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Convert to numpy array if it's a list
    if isinstance(action, list):
        action = np.array(action)
    
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action

def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    # Convert to numpy array if it's a list
    if isinstance(action, list):
        action = np.array(action)
    
    action[..., -1] = action[..., -1] * -1.0
    return action

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

def scale_pt(self, pt, w, h):
    """
    Convert a point whose coordinates are in 0–255 space
    to image-pixel space (0‥w-1, 0‥h-1).
    """
    x, y = pt
    return (int(round(x / 255.0 * (w - 1))),
            int(round(y / 255.0 * (h - 1))))
    

def step(img, wrist_img, language_instruction, model, processor, unnorm_key):
    """
    Run the multimodal model to get a text, parse out the 8×7 action matrix,
    unnormalize, then temporally aggregate the first 6 DOFs (dims 0–5) while using
    the latest value for DOF 6. Return a single aggregated 7-D action vector and
    the annotated image.
    """
         
    image = Image.fromarray(img)
    wrist = Image.fromarray(wrist_img)
    image = center_crop_image(image)
    wrist = center_crop_image(wrist)
    imgs = [image, wrist]


    prompt = (
        f"The task is {language_instruction}. "
        "What is the action that the robot should take. "
        f"To figure out the action that the robot should take to {language_instruction}, "
        "let's think through it step by step. "
        "First, what is the depth map for the first image? "
        "Second, what is the trajectory of the end effector in the first image? "
        "Based on the depth map of the first image and the trajectory of the end effector in the first image, "
        "along with other images from different camera views as additional information, "
        "what is the action that the robot should take?"
    )
    
        
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
        
    inputs = processor(
        images=[imgs],
        text=text,
        padding=True,
        return_tensors="pt",
    )


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


    depth = model.parse_depth(generated_text)
    print(f"generated depth perception tokens: {depth}")
    
    trace = model.parse_trace(generated_text)
    print(f"generated visual reasoning trace: {trace}")


    action = model.parse_action(generated_text, unnorm_key=unnorm_key)
    print(f"generated action: {action}")

    if (
        action is None
        or (isinstance(action, (list, tuple)) and len(action) == 0)
        or (isinstance(action, np.ndarray) and action.size == 0)
    ):
        raise ValueError("parse_action produced no action (None/empty).")
    annotated = np.array(img.copy())



    return action, annotated, trace



# @draccus.wrap()
def eval_libero(args, processor, model, task_suite_name, checkpoint, seed, model_family, num_trials_per_task, num_steps_wait) -> None:

    set_seed_everywhere(seed)



    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    print(f"Task suite: {task_suite_name}")


    # Get expected image dimensions
    resize_size = get_image_resize_size()


    # Start evaluation
    total_episodes, total_successes = 0, 0
    for _ in tqdm.tqdm(range(1)):
        # Get task
        task_id = args.task_id
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, model_family, resolution=256)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(num_trials_per_task)):
            last_gripper_state = -1
       
       
         
            print(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            if task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
                unnorm_key = "libero_spatial_no_noops_modified"
                print(f"Max steps: {max_steps}")
            elif task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
                unnorm_key = "libero_object_no_noops_modified"
                print(f"Max steps: {max_steps}")
            elif task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
                unnorm_key = "libero_goal_no_noops_modified"
                print(f"Max steps: {max_steps}")
            elif task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
                unnorm_key = "libero_10_no_noops_modified"
                print(f"Max steps: {max_steps}")
            elif task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps
                print(f"Max steps: {max_steps}")

            print(f"Starting episode {task_episodes+1}...")
   
            
            timestep = 0
            outer_done = False
         
            while t < max_steps + num_steps_wait and not outer_done:
                # 1) Warm-up: ignore its 'done'
                if t < num_steps_wait:
                    obs, _, _, _ = env.step(get_libero_dummy_action(model_family))
                    t += 1
                    continue

                # 2) step action
                img = get_libero_image(obs, resize_size)
                wrist_img = get_libero_wrist_image(obs, resize_size)
                wait = False
                try:
                    action_matrix, annotated_image, traj = step(img, wrist_img, task_description, model, processor, unnorm_key)
                except Exception as e:
                    print(e)
                    action_matrix = np.zeros((1, 7), dtype=float)
                    action_matrix[:, -1] = last_gripper_state
                    annotated_image = img
                    wait = True
                    print(f"error: {e}")

                if annotated_image is None:
                    annotated_image = img
                replay_images.append(annotated_image)


                action_num = 0
                # 3) Execute each of the N actions until done
                for single_action in action_matrix:
                    
                    if isinstance(single_action, str):
                        single_action = ast.literal_eval(single_action)
                    single_action = normalize_gripper_action(single_action, binarize=True)
                    single_action = invert_gripper_action(single_action)
                    obs, _, done, _ = env.step(single_action)
                    visualize = get_libero_image(obs, resize_size)

                    try:
                        visualize_annotated = np.array(visualize.copy())
                        if traj is not None:
                            for i in range(len(traj) - 1):
                                p1 = tuple(map(int, traj[i]))
                                p2 = tuple(map(int, traj[i + 1]))
                                cv2.line(visualize_annotated, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
                    except Exception as e:
                        print(f"step() trajectory annotation failed, returning unannotated image: {e}")
                        visualize_annotated = np.array(visualize)

                    replay_images.append(visualize_annotated)

                    action_num += 1
   
                    if done:
                        outer_done = True
                        break
                
                # 4) Advance your loop counters
                timestep += 1
                print(f"wait: {wait}")
                if wait:
                    action_num = 1
                    
                print(f"action num: {action_num}")
                t += action_num


                if done:
                    task_successes += 1
                    total_successes += 1
                    break


            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, checkpoint=checkpoint, task=task_suite_name
            )

            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

    




def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",     type=str, required=True)
    p.add_argument("--task_id",  type=int, required=False, default=None, 
                   help="Specific task ID (0-9). If not provided, will run all task IDs 0-9 for the specified task type.")
    p.add_argument("--checkpoint", type=str, required=True)
    return p.parse_args()

def main():
    args = parse_args()
    task_suite_name = f"libero_{args.task}"
    ckpt       = args.checkpoint
    seed = 7

    set_seed_everywhere(seed)
    


    processor = AutoProcessor.from_pretrained(
        ckpt,
        trust_remote_code=True,
        torch_dtype="bfloat16",
        device_map="auto",
        padding_side="left",
    )

    model = AutoModelForImageTextToText.from_pretrained(
        ckpt,
        trust_remote_code=True,
        torch_dtype="bfloat16",
        device_map="auto",
    )

    model_family = ckpt.replace("/", "-")
    num_trials_per_task = 50
    num_steps_wait = 10  
    
    if args.task_id is not None:
        print(f"Running single task ID: {args.task_id}")
        eval_libero(args, processor, model, task_suite_name, ckpt, seed, model_family, num_trials_per_task, num_steps_wait)
    else:
        # Run all task IDs 0-9 for the specified task type
        print(f"Running all task IDs 0-9 for task type: {args.task}")
        for task_id in range(10):
            print(f"\n{'='*50}")
            print(f"Running task ID: {task_id}")
            print(f"{'='*50}")
            args.task_id = task_id
            eval_libero(args, processor, model, task_suite_name, ckpt, seed, model_family, num_trials_per_task, num_steps_wait)

if __name__ == "__main__":
    main()
