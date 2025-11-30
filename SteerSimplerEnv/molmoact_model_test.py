from typing import Optional, Sequence
import os
import re
import ast
import json
import numpy as np
import torch
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoProcessor
from transforms3d.euler import euler2axangle
from vllm import LLM, ModelRegistry
from vllm.model_executor.models.registry import _MULTIMODAL_MODELS
from vllm.sampling_params import SamplingParams
from simpler_env.policies.molmoact.molmoact import MolmoActForActionReasoning, MolmoActParser
ModelRegistry.register_model("MolmoActForActionReasoning", MolmoActForActionReasoning)
_MULTIMODAL_MODELS["MolmoActForActionReasoning"] = ("molmoact", "MolmoActForActionReasoning")



class MolmoActInferenceTEST:
    def __init__(
        self,
        saved_model_path: str = "",
        unnorm_key: Optional[str] = None,
        policy_setup: str = "google_robot",
        image_size: list[int] = [256, 256],
        action_scale: float = 1.0,

    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if policy_setup == "widowx_bridge":
            unnorm_key = "bridge_orig" if unnorm_key is None else unnorm_key
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == "google_robot":
            unnorm_key = "fractal20220817_data" if unnorm_key is None else unnorm_key
            self.sticky_gripper_num_repeat = 15
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported for octo models."
            )
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        self.processor = AutoProcessor.from_pretrained(
            saved_model_path,
            trust_remote_code=True,
            torch_dtype="float16",
            device_map="auto",
            padding_side="left",
        )

        # self.processor = AutoProcessor.from_pretrained(
        #     saved_model_path,
        #     trust_remote_code=True,
        #     torch_dtype="bfloat16",
        #     device_map="auto",
        #     padding_side="left",
        # )

        # self.model = LLM(
        #     model=saved_model_path,
        #     trust_remote_code=True,
        #     tensor_parallel_size=torch.cuda.device_count(),
        #     gpu_memory_utilization=0.95,
        #     dtype="bfloat16",
        # )
        self.model = LLM(
            model=saved_model_path,
            trust_remote_code=True,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.95,
            dtype="float16",
        )

        self.sampling_params = SamplingParams(
            max_tokens=256,
            temperature=0
        )

        self.parser = MolmoActParser.from_pretrained(saved_model_path)


        self.image_size = image_size
        self.action_scale = action_scale


        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task = None
        self.task_description = None


        stats_path = './simpler_env/policies/molmoact/dataset_statistics.json'
        

        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Dataset statistics file not found at {stats_path}")
        with open(stats_path, 'r') as f:
            self.dataset_stats = json.load(f)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None


    def scale_pt(self, pt, w, h):
        """
        Convert a point whose coordinates are in 0–255 space
        to image-pixel space (0‥w-1, 0‥h-1).
        """
        x, y = pt
        return (int(round(x / 255.0 * (w - 1))),
                int(round(y / 255.0 * (h - 1))))
    
    def scale_pt2(self, pt, orig_w, orig_h, new_w, new_h):
        """
        Convert a point whose coordinates are in 0–255 space
        to image-pixel space (0‥w-1, 0‥h-1).
        """
        x, y = pt
        # keep float math until the very end, then round & cast to int
        return (int(round(x / orig_w * (new_w - 1))),
                int(round(y / orig_h * (new_h - 1))))

    def draw_user_trajectory(self, image: np.ndarray, trajectory: Optional[Sequence[Sequence[int]]], orig_w: int, orig_h: int, new_h: int, new_w: int) -> np.ndarray:
        """
        Draw user-provided trajectory on the image.
        """
        annotated_image = image.copy()
        if trajectory:
            print(trajectory)
            for i in range(len(trajectory) - 1):
                pt1 = tuple(map(int, trajectory[i]))
                pt2 = tuple(map(int, trajectory[i + 1]))
                pt1 = self.scale_pt2(pt1, orig_w, orig_h, new_w, new_h)
                pt2 = self.scale_pt2(pt2, orig_w, orig_h, new_w, new_h)
                cv.line(annotated_image, pt1, pt2, (0, 255, 255), thickness=3, lineType=cv.LINE_AA)  # 蓝色轨迹

        return annotated_image
    
    def step(
        self, image: np.ndarray, task_description: Optional[str] = None, user_trajectory: Optional[list] = None,*args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)
    
        assert image.dtype == np.uint8
        orig_h, orig_w = image.shape[:2]
        image = self._resize_image(image)
        new_h, new_w = image.shape[:2]
        # img = Image.fromarray(image)
        image_for_model = self.draw_user_trajectory(image, user_trajectory, orig_w, orig_h, new_h, new_w)
        img = Image.fromarray(image_for_model)
        language_instruction = self.task_description
        # print(language_instruction)
        # raise KeyboardInterrupt()
        if user_trajectory:
            prompt = (
                f"The task is {task_description}. "
                "Notice that the trajectory of the end effector is annotated on the image. "
                "Based on the trajectory annotated on the image, what is the action that the robot should take?"
            )
        else:
            prompt = (
                f"The task is {language_instruction}. "
                "What is the action that the robot should take. "
                f"To figure out the action that the robot should take to {language_instruction}, "
                "let's think through it step by step. "
                "First, what is the depth map for this image? "
                "Second, what is the trajectory of the end effector? "
                "Based on the depth map of the image and the trajectory of the end effector, "
                "what is the action that the robot should take?"
            )
        text = self.processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [dict(type="text", text=prompt)]
                }
            ], 
            tokenize=False, 
            add_generation_prompt=True,
        )

        inputs = [
            {
                "prompt": text,
                "multi_modal_data": {
                    "image": [img]
                },
            },
        ]

        # plt.figure(figsize=(12, 8))
        # plt.imshow(np.array(img))
        # plt.savefig("DEBUG_model_input_plot.png", bbox_inches='tight', dpi=150)
        # plt.show()

        outputs = self.model.generate(inputs, sampling_params=self.sampling_params)
        generated_text = outputs[0].outputs[0].text
        print(generated_text)

        # depth = self.parser.parse_depth(generated_text)
        # print(f"Depth: {depth}")

        # trace = self.parser.parse_trace(generated_text)
        # print(f"Trace: {trace}")

        action = self.parser.parse_action(generated_text, unnorm_key="fractal20220817_data")
        print(f"Action: {action}")
        
        
        annotated_image = image_for_model.copy()

        if user_trajectory:
            trajectory = user_trajectory
        else:
            trajectory = None



        unnormalized_action = action[0] if isinstance(action, list) and len(action) == 1 else action
        unnormalized_action = np.array(unnormalized_action, dtype=np.float64)
        
        if user_trajectory:
            print ()

        else:
            if "The trajectory of the end effector is" in generated_text:
                try:
                    traj_part = generated_text.split("The trajectory of the end effector is")[-1]
                    traj_part = traj_part.split("Based on")[0].strip()
                    traj_str = traj_part.rstrip('.').strip()
                    trajectory = ast.literal_eval(traj_str)
                    traj_digits = []
                    for num in trajectory:
                        for digit in str(num):
                            traj_digits.append(digit)

                    for i in range(len(trajectory) - 1):
                        pt1 = tuple(map(int, trajectory[i]))
                        pt2 = tuple(map(int, trajectory[i + 1]))
                        pt1 = self.scale_pt(pt1, new_w, new_h)
                        pt2 = self.scale_pt(pt2, new_w, new_h)
                        cv.line(annotated_image, pt1, pt2, (0, 255, 255), thickness=2, lineType=cv.LINE_AA)
    
                except Exception as e:
                    print("Failed to parse trajectory:", e)
            else:
                print("No trajectory found in generated text.")
            
            
        (h, w) = annotated_image.shape[:2]
        raw_action = {
            "world_vector": unnormalized_action[:3],
            "rotation_delta": unnormalized_action[3:6],
            "open_gripper": unnormalized_action[6:7],  # assuming the last value is gripper action
        }
        annotated_image = cv.resize(annotated_image, (orig_w, orig_h), interpolation=cv.INTER_LINEAR)
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale
    
        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            self.previous_gripper_action = current_gripper_action
    
            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action
    
            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action
    
            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0
    
            action["gripper"] = relative_gripper_action
    
        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
    
        action["terminate_episode"] = np.array([0.0])
    
        return raw_action, action, annotated_image

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image
    
   
        
    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        # images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]
    
        img_strip = np.concatenate(np.array(images[::3]), axis=1)
    
        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])
    
        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")
    
        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)












