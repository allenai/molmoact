from typing import Any, Dict, Iterable, List, Optional, Tuple
from PIL import Image, ImageOps
from io import BytesIO
import os
import ast
import json
import datasets
import numpy as np

from olmo.data.dataset import Dataset

CAMERA_NAMES = ["image", "wrist_image"]

class CustomLeRobotDataset(Dataset):
    def __init__(self, path: str, high_res: bool = True, style: str = "demo", keep_in_memory: bool = False):
        self.dataset = datasets.load_from_disk(path, keep_in_memory=keep_in_memory)
        self.high_res = high_res
        self.style = style

    def __len__(self):
        return len(self.dataset)

    def _to_pil(self, x) -> Image.Image:
        """Best-effort conversion of common formats to PIL.Image."""
        # Already a PIL image
        if isinstance(x, Image.Image):
            return x

        # HF Image feature can return dicts with bytes/path/array
        if isinstance(x, dict):
            if "bytes" in x and isinstance(x["bytes"], (bytes, bytearray)):
                return Image.open(BytesIO(x["bytes"])).convert("RGB")
            if "path" in x and isinstance(x["path"], str) and os.path.exists(x["path"]):
                return Image.open(x["path"]).convert("RGB")
            if "array" in x and np is not None and isinstance(x["array"], np.ndarray):
                arr = x["array"]
                if arr.dtype != np.uint8:
                    arr = np.clip(arr * (255.0 if arr.dtype.kind == "f" else 1.0), 0, 255).astype(np.uint8)
                return Image.fromarray(arr)
            
        # Python list/tuple -> PIL.Image
        if isinstance(x, (list, tuple)):
            if np is None:
                raise TypeError("List inputs require NumPy to be available.")
            arr = np.array(x)

            # Case A: [[n*n], [n*n], [n*n]]  -> shape (3, n*n)
            if arr.ndim == 2 and arr.shape[0] == 3:
                n2 = arr.shape[1]
                n = int(round(n2 ** 0.5))
                if n * n != n2:
                    raise ValueError(f"Channel length {n2} is not a perfect square.")
                arr = arr.reshape(3, n, n).transpose(1, 2, 0)  # (H, W, 3)

            # Case B: flat grayscale vector [n*n] -> shape (n, n)
            elif arr.ndim == 1:
                n2 = arr.size
                n = int(round(n2 ** 0.5))
                if n * n != n2:
                    raise ValueError(f"Vector length {n2} is not a perfect square.")
                arr = arr.reshape(n, n)  # (H, W) grayscale

            # Case C: nested list already (H, W) or (H, W, C) or (C, H, W)
            elif arr.ndim == 3:
                # If channel-first (C, H, W) with common channel counts, move to (H, W, C)
                if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (3, 4):
                    arr = np.transpose(arr, (1, 2, 0))  # (H, W, C)
                # else assume already (H, W, C)
            elif arr.ndim == 2:
                # (H, W) grayscale — already fine
                pass
            else:
                raise TypeError(f"Unsupported list/tuple structure with shape {arr.shape}.")

            # Match numpy-array handling above
            if arr.dtype != np.uint8:
                arr = np.clip(arr * (255.0 if arr.dtype.kind == "f" else 1.0), 0, 255).astype(np.uint8)
            return Image.fromarray(arr)
            
        # Raw bytes / path
        if isinstance(x, (bytes, bytearray)):
            return Image.open(BytesIO(x)).convert("RGB")
        if isinstance(x, str) and os.path.exists(x):
            return Image.open(x).convert("RGB")

        # Numpy array
        if np is not None and isinstance(x, np.ndarray):
            arr = x
            if arr.dtype != np.uint8:
                arr = np.clip(arr * (255.0 if arr.dtype.kind == "f" else 1.0), 0, 255).astype(np.uint8)
            return Image.fromarray(arr)

        raise TypeError(f"Cannot convert type {type(x)} to PIL.Image")

    def resize_image(self, img: Image.Image, max_size: int = 378) -> Image.Image:
        """
        Resize `img` so that max(width, height) <= max_size while preserving aspect ratio.
        Does not upscale smaller images.
        """
        # Honor orientation tags and ensure RGB
        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size
        if w <= max_size and h <= max_size:
            return img  # no upscaling

        scale = min(max_size / float(w), max_size / float(h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def extract_images(self, example: Dict, high_res: bool = True, camera_names: List[str] = CAMERA_NAMES) -> List[Image.Image]:
        """
        Collect images from `example` for the provided camera names and return as a list of PIL.Image.
        Missing keys are skipped.
        """
        images = []
        for name in camera_names:
            if name in example and example[name] is not None:
                try:
                    img = self._to_pil(example[name]).convert("RGB")
                    if not high_res:
                        img = self.resize_image(img)
                    images.append(img)
                except Exception:
                    # Skip unconvertible entries rather than crashing; you can `raise` if you prefer.
                    continue
        return images

    def format_action_reasoning(self, task: str, depth: str, trace: str, action: str) -> Tuple:
        question = (
            f"The task is {task}. What is the action that the robot should take. "
            f"To figure out the action that the robot should take to {task}, let's think through it step by step. "
            "First, what is the depth map for the first image? "
            "Second, what is the trajectory of the end effector in the first image? "
            "Based on the depth map of the first image and the trajectory of the end effector in the first image, "
            "along with other images from different camera views as additional information, what is the action that the robot should take?"
        )

        answer = (
            f"The depth map of the image is {depth}. "
            f"The trajectory of the left end effector is {trace}. "
            f"The trajectory of the right end effector is {trace}. "
            f"Based on this information, the action that the robot should take is {action}."
        )

        annotation = None

        return question, answer, annotation

    def format_trajectory_conditioned(self, task: str, trace: str, action: str) -> Tuple:

        question = (
            f"The task is {task}. Notice that the trajectory of the end effector is annotated on the first image. "
            "Based on the trajectory annotated on the image, along with other images from different camera views as "
            "additional information, what is the action that the robot should take?"
        )

        answer = (
            f"Based on the image and the annotated trajectory of the end effector, "
            f"the action that the robot should take is {action}."
        )

        annotation = trace

        return question, answer, annotation

    def get(self, item, rng):
        example = self.dataset[item]

        task = example['language_instruction']
        depth = example['depth']
        trace = example['trace']
        action = json.loads(example['processed_action'])

        # gather all images
        image_out = self.extract_images(example, high_res=self.high_res)

        # randomly choose action reasoning or trajectory-conditioned
        r = rng.random_sample(1)[0]

        if r < 0.5:
            question, answer, annotation = self.format_action_reasoning(task, depth, trace, action["chunked_action"])
        else:
            question, answer, annotation = self.format_trajectory_conditioned(task, trace, action["chunked_action"])
        
        if annotation:
            annotation = ast.literal_eval(annotation)

        assert annotation is None or type(annotation) is list

        # return the final output dict
        out = dict(
            style=self.style,
            image=image_out,
            question=question,
            answers=answer,
            annotation=annotation,
        )

        return out


if __name__ == "__main__":

    ds = CustomLeRobotDataset(path="/weka/oe-training-default/hqfang/molmoact-debug/datasets/libero_processed")

    rng = np.random.RandomState(2) # 2 is reasoning, 3 is trajectory
    sample = ds.get(0, rng)
    
    print(sample)

    # imgs = sample["image"]
    # imgs[0].save("/weka/oe-training-default/hqfang/molmoact-debug/tests/img1.png")
    # imgs[1].save("/weka/oe-training-default/hqfang/molmoact-debug/tests/img2.png")
