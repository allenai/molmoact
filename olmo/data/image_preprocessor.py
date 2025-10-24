import dataclasses
import ast
import math
import warnings
import os
import logging
import multiprocessing
from io import BytesIO
from tqdm import tqdm
from typing import List, Optional, Union, Any, Tuple, Iterable, Set
import cv2
import PIL
from PIL import ImageFile, ImageOps, Image

from olmo.data.dataset import DATA_HOME
from olmo.io import get_bytes_range, write_file, file_exists


def setup_pil():
    PIL.Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True


import numpy as np
import torch
import torchvision.transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import convert_image_dtype

from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ImageInput,
)

from olmo.models.molmo.data_formatter import DataFormatter

DEFAULT_IMAGE_PATH = "/weka/oe-training-default/mm-olmo/torch_datasets"


def get_image_collage(frames: np.ndarray, frame_size: int = 128) -> np.ndarray:
    """
    Creates a collage of frames arranged in a Nx4 grid in reading order.
    Each frame is resized to 224x224 while maintaining aspect ratio.

    Args:
        frames: numpy array of shape (num_frames, height, width, channels)
        frame_size: size of each frame in the collage (default: 224)

    Returns:
        collage: numpy array of shape (N*224, 896, 3) where N is ceil(num_frames/4)
    """
    num_frames = len(frames)
    num_rows = (num_frames + 3) // 4  # Ceiling division for number of columns
    # Create black canvas of appropriate size
    canvas = np.zeros((num_rows * frame_size, 4 * frame_size, 3), dtype=np.uint8)

    for idx, frame in enumerate(frames):
        row = idx // 4
        col = idx % 4

        largest_dim = max(frame.shape[0], frame.shape[1])
        square_frame = np.zeros((largest_dim, largest_dim, 3), dtype=np.uint8)

        # Center the frame in the square
        square_frame[
            (largest_dim - frame.shape[0]) // 2:(largest_dim - frame.shape[0]) // 2 + frame.shape[0],
            (largest_dim - frame.shape[1]) // 2:(largest_dim - frame.shape[1]) // 2 + frame.shape[1]
        ] = frame

        resized = Image.fromarray(square_frame).resize((frame_size, frame_size), Image.Resampling.BILINEAR)
        resized = np.array(resized)

        canvas[row * frame_size:(row + 1) * frame_size, col * frame_size:(col + 1) * frame_size] = resized

    return canvas


def load_pil_image(image_path: str) -> PIL.Image.Image:
    setup_pil()  # Call here so the setting is applied in multi-processing contexts
    # This a bit of hack to handle cases where the image path was hard-coded
    # into the dataset to the weka path
    if DATA_HOME != DEFAULT_IMAGE_PATH and DEFAULT_IMAGE_PATH in image_path:
        image_path = image_path.replace(DEFAULT_IMAGE_PATH, DATA_HOME)
    # Ignore image loading warning
    with warnings.catch_warnings(record=True) as w:
        if image_path.startswith("gs://"):
            image_bytes = get_bytes_range(image_path, 0, None)
            return PIL.Image.open(BytesIO(image_bytes))
        else:
            return PIL.Image.open(image_path)


def load_image(image_path, annotation_list):
    setup_pil()  # Call here so the setting is applied in multi-processing contexts
    if isinstance(image_path, PIL.Image.Image):
        # Avoid annoying palette transparency warnings filling up the logs
        with warnings.catch_warnings(record=True) as w:
            image = image_path.convert("RGB")
            
        try:
            image = ImageOps.exif_transpose(image)
        except Exception as e:
            pass
        

        annotated_image = np.array(image)
        w, h = image.size

        # Scale factors: map 0→0, 255→(w-1)/(h-1) so 255 lands on last valid pixel index.
        sx = (w - 1) / 255.0
        sy = (h - 1) / 255.0
        scale = np.array([sx, sy], dtype=np.float32)

        for annotation in annotation_list:
            if annotation is not None:
                try:
                    coords = annotation

                    # to array float, scale, round->int
                    pts = np.asarray(coords, dtype=np.float32) * scale
                    pts = np.rint(pts).astype(int)  # shape (N,2)

                    # guard against weird shapes
                    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
                        continue

                    # clip just in case (avoid out-of-bounds)
                    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

                    # Convert PIL image to a numpy array for OpenCV (note: array is in RGB)
                    # Draw lines connecting consecutive points in yellow (BGR: (0,255,255))
                    for i in range(len(pts) - 1):
                        pt1 = tuple(pts[i])
                        pt2 = tuple(pts[i + 1])
                        cv2.line(annotated_image, pt1, pt2, (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

                    # # ——— SANITY CHECK DUMP ———
                    # from datetime import datetime
                    # annotated_image_pil = Image.fromarray(annotated_image)
                    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    # out_dir = "/weka/oe-training-default/jiafeid/sanitycheck"
                    # os.makedirs(out_dir, exist_ok=True)
                    # out_path = os.path.join(out_dir, f"{timestamp}_line.png")
                    # annotated_image_pil.save(out_path)
                    # print(f"[sanity] wrote annotated image to {out_path}")
                    # # ————————————————

                except Exception as e:
                    print(f"Error processing annotation: {e}")
        
        image = Image.fromarray(annotated_image)
                
        return np.array(image)
    elif isinstance(image_path, np.ndarray):
        assert len(image_path.shape) == 3, "Image should have 3 dimensions"
        assert image_path.shape[2] == 3, "Image should have 3 channels"
        assert image_path.dtype == np.uint8, "Image should have uint8 type"
        return image_path
    else:
        # This a bit of hack to handle cases where the image path was hard-coded
        # into the dataset to the weka path
        if DATA_HOME != DEFAULT_IMAGE_PATH and DEFAULT_IMAGE_PATH in image_path:
            image_path = image_path.replace(DEFAULT_IMAGE_PATH, DATA_HOME)

        # Ignore image loading warning
        with warnings.catch_warnings(record=True) as w:
            if image_path.startswith("gs://"):
                image_bytes = get_bytes_range(image_path, 0, None)
                with PIL.Image.open(BytesIO(image_bytes)) as image:
                    return load_image(image, annotation_list)
            else:
                with PIL.Image.open(image_path) as image:
                    return load_image(image, annotation_list)


def save_image(args) -> Tuple[str, bool]:
    image: PIL.Image.Image = args[0]
    filename: str = args[1]

    assert isinstance(image, PIL.PngImagePlugin.PngImageFile), \
        f"{filename}: Expected a PIL image, got {type(image)}"

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    write_file(
        os.path.dirname(filename),
        os.path.basename(filename),
        image_bytes,
        save_overwrite=True
    )
    return filename, file_exists(filename)


def save_images(
    pil_images: Iterable[PIL.Image.Image],
    filenames: List[str],
    n_procs: int = 1,
) -> Set[str]:
    if n_procs != 1:
        def _iter():
            with multiprocessing.Pool(processes=n_procs, initializer=setup_pil) as pool:
                for val in pool.imap_unordered(save_image, zip(pil_images, filenames)):
                    yield val
    else:
        setup_pil()
        def _iter():
            for val in zip(pil_images, filenames):
                yield save_image(val)
    
    pbar = tqdm(total=len(filenames), desc="Saving images")
    saved_images = set()
    for val in _iter():
        if val[1]:
            saved_images.add(val[0])
        pbar.update(1)
    pbar.close()
    logging.info(
        f"Saved {len(saved_images)}/{len(filenames)} ({len(saved_images)/len(filenames) * 100:0.2f}%) images")
    return saved_images


def resize_and_pad(
    image,
    desired_output_size,
    is_training=False,
    resize_method="torch-bilinear",
    pad_value=0,
    rng=np.random
):
    """Resize an image while padding to preserve uts aspect ratio."""
    desired_height, desired_width = desired_output_size
    height, width = image.shape[:2]

    # Cast into float32 since the training code did this in float32 and it (very rarely) effects
    # the results after rounding.
    image_scale_y = np.array(desired_height, np.float32) / np.array(height, np.float32)
    image_scale_x = np.array(desired_width, np.float32) / np.array(width, np.float32)
    image_scale = min(image_scale_x, image_scale_y)
    scaled_height = int(np.array(height, np.float32) * image_scale)
    scaled_width = int(np.array(width, np.float32) * image_scale)

    if resize_method in ["tensorflow", "tensorflow-random"]:
        # This how the original training code did resizing, it can produce slightly different
        # results then using torch resize so we keep it just in case
        import tensorflow as tf
        if resize_method == "tensorflow-random" and is_training:
            resize_methods = sorted([k for k in tf.image.ResizeMethod.__dict__.keys() if k.isupper()])
            mode = resize_methods[rng.randint(len(resize_methods))]
            mode = getattr(tf.image.ResizeMethod, mode)
        else:
            mode = tf.image.ResizeMethod.BILINEAR
        image = tf.image.convert_image_dtype(tf.constant(image), dtype=tf.float32)
        image = tf.image.resize(
            image,
            [scaled_height, scaled_width],
            method=mode,
            antialias=True,
        )
        image = tf.clip_by_value(image, 0.0, 1.0)
        image = image.numpy()
    elif resize_method in ["torch-bilinear", "torch-rng"]:
        image = torch.permute(torch.from_numpy(image), [2, 0, 1])
        image = convert_image_dtype(image)  # resize in float32 to match the training code
        if resize_method == "torch-rng"  and is_training:
            options = [InterpolationMode.BILINEAR, InterpolationMode.NEAREST_EXACT,
                       InterpolationMode.BICUBIC, InterpolationMode.LANCZOS, InterpolationMode.HAMMING]
            mode = options[rng.randint(len(options))]
        else:
            mode = InterpolationMode.BILINEAR
        image = torchvision.transforms.Resize([scaled_height, scaled_width], mode, antialias=True)(image)
        image = torch.clip(image, 0.0, 1.0)
        image = torch.permute(image, [1, 2, 0]).numpy()
    else:
        raise NotImplementedError(resize_method)

    top_pad = (desired_height - scaled_height) // 2
    left_pad = (desired_width - scaled_width) // 2
    padding = [
        [top_pad, desired_height - scaled_height - top_pad],
        [left_pad, desired_width - scaled_width - left_pad],
        [0, 0]
    ]
    image_mask = np.pad(np.ones_like(image[:, :, 0], dtype=bool), padding[:2])
    image = np.pad(image, padding, constant_values=pad_value)
    return image, image_mask


def metaclip_resize(image, desired_output_size):
    image = torch.permute(torch.from_numpy(image), [2, 0, 1])
    if torch.is_floating_point(image):
        image = torchvision.transforms.Resize(
            desired_output_size, InterpolationMode.BICUBIC, antialias=True)(image)
        image = torch.clip(image, 0.0, 1.0)
    else:
        assert image.dtype == torch.uint8, "Expected float images or uint8 images, but got {}".format(image.dtype)
        image = torchvision.transforms.Resize(
            desired_output_size, InterpolationMode.BICUBIC, antialias=True)(image)
        image = image.to(torch.float32)
        image = torch.clip(image, 0, 255)
        image = image / 255.0
    resized = torch.permute(image, [1, 2, 0]).numpy()
    image_mask = np.ones_like(resized[:, :, 0], dtype=np.bool_)
    return resized, image_mask


def siglip_resize_and_pad(
    image: np.ndarray,
    desired_output_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    # by default, image is a single image
    image = torch.permute(torch.from_numpy(image), [2, 0, 1])
    dtype = image.dtype
    if torch.is_floating_point(image):
        in_min = 0.0
        in_max = 1.0
        resized = torchvision.transforms.Resize(
            desired_output_size,
            InterpolationMode.BILINEAR,
            antialias=False,
        )(image)
        resized = torch.clip(resized, 0.0, 1.0).to(dtype)
    else:
        assert image.dtype == torch.uint8, "SigLIP expects float images or uint8 images, but got {}".format(image.dtype)
        in_min = 0.0
        in_max = 255.0
        resized = torchvision.transforms.Resize(
            desired_output_size,
            InterpolationMode.BILINEAR,
            antialias=False,
        )(image)
        resized = torch.clip(resized, 0, 255).to(dtype)

    resized = resized.to(torch.float32)
    resized = (resized - in_min) / (in_max - in_min)

    resized = torch.permute(resized, [1, 2, 0]).numpy()
    image_mask = np.ones_like(resized[:, :, 0], dtype=np.bool_)

    return resized, image_mask


def dino_resize_and_pad(
    image: np.ndarray,
    desired_output_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    image = torch.permute(torch.from_numpy(image), [2, 0, 1])
    dtype = image.dtype
    if torch.is_floating_point(image):
        resized = torchvision.transforms.Resize(
            desired_output_size,
            InterpolationMode.BICUBIC,
            antialias=True,
        )(image)
        resized = torch.clip(resized, 0.0, 1.0).to(torch.float32)
    else:
        assert image.dtype == torch.uint8, "DINOv2 expects float images or uint8 images, but got {}".format(image.dtype)
        resized = torchvision.transforms.Resize(
            desired_output_size,
            InterpolationMode.BICUBIC,
            antialias=True,
        )(image)
        resized = torch.clip(resized, 0, 255).to(torch.float32)
        resized = resized / 255.0

    resized = torch.permute(resized, [1, 2, 0]).numpy()
    image_mask = np.ones_like(resized[:, :, 0], dtype=np.bool_)

    return resized, image_mask


def select_tiling(h, w, patch_size, max_num_crops):
    """Divide in image of size [w, h] in up to max_num_patches of size patch_size"""
    original_size = np.stack([h, w])  # [1, 2]
    original_res = h * w
    tilings = []
    for i in range(1, max_num_crops + 1):
        for j in range(1, max_num_crops + 1):
            if i*j <= max_num_crops:
                tilings.append((i, j))
    # sort so argmin and argmax favour smaller tilings in the event of a tie
    tilings.sort(key=lambda x: (x[0]*x[1], x[0]))
    candidate_tilings = np.array(tilings, dtype=np.int32)  # [n_resolutions, 2]
    candidate_resolutions = candidate_tilings * patch_size  # [n_resolutions, 2]

    # How much we would need to scale the image to fit exactly in each tiling
    original_size = np.stack([h, w], dtype=np.float32)  # [1, 2]

    # The original size can be zero in rare cases if the image is smaller than the margin
    # In those cases letting the scale become infinite means the tiling is based on the
    # other side, or falls back to the smallest tiling
    with np.errstate(divide='ignore'):
        required_scale_d = candidate_resolutions.astype(np.float32) / original_size,
    required_scale = np.min(required_scale_d, axis=-1, keepdims=True)  # [n_resolutions, 1]
    if np.all(required_scale < 1):
        # We are forced to downscale, so try to minimize the amount of downscaling
        ix = np.argmax(required_scale)
    else:
        # Pick the resolution that required the least upscaling so that it most closely fits the image
        required_scale = np.where(required_scale < 1.0, 10e9, required_scale)
        ix = np.argmin(required_scale)
    return candidate_tilings[ix]


@dataclasses.dataclass
class ImagePreprocessor:
    """Generic image preprocessor that supports multi-cropping an image"""

    normalize: str = "openai"
    resize: str = "default"
    pad_value: float = 0
    max_crops: int = 6
    max_multi_image_crops: int = 6
    overlap_margins: Tuple[int, int] = (4, 4)
    use_col_tokens: bool = True
    base_image_input_size: Tuple[int, int] = (336, 336)
    image_patch_size: int = 14

    def unnormalize_image(self, image: np.ndarray):
        if self.normalize == "openai":
            return (image * np.array(OPENAI_CLIP_STD, dtype=np.float32)[None, None, :] +
                    np.array(OPENAI_CLIP_MEAN, dtype=np.float32)[None, None, :])
        elif self.normalize == "siglip":
            return (image + 1) / np.asarray(2.0, dtype=np.float32)
        else:
            raise NotImplementedError()

    def normalize_image(self, image):
        if self.normalize == "openai":
            image -= np.array(OPENAI_CLIP_MEAN, dtype=np.float32)[None, None, :]
            image /= np.array(OPENAI_CLIP_STD, dtype=np.float32)[None, None, :]
        elif self.normalize == "siglip":
            image = np.asarray(-1.0, dtype=np.float32) + image * np.asarray(2.0, dtype=np.float32)
        elif self.normalize == "dino":
            image -= np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, None, :]
            image /= np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, None, :]
        else:
            raise NotImplementedError(self.normalize)
        return image

    def resize_image(self, image, output_size, is_training, rng):
        if self.resize == "siglip":
            return siglip_resize_and_pad(image, output_size)
        elif self.resize == "dino":
            return dino_resize_and_pad(image, output_size)
        elif self.resize == "metaclip":
            return metaclip_resize(image, output_size)
        else:
            resize = "torch-bilinear" if self.resize == "default" else self.resize
            return resize_and_pad(
                image, output_size, pad_value=self.pad_value, rng=rng, is_training=is_training,
                resize_method=resize)

    def build_resized_image(self, image, is_training, rng, image_size=None):
        image_size = image_size or self.base_image_input_size
        resized, resized_mask = self.resize_image(image, image_size, is_training, rng)

        # def safe_image_from_array(arr):
        #     # Clone the array to avoid modifying original data
        #     arr = arr.copy()

        #     # Convert float32 → uint8 if necessary
        #     if arr.dtype != np.uint8:
        #         arr = np.clip(arr, 0.0, 1.0)
        #         arr = (arr * 255).astype(np.uint8)

        #     # Handle shape like (1, 1, 3)
        #     if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[1] == 1:
        #         arr = arr[0, 0][np.newaxis, np.newaxis, :]

        #     return Image.fromarray(arr)

        # # ——— SANITY CHECK DUMP ———
        # from datetime import datetime
        # annotated_image_pil = safe_image_from_array(resized)
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # out_dir = "/weka/oe-training-default/jiafeid/sanitycheck"
        # os.makedirs(out_dir, exist_ok=True)
        # out_path = os.path.join(out_dir, f"{timestamp}_resize.png")
        # annotated_image_pil.save(out_path)
        # print(f"[sanity] wrote resized image to {out_path}")
        # # ————————————————

        resized = self.normalize_image(resized)
        if len(resized.shape) == 3:
            resized = np.expand_dims(resized, 0)
        resized_mask = np.expand_dims(resized_mask, 0)
        crop_patch_w = image_size[1] // self.image_patch_size
        crop_patch_h = image_size[0] // self.image_patch_size
        resize_idx = np.arange(crop_patch_w*crop_patch_h).reshape([crop_patch_h, crop_patch_w])
        return resized, resized_mask, resize_idx

    def build_overlapping_crops(self, image, is_training, rng, is_multi_image: bool = False):
        """Decompose an image into a set of overlapping crops

        :return crop_arr: [n_crops, h, w, 3] The crops
        :return mask_arr: [n_crops, h, w] The padding masks
        :return patch_idx: [overlap_patch_h, overlap_patch_w] For each patch in the resized image
                           the crops were extracted from, what patch in `crop_arr` it corresponds to
        """
        original_image_h, original_image_w = image.shape[:2]
        max_crops = self.max_multi_image_crops if is_multi_image else self.max_crops
        overlap_margins = self.overlap_margins
        base_image_input_size = self.base_image_input_size
        image_patch_size = self.image_patch_size
        crop_size = base_image_input_size[0]
        assert base_image_input_size[0] == base_image_input_size[1]

        left_margin, right_margin = overlap_margins
        total_margin_pixels = image_patch_size*(right_margin + left_margin)  # pixels removed per dim
        crop_patches = base_image_input_size[0] // image_patch_size  # patches per crop dim
        crop_window_patches = crop_patches - (right_margin + left_margin)  # usable patches
        crop_window_size = crop_window_patches * image_patch_size
        crop_patch_w = base_image_input_size[1] // image_patch_size
        crop_patch_h = base_image_input_size[0] // image_patch_size
        original_image_h, original_image_w = image.shape[:2]
        crop_size = base_image_input_size[0]

        # Decide how to tile the image, to account for the overlap margins we compute the tiling
        # as if we had an image without the margins and were using a crop size without the margins
        tiling = select_tiling(
            original_image_h - total_margin_pixels,
            original_image_w - total_margin_pixels,
            crop_window_size,
            max_crops
        )
        src, img_mask = self.resize_image(
            image,
            [tiling[0]*crop_window_size+total_margin_pixels, tiling[1]*crop_window_size+total_margin_pixels],
            is_training,
            rng
        )
        src = self.normalize_image(src)

        # Now we have to split the image into crops, and track what patches came from
        # where in `patch_idx_arr`
        n_crops = tiling[0] * tiling[1]
        crop_arr = np.zeros([n_crops, crop_size, crop_size, 3], dtype=src.dtype)
        mask_arr = np.zeros([n_crops, crop_size, crop_size], dtype=img_mask.dtype)
        patch_idx_arr = np.zeros([n_crops, crop_patch_h, crop_patch_w], dtype=np.int32)
        on = 0
        on_crop = 0
        for i in range(tiling[0]):
            # Slide over `src` by `crop_window_size` steps, but extract crops of size `crops_size`
            # which results in overlapping crop windows
            y0 = i*crop_window_size
            for j in range(tiling[1]):
                x0 = j*crop_window_size
                crop_arr[on_crop] = src[y0:y0+crop_size, x0:x0+crop_size]
                mask_arr[on_crop] = img_mask[y0:y0+crop_size, x0:x0+crop_size]
                patch_idx = np.arange(crop_patch_w*crop_patch_h).reshape(crop_patch_h, crop_patch_w)
                patch_idx += on_crop * crop_patch_h * crop_patch_w

                # Mask out idx that are in the overlap region
                if i != 0:
                    patch_idx[:left_margin, :] = -1
                if j != 0:
                    patch_idx[:, :left_margin] = -1
                if i != tiling[0]-1:
                    patch_idx[-right_margin:, :] = -1
                if j != tiling[1]-1:
                    patch_idx[:, -right_margin:] = -1
                patch_idx_arr[on_crop] = patch_idx
                on_crop += 1

        # `patch_idx_arr` is ordered crop-by-crop, here we transpose `patch_idx_arr`
        # so it is ordered left-to-right order
        patch_idx_arr = np.reshape(
            patch_idx_arr,
            [tiling[0], tiling[1], crop_patch_h, crop_patch_w]
        )
        patch_idx_arr = np.transpose(patch_idx_arr, [0, 2, 1, 3])
        patch_idx_arr = np.reshape(patch_idx_arr, [-1])

        # Now get the parts not in the overlap region, so it should map each patch in `src`
        # to the correct patch it should come from in `crop_arr`
        patch_idx_arr = patch_idx_arr[patch_idx_arr >= 0].reshape(
            src.shape[0]//image_patch_size,
            src.shape[1]//image_patch_size,
        )
        return crop_arr, mask_arr, patch_idx_arr

    def compute_overlapping_crops_size(self, image_h, image_w, is_multi_image: bool = False) -> Tuple[int, int]:
        """Returns the number of patches the multi-crop image would have"""
        image_patch_size = self.image_patch_size
        crop_patch_w = self.base_image_input_size[1] // image_patch_size
        crop_patch_h = self.base_image_input_size[0] // image_patch_size
        margin_patches = sum(self.overlap_margins)
        margin_pixels = image_patch_size*margin_patches  # pixels removed per dim
        assert crop_patch_w == crop_patch_h
        crop_window_patches = crop_patch_w - margin_patches
        crop_window_size = crop_window_patches * image_patch_size
        max_crops = self.max_multi_image_crops if is_multi_image else self.max_crops
        tiling = select_tiling(
            image_h - margin_pixels,
            image_w - margin_pixels,
            crop_window_size,
            max_crops
        )
        h, w = [tiling[0]*crop_window_size+margin_pixels, tiling[1]*crop_window_size+margin_pixels]
        return h//image_patch_size, w//image_patch_size
