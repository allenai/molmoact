"""
Robot datasets that load directly from the HuggingFace Hub using HfDataset.

Key points:
- Uses HfDataset (no pre-saving to local path-indexed datasets).
- Preserves class organization (BC_Z, BridgeDataV2, RT_1, AuxiliaryDepthData, AuxiliaryTraceData,
  and MolmoAct* variants).
- get():
  * Pretraining configs (single camera: "image") -> return a single PIL image by default.
  * Midtraining configs (multi-camera: e.g., 'primary'/'secondary', 'wrist') -> return a list
    of PIL images in the order of camera_fields.
  * Optional resizing via width/height (if provided).
"""

import io
import os
from os.path import exists, join
from typing import Any, Dict, Iterable, List, Optional, Tuple
import ast

import datasets
from datasets import Image as HFImage
import numpy as np
from PIL import Image

from olmo.data.dataset import Dataset, HfDataset


# --------------------------
# Helpers
# --------------------------

def _to_pil(x: Any) -> Image.Image:
    """Convert common containers to PIL.Image (RGB)."""
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, (bytes, bytearray)):
        return Image.open(io.BytesIO(x)).convert("RGB")
    if isinstance(x, str) and os.path.exists(x):
        return Image.open(x).convert("RGB")
    if isinstance(x, dict):
        if x.get("path"):
            return Image.open(x["path"]).convert("RGB")
        if x.get("bytes") is not None:
            return Image.open(io.BytesIO(x["bytes"])).convert("RGB")
    if isinstance(x, np.ndarray):
        arr = x
        if arr.dtype in (np.float32, np.float64):
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        return Image.fromarray(arr)
    # Some HF Image features already return PIL; fallback to identity error if unknown
    raise ValueError(f"Unsupported image container: {type(x)}")


def _infer_cameras(dataset_name: str) -> Tuple[str, ...]:
    """Infer (body_cam, 'wrist') from dataset_name (primary or secondary or libero)."""
    if "primary" in dataset_name:
        return ("primary", "wrist")
    elif "secondary" in dataset_name:
        return ("secondary", "wrist")
    elif "libero" in dataset_name:
        return ("image", "wrist")
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")


def _cast_if_present(ds: datasets.Dataset, cols: Iterable[str]) -> datasets.Dataset:
    """Cast existing columns among `cols` to HF Image(decode=True)."""
    for c in cols:
        if c in ds.column_names:
            ds = ds.cast_column(c, HFImage(decode=True))
    return ds


# =============================================================================
# Generic base that loads directly from HF
# =============================================================================

class RobotHfDataset(HfDataset):
    """
    Generic loader over HfDataset with camera-aware get():

    - Subclasses set PATH (Hub repo like 'allenai/MolmoAct-Pretraining-Mixture').
    - `dataset_name` is passed as HF `name` (config).
    - `camera_fields` controls which image columns are read:
        * default: ('image',)
        * midtraining: ('primary' or 'secondary', 'wrist')
    - get():
        * single camera -> returns a PIL.Image (optionally resized)
        * multi camera  -> returns [PIL.Image, ...] (optionally resized)
    """

    PATH: str = ""  # must be overridden

    @classmethod
    def default_camera_fields(cls, dataset_name: str) -> Tuple[str, ...]:
        return ("image",)

    @classmethod
    def download(cls, dataset_name: str, n_procs: Optional[int] = None):
        """
        Prefetch the HF dataset (builder) for this config to the local HF cache.
        This mirrors HfDataset.download but passes the config name.
        """
        assert cls.PATH, "Subclass must define PATH"
        builder = datasets.load_dataset_builder(cls.PATH, name=dataset_name)
        # HF handles parallelism internally; n_procs unused here but kept for API parity
        builder.download_and_prepare()

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        camera_fields: Optional[Iterable[str]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        style: str = "demo",
        keep_in_memory: bool = False,
        num_proc: int = 16,
        **hf_kwargs,
    ):
        """
        Args:
            dataset_name: HF config name for this repo.
            split: HF split (default 'train').
            camera_fields: which image columns to read. If None, default_camera_fields(dataset_name) is used.
            width, height: if both provided, images are resized in get().
            style: string passed through in get() output.
            keep_in_memory: passed through to HfDataset (speeds up random access on smaller shards).
            **hf_kwargs: forwarded to datasets.load_dataset (e.g., streaming=True).
        """
        assert self.PATH, "Subclass must define PATH"

        self.dataset_name = dataset_name
        self.split = split
        self.camera_fields = tuple(camera_fields) if camera_fields is not None else self.default_camera_fields(dataset_name)
        self.style = style
        self.width = width
        self.height = height

        # Load from HF directly (no materialization to a local path-indexed dataset)
        super().__init__(split=split, keep_in_memory=keep_in_memory, name=dataset_name, num_proc=num_proc, **hf_kwargs)

        # Cast image columns that exist (top-level). For nested structures, we handle in get().
        self.dataset = _cast_if_present(self.dataset, self.camera_fields)

    def __len__(self):
        return len(self.dataset)

    # ---- runtime helpers ----
    def _maybe_resize(self, img: Any) -> Image.Image:
        pil = img if isinstance(img, Image.Image) else _to_pil(img)
        if self.width is not None and self.height is not None:
            return pil.resize((self.width, self.height), resample=Image.BILINEAR)
        return pil

    def _extract_images(self, ex: Dict) -> List[Image.Image]:
        """Collect PILs for all camera_fields, supporting both top-level and nested 'images' layouts."""
        out: List[Image.Image] = []
        images_nested = ex.get("images") if isinstance(ex, dict) else None  # some configs store under ex['images'][cam]
        for cam in self.camera_fields:
            val = None
            if cam in ex and ex[cam] is not None:
                val = ex[cam]
            elif isinstance(images_nested, dict) and cam in images_nested and images_nested[cam] is not None:
                val = images_nested[cam]
            if val is not None:
                out.append(self._maybe_resize(val))
        return out

    # ---- API ----
    def get(self, item, rng):
        ex = self.dataset[item]
        conv = ex["conversations"]
        annotation = ex.get("annotation", None)

        if annotation:
            annotation = ast.literal_eval(annotation)

        assert annotation is None or type(annotation) is list

        # Single camera -> return single PIL; Multi -> list of PIL
        image_out = self._extract_images(ex)

        return dict(
            style=self.style,
            image=image_out,
            question=conv["value"][0],
            answers=conv["value"][1],
            annotation=annotation,
        )


# =============================================================================
# Pre-training specializations (single camera: "image")
# =============================================================================

class RobotHfDatasetPretrain(RobotHfDataset):
    PATH = "allenai/MolmoAct-Pretraining-Mixture"

    @classmethod
    def default_camera_fields(cls, dataset_name: str) -> Tuple[str, ...]:
        return ("image",)

def _prep_pretrain_config(dataset_name: str, split: str = "train"):
    # Trigger download/prepare into HF cache
    RobotHfDatasetPretrain.download(dataset_name=dataset_name)


class _RobotPretrainBase(Dataset):
    """Thin wrapper that fixes dataset_name and keeps original import ergonomics."""
    DATASET_NAME: str = ""

    @classmethod
    def download(cls, n_procs: int = 1):
        assert cls.DATASET_NAME, "Set DATASET_NAME on the subclass"
        _prep_pretrain_config(cls.DATASET_NAME)

    def __init__(
        self,
        split: str = "train",
        style: str = "demo",
        keep_in_memory: bool = False,
        num_proc: int = 16,
        **hf_kwargs,
    ):
        assert self.DATASET_NAME, "Set DATASET_NAME on the subclass"
        self._inner = RobotHfDatasetPretrain(
            dataset_name=self.DATASET_NAME,
            split=split,
            camera_fields=None,       # default -> ("image",)
            width=None, height=None,  # no resize by default for pretraining
            style=style,
            keep_in_memory=keep_in_memory,
            num_proc=num_proc,
            **hf_kwargs,
        )

    def __len__(self): return len(self._inner)

    def get(self, item, rng): return self._inner.get(item, rng)


# Concrete pretraining datasets (unchanged names)
class BC_Z(_RobotPretrainBase):
    DATASET_NAME = "bc_z"


class BridgeDataV2(_RobotPretrainBase):
    DATASET_NAME = "bridge_dataset"


class RT_1(_RobotPretrainBase):
    DATASET_NAME = "fractal20220817_data"


class AuxiliaryDepthData(_RobotPretrainBase):
    DATASET_NAME = "auxiliary_depth"


class AuxiliaryTraceData(_RobotPretrainBase):
    DATASET_NAME = "auxiliary_trace"


# =============================================================================
# Mid-training specializations (multi-camera + optional resize)
# =============================================================================

class RobotHfDatasetMidtrain(RobotHfDataset):
    PATH = "allenai/MolmoAct-Midtraining-Mixture"

    @classmethod
    def default_camera_fields(cls, dataset_name: str) -> Tuple[str, ...]:
        return _infer_cameras(dataset_name)

def _prep_mid_config(dataset_name: str, split: str = "train"):
    RobotHfDatasetMidtrain.download(dataset_name=dataset_name)


class _MolmoActHfBase(Dataset):
    """Wrapper that infers camera set from dataset_name and exposes resize knobs."""
    DATASET_NAME: str = ""

    @classmethod
    def download(cls, n_procs: int = 1):
        assert cls.DATASET_NAME, "Set DATASET_NAME on the subclass"
        _prep_mid_config(cls.DATASET_NAME)

    def __init__(
        self,
        split: str = "train",
        width: Optional[int] = 320,
        height: Optional[int] = 240,
        style: str = "demo",
        keep_in_memory: bool = False,
        num_proc: int = 16,
        **hf_kwargs,
    ):
        assert self.DATASET_NAME, "Set DATASET_NAME on the subclass"
        cams = RobotHfDatasetMidtrain.default_camera_fields(self.DATASET_NAME)
        self._inner = RobotHfDatasetMidtrain(
            dataset_name=self.DATASET_NAME,
            split=split,
            camera_fields=cams,
            width=width,
            height=height,
            style=style,
            keep_in_memory=keep_in_memory,
            num_proc=num_proc,
            **hf_kwargs,
        )

    def __len__(self): return len(self._inner)

    def get(self, item, rng): return self._inner.get(item, rng)


# Concrete midtraining datasets (unchanged names)
class MolmoActDatasetHomePrimary(_MolmoActHfBase):
    DATASET_NAME = "molmoact_home_primary"


class MolmoActDatasetHomeSecondary(_MolmoActHfBase):
    DATASET_NAME = "molmoact_home_secondary"


class MolmoActDatasetTabletopPrimary(_MolmoActHfBase):
    DATASET_NAME = "molmoact_tabletop_primary"


class MolmoActDatasetTabletopSecondary(_MolmoActHfBase):
    DATASET_NAME = "molmoact_tabletop_secondary"


# =============================================================================
# Libero Post-training specializations (multi-camera)
# =============================================================================

class RobotHfDatasetLIBERO(RobotHfDataset):
    PATH = "allenai/libero"

    @classmethod
    def default_camera_fields(cls, dataset_name: str) -> Tuple[str, ...]:
        return _infer_cameras(dataset_name)

def _prep_libero_config(dataset_name: str, split: str = "train"):
    RobotHfDatasetLIBERO.download(dataset_name=dataset_name)


class _LIBEROHfBase(Dataset):
    """Wrapper that infers camera set from dataset_name and exposes resize knobs."""
    DATASET_NAME: str = ""

    @classmethod
    def download(cls, n_procs: int = 1):
        assert cls.DATASET_NAME, "Set DATASET_NAME on the subclass"
        _prep_libero_config(cls.DATASET_NAME)

    def __init__(
        self,
        split: str = "train",
        style: str = "demo",
        keep_in_memory: bool = False,
        num_proc: int = 16,
        **hf_kwargs,
    ):
        assert self.DATASET_NAME, "Set DATASET_NAME on the subclass"
        cams = RobotHfDatasetLIBERO.default_camera_fields(self.DATASET_NAME)
        self._inner = RobotHfDatasetLIBERO(
            dataset_name=self.DATASET_NAME,
            split=split,
            camera_fields=cams,
            style=style,
            keep_in_memory=keep_in_memory,
            num_proc=num_proc,
            **hf_kwargs,
        )

    def __len__(self): return len(self._inner)

    def get(self, item, rng): return self._inner.get(item, rng)


# Concrete midtraining datasets (unchanged names)
class LIBEROSpatial(_LIBEROHfBase):
    DATASET_NAME = "libero_spatial"


class LIBEROObject(_LIBEROHfBase):
    DATASET_NAME = "libero_object"


class LIBEROGoal(_LIBEROHfBase):
    DATASET_NAME = "libero_goal"


class LIBEROLong(_LIBEROHfBase):
    DATASET_NAME = "libero_10"


__all__ = [
    # Pretraining wrappers
    "BC_Z", "BridgeDataV2", "RT_1", "AuxiliaryDepthData", "AuxiliaryTraceData",
    # Midtraining wrappers
    "MolmoActDatasetHomePrimary", "MolmoActDatasetHomeSecondary",
    "MolmoActDatasetTabletopPrimary", "MolmoActDatasetTabletopSecondary",
    # Libero wrappers
    "LIBEROSpatial", "LIBEROObject", "LIBEROGoal", "LIBEROLong",
]