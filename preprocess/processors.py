#!/usr/bin/env python3
"""
Modular processors for depth, tokenization, point detection, and trace generation.
"""
import cv2
import json
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from depth_anything_v2.dpt import DepthAnythingV2
from line_utils import point_at_gripper
from models.VQVAE import VQVAE
from PIL import Image
from transformers import Qwen2Tokenizer
from typing import Dict, List, Optional, Tuple, Union


class Depth:
    """Handle depth estimation using Depth-Anything-V2."""
    
    def __init__(self, encoder: str = "vitb", ckpt_dir: str = "checkpoints"):
        """Initialize Depth-Anything-V2 model.
        
        Args:
            encoder: Model encoder type ('vits', 'vitb', 'vitl', 'vitg')
            ckpt_dir: Directory containing model checkpoints
        """
        self.device = 'cuda' if torch.cuda.is_available() else \
                     'mps' if torch.backends.mps.is_available() else 'cpu'
        
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
        }
        
        self.model = DepthAnythingV2(**model_configs[encoder])
        state_dict_path = f"{ckpt_dir}/depth_anything_v2_{encoder}.pth"
        state = torch.load(state_dict_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device).eval()
        
    def inference_depth_from_rgb(self, rgb_image: np.ndarray, input_size: int = 518) -> np.ndarray:
        """Infer depth from RGB image.
        
        Args:
            rgb_image: RGB image as numpy array
            input_size: Input size for the model
            
        Returns:
            Depth image as 3-channel uint8 numpy array
        """
        depth = self.model.infer_image(rgb_image, input_size)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255.0
        depth = depth.astype(np.uint8)
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        return depth


class DepthTokens:
    """Handle depth tokenization using VQVAE."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize VQVAE model for depth tokenization.
        
        Args:
            model_path: Path to VQVAE model checkpoint. If None, uses default or env var.
        """
        cfg_model = dict(
            image_size=320,
            num_resnet_blocks=2,
            downsample_ratio=32,
            num_tokens=128,
            codebook_dim=512,
            hidden_dim=16,
            use_norm=False,
            channels=1,
            train_objective='regression',
            max_value=10.,
            residul_type='v1',
            loss_type='mse',
        )
        
        self.vae = VQVAE(**cfg_model).cuda()
        
        # Use provided path or environment variable or default
        if model_path is None:
            model_path = os.environ.get('VQVAE_MODEL_PATH')
        
        ckpt = torch.load(model_path)['weights']
        if 'module' in list(ckpt.keys())[0]:
            new_ckpt = {key[7:]: val for key, val in ckpt.items()}
            self.vae.load_state_dict(new_ckpt)
        else:
            self.vae.load_state_dict(ckpt)
        
        self.vae = torch.nn.DataParallel(self.vae).cuda()
        self.vae.eval()
        
        # Setup transform
        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
        ])
    
    def inference_depth_tokens(self, depth_image_gray: Union[np.ndarray, Image.Image]) -> str:
        """Convert grayscale depth image to tokens.
        
        Args:
            depth_image_gray: Grayscale depth image as numpy array or PIL Image
            
        Returns:
            String representation of depth tokens
        """
        # Convert to PIL if needed
        if isinstance(depth_image_gray, np.ndarray):
            depth_pil = Image.fromarray(depth_image_gray)
        else:
            depth_pil = depth_image_gray
        
        # Transform to tensor
        image_tensor = self.transform(depth_pil).unsqueeze(0).cuda()
        
        with torch.no_grad():
            codes = self.vae(img=image_tensor, return_indices=True)
            tokens = codes.flatten().cpu().tolist()
        
        depth_string = "<DEPTH_START>" + "".join(f"<DEPTH_{n}>" for n in tokens) + "<DEPTH_END>"
        return depth_string


class Point:
    """Handle gripper point detection using Molmo."""
    
    def __init__(self):
        """Initialize Molmo for gripper detection."""
        self.point_at_gripper = point_at_gripper
        self._temp_counter = 0
    
    def inference_point(self, image: Union[np.ndarray, Image.Image, str]) -> Optional[List[int]]:
        """Detect gripper point in image.
        
        Args:
            image: Input image as numpy array, PIL Image, or file path
            
        Returns:
            [x, y] coordinates of gripper point, or None if not found
        """

        # Need to save temporarily
        self._temp_counter += 1
        
        if isinstance(image, np.ndarray):
            # Convert numpy to PIL
            if image.ndim == 3 and image.shape[2] == 3:
                # RGB image
                pil_image = Image.fromarray(image)
            else:
                # Assume BGR from OpenCV
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
        elif hasattr(image, 'numpy'):  # Handle Tensor objects (PyTorch, TensorFlow, etc.)
            # Convert tensor to numpy array first
            if hasattr(image, 'detach'):  # PyTorch tensor
                # LeRobotDataset returns images as tensors in channel-first format (C, H, W)
                # Convert from (C, H, W) to (H, W, C) and from [0, 1] to [0, 255]
                image_array = (image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_array)
            else:  # Other tensor types
                image_np = image.numpy()
                pil_image = Image.fromarray(image_np)
        else:
            # Already PIL Image
            pil_image = image


        
        try:
            # Get gripper point
            gripper_result = self.point_at_gripper(pil_image)
            

            print(f"Debug gripper result {gripper_result}")

            
            # Extract point
            if gripper_result['x_px'] is not None and gripper_result['y_px'] is not None:
                return [gripper_result['x_px'], gripper_result['y_px']]
            else:
                return None
        finally:
            pass

class Trace:
    """Handle trajectory trace generation from points."""
    
    def __init__(self, points: Optional[List[Optional[List[int]]]] = None):
        """Initialize with a list of points.
        
        Args:
            points: List of [x, y] points, can contain None values
        """
        self.points = points if points is not None else []
    
    def subsample_to_line(self, k: int, last_valid_pt: Optional[List[int]] = None) -> List[List[int]]:
        """Subsample points to create a line of k points.
        
        Args:
            k: Number of points to return
            last_valid_pt: Fallback point if no valid points available
            
        Returns:
            List of ≤k evenly-spaced points
        """
        # Filter out None values
        valid_pts = [pt for pt in self.points if pt is not None]
        
        m = len(valid_pts)
        if m == 0:
            return [last_valid_pt] if last_valid_pt is not None else []
        if m <= k:
            return valid_pts
        
        # Evenly space k points
        idxs = np.linspace(0, m - 1, num=k, dtype=int)
        return [valid_pts[j] for j in idxs]
    
    def set_points(self, points: List[Optional[List[int]]]):
        """Update the list of points.
        
        Args:
            points: New list of points
        """
        self.points = points
    
    def get_future_trace(self, start_idx: int, length: int, 
                        fallback_point: Optional[List[int]] = None) -> List[List[int]]:
        """Get a trace of future points from a starting index.
        
        Args:
            start_idx: Starting index in points list
            length: Desired length of trace
            fallback_point: Point to use as fallback
            
        Returns:
            Subsampled list of future points
        """
        # Get future points from start_idx
        future_points = self.points[start_idx:]
        
        # Create temporary trace object with future points
        future_trace = Trace(future_points)
        return future_trace.subsample_to_line(length, fallback_point)


# Example usage function showing how to use all classes together
def process_frame_with_classes(pil_image: Image.Image, 
                              depth_processor: Depth,
                              token_processor: DepthTokens,
                              point_processor: Point) -> Tuple[str, Optional[List[int]]]:
    """Process a single frame using all processors.
    
    Args:
        pil_image: Input PIL Image
        depth_processor: Initialized Depth processor
        token_processor: Initialized DepthTokens processor
        point_processor: Initialized Point processor
        
    Returns:
        Tuple of (depth_tokens_string, gripper_point)
    """
    # Convert PIL to numpy for depth processing
    rgb_array = np.array(pil_image.convert('RGB'))
    bgr_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    
    # 1. Generate depth image
    depth_image = depth_processor.inference_depth_from_rgb(bgr_image)
    
    # 2. Convert to grayscale and tokenize
    depth_gray = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
    depth_tokens = token_processor.inference_depth_tokens(depth_gray)
    
    # 3. Detect gripper point
    gripper_point = point_processor.inference_point(pil_image)
    
    return depth_tokens, gripper_point


class ActionProcessor:
    """Handle action processing: statistics, normalization, tokenization, and chunking."""
    
    def __init__(self, tokenizer_model: str = "Qwen/Qwen2-7B", 
                 bins: int = 256, 
                 min_action: float = -1.0, 
                 max_action: float = 1.0,
                 action_dims: int = 7,
                 chunk_size: int = 8):
        """Initialize action processor.
        
        Args:
            tokenizer_model: Pretrained tokenizer model name
            bins: Number of discretization bins
            min_action: Minimum action value for tokenization
            max_action: Maximum action value for tokenization
            action_dims: Number of action dimensions
            chunk_size: Size of action chunks for horizon
        """
        self.action_dims = action_dims
        self.chunk_size = chunk_size
        self.json = json
        
        # Initialize tokenizer
        self.base_tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_model)
        
        # Initialize action tokenizer
        self.n_bins = bins
        self.min_action = min_action
        self.max_action = max_action
        
        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        
        # Action tokens start near the end of the vocabulary
        self.action_token_begin_idx = int(self.base_tokenizer.vocab_size - (self.n_bins + 1))
        
        # Statistics will be computed from dataset
        self.stats = None
    
    def compute_dataset_statistics(self, dataset, normalize_dims: int = 6) -> Dict:
        """Compute action statistics from dataset.
        
        Args:
            dataset: HuggingFace dataset with 'action' field
            normalize_dims: Number of dimensions to normalize (rest get mask=False)
            
        Returns:
            Dictionary with statistics
        """
        all_actions = []
        all_proprios = []
        
        # Count unique trajectories/episodes
        episode_ids = set()
        
        for sample in dataset:
            # Handle different action field formats
            if 'action' in sample:
                action = sample['action']
            elif 'actions' in sample:  # Add this line for LeRobot format
                action = sample['actions']
            elif 'raw_action' in sample:
                action = sample['raw_action']
            else:
                continue
            
            # Parse if string
            if isinstance(action, str):
                action = self.json.loads(action)
            
            # Convert to numpy array
            action_vec = np.array(action, dtype=np.float32)
            all_actions.append(action_vec)
            
            # Try to extract proprio if available
            if 'observation.state' in sample:
                proprio = sample['observation.state']
                if isinstance(proprio, str):
                    proprio = self.json.loads(proprio)
                all_proprios.append(np.array(proprio, dtype=np.float32))
            
            # Try to count episodes
            if 'episode_index' in sample:
                episode_ids.add(sample['episode_index'])
            elif 'episode_id' in sample:
                episode_ids.add(sample['episode_id'])
        
        if not all_actions:
            raise ValueError("No actions found in dataset")
        
        actions_np = np.array(all_actions, dtype=np.float32)
        
        # Create mask: True for dimensions to normalize, False for others (e.g., gripper)
        action_dim = actions_np.shape[1]
        mask = [True] * min(normalize_dims, action_dim) + [False] * max(0, action_dim - normalize_dims)
        
        self.stats = {
            "action": {
                "mean": actions_np.mean(axis=0).tolist(),
                "std": actions_np.std(axis=0).tolist(),
                "min": actions_np.min(axis=0).tolist(),
                "max": actions_np.max(axis=0).tolist(),
                "q01": np.quantile(actions_np, 0.01, axis=0).tolist(),
                "q99": np.quantile(actions_np, 0.99, axis=0).tolist(),
                "mask": mask,
            },
            "num_transitions": int(actions_np.shape[0]),
            "num_trajectories": len(episode_ids) if episode_ids else 0
        }
        
        # Add proprio stats if available
        if all_proprios:
            proprios_np = np.array(all_proprios, dtype=np.float32)
            self.stats["proprio"] = {
                "mean": proprios_np.mean(axis=0).tolist(),
                "std": proprios_np.std(axis=0).tolist(),
                "min": proprios_np.min(axis=0).tolist(),
                "max": proprios_np.max(axis=0).tolist(),
                "q01": np.quantile(proprios_np, 0.01, axis=0).tolist(),
                "q99": np.quantile(proprios_np, 0.99, axis=0).tolist(),
            }
        else:
            # Add zeros if no proprio data
            proprio_dim = action_dim  # Default to action dimension
            self.stats["proprio"] = {
                "mean": [0.0] * proprio_dim,
                "std": [0.0] * proprio_dim,
                "min": [0.0] * proprio_dim,
                "max": [0.0] * proprio_dim,
                "q01": [0.0] * proprio_dim,
                "q99": [0.0] * proprio_dim,
            }
        
        return self.stats
    
    def save_dataset_statistics(self, output_path: str, dataset_name: str, 
                               dataset=None, stats: Optional[Dict] = None):
        """Save dataset statistics to JSON file.
        
        Args:
            output_path: Path to save the statistics JSON file
            dataset_name: Name/key for this dataset in the stats file
            dataset: Dataset to compute stats from (if stats not provided)
            stats: Pre-computed statistics (if not provided, will compute from dataset)
        """
        # Load existing stats file if it exists
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                all_stats = json.load(f)
        else:
            all_stats = {}
        
        # Get or compute stats
        if stats is None:
            if dataset is None:
                raise ValueError("Must provide either dataset or stats")
            stats = self.compute_dataset_statistics(dataset)
        
        # Add to all_stats under dataset_name
        all_stats[dataset_name] = stats
        
        # Save to file
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
        
        print(f"Saved statistics for '{dataset_name}' to {output_path}")
        return all_stats
    
    def normalize_action_bounds_q99(self, action: Union[np.ndarray, List], 
                                   normalize_dims: int = 6) -> np.ndarray:
        """Apply BOUNDS_Q99 normalization to action.
        
        Args:
            action: Action vector
            normalize_dims: Number of dimensions to normalize (default: 6, leaving gripper)
            
        Returns:
            Normalized action array
        """
        if self.stats is None:
            raise ValueError("Must compute dataset statistics first")
        
        action = np.array(action, dtype=np.float32)
        q01 = np.array(self.stats["action"]["q01"], dtype=np.float32)
        q99 = np.array(self.stats["action"]["q99"], dtype=np.float32)
        
        normalized = action.copy()
        
        # Normalize first N dimensions
        if normalize_dims > 0:
            q01_n = q01[:normalize_dims]
            q99_n = q99[:normalize_dims]
            
            eps = 1e-8
            denom = q99_n - q01_n + eps
            raw_norm = 2 * (action[:normalize_dims] - q01_n) / denom - 1.0
            clipped = np.clip(raw_norm, -1.0, 1.0)
            
            # Handle zero variance dimensions
            zeros_mask = np.isclose(q99_n, q01_n)
            clipped[zeros_mask] = 0.0
            
            normalized[:normalize_dims] = clipped
        
        # Optionally invert gripper (last dimension)
        if len(action) > normalize_dims:
            normalized[normalize_dims] = 1.0 - action[normalize_dims]
        
        return normalized
    
    def tokenize_action(self, action: np.ndarray) -> List[str]:
        """Tokenize a continuous action vector.
        
        Args:
            action: Normalized action array
            
        Returns:
            List of action token strings
        """
        # Clip & bin actions to discrete tokens
        action = np.clip(action, a_min=self.min_action, a_max=self.max_action)
        discretized_action = np.digitize(action, self.bins)
        
        # Convert to token IDs
        action_token_ids = (self.base_tokenizer.vocab_size - discretized_action).tolist()
        
        # Decode to token strings
        return self.base_tokenizer.convert_ids_to_tokens(action_token_ids)
    
    def chunk_actions(self, episode_actions: List[List[str]]) -> List[str]:
        """Create chunked action sequences with sliding window.
        
        Args:
            episode_actions: List of tokenized actions for an episode
            
        Returns:
            List of chunked action strings for each timestep
        """
        chunked_actions = []
        n_steps = len(episode_actions)
        
        for i in range(n_steps):
            # Get window of actions
            window = episode_actions[i:i + self.chunk_size]
            
            # Pad with last action if needed
            if len(window) < self.chunk_size and window:
                window += [window[-1]] * (self.chunk_size - len(window))
            
            # Format as nested list string
            if window:
                chunk_str = '[' + ', '.join(
                    '[' + ', '.join(act) + ']' for act in window
                ) + ']'
            else:
                chunk_str = '[]'
            
            chunked_actions.append(chunk_str)
        
        return chunked_actions
    
    def process_action(self, action: Union[str, List, np.ndarray]) -> Dict[str, str]:
        """Process a single action through full pipeline.
        
        Args:
            action: Raw action data
            
        Returns:
            Dictionary with normalized, tokenized, and chunked versions
        """
        # Parse if string
        if isinstance(action, str):
            action = self.json.loads(action)
        action = np.array(action, dtype=np.float32)
        
        # Normalize
        normalized = self.normalize_action_bounds_q99(action)
        
        # Tokenize
        tokens = self.tokenize_action(normalized)
        
        return {
            'normalized_action': normalized.tolist(),
            'tokenized_action': '[' + ', '.join(tokens) + ']'
        }
            
