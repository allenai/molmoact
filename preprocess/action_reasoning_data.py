#!/usr/bin/env python3
"""
Refactored main processing script using modular processor classes.
"""
import argparse
import cv2
import io
import json
import numpy as np
import os
import shutil
import sys
import torch
from datasets import load_from_disk
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Optional

# Get base directory dynamically
current_file = os.path.abspath(__file__)
data_preprocess_dir = os.path.dirname(current_file)
molmoact_dir = os.path.dirname(data_preprocess_dir)
project_root = os.path.dirname(molmoact_dir)

# Add paths dynamically
sys.path.append(molmoact_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "Depth-Anything-V2"))
sys.path.append(os.path.join(project_root, "AiT", "vae"))

# Import processor classes
from processors import Depth, DepthTokens, Point, Trace, ActionProcessor


class DatasetProcessor:
    """Main dataset processor that coordinates all processing steps."""
    
    def __init__(self, 
                 depth_encoder: str = "vitb",
                 depth_checkpoint_dir: Optional[str] = None,
                 vqvae_model_path: Optional[str] = None,
                 line_length: int = 5,
                 tokenizer_model: str = "Qwen/Qwen2-7B",
                 action_bins: int = 256,
                 action_chunk_size: int = 8,
                 process_actions: bool = True):
        """Initialize all processors.
        
        Args:
            depth_encoder: Encoder type for depth model
            depth_checkpoint_dir: Directory containing depth model checkpoints
            vqvae_model_path: Path to VQVAE model checkpoint
            line_length: Number of points in trajectory lines
            tokenizer_model: Model name for action tokenizer
            action_bins: Number of bins for action discretization
            action_chunk_size: Size of action chunks for horizon
            process_actions: Whether to process actions
        """
        print("Initializing processors...")
        
        # Set default checkpoint directory if not provided
        if depth_checkpoint_dir is None:
            depth_checkpoint_dir = os.path.join(project_root, "Depth-Anything-V2", "checkpoints")
        
        # Initialize processors
        self.depth_processor = Depth(encoder=depth_encoder, ckpt_dir=depth_checkpoint_dir)
        self.token_processor = DepthTokens(model_path=vqvae_model_path)
        self.point_processor = Point()
        self.line_length = line_length
        
        # Initialize action processor if needed
        self.process_actions = process_actions
        if process_actions:
            self.action_processor = ActionProcessor(
                tokenizer_model=tokenizer_model,
                bins=action_bins,
                chunk_size=action_chunk_size
            )
        else:
            self.action_processor = None
        
        print("All processors initialized successfully!")
    
    
    def _get_image_from_example(self, example: Dict) -> Optional[object]:
        """Extract image from dataset example and convert to PIL Image.
        
        Args:
            example: Dataset example dictionary
            
        Returns:
            PIL Image or None if not found
        """
        # Try common field names
        image_data = None
        if 'image' in example:
            image_data = example['image']
        elif 'observation.image' in example:
            image_data = example['observation.image']
        else:
            # Try to find any image field (excluding wrist camera)
            image_keys = [k for k in example.keys() 
                        if 'image' in k.lower() and 'wrist' not in k.lower()]
            if image_keys:
                image_data = example[image_keys[0]]
        
        if image_data is None:
            return None
        
        # Convert to PIL Image based on type
        if isinstance(image_data, Image.Image):
            return image_data
        elif isinstance(image_data, torch.Tensor):
            # Convert from (C, H, W) to (H, W, C) and from [0, 1] to [0, 255]
            image_array = (image_data.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            return Image.fromarray(image_array)
        elif isinstance(image_data, np.ndarray):
            # Handle numpy arrays
            if image_data.dtype in (np.float32, np.float64):
                image_array = (np.clip(image_data, 0, 1) * 255).astype(np.uint8)
            else:
                image_array = image_data.astype(np.uint8)
            return Image.fromarray(image_array)
        elif isinstance(image_data, dict):
            # HuggingFace datasets return images as dict after reset_format
            if 'bytes' in image_data:
                return Image.open(io.BytesIO(image_data['bytes'])).convert('RGB')
            elif 'path' in image_data:
                return Image.open(image_data['path']).convert('RGB')
        elif isinstance(image_data, bytes):
            return Image.open(io.BytesIO(image_data)).convert('RGB')
        elif isinstance(image_data, list):
            # HuggingFace may return decoded images as nested lists (could be C,H,W or H,W,C)
            try:
                image_array = np.array(image_data, dtype=np.uint8)
                # Check if it's channel-first (C, H, W) and transpose to (H, W, C)
                if len(image_array.shape) == 3:
                    if image_array.shape[0] in [1, 3, 4]:  # Channels first: (C, H, W)
                        image_array = np.transpose(image_array, (1, 2, 0))  # Convert to (H, W, C)
                    
                    if image_array.shape[2] in [1, 3, 4]:  # Valid channel-last format
                        return Image.fromarray(image_array)
                    else:
                        print(f"Warning: Invalid image shape after transpose: {image_array.shape}")
                        return None
                else:
                    print(f"Warning: Invalid image dimensions: {image_array.shape}")
                    return None
            except Exception as e:
                print(f"Warning: Could not convert list to image: {e}")
                return None
        else:
            print(f"Warning: Unknown image type: {type(image_data)}")
            return None
        
        
    def _get_episode_id(self, example: Dict) -> int:
        """Extract episode ID from dataset example.
        
        Args:
            example: Dataset example dictionary
            
        Returns:
            Episode ID (defaults to 0 if not found)
        """
        # return example.get('episode_index', 
        #                   example.get('episode_idx', 
        #                              example.get('episode', 0)))
        ep_id = example.get('episode_index', 
                       example.get('episode_idx', 
                                  example.get('episode', 0)))
    
        # Convert tensor to int if needed
        if hasattr(ep_id, 'item'):  # PyTorch tensor
            return ep_id.item()
        return int(ep_id)
    
    
    def collect_gripper_points(self, dataset) -> Dict[int, List[Dict]]:
        """Collect gripper points for all frames grouped by episode.
        
        Args:
            dataset: HuggingFace dataset
            
        Returns:
            Dictionary mapping episode_id to list of frame info
        """
        print("Collecting gripper points...")
        episodes = {}
        
        for idx, example in enumerate(tqdm(dataset, desc="Getting gripper points")):

            episode_id = self._get_episode_id(example)
            if idx < 5:  # Debug first 5 frames
                print(f"Frame {idx}: episode_id = {episode_id}")
            pil_image = self._get_image_from_example(example)
            
            if pil_image is None:
                print(f"Warning: No image found in frame {idx}")
                point = None
            else:
                # Get gripper point using Point processor
                point = self.point_processor.inference_point(pil_image)
            
            # Group by episode
            if episode_id not in episodes:
                episodes[episode_id] = []
            
            episodes[episode_id].append({
                'idx': idx,
                'point': point
            })
        

        print(f"Episode structure:")
        print(f"Episodes dict keys: {list(episodes.keys())}")  # Add this line
        print(f"Key types: {[type(k) for k in episodes.keys()]}")  # And this
        for ep_id, frames in episodes.items():
            print(f"  Episode {ep_id} (type: {type(ep_id)}): {len(frames)} frames")
        for ep_id, frames in episodes.items():
            print(f"  Episode {ep_id}: {len(frames)} frames")
            if len(frames) > 0:
                print(f"    First 5 frame indices: {[f['idx'] for f in frames[:5]]}")
                print(f"    First 5 points: {[f['point'] for f in frames[:5]]}")

        return episodes
    
    def build_trajectory_lines(self, episodes: Dict[int, List[Dict]]) -> Dict[int, List[List[int]]]:
        """Build trajectory lines for each frame.
        
        Args:
            episodes: Dictionary of episodes with frame info
            
        Returns:
            Dictionary mapping frame index to trajectory line
        """
        print("Building trajectory lines...")
        frame_lines = {}
        
        for episode_id, frames in episodes.items():
            n = len(frames)
            points = [f['point'] for f in frames]
            
            # For each frame in episode, build line of future points
            for i in range(n):
                frame_idx = frames[i]['idx']
                current_point = points[i]
                
                # Create Trace object with future points
                print(f"Debug points {points[i:]}")
                trace = Trace(points[i:])
                print(f"Debug trace {trace}")
                
                # Get subsampled line
                line = trace.subsample_to_line(self.line_length, current_point)
                frame_lines[frame_idx] = line
        
        return frame_lines
    
    def collect_episode_actions(self, dataset) -> Dict[int, List[Dict]]:
        """Collect actions for all frames grouped by episode.
        
        Args:
            dataset: HuggingFace dataset
            
        Returns:
            Dictionary mapping episode_id to list of frame info with actions
        """
        if not self.process_actions:
            return {}
        
        print("Collecting actions by episode...")
        episodes = {}
        
        for idx, example in enumerate(tqdm(dataset, desc="Collecting actions")):
            episode_id = self._get_episode_id(example)
            
            action = None
            if 'action' in example:
                action = example['action']
            elif 'actions' in example:  # Add this line to handle the 'actions' field
                action = example['actions']
            elif 'raw_action' in example:
                action = example['raw_action']
            
            # Group by episode
            if episode_id not in episodes:
                episodes[episode_id] = []
            
            episodes[episode_id].append({
                'idx': idx,
                'action': action
            })
        
        return episodes
    
    def process_episode_actions(self, episodes: Dict[int, List[Dict]]) -> Dict[int, str]:
        """Process actions for each episode with chunking.
        
        Args:
            episodes: Dictionary of episodes with frame info and actions
            
        Returns:
            Dictionary mapping frame index to processed action string
        """
        if not self.process_actions or not episodes:
            return {}
        
        print("Processing actions with normalization and chunking...")
        frame_processed_actions = {}
        
        # First compute statistics from all actions
        all_actions = []
        for episode_id, frames in episodes.items():
            for frame in frames:
                if frame['action'] is not None:
                    all_actions.append(frame)
        
        # Create temporary dataset-like structure for statistics
        temp_dataset = all_actions
        self.action_processor.compute_dataset_statistics(temp_dataset)
        
        # Process each episode
        for episode_id, frames in tqdm(episodes.items(), desc="Processing episodes"):
            n = len(frames)
            
            # Process all actions in episode
            episode_tokenized_actions = []
            for frame in frames:
                if frame['action'] is not None:
                    processed = self.action_processor.process_action(frame['action'])
                    tokenized = processed['tokenized_action']
                    # Parse tokenized action string to list
                    tokens = tokenized.strip('[]').split(', ') if tokenized != '[]' else []
                    episode_tokenized_actions.append(tokens)
                else:
                    episode_tokenized_actions.append([])
            
            # Chunk the tokenized actions
            chunked_actions = self.action_processor.chunk_actions(episode_tokenized_actions)
            
            # Assign to frames
            for i, frame in enumerate(frames):
                frame_idx = frame['idx']
                if frame['action'] is not None:
                    # Create processed action with all components
                    processed = self.action_processor.process_action(frame['action'])
                    processed_action = {
                        'normalized_action': processed['normalized_action'],
                        'tokenized_action': processed['tokenized_action'],
                        'chunked_action': chunked_actions[i]
                    }
                    frame_processed_actions[frame_idx] = processed_action
                else:
                    frame_processed_actions[frame_idx] = None
        
        return frame_processed_actions
    
    def process_frame(self, example: Dict, idx: int, frame_lines: Dict[int, List[List[int]]], 
                     frame_processed_actions: Optional[Dict[int, Dict]] = None) -> Dict:
        """Process a single frame to add depth and trace information.
        
        Args:
            example: Dataset example
            idx: Frame index
            frame_lines: Pre-computed trajectory lines
            
        Returns:
            Updated example with depth and trace
        """
        try:
            pil_image = self._get_image_from_example(example)
            
            if pil_image is None:
                print(f"Warning: No image found in frame {idx}")
                example['depth'] = ""
                example['trace'] = "[]"
                return example
            
            # Convert PIL to cv2 format for depth model
            rgb_array = np.array(pil_image.convert('RGB'))
            bgr_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            
            # 1. Generate depth image
            depth_image = self.depth_processor.inference_depth_from_rgb(bgr_image)
            
            # 2. Convert depth to grayscale and get tokens
            depth_gray = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
            depth_tokens = self.token_processor.inference_depth_tokens(depth_gray)
            
            # 3. Get the pre-computed line for this frame
            line = frame_lines.get(idx, [])
            
            # Update example
            example['depth'] = depth_tokens
            example['trace'] = str(line)  # Store as string representation of list
            
            # 4. Add processed action if available
            if frame_processed_actions and idx in frame_processed_actions:
                processed_action = frame_processed_actions[idx]
                if processed_action:
                    # Store as JSON string
                    example['processed_action'] = json.dumps(processed_action)
                else:
                    example['processed_action'] = "{}"
            elif self.process_actions:
                example['processed_action'] = "{}"
            
            if idx % 100 == 0:
                status_msg = f"Processed frame {idx}: depth tokens={len(depth_tokens)}, trace points={len(line)}"
                if self.process_actions and 'processed_action' in example:
                    status_msg += f", action processed={example['processed_action'] != '{}'}"
                print(status_msg)
            
        except Exception as e:
            print(f"Error processing frame {idx}: {e}")
            example['depth'] = ""
            example['trace'] = "[]"
            if self.process_actions:
                example['processed_action'] = "{}"
        
        return example
    
    def process_dataset(self, dataset_path: str, output_path: Optional[str] = None, num_proc: int = 1, episodes: Optional[List[int]] = None):
        """Process entire dataset with depth and trace information.
        
        Args:
            dataset_path: Path to input dataset
            output_path: Path for output dataset (defaults to overwriting input)
            num_proc: Number of parallel workers for gripper point detection
            episodes: List of episode indices to process (None = all episodes)
        """
        print(f"Loading dataset from {dataset_path}...")
        if episodes is not None:
            print(f"Filtering to episodes: {episodes}")
        
        # Try loading with LeRobotDataset first

        if episodes is not None:
            lerobot_ds = LeRobotDataset(repo_id=dataset_path, episodes=episodes)
        else:
            lerobot_ds = LeRobotDataset(repo_id=dataset_path)
        
        # Extract task mapping from metadata
        task_mapping = {}
        if hasattr(lerobot_ds, 'meta') and hasattr(lerobot_ds.meta, 'tasks'):
            print("Tasks:")
            print(lerobot_ds.meta.tasks)
            # Check if tasks is already a dict or a list of dicts
            if isinstance(lerobot_ds.meta.tasks, dict):
                # Already in the format {task_index: task_string}
                task_mapping = lerobot_ds.meta.tasks
            elif isinstance(lerobot_ds.meta.tasks, list):
                # List of dicts with 'task_index' and 'task' keys
                for task_info in lerobot_ds.meta.tasks:
                    task_mapping[task_info['task_index']] = task_info['task']
            print(f"Loaded {len(task_mapping)} tasks from metadata")
        else:
            print("Warning: No task metadata found in dataset")
        
        dataset = lerobot_ds.hf_dataset
            
            
        print(f"Dataset loaded: {len(dataset)} frames")
        print(f"Features: {list(dataset.features.keys())}")
        
        # Ensure required fields exist
        required_fields = ['depth', 'trace', 'language_instruction']
        if self.process_actions:
            required_fields.append('processed_action')
        
        missing_fields = [f for f in required_fields if f not in dataset.features]
        if missing_fields:
            print(f"Adding missing fields: {missing_fields}")
            
            def add_fields(x):
                result = {**x}
                if 'depth' not in x:
                    result['depth'] = ""
                if 'trace' not in x:
                    result['trace'] = "[]"
                if 'processed_action' not in x:
                    result['processed_action'] = "{}"
                if 'language_instruction' not in x:
                    # Get task_index and look up instruction
                    task_idx = x.get('task_index', 0)
                    # Convert tensor to int if needed
                    if hasattr(task_idx, 'item'):
                        task_idx = task_idx.item()
                    result['language_instruction'] = task_mapping.get(int(task_idx), "")
                return result
            
            dataset = dataset.map(
                add_fields,
                desc="Adding missing fields"
            )
        
        # Print sample frame fields for debugging
        if len(dataset) > 0:
            print(f"Sample frame fields: {list(dataset[0].keys())}")
        
        # Phase 1: Collect all gripper points
        episodes = self.collect_gripper_points(dataset)
        
        # Phase 2: Build trajectory lines
        frame_lines = self.build_trajectory_lines(episodes)
        
        # Phase 3: Process actions if enabled
        frame_processed_actions = None
        if self.process_actions:
            action_episodes = self.collect_episode_actions(dataset)
            frame_processed_actions = self.process_episode_actions(action_episodes)
        
        # Phase 4: Process all frames with depth, traces, and actions
        print("Processing frames with all features...")
        dataset.reset_format()
        processed_dataset = dataset.map(
            lambda example, idx: self.process_frame(example, idx, frame_lines, frame_processed_actions),
            with_indices=True,
            desc="Adding depth, trace, and action data",
            num_proc=1  
        )
        
        # Save processed dataset
        if output_path is None:
            output_path = dataset_path
        
        # Handle case where output_path is same as input path
        if os.path.abspath(output_path) == os.path.abspath(dataset_path):
            temp_path = dataset_path + "_temp"
            print(f"Saving to temporary location: {temp_path}...")
            processed_dataset.save_to_disk(temp_path)
            
            print(f"Moving processed dataset back to {dataset_path}...")
            shutil.rmtree(dataset_path)
            shutil.move(temp_path, dataset_path)
            print("Done!")
        else:
            print(f"Saving processed dataset to {output_path}...")
            processed_dataset.save_to_disk(output_path)
            print("Done!")
        
        # Save dataset statistics if action processing is enabled
        if self.process_actions and self.action_processor.stats:
            stats_output = os.path.join(
                output_path if output_path else dataset_path,
                "dataset_statistics.json"
            )
            dataset_name = os.path.basename(dataset_path)
            self.action_processor.save_dataset_statistics(
                output_path=stats_output,
                dataset_name=dataset_name,
                stats=self.action_processor.stats
            )
        
        # Print statistics
        print("\nProcessing complete!")
        print(f"Total frames processed: {len(processed_dataset)}")
        
        # Sample check
        sample = processed_dataset[0]
        print(f"\nSample frame 0:")
        print(f"  Depth tokens (first 100 chars): {sample['depth'][:100] if sample['depth'] else 'None'}...")
        print(f"  Trace line: {sample['trace']}")
        if 'language_instruction' in sample:
            print(f"  Language instruction: {sample['language_instruction']}")
        if self.process_actions and 'processed_action' in sample:
            print(f"  Processed action: {sample['processed_action'] if sample['processed_action'] else 'None'}...")


def main():
    """Main entry point."""
    # Get default dataset path relative to project structure
    default_dataset_path = os.path.join(molmoact_dir, "data", "libero")
    
    parser = argparse.ArgumentParser(
        description="Process LeRobot dataset with depth and trace information (refactored version)"
    )
    parser.add_argument("--dataset-path", type=str, 
                        default=default_dataset_path,
                        help="Path to LeRobot dataset to process")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Output path for processed dataset (default: overwrites input dataset)")
    parser.add_argument("--line-length", type=int, default=5,
                        help="Number of points in trajectory line (default: 5)")
    parser.add_argument("--depth-encoder", type=str, default="vitb",
                        choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help="Depth model encoder type (default: vitb)")
    parser.add_argument("--depth-checkpoint-dir", type=str, default=None,
                        help="Path to Depth-Anything-V2 checkpoints (default: auto-detect)")
    parser.add_argument("--vqvae-model-path", type=str, default=None,
                        help="Path to VQVAE model checkpoint (default: use env var or built-in path)")
    
    # Action processing arguments
    parser.add_argument("--process-actions", action="store_true",
                        help="Enable action processing (normalization, tokenization, chunking)")
    parser.add_argument("--no-process-actions", dest="process_actions", action="store_false",
                        help="Disable action processing")
    parser.set_defaults(process_actions=True)
    parser.add_argument("--tokenizer-model", type=str, default="Qwen/Qwen2-7B",
                        help="Pretrained tokenizer model for actions (default: Qwen/Qwen2-7B)")
    parser.add_argument("--action-bins", type=int, default=256,
                        help="Number of discretization bins for actions (default: 256)")
    parser.add_argument("--action-chunk-size", type=int, default=8,
                        help="Size of action chunks for horizon (default: 8)")
    parser.add_argument("--episodes", type=str, default=None,
                        help="Comma-separated list of episode indices to process (e.g., '0,1,2' or '0' for testing). Default: process all episodes")
    
    args = parser.parse_args()
    
    # Parse episodes argument
    episodes = None
    if args.episodes:
        episodes = [int(e.strip()) for e in args.episodes.split(',')]
    
    # Create processor
    processor = DatasetProcessor(
        depth_encoder=args.depth_encoder,
        depth_checkpoint_dir=args.depth_checkpoint_dir,
        vqvae_model_path=args.vqvae_model_path,
        line_length=args.line_length,
        tokenizer_model=args.tokenizer_model,
        action_bins=args.action_bins,
        action_chunk_size=args.action_chunk_size,
        process_actions=args.process_actions
    )
    
    # Process the dataset
    processor.process_dataset(args.dataset_path, args.output_path, episodes=episodes)


if __name__ == "__main__":
    main()
