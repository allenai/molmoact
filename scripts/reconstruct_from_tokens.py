import argparse
import math
import os
from typing import List

import torch
import torchvision.utils as vutils

from vqvae import VQVAE


def parse_depth_tokens(text: str) -> List[int]:
    """
    Parse depth tokens from input text.

    Supported formats:
      1) Old tagged format (e.g., "<DEPTH_START><DEPTH_12><DEPTH_7>...<DEPTH_END>")
      2) Comma/space separated ints (e.g., "12, 7, 3, 5" or "12 7 3 5")

    Returns:
        List[int]: parsed token ids.
    """
    s = text.strip()

    if "<DEPTH" in s:
        # Remove bookends if present
        s = s.replace("<DEPTH_START>", "").replace("<DEPTH_END>", "")
        # Normalize markers like "<DEPTH_123>" -> "123>"
        s = s.replace("<DEPTH_", "")
        # Now split on '>' and filter empties
        pieces = [p for p in s.split(">") if p.strip() != ""]
        try:
            return [int(p) for p in pieces]
        except ValueError as e:
            raise ValueError(f"Failed to parse tagged depth tokens: {e}") from e
    else:
        # Accept comma or whitespace separated integers
        # Replace commas with spaces, then split
        s = s.replace(",", " ")
        pieces = [p for p in s.split() if p.strip() != ""]
        try:
            return [int(p) for p in pieces]
        except ValueError as e:
            raise ValueError(f"Failed to parse depth tokens (expect ints): {e}") from e


def load_vae(ckpt_path: str) -> VQVAE:
    cfg_model = dict(
        image_size=320,
        num_resnet_blocks=2,
        downsample_ratio=32,
        num_tokens=128,
        codebook_dim=512,
        hidden_dim=16,
        use_norm=False,
        channels=1,  # depth = 1 channel
        train_objective="regression",
        max_value=10.0,
        residul_type="v1",
        loss_type="mse",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = VQVAE(**cfg_model).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)

    # Allow both raw state_dict or {'weights': state_dict}
    state = ckpt.get("weights", ckpt)

    # Strip 'module.' if present
    if len(state) > 0 and next(iter(state)).startswith("module."):
        state = {k[len("module.") :]: v for k, v in state.items()}

    vae.load_state_dict(state, strict=True)
    vae.eval()
    return vae


def infer_grid_hw(n_tokens: int) -> (int, int):
    """
    Infer an HxW grid from number of tokens.
    Prefers a square if possible; otherwise tries to factor into a near-square grid.
    """
    r = int(math.isqrt(n_tokens))
    if r * r == n_tokens:
        return r, r
    # Find factors closest to a square
    for h in range(r, 0, -1):
        if n_tokens % h == 0:
            w = n_tokens // h
            return h, w
    # Fallback (shouldn't happen)
    return 1, n_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Decode depth tokens with VQ-VAE and save reconstruction."
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to VQ-VAE checkpoint (.pt) containing 'weights' or a state_dict.",
    )
    parser.add_argument(
        "--depth_tokens",
        type=str,
        required=True,
        help=(
            "Depth tokens as a string. Either tagged format like "
            "'<DEPTH_START><DEPTH_1><DEPTH_2>...<DEPTH_END>' "
            "or comma/space-separated ints like '1,2,3,...'."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Where to save the reconstructed image (e.g., /path/to/out.png).",
    )
    args = parser.parse_args()

    # Parse tokens
    tokens = parse_depth_tokens(args.depth_tokens)
    if len(tokens) == 0:
        raise ValueError("No depth tokens parsed from input.")

    # Arrange tokens into (1, H, W) code grid
    H, W = infer_grid_hw(len(tokens))
    if H * W != len(tokens):
        raise ValueError(
            f"Token count {len(tokens)} cannot be reshaped to a grid. Got H={H}, W={W}."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    codes = torch.tensor(tokens, dtype=torch.long, device=device).view(1, H, W)

    # Load VAE
    vae = load_vae(args.ckpt_path)

    # Decode
    with torch.no_grad():
        out = vae.decode(codes)

    # Normalize to [0,1]
    out_min = out.amin(dim=(1, 2, 3), keepdim=True)
    out_max = out.amax(dim=(1, 2, 3), keepdim=True)
    # Avoid divide-by-zero if flat
    denom = torch.clamp(out_max - out_min, min=1e-8)
    out_norm = (out - out_min) / denom

    # Ensure directory exists
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    # Save as grayscale image
    vutils.save_image(out_norm, args.output_path)
    print(f"Saved reconstruction to: {args.output_path}")


if __name__ == "__main__":
    main()