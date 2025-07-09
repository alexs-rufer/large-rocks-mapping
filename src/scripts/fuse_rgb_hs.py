import argparse
from pathlib import Path
from tifffile import imread, imwrite
from tqdm import tqdm
import numpy as np

"""
Fuse RGB patches with hill-shade by replacing one RGB band. This is a necesseray step for inference with models which were trained using this image fusion technique.

Example usage:
python src/scripts/fuse_rgb_hs.py --rgb_dir data/raw/swissImage_50cm_patches --hs_dir data/raw/swissSURFACE3D_hillshade_patches --out_dir data/processed/images_hs_fusion --channel 1
"""

def fuse(rgb_path: Path, hs_path: Path, ch: int) -> np.ndarray:
    """Return fused image where RGB[..., ch] is replaced by hill-shade."""
    rgb = imread(rgb_path)
    hs  = imread(hs_path)
    if hs.ndim != 2:
        hs = hs.squeeze()
    fused = rgb.copy()
    fused[..., ch] = hs.astype(rgb.dtype)
    return fused


def main() -> None:
    p = argparse.ArgumentParser("Fuse RGB with hill-shade (channel replacement).")
    p.add_argument("--rgb_dir", required=True, type=Path,
                   help="Directory with RGB .tif patches")
    p.add_argument("--hs_dir",  required=True, type=Path,
                   help="Directory with hill-shade .tif patches")
    p.add_argument("--out_dir", required=True, type=Path,
                   help="Destination directory for fused images")
    p.add_argument("--channel", type=int, default=1, choices=[0, 1, 2],
                   help="RGB channel to replace (0=R, 1=G, 2=B)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rgb_files = sorted(args.rgb_dir.glob("*.tif"))
    if not rgb_files:
        raise FileNotFoundError(f"No .tif images found in {args.rgb_dir}")

    print(f"Fusing {len(rgb_files)} patches …")
    for rgb_path in tqdm(rgb_files, unit="img"):
        hs_path = args.hs_dir / rgb_path.name
        if not hs_path.exists():
            print(f"[WARN] No hill-shade for {rgb_path.name} – skipped.")
            continue
        fused = fuse(rgb_path, hs_path, args.channel)
        imwrite(args.out_dir / rgb_path.name, fused)

    print(f"Done. Fused images saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
