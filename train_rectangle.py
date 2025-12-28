import argparse
import os
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from train import PointRegressor, chamfer_loss, get_device
import geotnf.point_tnf


@dataclass
class RectangleConfig:
    """Configuration for rectangle point set generation."""
    rows: int = 6
    cols: int = 5
    spacing: float = 100.0


def make_rectangle_points(cfg: RectangleConfig) -> np.ndarray:
    """
    Create a grid-like rectangle point set.

    The grid uses roughly 30 points (rows * cols) and spans ~500 units
    along the longer axis, matching the user's requested scale.
    """
    xs = np.arange(cfg.cols, dtype=np.float32) * cfg.spacing
    ys = np.arange(cfg.rows, dtype=np.float32) * cfg.spacing
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)
    return points


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Normalize points to roughly [-1, 1] for TPS stability.

    Returns normalized points, center used for normalization, and scale factor.
    """
    center = points.mean(axis=0, keepdims=True)
    max_extent = np.max(np.ptp(points, axis=0))
    scale = max_extent / 2.0 if max_extent > 0 else 1.0
    normalized = (points - center) / scale
    return normalized, center, scale


def denormalize_points(points: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    """Undo normalize_points by restoring the original scale and center."""
    return points * scale + center


def apply_random_tps(points: np.ndarray, deform_level: float, use_cuda: bool) -> np.ndarray:
    """
    Apply a random TPS warp to the input points.

    This mirrors the deformation logic in LoaderFish: start from a 3x3 TPS
    control grid in [-1, 1] and add random offsets scaled by deform_level.
    """
    normalized, center, scale = normalize_points(points)

    # TPS expects points in [B, 2, N].
    points_tensor = torch.tensor(normalized.T[None, ...], dtype=torch.float32)
    if use_cuda:
        points_tensor = points_tensor.cuda()

    # 3x3 control points (9 for x, 9 for y). Same base as original code.
    base_theta = np.array(
        [-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1],
        dtype=np.float32,
    )
    noise = (np.random.rand(18).astype(np.float32) - 0.5) * 2.0 * deform_level
    theta = base_theta + noise
    theta_tensor = torch.tensor(theta[None, ...], dtype=torch.float32)
    if use_cuda:
        theta_tensor = theta_tensor.cuda()

    tps = geotnf.point_tnf.PointTnf(use_cuda=use_cuda)
    warped = tps.tpsPointTnf(theta_tensor, points_tensor).cpu().numpy()[0]
    warped = warped.T

    return denormalize_points(warped, center, scale)


class RectanglePairDataset(Dataset):
    """
    Dataset that yields (source, target) pairs for the rectangle grid.

    Source points are the original grid; target points are TPS-warped versions.
    """

    def __init__(self, total_samples: int, deform_level: float, config: RectangleConfig):
        self.total_samples = total_samples
        self.deform_level = deform_level
        self.base_points = make_rectangle_points(config)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Use the same base points for each sample and apply a random TPS warp.
        source = self.base_points
        target = apply_random_tps(source, self.deform_level, use_cuda=torch.cuda.is_available())
        return (
            torch.tensor(source, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )


def train_rectangle(args: argparse.Namespace) -> None:
    """Train CPD-Net on the rectangle grid using TPS-warped targets."""
    device = get_device()

    config = RectangleConfig(rows=args.rows, cols=args.cols, spacing=args.spacing)
    dataset = RectanglePairDataset(
        total_samples=args.train_samples,
        deform_level=args.deform_levels[args.deform_key],
        config=config,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = PointRegressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    os.makedirs(args.save_dir, exist_ok=True)

    step = 0
    model.train()
    while step < args.max_steps:
        for source, target in loader:
            if step >= args.max_steps:
                break

            source = source.to(device)
            target = target.to(device)

            optimizer.zero_grad(set_to_none=True)
            displacement = model(source, target)
            pred = source + displacement
            loss = chamfer_loss(pred, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % args.log_every == 0:
                print(f"Step {step} | Loss: {loss.item():.6f}")

            if step % args.save_every == 0:
                checkpoint_path = os.path.join(args.save_dir, f"rect_model_step_{step}.pt")
                torch.save({"model_state": model.state_dict(), "step": step}, checkpoint_path)

            step += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CPD-Net on rectangle TPS pairs.")
    parser.add_argument("--rows", type=int, default=6, help="Number of rows in rectangle grid.")
    parser.add_argument("--cols", type=int, default=5, help="Number of columns in rectangle grid.")
    parser.add_argument(
        "--spacing",
        type=float,
        default=100.0,
        help="Spacing between grid points in units (controls ~500 unit span).",
    )
    parser.add_argument("--train-samples", type=int, default=5000, help="Training samples.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--max-steps", type=int, default=10000, help="Max training steps.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--lr-decay", type=float, default=0.999, help="LR decay per step.")
    parser.add_argument("--log-every", type=int, default=100, help="Log frequency.")
    parser.add_argument("--save-every", type=int, default=1000, help="Checkpoint frequency.")
    parser.add_argument("--save-dir", type=str, default="./rect_checkpoints", help="Save dir.")
    parser.add_argument(
        "--deform-key",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
        help="Select deformation level from the predefined dict.",
    )

    args = parser.parse_args()

    # Deformation levels mimic the original dict-like configuration.
    args.deform_levels = {
        "low": 0.2,
        "medium": 0.4,
        "high": 0.8,
    }
    return args


if __name__ == "__main__":
    train_rectangle(parse_args())
