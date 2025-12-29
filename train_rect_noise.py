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
    rows: int = 4
    cols: int = 8
    spacing: float = 1.0  # spacing in normalized units


def make_rectangle_points(rows: int, cols: int, spacing: float) -> np.ndarray:
    xs = np.arange(cols, dtype=np.float32) * spacing
    ys = np.arange(rows, dtype=np.float32) * spacing
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)


def normalize_to_unit(points: np.ndarray) -> np.ndarray:
    """
    Scale the rectangle to [0,1]x[0,1].
    """
    min_xy = points.min(axis=0, keepdims=True)
    max_xy = points.max(axis=0, keepdims=True)
    range_xy = np.maximum(max_xy - min_xy, 1e-6)
    return (points - min_xy) / range_xy


def apply_random_tps(points_unit: np.ndarray, deform_level: float, use_cuda: bool) -> np.ndarray:
    """
    Apply TPS on points in [0,1]x[0,1] by mapping to [-1,1].
    """
    # map [0,1] -> [-1,1]
    points_centered = (points_unit * 2.0) - 1.0
    points_tensor = torch.tensor(points_centered.T[None, ...], dtype=torch.float32)
    if use_cuda:
        points_tensor = points_tensor.cuda()

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
    warped = tps.tpsPointTnf(theta_tensor, points_tensor).cpu().numpy()[0].T

    # map back [-1,1] -> [0,1]
    return (warped + 1.0) / 2.0


def replace_points_with_noise(points: np.ndarray, num_replace: int, noise_scale: float) -> np.ndarray:
    if num_replace <= 0:
        return points
    noisy = points.copy()
    n = noisy.shape[0]
    num_replace = min(num_replace, n)

    min_xy = noisy.min(axis=0)
    max_xy = noisy.max(axis=0)
    span = max_xy - min_xy
    low = min_xy - noise_scale * span
    high = max_xy + noise_scale * span

    replace_idx = np.random.choice(n, num_replace, replace=False)
    noisy[replace_idx] = np.random.uniform(low, high, size=(num_replace, 2))
    return noisy


def random_shift(points: np.ndarray, t: float) -> np.ndarray:
    """
    Apply independent planar translation in range (-t, t) for x and y.
    """
    shift = np.random.uniform(-t, t, size=(1, 2)).astype(np.float32)
    return points + shift


class RectangleNoisyPairDataset(Dataset):
    """
    Dataset yielding (source, target) pairs with:
      - fixed grid size (rows/cols) so batch tensors align
      - normalization to [0,1]x[0,1]
      - TPS applied once to get target_unshifted
      - independent planar shifts applied to source and target
      - optional point replacement noise
    """

    def __init__(
        self,
        total_samples: int,
        deform_level: float,
        config: RectangleConfig,
        replace_count: int,
        noise_scale: float,
        shift_t: float,
    ):
        self.total_samples = total_samples
        self.deform_level = deform_level
        self.config = config
        self.replace_count = replace_count
        self.noise_scale = noise_scale
        self.shift_t = shift_t
        self.base = make_rectangle_points(config.rows, config.cols, config.spacing)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 1) Normalize rectangle to [0,1]x[0,1]
        source_unshifted = normalize_to_unit(self.base)

        # 2) TPS on source_unshifted to get target_unshifted
        target_unshifted = apply_random_tps(
            source_unshifted, self.deform_level, use_cuda=torch.cuda.is_available()
        )

        # 3) Independent planar shifts on source/target
        source = random_shift(source_unshifted, self.shift_t)
        target = random_shift(target_unshifted, self.shift_t)

        # 4) Optional noise replacement
        source = replace_points_with_noise(source, self.replace_count, self.noise_scale)
        target = replace_points_with_noise(target, self.replace_count, self.noise_scale)

        return (
            torch.tensor(source, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )


def train_rect_noise(args: argparse.Namespace) -> None:
    device = get_device()

    config = RectangleConfig(rows=args.rows, cols=args.cols, spacing=args.spacing)
    dataset = RectangleNoisyPairDataset(
        total_samples=args.train_samples,
        deform_level=args.deform_levels[args.deform_key],
        config=config,
        replace_count=args.replace_count,
        noise_scale=args.noise_scale,
        shift_t=args.shift_t,
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
                checkpoint_path = os.path.join(args.save_dir, f"rect_noise_step_{step}.pt")
                torch.save({"model_state": model.state_dict(), "step": step}, checkpoint_path)

            step += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CPD-Net on normalized noisy rectangle TPS pairs.")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--spacing", type=float, default=1.0)
    parser.add_argument("--shift-t", type=float, default=0.1, help="Translation range t for planar shifts.")
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lr-decay", type=float, default=0.999)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--save-dir", type=str, default="./rect_checkpoints")
    parser.add_argument("--deform-key", type=str, default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--replace-count", type=int, default=3)
    parser.add_argument("--noise-scale", type=float, default=0.1)

    args = parser.parse_args()
    args.deform_levels = {"low": 0.2, "medium": 0.4, "high": 0.8}
    return args


if __name__ == "__main__":
    train_rect_noise(parse_args())
