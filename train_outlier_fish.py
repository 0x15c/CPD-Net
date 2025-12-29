import argparse
import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import LoaderFish
from train import PointRegressor, get_device


def to_point_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convert LoaderFish arrays into (N, 2) float32 tensors."""
    if arr.shape[0] == 2 and arr.shape[1] != 2:
        arr = arr.T
    return torch.tensor(arr, dtype=torch.float32)


def trimmed_chamfer_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    trim_ratio: float,
) -> torch.Tensor:
    """
    Trimmed Chamfer loss to reduce the influence of outliers.

    We compute symmetric nearest-neighbor distances, then keep only the
    smallest (1 - trim_ratio) fraction before averaging.
    """
    diff = pred.unsqueeze(2) - target.unsqueeze(1)
    dist = (diff ** 2).sum(dim=-1)

    min_pred = dist.min(dim=2).values
    min_target = dist.min(dim=1).values

    def trimmed_mean(values: torch.Tensor) -> torch.Tensor:
        if trim_ratio <= 0:
            return values.mean(dim=1)
        k = max(1, int((1.0 - trim_ratio) * values.shape[1]))
        kept = torch.topk(values, k, largest=False).values
        return kept.mean(dim=1)

    loss_pred = trimmed_mean(min_pred)
    loss_target = trimmed_mean(min_target)
    return ((loss_pred + loss_target) / 2.0).mean()


class FishOutlierDataset(Dataset):
    """Dataset wrapping LoaderFish with outlier injection."""

    def __init__(
        self,
        total_samples: int,
        deform_level: float,
        outlier_ratio: float,
        outlier_s: bool,
        outlier_t: bool,
        point_size: int = 91,
        clas: int = 1,
    ) -> None:
        self.dataset = LoaderFish.PointRegDataset(
            total_data=total_samples,
            point_size=point_size,
            deform_level=deform_level,
            outlier_ratio=outlier_ratio,
            outlier_s=outlier_s,
            outlier_t=outlier_t,
            noise_ratio=0,
            noise_s=False,
            noise_t=False,
            missing_points=0,
            miss_source=False,
            miss_targ=False,
            clas=clas,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        target, source, _, _ = self.dataset[index]
        source_tensor = to_point_tensor(source)
        target_tensor = to_point_tensor(target)
        return source_tensor, target_tensor


def train(args: argparse.Namespace) -> None:
    device = get_device()

    dataset = FishOutlierDataset(
        total_samples=args.train_samples,
        deform_level=args.deform_level,
        outlier_ratio=args.outlier_ratio,
        outlier_s=args.outlier_source,
        outlier_t=args.outlier_target,
        point_size=args.point_size,
        clas=args.clas,
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
            loss = trimmed_chamfer_loss(pred, target, trim_ratio=args.trim_ratio)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % args.log_every == 0:
                print(f"Step {step} | Loss: {loss.item():.6f}")

            if step % args.save_every == 0:
                checkpoint_path = os.path.join(args.save_dir, f"outlier_model_step_{step}.pt")
                torch.save({"model_state": model.state_dict(), "step": step}, checkpoint_path)

            step += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CPD-Net with outlier-robust Chamfer loss.")
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lr-decay", type=float, default=0.999)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--save-dir", type=str, default="./outlier_checkpoints")

    parser.add_argument("--deform-level", type=float, default=0.4)
    parser.add_argument("--outlier-ratio", type=float, default=0.1)
    parser.add_argument("--outlier-source", action="store_true")
    parser.add_argument("--outlier-target", action="store_true")
    parser.add_argument("--trim-ratio", type=float, default=0.1)

    parser.add_argument("--point-size", type=int, default=91)
    parser.add_argument("--clas", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
