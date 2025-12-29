import argparse
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from train import PointRegressor, chamfer_loss, get_device


class H5PointSetDataset(Dataset):
    """
    Dataset that reads point-set pairs from an H5 file.

    Expected structure:
      - Root keys like "frame_1", "frame_2", ...
      - Each key contains datasets "c0" (source) and "c1" (target)

    Indexing starts from 1 in the file (e.g., frame_1, frame_2, ...).
    """

    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        with h5py.File(self.h5_path, "r") as f:
            self.keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = self.keys[index]
        with h5py.File(self.h5_path, "r") as f:
            source = np.array(f[key]["c0"], dtype=np.float32)
            target = np.array(f[key]["c"], dtype=np.float32)

        # Ensure shape is (N, 2)
        if source.shape[0] == 2 and source.shape[1] != 2:
            source = source.T
        if target.shape[0] == 2 and target.shape[1] != 2:
            target = target.T

        return torch.tensor(source, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


def train_with_existing_data(args: argparse.Namespace) -> None:
    device = get_device()

    dataset = H5PointSetDataset(args.h5_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = PointRegressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for source, target in loader:
            source = source.to(device)
            target = target.to(device)

            optimizer.zero_grad(set_to_none=True)
            displacement = model(source, target)
            pred = source + displacement
            loss = chamfer_loss(pred, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        scheduler.step()

        avg_loss = epoch_loss / max(batch_count, 1)
        if epoch % args.log_every == 0:
            print(f"Epoch {epoch} | Avg Loss: {avg_loss:.6f}")

        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(args.save_dir, f"h5_model_epoch_{epoch}.pt")
            torch.save({"model_state": model.state_dict(), "epoch": epoch}, checkpoint_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CPD-Net using H5 point-set data (epoch-based).")
    parser.add_argument("--h5-path", type=str, default="training_biotactip.h5", help="Path to training_pointset.h5")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-2, help="Adam learning rate.")
    parser.add_argument("--lr-decay", type=float, default=0.999, help="LR decay per epoch.")
    parser.add_argument("--log-every", type=int, default=1, help="Log frequency in epochs.")
    parser.add_argument("--save-every", type=int, default=10, help="Checkpoint frequency in epochs.")
    parser.add_argument("--save-dir", type=str, default="./h5_checkpoints", help="Checkpoint directory.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_with_existing_data(args)
