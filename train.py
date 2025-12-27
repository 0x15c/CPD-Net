import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import LoaderFish


def set_seed(seed: int) -> None:
    """Set seeds for reproducible runs (as far as the data generator allows)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SharedMLP(nn.Module):
    """
    Shared MLP implemented with 1x1 convolutions.

    In the original TensorFlow+sugartensor code, the network used sg_conv with
    kernel size (1, 1) applied to per-point features. A 1x1 Conv1d does the
    same thing when the data is shaped as (B, C, N).
    """

    def __init__(self, in_channels: int, channels: list[int]) -> None:
        super().__init__()
        layers = []
        for out_channels in channels:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            in_channels = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the stacked 1x1 convolutions."""
        return self.net(x)


class PointRegressor(nn.Module):
    """
    PyTorch reimplementation of the original TensorFlow CPD-Net.

    The model takes a source point set and a target point set and predicts a
    per-point displacement for the source. The architecture preserves the two
    separate MLP branches from the original code (gen* and gen9* blocks), then
    fuses the point-wise source coordinates with global features from both
    branches before regressing the displacement vectors.
    """

    def __init__(self) -> None:
        super().__init__()
        # Target branch (corresponds to gen9/gen1/gen2/gen3/gen4 in TF code).
        self.target_mlp = SharedMLP(2, [16, 64, 128, 256, 512])
        # Source branch (corresponds to gen99/gen11/gen22/gen33/gen44 in TF code).
        self.source_mlp = SharedMLP(2, [16, 64, 128, 256, 512])

        # Fusion head (f1/f2/f3 in TF code).
        self.fusion = nn.Sequential(
            nn.Conv1d(2 + 512 + 512, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(256, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(128, 2, kernel_size=1),
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            source: (B, N, 2) tensor of source points.
            target: (B, N, 2) tensor of target points.

        Returns:
            (B, N, 2) tensor of displacement vectors to apply to source points.
        """
        # Convert to (B, C, N) layout for Conv1d layers.
        source_channels = source.transpose(1, 2)
        target_channels = target.transpose(1, 2)

        # Extract per-point features for the target branch.
        target_features = self.target_mlp(target_channels)
        target_global = torch.max(target_features, dim=2, keepdim=True).values
        target_global = target_global.expand(-1, -1, target_features.shape[2])

        # Extract per-point features for the source branch.
        source_features = self.source_mlp(source_channels)
        source_global = torch.max(source_features, dim=2, keepdim=True).values
        source_global = source_global.expand(-1, -1, source_features.shape[2])

        # Fuse raw coordinates with global features from both branches.
        fused = torch.cat([source_channels, source_global, target_global], dim=1)
        displacement = self.fusion(fused)

        # Return to (B, N, 2) layout.
        return displacement.transpose(1, 2)


def chamfer_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the symmetric Chamfer distance between two point sets.

    This mirrors the TensorFlow implementation in the original repo by
    calculating pairwise distances, taking the minimum in each direction, and
    averaging the results.
    """
    # pred/target: (B, N, 2)
    diff = pred.unsqueeze(2) - target.unsqueeze(1)
    dist = (diff ** 2).sum(dim=-1)
    min_pred = dist.min(dim=2).values
    min_target = dist.min(dim=1).values
    return ((min_pred + min_target) / 2.0).mean()


def _ensure_point_shape(points: np.ndarray) -> np.ndarray:
    """Ensure point set is shaped as (N, 2) for downstream PyTorch code."""
    if points.shape[0] == 2 and points.shape[1] != 2:
        return points.T
    return points


class RandomPairDataset(Dataset):
    """
    Dataset that returns (source, target) pairs using the original training logic.

    The TensorFlow pipeline built source/target pairs from the same list of
    synthesized point sets by randomly reordering the sources. We replicate the
    same idea here by pairing each target with a randomly sampled source.
    """

    def __init__(self, target_list: list[np.ndarray]) -> None:
        self.targets = [
            torch.tensor(_ensure_point_shape(item), dtype=torch.float32)
            for item in target_list
        ]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        target = self.targets[index]
        source = self.targets[random.randrange(len(self.targets))]
        return source, target


def train():
    set_seed(888)
    print("*****************************************")
    print("Training started with random seed: {}".format(111))
    print("Batch started with random seed: {}".format(111))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate synthetic training data using the original LoaderFish pipeline.
    dataset = LoaderFish.PointRegDataset(
        total_data=train_num,
        deform_level=def_level,
        noise_ratio=0,
        outlier_ratio=0,
        outlier_s=False,
        outlier_t=False,
        noise_s=False,
        noise_t=False,
        missing_points=0,
        miss_source=False,
        miss_targ=False,
        clas=1,
    )

    train_dataset = RandomPairDataset(dataset.target_list)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batSize,
        shuffle=True,
        drop_last=True,
    )

    model = PointRegressor().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learningRate,
        betas=(adam_beta1, adam_beta2),
    )

    # Match TensorFlow's exponential decay schedule.
    lr_lambda = lambda step: learningRateDecay ** (step / batSize)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    start_step = 1
    if len(conWeightPath) > 0:
        checkpoint_path = conWeightPath
        print("Continue Training...")
        print("Reading Weight:{}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_step = checkpoint["step"] + 1

    if not os.path.exists(dirSave):
        os.makedirs(dirSave)

    step = start_step
    model.train()
    while step <= maxStep:
        for source, target in train_loader:
            if step > maxStep:
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

            if step % printStep == 0:
                print(f"Step {step} | Loss: {loss.item():.6f}")

            if step % saveStep == 0:
                checkpoint = {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "step": step,
                }
                torch.save(checkpoint, os.path.join(dirSave, f"model_step_{step}.pt"))

            step += 1


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # change it to the number of gpu.
    train_num = 20000  # number of synthesized training pairs
    deformation_list = [0.4]
    batSize = 8
    maxStep = 100000  # fixed with learningRate and learningRateDecay
    learningRate = 0.001
    learningRateDecay = 0.999
    adam_beta1 = 0.9  # check adam optimization
    adam_beta2 = 0.99
    saveStep = 20000  # frequency to save weight
    maxKeepWeights = 2000  # how many records to save (for disk)
    stepsContinue = -1  # from which steps continue.
    # For Debug and results printing
    keepProb = 0.99999
    printStep = 1000
    s1 = 91
    s2 = 91
    clas = "fish"

    for def_level in deformation_list:
        dat = "Exp1"
        dirSave = "./UnSup-{}/{}_Def_{}_trNum_{}_maxStep_{}".format(
            clas, dat, def_level, train_num, maxStep
        )
        conWeightPath = ""
        train()
