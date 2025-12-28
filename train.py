import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import LoaderFish
from model import PointRegressor, chamfer_loss


def set_seed(seed: int) -> None:
    """Set seeds for reproducible runs (as far as the data generator allows)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    Resolve the training device and emit a clear log.

    GPU usage can be disabled if CUDA_VISIBLE_DEVICES hides GPUs or if PyTorch
    is installed without CUDA support. We print the selected device so users
    can verify that training is actually using the GPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
        return device
    device = torch.device("cpu")
    print("CUDA not available; using CPU. Check CUDA_VISIBLE_DEVICES and your PyTorch build.")
    return device


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

    device = get_device()

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
    # Optionally set CUDA_VISIBLE_DEVICES in your shell to select a specific GPU.
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
