import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

import LoaderFish
from train import PointRegressor
from train_outlier_fish import to_point_tensor


def load_model(checkpoint_path: str, device: torch.device) -> PointRegressor:
    model = PointRegressor().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def visualize_pair(source: np.ndarray, target: np.ndarray, pred: np.ndarray, save_path: str) -> None:
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.scatter(source[:, 0], source[:, 1], c="r", s=8)
    ax1.set_title("Source (with outliers)")
    ax1.axis("equal")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter(target[:, 0], target[:, 1], c="b", s=8)
    ax2.set_title("Target (with outliers)")
    ax2.axis("equal")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.scatter(target[:, 0], target[:, 1], c="b", s=8, label="Target")
    ax3.scatter(pred[:, 0], pred[:, 1], c="r", s=8, label="Predicted")
    for i in range(min(len(source), len(pred))):
        ax3.arrow(
            source[i, 0],
            source[i, 1],
            pred[i, 0] - source[i, 0],
            pred[i, 1] - source[i, 1],
            head_width=0.02,
            head_length=0.02,
            fc="k",
            ec="k",
            length_includes_head=True,
        )
    ax3.legend()
    ax3.set_title("Predicted displacement")
    ax3.axis("equal")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved visualization to {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize outlier-aware CPD-Net predictions.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save", type=str, default="result_visuliation/outlier_vis.png")

    parser.add_argument("--deform-level", type=float, default=0.4)
    parser.add_argument("--outlier-ratio", type=float, default=0.1)
    parser.add_argument("--outlier-source", action="store_true")
    parser.add_argument("--outlier-target", action="store_true")
    parser.add_argument("--point-size", type=int, default=91)
    parser.add_argument("--clas", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LoaderFish.PointRegDataset(
        total_data=1,
        point_size=args.point_size,
        deform_level=args.deform_level,
        outlier_ratio=args.outlier_ratio,
        outlier_s=args.outlier_source,
        outlier_t=args.outlier_target,
        noise_ratio=0,
        noise_s=False,
        noise_t=False,
        missing_points=0,
        miss_source=False,
        miss_targ=False,
        clas=args.clas,
    )

    target, source, _, _ = dataset[0]
    source_tensor = to_point_tensor(source).unsqueeze(0).to(device)
    target_tensor = to_point_tensor(target).unsqueeze(0).to(device)

    model = load_model(args.checkpoint, device)
    with torch.no_grad():
        displacement = model(source_tensor, target_tensor)
        pred = source_tensor + displacement

    visualize_pair(
        source_tensor.squeeze(0).cpu().numpy(),
        target_tensor.squeeze(0).cpu().numpy(),
        pred.squeeze(0).cpu().numpy(),
        args.save,
    )
