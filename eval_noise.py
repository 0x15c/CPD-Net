import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

import geotnf.point_tnf
from train_rectangle import RectangleConfig, make_rectangle_points, normalize_points, denormalize_points


def apply_random_tps(points: np.ndarray, deform_level: float, use_cuda: bool) -> np.ndarray:
    """Apply a random TPS warp to the input points for visualization."""
    normalized, center, scale = normalize_points(points)
    points_tensor = torch.tensor(normalized.T[None, ...], dtype=torch.float32)
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
    warped = tps.tpsPointTnf(theta_tensor, points_tensor).cpu().numpy()[0]
    warped = warped.T

    return denormalize_points(warped, center, scale)


def replace_points_with_noise(points: np.ndarray, num_replace: int, noise_scale: float) -> np.ndarray:
    """Replace a subset of points with noisy samples around the cloud bounds."""
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


def plot_displacement(base_points: np.ndarray, warped_points: np.ndarray, save_path: str | None) -> None:
    """Plot displacement vectors from base points to warped points."""
    displacement = warped_points - base_points
    plt.figure(figsize=(6, 6))
    plt.quiver(
        base_points[:, 0],
        base_points[:, 1],
        displacement[:, 0],
        displacement[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="tab:red",
    )
    plt.scatter(base_points[:, 0], base_points[:, 1], c="tab:blue", label="Noisy Source")
    plt.scatter(warped_points[:, 0], warped_points[:, 1], c="tab:orange", label="Noisy Target")
    plt.legend()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Rectangle TPS displacement vectors with noise")
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize TPS displacement on noisy rectangle grid.")
    parser.add_argument("--rows", type=int, default=6, help="Number of rows in rectangle grid.")
    parser.add_argument("--cols", type=int, default=5, help="Number of columns in rectangle grid.")
    parser.add_argument("--spacing", type=float, default=100.0, help="Grid spacing in units.")
    parser.add_argument("--deform", type=float, default=0.4, help="TPS deformation level.")
    parser.add_argument("--replace-count", type=int, default=3, help="Points replaced by noise.")
    parser.add_argument("--noise-scale", type=float, default=0.1, help="Noise scale relative to span.")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save plot.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = RectangleConfig(rows=args.rows, cols=args.cols, spacing=args.spacing)
    base = make_rectangle_points(cfg)
    warped = apply_random_tps(base, deform_level=args.deform, use_cuda=torch.cuda.is_available())
    noisy_base = replace_points_with_noise(base, args.replace_count, args.noise_scale)
    noisy_warped = replace_points_with_noise(warped, args.replace_count, args.noise_scale)
    plot_displacement(noisy_base, noisy_warped, args.save)