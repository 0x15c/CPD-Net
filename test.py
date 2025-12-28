import glob
import os
import numpy as np
import matplotlib.pylab as plt
import torch
from torch.utils.data import DataLoader
import LoaderFish
from model import PointRegressor, chamfer_loss


def _ensure_point_shape(points: np.ndarray) -> np.ndarray:
    """Ensure point set is shaped as (N, 2) for downstream PyTorch code."""
    if points.shape[0] == 2 and points.shape[1] != 2:
        return points.T
    return points


def chamfer_loss_np(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Numpy Chamfer distance used for reporting results."""
    r = np.sum(A * A, 2)
    r = np.reshape(r, [int(r.shape[0]), int(r.shape[1]), 1])
    r2 = np.sum(B * B, 2)
    r2 = np.reshape(r2, [int(r.shape[0]), int(r.shape[1]), 1])
    t = r - 2 * np.matmul(A, np.transpose(B, (0, 2, 1))) + np.transpose(r2, (0, 2, 1))
    return np.mean((np.min(t, axis=1) + np.min(t, axis=2)) / 2.0, axis=-1)


def vis_Bat(xx, yy, yyp, name):
    """Visualize a batch of point sets with arrows showing the deformation."""
    fig = plt.figure(1, figsize=(20, 40))
    for i in range(8):
        x = xx[i]
        y = yy[i]
        yp = yyp[i]

        ax = fig.add_subplot(8, 4, i * 4 + 1)
        plt.scatter(x[:, 0], x[:, 1], label="source", s=5, c="r")
        plt.scatter(y[:, 0], y[:, 1], label="target", s=20, c="b", marker="x")
        plt.ylim(-3.5, 3.5)
        plt.xlim(-3.5, 3.5)
        ax.axis("off")

        ax = fig.add_subplot(8, 4, i * 4 + 2)
        plt.scatter(x[:, 0], x[:, 1], label="source", s=5, c="r")
        plt.scatter(yp[:, 0], yp[:, 1], label="transformed", s=5, c="r")
        for ii in range(len(x)):
            plt.arrow(
                x[ii, 0],
                x[ii, 1],
                (yp[ii, 0] - x[ii, 0]),
                (yp[ii, 1] - x[ii, 1]),
                head_width=0.03,
                head_length=0.08,
                fc="k",
                ec="k",
            )

        plt.ylim(-3.5, 3.5)
        plt.xlim(-3.5, 3.5)
        ax.axis("off")

        ax = fig.add_subplot(8, 4, i * 4 + 3)
        plt.scatter(y[:, 0], y[:, 1], label="target", s=20, c="b", marker="x")
        plt.scatter(yp[:, 0], yp[:, 1], label="transformed", s=5, c="r")
        plt.ylim(-3.5, 3.5)
        plt.xlim(-3.5, 3.5)
        ax.axis("off")

        ax = fig.add_subplot(8, 4, i * 4 + 4)
        plt.scatter(x[:20, 0], x[:20, 1], label="source", s=5, c="r")
        plt.scatter(yp[:20, 0], yp[:20, 1], label="transformed", s=5, c="r")
        for ii in range(20):
            plt.arrow(
                x[ii, 0],
                x[ii, 1],
                (yp[ii, 0] - x[ii, 0]),
                (yp[ii, 1] - x[ii, 1]),
                head_width=0.04,
                head_length=0.04,
                fc="k",
                ec="k",
            )

        ax.axis("off")

    plt.savefig(name, transparent=True)
    plt.close("all")


def load_latest_checkpoint(weight_dir: str) -> str | None:
    """Return the most recent PyTorch checkpoint in a directory, if any."""
    checkpoints = sorted(glob.glob(os.path.join(weight_dir, "model_step_*.pt")))
    if not checkpoints:
        return None
    return checkpoints[-1]


def test(weight_path: str, deform_level: float, test_num: int = 200):
    """Run evaluation on synthetic data and return source/target/predicted sets."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available; using CPU. Check CUDA_VISIBLE_DEVICES and your PyTorch build.")

    model = PointRegressor().to(device)
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = LoaderFish.PointRegDataset(
        total_data=test_num,
        deform_level=deform_level,
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

    # Use the same random pairing logic as training for evaluation.
    target_list = [_ensure_point_shape(item) for item in dataset.target_list]
    sources = np.stack(target_list, axis=0)
    targets = np.stack(target_list, axis=0)

    # Convert to torch tensors for inference.
    source_tensor = torch.tensor(sources, dtype=torch.float32, device=device)
    target_tensor = torch.tensor(targets, dtype=torch.float32, device=device)

    with torch.no_grad():
        displacement = model(source_tensor, target_tensor)
        pred = source_tensor + displacement

    return (
        sources,
        targets,
        pred.cpu().numpy(),
    )


if __name__ == "__main__":
    # Optionally set CUDA_VISIBLE_DEVICES in your shell to select a specific GPU.

    weight_dirs = sorted(glob.glob("./UnSup-fish/*"))
    if not weight_dirs:
        raise RuntimeError("No weight directories found under ./UnSup-fish")

    print("Weight directories found:", weight_dirs)

    Bef_te = []
    Aft_te = []

    for weight_dir in weight_dirs:
        def_level = float(weight_dir.split("Def_")[-1].split("_")[0])
        checkpoint_path = load_latest_checkpoint(weight_dir)
        if checkpoint_path is None:
            print(f"No PyTorch checkpoints found in {weight_dir}, skipping.")
            continue

        print("deformation level:", def_level)
        print("Weight is loaded from:", checkpoint_path)

        S, T, TS = test(checkpoint_path, def_level, test_num=200)

        org = chamfer_loss_np(T, S)
        aft = chamfer_loss_np(T, TS)
        Bef_te.append([np.mean(org), np.std(org)])
        Aft_te.append([np.mean(aft), np.std(aft)])

        if not os.path.exists("result_visuliation"):
            os.makedirs("result_visuliation")
        vis_Bat(S, T, TS, "result_visuliation/test.png")
        print("#####################################################")

    print("C.D. for Inputs (mean+-std):", Bef_te)
    print("C.D. for Outputs (mean+-std):", Aft_te)
