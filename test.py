import glob
import os
import numpy as np
import matplotlib.pylab as plt
import torch
from torch.utils.data import DataLoader
import LoaderFish
from train import PointRegressor, chamfer_loss


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
            plt.arrow(x[ii,0],x[ii,1],(yp[ii,0]-x[ii,0]),(yp[ii,1]-x[ii,1]), head_width=0.04, head_length=0.04, fc='k', ec='k')
        
#         plt.ylim(-3.5, 3.5)
#         plt.xlim(-3.5, 3.5)
        ax.axis('off')
        
#     plt.show()
    plt.savefig(name,transparent=True)
    plt.close('all')
    
    
os.environ['CUDA_VISIBLE_DEVICES']="1"

DaTf=sorted(glob.glob("./Def_train_*.*_20000.tfrecords"))[:8]
Weig=sorted(glob.glob("./UnSup-fish/*"))

print("Data for training: ",DaTf)
print("Weight is loaded from: ", Weig)

Def_lv=[]
Bef_tr=[]
Aft_tr=[]
Bef_te=[]
Aft_te=[]

for i in range(len(Weig)):
    
    def_level=float(Weig[i].split("Def_")[-1].split("_")[0])
    print("deformation level : ", def_level)
    tr_d=DaTf[i]
    wt=Weig[i]

    Def_lv.append(def_level)
    
    test_num=200
    a=LoaderFish.PointRegDataset(total_data=test_num, 
                  deform_level=def_level,
                  noise_ratio=0, 
                  outlier_ratio=0, 
                  outlier_s=False,
                    outlier_t=False, 
                    noise_s=False, 
                    noise_t=False,
                  missing_points=0,
                  miss_source=False,
                    miss_targ=False)

    try:
        os.remove("temp_test_1.tfrecords")
    except:
        print("fine, you don't have such files")
    write_to_tfrecords({"source":np.asanyarray([i.T for i in a.target_list])[np.random.choice(range(test_num),test_num)],
                        "target":np.asanyarray([i.T for i in a.target_list])},"temp_test_1.tfrecords")

    S,T,TS=test("temp_test_1.tfrecords",wt+"/", lll=2)
    S=np.asanyarray(S).reshape(-1,91,2)
    T=np.asanyarray(T).reshape(-1,91,2)
    TS=np.asanyarray(TS).reshape(-1,91,2)
    org=chamfer_loss_np(T,S)
    aft=chamfer_loss_np(T,TS)
    Bef_te.append([np.mean(org), np.std(org)])
    Aft_te.append([np.mean(aft),np.std(aft)])
    if not os.path.exists("result_visuliation"):
        os.makedirs("result_visuliation")
    vis_Bat(S,T,TS,"result_visuliation/test.png")
    print("#####################################################")
    
print("C.D. for Inputs (mean+-std): ", Bef_te)
print("C.D. for Outputs (mean+-std): ", Aft_te)