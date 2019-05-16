import os
from pathlib import Path
from matplotlib import cm

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import pickle
import cv2
from PIL import Image


def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguemnt for creating 3d")
    parser.add_argument(
        "--root", type=str,
        default="./data/davis_tempered/gt",
        help="ground truth folder"
    )

    args = parser.parse_args()
    print(args)

    root = Path(args.root)

    scale = 1/10

    color1 = (128, 0, 0)
    color2 = (0, 128, 0)

    for each_npy in tqdm(root.iterdir()):
        with each_npy.open("rb") as fp:
            Data = pickle.load(fp)

        T = len(Data)
        for i, fname in tqdm(enumerate(Data)):
            mask_orig = Data[fname]["mask_orig"]
            mask_new = Data[fname]["mask_new"]

            if i == 0:
                # r, c = int(mask_new.shape[0]*scale), int(mask_new.shape[1]*scale)
                r, c = 30, 30
                X1 = np.zeros((T, r, c), dtype=bool)
                X2 = np.zeros((T, r, c), dtype=bool)

            mask_orig = cv2.resize(mask_orig, (c, r),
                                   interpolation=cv2.INTER_NEAREST)
            mask_new = cv2.resize(mask_new, (c, r),
                                   interpolation=cv2.INTER_NEAREST)
            X1[i] = mask_orig > 0
            X2[i] = mask_new > 0

            # im_mask = np.empty((r, c, 3), dtype = np.uint8)

            # # im_mask[mask_orig] = color1
            # mask_new = mask_new > 0
            # mask_orig = mask_orig > 0

            # im_mask[mask_new] = color2
            # mask_total = (mask_new)
            # mask_total = Image.fromarray((mask_total*255).astype(np.uint8))
            # mask_total = mask_total.convert("L")

            # im = Image.fromarray(im_mask)
            # im.putalpha(mask_total)

            # filename = Path("./data/davis_tempered/tmp") / fname.parts[-2] / \
            #     f"{i}.png"
            # filename.parent.mkdir(exist_ok=True, parents=True)
            # im.save(filename)

        ind = np.random.choice(T, size=int(T), replace=False)
        ind.sort()

        X1 = X1[ind]
        X2 = X2[ind]

        X1 = X1.transpose(2, 0, 1)
        X2 = X2.transpose(2, 0, 1)


        Colors1 = np.empty(X1.shape, dtype=object)
        Colors2 = np.empty(X2.shape, dtype=object)

        Colors1[X1] = 'blue'
        Colors2[X2] = 'red'

        print(X1.shape)

        # sns.set_style("white")
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.voxels(X1, facecolors=Colors1)
        ax.voxels(X2, facecolors=Colors2)
        # ax.scatter(*X1.nonzero())
        # ax.scatter(*X2.nonzero())
        ax.set_xlabel("x")
        ax.set_ylabel("T")
        ax.set_zlabel("y")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])



        plt.show()
        plt.close('all')
        break
