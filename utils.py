import os
import cv2
from PIL import Image
import numpy as np
import skimage.io as io
import skimage
from PIL import Image

from builtins import range
import scipy.sparse
import pyamg

from poisson_edit import poisson_edit

def splice(img_target, img_source, img_mask, do_blend=False):

    if img_target.shape != img_source.shape:
        img_target = skimage.transform.resize(
            img_target, img_mask.shape[:2], anti_aliasing=True, mode='reflect'
        )
        img_target = skimage.img_as_ubyte(img_target)

    if do_blend:
        # img_mani = blend(
        #     img_target, img_source, img_mask
        # )
        img_mani = poisson_edit(
            img_source, img_target, img_mask, offset=(0, 0)
        )
        return img_mani

    if len(img_mask.shape) < 3:
        img_mask = img_mask[..., None]

    img_mask = (img_mask > 0)
    if img_mask.dtype != np.float:
        img_mask = img_mask.astype(np.float)
    img_mani = img_mask * img_source + img_target * (1 - img_mask)
    img_mani = img_mani.astype(np.uint8)


    return img_mani


def blend(img_target, img_source, img_mask, offset=(0, 0)):
    # compute regions to be blended
    region_source = (
        max(-offset[0], 0),
        max(-offset[1], 0),
        min(img_target.shape[0] - offset[0], img_source.shape[0]),
        min(img_target.shape[1] - offset[1], img_source.shape[1]),
    )
    region_target = (
        max(offset[0], 0),
        max(offset[1], 0),
        min(img_target.shape[0], img_source.shape[0] + offset[0]),
        min(img_target.shape[1], img_source.shape[1] + offset[1]),
    )
    region_size = (
        region_source[2] - region_source[0],
        region_source[3] - region_source[1],
    )

    # clip and normalize mask image
    img_mask = img_mask[
        region_source[0] : region_source[2], region_source[1] : region_source[3]
    ]
    img_mask[img_mask == 0] = False
    img_mask[img_mask != False] = True

    # create coefficient matrix
    A = scipy.sparse.identity(np.prod(region_size), format="lil")
    for y in range(region_size[0]):
        for x in range(region_size[1]):
            if img_mask[y, x]:
                index = x + y * region_size[1]
                A[index, index] = 4
                if index + 1 < np.prod(region_size):
                    A[index, index + 1] = -1
                if index - 1 >= 0:
                    A[index, index - 1] = -1
                if index + region_size[1] < np.prod(region_size):
                    A[index, index + region_size[1]] = -1
                if index - region_size[1] >= 0:
                    A[index, index - region_size[1]] = -1
    A = A.tocsr()

    # create poisson matrix for b
    P = pyamg.gallery.poisson(img_mask.shape)

    # for each layer (ex. RGB)
    for num_layer in range(img_target.shape[2]):
        # get subimages
        t = img_target[
            region_target[0] : region_target[2],
            region_target[1] : region_target[3],
            num_layer,
        ]
        s = img_source[
            region_source[0] : region_source[2],
            region_source[1] : region_source[3],
            num_layer,
        ]
        t = t.flatten()
        s = s.flatten()

        # create b
        b = P * s
        for y in range(region_size[0]):
            for x in range(region_size[1]):
                if not img_mask[y, x]:
                    index = x + y * region_size[1]
                    b[index] = t[index]

        # solve Ax = b
        x = pyamg.solve(A, b, verb=False, tol=1e-10)

        # assign x to target image
        x = np.reshape(x, region_size)
        x[x > 255] = 255
        x[x < 0] = 0
        x = np.array(x, img_target.dtype)
        img_target[
            region_target[0] : region_target[2],
            region_target[1] : region_target[3],
            num_layer,
        ] = x

    return img_target

