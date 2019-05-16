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

