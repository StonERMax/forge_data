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



def get_max_bb(mask):

    h, w = mask.shape[:2]
    m_y, m_x = np.where(mask > 0)
    x1, x2 = np.min(m_x), np.max(m_x)
    y1, y2 = np.min(m_y), np.max(m_y)

    mask_orig_bb = (x1, y1, x2, y2)

    BBs = [
        (0, 0, x1, y1),
        (x1, 0, x2, y1),
        (x2, 0, w, y1),
        (0, y1, x1, y2),
        (x2, y1, w, y2),
        (0, y2, x1, h),
        (x1, y2, x2, h),
        (x2, y2, w, h)
    ]

    max_bb = max(BBs, key = lambda x: min(wh_bb(x)))

    return max_bb, mask_orig_bb


def transform_mask(max_bb, mask_bb, mask):
    w, h = wh_bb(max_bb)
    pad = 0.1 * min(w, h)
    effective_bb = np.array([
        max_bb[0] + pad,
        max_bb[1] + pad,
        max_bb[2] - pad,
        max_bb[3] - pad
    ], dtype=np.int)

    centroid = centroid_bb(effective_bb)

    centroid_orig = centroid_bb(mask_bb)

    translate = centroid - centroid_orig

    w_n, h_n = wh_bb(effective_bb)
    w_o, h_o = wh_bb(mask_bb)

    fx, fy = w_n / w_o, h_n / h_o

    scale = min(fx, fy)

    if scale <= 0:
        import pdb
        pdb.set_trace()

    return translate, scale, centroid


def patch_transform(im_mask, mask_bb, new_centroid, translate=None, scale=None):
    patch_mask = im_mask[mask_bb[1]:mask_bb[3], mask_bb[0]:mask_bb[2]]
    resized_patch = cv2.resize(patch_mask, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_NEAREST)

    hp, wp = resized_patch.shape[:2]

    topx = int(max(0, new_centroid[0] - wp/2))
    topy = int(max(0, new_centroid[1] - hp/2))

    new_mask = np.zeros(im_mask.shape, dtype=np.float)

    new_mask[topy:topy+hp, topx:topx+wp] = resized_patch

    return new_mask


def area_bb(x):
    return x[0] * x[1] * x[2] * x[3]

def centroid_bb(x):
    return np.array([int((x[0]+x[2])/2), int((x[1]+x[3])/2)])

def wh_bb(x):
    return x[2]-x[0], x[3]-x[1]