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
        # img_mani = poisson_edit(
        #     img_source, img_target, img_mask, offset=(0, 0)
        # )
        img_mani = Laplacian_Pyramid_Blending_with_mask(
            img_source, img_target, img_mask, num_levels=3
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

def check_same_side(ind, prev):
    left = (0, 3, 5)
    top = (0, 1, 2)
    bottom = (5, 6, 7)
    right = (2, 4, 7)

    Pos = {'left': left, 'right':right, 'top':top, 'bottom':bottom}

    for k, _pos in Pos.items():
        if np.any(ind in _pos and prev in _pos):
            if k in ('left', 'right') and abs(ind - prev) > 2:
                return False
            if k in ('top', 'bottom') and abs(ind - prev) > 1:
                return False
            return True
    return False


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

    key_BB = [min(wh_bb(x)) for x in BBs]

    max_ind = np.argmax(key_BB)
    max_bb = BBs[max_ind]

    return max_bb, mask_orig_bb, max_ind


def fn_argmax(key_BB, ratio_thres=0.8):
    _sort = np.argsort(key_BB)[::-1]
    if key_BB[_sort[1]] / key_BB[_sort[0]] > ratio_thres:
        if np.random.rand() > 0.2:
            return _sort[0]
        else:
            return _sort[1]
    else:
        return _sort[0]


def tmp_fn_max_trans(mask, prev_ind=-1, prev_scale=-1, prev_xy=(None, None)):
    h, w = mask.shape[:2]
    m_y, m_x = np.where(mask > 0)
    x1, x2 = np.min(m_x), np.max(m_x)
    y1, y2 = np.min(m_y), np.max(m_y)

    mask_orig_bb = (x1, y1, x2, y2)

    centroid_orig = centroid_bb(mask_orig_bb)
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

    key_BB = [(wh_bb(x)) for x in BBs]

    flag_BB = [((mw>w) & (mh>h))   for mw, mh in key_BB]

    cmp = [key_BB[i][0]*key_BB[i][1]  for i in range(len(flag_BB))]
    max_ind = np.argmax(cmp)
    max_bb = BBs[max_ind]

    if cmp[max_ind] == 0:
        return None


    def tmp_transform_mask(max_bb, mask_orig_bb, mask):
        centroid = centroid_bb(max_bb)
        centroid_orig = centroid_bb(mask_orig_bb)
        translate = centroid - centroid_orig
        return translate, 1, centroid

    translate, scale, centroid = tmp_transform_mask(max_bb, mask_orig_bb, mask)

    return translate, scale, centroid, mask_orig_bb


def fn_max_trans(mask, prev_ind=-1, prev_scale=-1):
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

    key_BB = [min(wh_bb(x)) for x in BBs]

    max_ind = fn_argmax(key_BB) # np.argmax(key_BB)
    max_bb = BBs[max_ind]

    try:
        translate, scale, centroid = transform_mask(max_bb, mask_orig_bb, mask)
        if prev_ind >= 0:
            translate_, scale_, centroid_ = transform_mask(
                BBs[prev_ind], mask_orig_bb, mask)
            if scale_ / prev_scale > 0.8:
                max_bb = BBs[prev_ind]
                max_ind = prev_ind
                return translate_, scale_, centroid_, mask_orig_bb, max_ind
            elif check_same_side(max_ind, prev_ind):
                return translate, scale, centroid, mask_orig_bb, max_ind
            else:
                return None
        elif prev_ind == -1:
            return translate, scale, centroid, mask_orig_bb, max_ind
        else:
            return None
    except ValueError:
        return None


def transform_mask(max_bb, mask_bb, mask, max_scale=1.2):
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

    scale = min(max_scale, fx, fy)

    # random scaling
    scale = (np.random.rand() * 0.05 + 0.95) * scale

    if scale <= 0:
        raise ValueError
        # import pdb
        # pdb.set_trace()

    return translate, scale, centroid


def patch_transform(im_mask, mask_bb, new_centroid, translate=None, scale=None):
    patch_mask = im_mask[mask_bb[1]:mask_bb[3], mask_bb[0]:mask_bb[2]]
    resized_patch = cv2.resize(patch_mask, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_NEAREST)

    hp, wp = resized_patch.shape[:2]

    topx = int(max(0, new_centroid[0] - wp/2))
    topy = int(max(0, new_centroid[1] - hp/2))

    new_mask = np.zeros(im_mask.shape, dtype=patch_mask.dtype)

    new_mask[topy:topy+hp, topx:topx+wp] = resized_patch

    return skimage.img_as_ubyte(new_mask)


def area_bb(x):
    return x[0] * x[1] * x[2] * x[3]

def centroid_bb(x):
    return np.array([int((x[0]+x[2])/2), int((x[1]+x[3])/2)])

def wh_bb(x):
    return x[2]-x[0], x[3]-x[1]


def Laplacian_Pyramid_Blending_with_mask(A, B, m, num_levels = 6):
    # assume mask is float32 [0,1]

    m = cv2.dilate(m, np.ones((2, 2)), iterations=1)

    m = m.astype(np.float)
    m[m > 0] = 1

    # generate Gaussian pyramid for A,B and mask
    GA = A.copy()
    GB = B.copy()
    GM = m.copy()
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in range(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(GA.astype(np.float))
        gpB.append(GB.astype(np.float))
        gpM.append(GM.astype(np.float))

    def same_shape(X, Y):
        r = min(X.shape[0], Y.shape[0])
        c = min(X.shape[1], Y.shape[1])
        return X[:r, :c], Y[:r, :c]

    # generate Laplacian Pyramids for A,B and masks
    lpA  = [gpA[num_levels-1]] # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lpB  = [gpB[num_levels-1]]
    gpMr = [gpM[num_levels-1]]
    for i in range(num_levels-1,0,-1):
        # Laplacian: subtarct upscaled version of lower level from current level
        # to get the high frequencies
        LA = np.subtract(*same_shape(gpA[i-1], cv2.pyrUp(gpA[i])))
        LB = np.subtract(*same_shape(gpB[i-1], cv2.pyrUp(gpB[i])))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i-1]) # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    for la,lb,gm in zip(lpA,lpB,gpMr):
        if len(gm.shape) < len(la.shape):
            gm = gm[..., None]
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1,num_levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(*same_shape(ls_, LS[i]))
    ls_ = ls_.astype(np.uint8)
    if B.shape != ls_.shape:
        ls_ = cv2.resize(ls_, (B.shape[1], B.shape[0]))

    return ls_

if __name__ == '__main__':
    ann = '/home/ashraful/dataset/DAVIS/Annotations/480p/bear/00000.png'
    fdest = '/home/ashraful/dataset/DAVIS/JPEGImages/480p/camel/00000.jpg'
    fim = '/home/ashraful/dataset/DAVIS/JPEGImages/480p/bear/00000.jpg'

    im1 = io.imread(fim)
    im2 = io.imread(fdest)
    mask = io.imread(ann, as_gray=True)
    mask[mask>0] = 1
    nim = Laplacian_Pyramid_Blending_with_mask(im1, im2, mask, num_levels=6)
    io.imshow(nim)
    io.show()
