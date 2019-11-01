from __future__ import print_function
import os
from pathlib import Path
import skimage.io as io
import skimage
import numpy as np
import argparse
import shutil
import utils
from tqdm import tqdm
import pickle
import sys


if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def it_repeat(ar, shuffle=True):
    if shuffle:
        np.random.shuffle(ar)
    _len = len(ar)
    i = 0
    while True:
        yield ar[i%_len]
        i += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguemnt for DAVIS synthetic dataset")
    parser.add_argument(
        "--num", "-n", type=int, default=-1, help="total manipulated video"
    )
    parser.add_argument("--seed", "-s", type=int, default=0,
                        help="random seed")

    parser.add_argument("--offset", type=int, default=0)

    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)

    HOME = Path(os.environ["HOME"])
    root = HOME / "dataset" / "DAVIS"

    ann_path = root / "Annotations" / "480p"
    im_sets_root = root / "ImageSets" / "2017"

    im_root = root / "JPEGImages" / "480p"

    _all_sets = [f.name for f in ann_path.iterdir()]
    all_sets = it_repeat(_all_sets)

    if args.num < 0:
        args.num = len(_all_sets)

    for i in tqdm(range(args.num)):
        # for k in range(28):
        #     next(all_sets)
        v_src = next(all_sets)
        v_tar = v_src  # Copy move, so same video

        print(v_src)

        this_write_dir = Path(f"./data/ttmp_davis_tempered/vid/{i}_{v_tar}")
        this_write_data_file = Path(f"./data/ttmp_davis_tempered/gt/{i}_{v_tar}.pkl")

        this_write_dir_gt_mask = Path(f"./data/ttmp_davis_tempered/gt_mask/{i}_{v_tar}")

        Data_dict = {}  # Data to save gt

        # create some directories if not exist
        try:
            this_write_dir.mkdir(parents=True)
        except Exception as e:
            shutil.rmtree(str(this_write_dir))
            this_write_dir.mkdir(parents=True)

        try:
            this_write_dir_gt_mask.mkdir(parents=True)
        except Exception as e:
            shutil.rmtree(str(this_write_dir_gt_mask))
            this_write_dir_gt_mask.mkdir(parents=True)

        this_write_data_file.parent.mkdir(parents=True, exist_ok=True)

        v_src_folder = im_root / v_src
        v_tar_folder = im_root / v_tar
        mask_folder = ann_path / v_src

        v_tar_folder_list = sorted(v_tar_folder.iterdir())
        v_src_folder_list = sorted(v_src_folder.iterdir())
        mask_folder_list = sorted(mask_folder.iterdir())

        while True:
            _str = np.random.randint(0, int(len(v_tar_folder_list)*0.3)+1)
            _end = np.random.randint(
                int(len(v_tar_folder_list)*0.7),
                len(v_tar_folder_list)
            )
            if _end - _str < 80:
                break

        tar_images = v_tar_folder_list[_str:_end]
        src_images = list(zip(
            v_src_folder_list[_str:_end],
            mask_folder_list[_str:_end]
        ))

        offset = np.random.randint(0, int(len(tar_images)/2))
        if offset > 0:
            src_images = [(None, None)] * offset + src_images
            # first offset frame has no source

        all_images = zip(tar_images, src_images)

        prev_ind = -1
        prev_scale = -1
        prev_xy = (None, None)
        translate = None

        for counter, (tar, src) in enumerate(all_images):
            im_t = skimage.img_as_float(io.imread(tar))

            if src[0] is None:  # do not manipulate
                im_s = im_t.copy()
                im_mask, im_mask_new = None, None
            else:
                im_s = io.imread(src[0])
                im_mask = io.imread(src[1], as_gray=True)
                im_mask = skimage.img_as_float(im_mask)

                # if there are several masks
                uniq = np.unique(im_mask)
                if uniq.size > 2:
                    try:
                        choice
                    except NameError:
                        # chose one, and reuse it for every frames next
                        choice = np.random.choice(uniq[1:])
                    finally:
                        if choice in uniq:
                            _mask = im_mask == choice
                            im_mask[:] = 0
                            im_mask[_mask] = 1
                        else:
                            src = (None, None)
                            im_mask, im_mask_new = None, None
                if im_mask is not None and not np.any(im_mask > 0):
                    src = (None, None)
                    im_mask, im_mask_new = None, None

            if src[0] is not None:
                if prev_ind == -1:
                    ret = utils.tmp_fn_max_trans(im_mask, prev_ind, prev_scale, prev_xy)
                    if ret is None:
                        break
                    else:
                        translate, scale, centroid, mask_orig_bb = ret
                if translate is not None:
                    prev_ind = 0
                    prev_scale = scale

                    # im_mask_new = utils.patch_transform(im_mask, mask_orig_bb, centroid,
                    #                                     translate, scale)

                    im_mask_b = im_mask > 0
                    im_mask_new = im_mask_b

                    im_mask_bool = im_mask_b[..., None].repeat(3, 2)
                    # im_s_masked = im_mask_bool[..., None] * im_s
                    #
                    tfm = skimage.transform.SimilarityTransform(
                        translation = -translate, scale=1)
                    im_s_tfm = skimage.transform.warp(im_s, tfm)
                    im_mask_bool = skimage.transform.warp(im_mask_bool, tfm)
                    im_mani = im_mask_bool * im_s_tfm + (1-im_mask_bool)*im_t
                    # get manipulated image
                    # im_mani = utils.splice(im_t, im_s_n, im_mask_new, do_blend=False)
                    im_s_new = np.zeros(im_mani.shape, dtype=np.uint8)
                    im_s_new[im_mask>0] = (255, 0, 0)
                    im_s_new[im_mask_bool[..., 0]>0.5] = (0, 0, 255)

                    im_mask_new = im_mask_bool[..., 0] > 0.5
            else:
                im_mani = im_t
                im_s_new = np.zeros(im_s.shape[:2], dtype=np.uint8)
                im_mask, im_mask_new = None, None

            fname = this_write_dir / f"{counter}.png"
            io.imsave(fname, skimage.img_as_ubyte(im_mani))

            fname2 = this_write_dir_gt_mask / f"{counter}.png"
            io.imsave(fname2, skimage.img_as_ubyte(im_s_new))

            Data_dict[fname] = {
                "source_image_file": src[0],
                "source_mask_file": src[1],
                "target_image_file": tar,
                "mask_orig": im_mask,
                "mask_new": im_mask_new,
                "offset": offset
            }

        with open(this_write_data_file, "wb") as fp:
            pickle.dump(Data_dict, fp)

        try:
            del choice
        except NameError:
            pass
