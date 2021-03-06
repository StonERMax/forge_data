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

    parser = argparse.ArgumentParser(
        description="Arguemnt for SegtrackV2 synthetic dataset"
    )
    parser.add_argument(
        "--num", "-n", type=int, default=-1, help="total manipulated video"
    )
    parser.add_argument("--seed", "-s", type=int, default=0, help="random seed")

    parser.add_argument("--blend", type=bool, default=False)

    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)

    HOME = Path(os.environ["HOME"])
    root = HOME / "dataset" / "SegTrackv2"

    ann_path = root / "GroundTruth"

    im_root = root / "JPEGImages"
    _all_sets = [f.name for f in ann_path.iterdir()]
    all_sets = it_repeat(_all_sets)

    if args.num < 0:
        args.num = len(_all_sets)

    for i in tqdm(range(args.num)):
        v_src = next(all_sets)
        v_tar = v_src  # Copy move, so same video

        print(v_src)

        this_write_dir = Path(f"./data/SegTrackv2_tempered/vid/{i}_{v_tar}")
        this_write_data_file = Path(f"./data/SegTrackv2_tempered/gt/{i}_{v_tar}.pkl")

        this_write_dir_gt_mask = Path(f"./data/SegTrackv2_tempered/gt_mask/{i}_{v_tar}")

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

        gt_str = ""

        v_src_folder = im_root / v_src
        v_tar_folder = im_root / v_tar
        mask_folder = ann_path / v_src

        mask_files = sorted(mask_folder.iterdir())

        if not any([flg.suffix in (".png", ".bmp", ".jpg") for flg in mask_files]):
            fldr = [x for x in mask_files if x.is_dir()]
            choice = np.random.choice(fldr)
            mask_folder = choice

        v_tar_folder_list = sorted(v_tar_folder.iterdir())
        v_src_folder_list = sorted(v_src_folder.iterdir())
        mask_folder_list = sorted(mask_folder.iterdir())

        _str = np.random.randint(0, int(len(v_tar_folder_list)*0.3)+1)
        _end = np.random.randint(
            int(len(v_tar_folder_list)*0.7),
            len(v_tar_folder_list)
        )

        tar_images = v_tar_folder_list[_str:_end+1]
        src_images = list(zip(
            v_src_folder_list[_str:_end+1],
            mask_folder_list[_str:_end+1]
        ))

        offset = np.random.randint(0, int(len(tar_images) * 0.4))
        if offset > 0:
            src_images = [(None, None)] * offset + src_images
            # first offset frame has no source

        all_images = zip(tar_images, src_images)

        prev_ind = -1
        prev_scale = -1

        for counter, (tar, src) in enumerate(all_images):
            im_t = io.imread(tar)

            if src[0] is None:  # do not manipulate
                im_s = im_t.copy()
                im_mask, im_mask_new = None, None
            else:
                im_s = io.imread(src[0])
                im_mask = io.imread(src[1], as_gray=True)
                im_mask = skimage.img_as_float(im_mask)

                if not np.any(im_mask > 0):
                    src = (None, None)
                    im_mask, im_mask_new = None, None

            if src[0] is not None:
                # Convert mask and masked image

                ret = utils.fn_max_trans(im_mask, prev_ind, prev_scale)

                # max_bb, mask_orig_bb, ind_bb = utils.get_max_bb(im_mask)
                # translate, scale, centroid = utils.transform_mask(
                #     max_bb, mask_orig_bb, im_mask
                # )
                # check if new bb is close to prev bb
                if ret is None:
                    # Do not manipulate the following frames
                    src = (None, None)
                    prev_ind = -2
                    im_mani = im_t
                    im_s_new = np.zeros(im_s.shape[:2], dtype=np.uint8)
                    im_mask = None
                    im_mask_new = None
                else:
                    translate, scale, centroid, mask_orig_bb, ind_bb = ret
                    prev_ind = ind_bb
                    prev_scale = scale
                    im_mask_new = utils.patch_transform(
                        im_mask, mask_orig_bb, centroid, translate, scale
                    )

                    im_mask_bool = im_mask > 0
                    im_s_masked = im_mask_bool[..., None] * im_s

                    im_s_n = utils.patch_transform(
                        im_s_masked, mask_orig_bb, centroid, translate, scale
                    )

                    # get manipulated image
                    im_mani = utils.splice(
                        im_t, im_s_n, im_mask_new, do_blend=args.blend
                    )
                    im_s_new = np.zeros(im_s_n.shape, dtype=np.uint8)
                    im_s_new[im_mask > 0] = (255, 0, 0)
                    im_s_new[im_mask_new > 0] = (0, 0, 255)
            else:
                im_mani = im_t
                im_s_new = np.zeros(im_s.shape[:2], dtype=np.uint8)

            fname = this_write_dir / f"{counter}.png"
            io.imsave(fname, im_mani)

            fname2 = this_write_dir_gt_mask / f"{counter}.jpg"
            io.imsave(fname2, im_s_new)

            Data_dict[fname] = {
                "source_image_file": src[0],
                "source_mask_file": src[1],
                "target_image_file": tar,
                "mask_orig": im_mask,
                "mask_new": im_mask_new,
                "offset": offset,
            }

        with open(this_write_data_file, "wb") as fp:
            pickle.dump(Data_dict, fp)
