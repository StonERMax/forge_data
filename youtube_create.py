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
import cv2
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

    parser = argparse.ArgumentParser(description="Arguemnt for Youtube synthetic dataset")
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
    root = HOME / "dataset" / "youtube_masks"

    ann_path = root
    im_root = root

    all_sets = [f.name for f in ann_path.iterdir()]

    i = 0
    # for i in tqdm(range(args.num)):

    progress = tqdm(total=args.num)

    _all_sets = [f.name for f in ann_path.iterdir()]
    all_sets = it_repeat(_all_sets)

    if args.num < 0:
        args.num = len(_all_sets)

    while i <= args.num:
        v_src = next(all_sets)
        v_tar = v_src  # Copy move, so same video

        gt_str = ""

        now_list = list((im_root / v_src / "data").iterdir())
        tmp = np.random.choice(list((im_root / v_src / "data").iterdir()))
        _num = tmp.stem

        for tmp in now_list:
            _num = tmp.stem

            v_src_folder = im_root / v_src/ "data" / _num / "shots" / "001" / "images"
            v_tar_folder = v_src_folder
            mask_folder = im_root / v_src / "data" / _num / "shots" / "001" / "labels"

            print(v_src_folder)

            v_tar_folder_list = sorted(v_tar_folder.iterdir())
            v_src_folder_list = sorted(v_src_folder.iterdir())
            mask_folder_list = sorted(mask_folder.iterdir())

            _str = np.random.randint(0, int(len(v_tar_folder_list)*0.2)+1)
            _end = np.random.randint(
                int(len(v_tar_folder_list)*0.8),
                len(v_tar_folder_list)
            )

            tar_images = v_tar_folder_list[_str:_end+1]
            src_images = list(zip(
                v_src_folder_list[_str:_end+1],
                mask_folder_list[_str:_end+1]
            ))

            if int(len(tar_images)/2) <=0:
                continue

            offset = min(np.random.randint(0, int(len(tar_images)/2)), 5)
            if offset > 0:
                src_images = [(None, None)] * offset + src_images
                # first offset frame has no source

            tmp = v_src_folder.parts
            _flg = tmp[-6] + "_" + tmp[-4]
            this_write_dir = Path(f"./data/youtube_tempered/vid/{i}_{_flg}")
            this_write_data_file = Path(f"./data/youtube_tempered/gt/{i}_{_flg}.pkl")

            this_write_dir_gt_mask = Path(f"./data/youtube_tempered/gt_mask/{i}_{_flg}")

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

            all_images = zip(tar_images, src_images)

            prev_ind = -1
            prev_scale = -1
            prev_xy = (None, None)
            translate = None
            prev_translate = None

            for counter, (tar, src) in enumerate(all_images):
                im_t = skimage.img_as_float(io.imread(tar))

                if src[0] is None:  # do not manipulate
                    im_s = im_t.copy()
                    im_mask, im_mask_new = None, None
                else:
                    im_s = skimage.img_as_float(io.imread(src[0]))
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
                    # if prev_ind == -1:
                    ret = utils.tmp_fn_max_trans(
                        im_mask, prev_ind, prev_scale, prev_xy
                    )
                    if ret is None:
                        break
                    else:
                        translate, max_scale, centroid_orig, mask_orig_bb = ret

                        if prev_translate is None:
                            prev_translate = translate
                        else:
                            translate = prev_translate

                        if max_scale < 0.75:
                            translate = None
                        else:
                            if prev_scale == -1:

                                # TODO: this is where you change the scale
                                scale = np.random.choice(
                                    np.linspace(0.75, min(2, max_scale), 30)
                                )
                            else:
                                scale = prev_scale
                                if scale > max_scale:
                                    break

                    if translate is not None:
                        prev_ind = 0
                        prev_scale = scale

                        im_mask_new, flag = utils.patch_transform_s(
                            im_mask, mask_orig_bb, centroid_orig, translate, scale
                        )

                        if not flag:
                            break

                        im_mask_bool = im_mask > 0
                        im_s_masked = im_mask_bool[..., None] * im_s

                        im_s_n, _ = utils.patch_transform_s(
                            im_s_masked, mask_orig_bb, centroid_orig, translate, scale
                        )

                        # get manipulated image
                        im_mani = utils.splice(
                            im_t, im_s_n, im_mask_new, do_blend=False
                        )
                        im_s_new = np.zeros(im_s_n.shape, dtype=np.float)
                        im_s_new[im_mask > 0] = (1.0, 0, 0)
                        im_s_new[im_mask_new > 0] = (0, 0, 1.0)
                else:
                    im_mani = im_t
                    im_s_new = np.zeros(im_s.shape[:2], dtype=np.float)
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
                    "offset": offset,
                    "scale": prev_scale
                }

            if prev_ind >= 0:
                with open(this_write_data_file, "wb") as fp:
                    pickle.dump(Data_dict, fp)

        i += 1
        progress.update()
