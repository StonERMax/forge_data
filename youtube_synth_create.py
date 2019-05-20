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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguemnt for Youtube synthetic dataset")
    parser.add_argument(
        "--num", "-n", type=int, default=10, help="total manipulated video"
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

    for i in tqdm(range(args.num)):
        v_src = np.random.choice(all_sets)
        v_tar = v_src  # Copy move, so same video

        print(v_src)


        gt_str = ""

        tmp = np.random.choice(list((im_root / v_src / "data").iterdir()))
        _num = tmp.stem


        v_src_folder = im_root / v_src/ "data" / _num / "shots" / "001" / "images"
        v_tar_folder = v_src_folder
        mask_folder = im_root / v_src / "data" / _num / "shots" / "001" / "labels"

        tar_images = sorted(v_tar_folder.iterdir())
        src_images = list(
            zip(sorted(v_src_folder.iterdir()), sorted(mask_folder.iterdir()))
        )

        if int(len(tar_images)/2) <=0:
            continue

        offset = np.random.randint(0, int(len(tar_images)/2))
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

        for counter, (tar, src) in enumerate(all_images):
            im_t = io.imread(tar)

            if src[0] is None:  # do not manipulate
                im_s = im_t.copy()
                im_mask, im_mask_new = None, None
            else:
                im_s = io.imread(src[0])
                im_mask = io.imread(src[1], as_gray=True)

                if im_s.shape[:2] != im_mask.shape[:2]:
                    im_mask = cv2.resize(im_mask, (im_s.shape[1], im_s.shape[0]))

                im_mask = skimage.img_as_float(im_mask)

                im_mask[im_mask > 0] = 1

            if src[0] is not None:
                # Convert mask and masked image
                max_bb, mask_orig_bb, ind_bb = utils.get_max_bb(im_mask)

                # check if new bb is close to prev bb
                if prev_ind != -1 and \
                        not utils.check_same_side(ind_bb, prev_ind):
                    # Do not manipulate the following frames
                    src = (None, None)
                    prev_ind = -2
                    im_mani = im_t
                    im_s_new = np.zeros(im_s.shape[:2], dtype=np.uint8)
                    im_mask = None
                    im_mask_new = None
                else:
                    prev_ind = ind_bb

                    translate, scale, centroid =  utils.transform_mask(
                                max_bb, mask_orig_bb, im_mask
                    )

                    im_mask_new = utils.patch_transform(im_mask, mask_orig_bb, centroid,
                                                        translate, scale)

                    im_mask_bool = im_mask > 0
                    im_s_masked = im_mask_bool[..., None] * im_s

                    im_s_n = utils.patch_transform(im_s_masked, mask_orig_bb,
                                                    centroid, translate, scale)

                    # get manipulated image
                    im_mani = utils.splice(im_t, im_s_n, im_mask_new, do_blend=False)
                    im_s_new = np.zeros(im_s_n.shape, dtype=np.uint8)
                    im_s_new[im_mask>0] = (255, 0, 0)
                    im_s_new[im_mask_new>0] = (0, 0, 255)
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
                "offset": offset
            }

        with open(this_write_data_file, "wb") as fp:
            pickle.dump(Data_dict, fp)

        try:
            del choice
        except NameError:
            pass