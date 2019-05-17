from __future__ import print_function
import os
from pathlib import Path
import skimage.io as io
import numpy as np
import os
import argparse
import shutil
import utils
from tqdm import tqdm
import pickle

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguemnt for DAVIS synthetic dataset")
    parser.add_argument(
        "--num", "-n", type=int, default=1, help="total manipulated video"
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

    all_sets = [f.name for f in ann_path.iterdir()]

    for i in tqdm(range(args.num)):
        v_src = np.random.choice(all_sets)
        v_tar = v_src

        this_write_dir = Path(f"./data/davis_tempered/vid/{i}")
        this_write_data_file = Path(f"./data/davis_tempered/gt/{i}.pkl")

        this_write_dir_gt_mask = Path(f"./data/davis_tempered/gt_mask/{i}")

        Data_dict = {}

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

        offset = args.offset # time

        tar_images = list(sorted(v_tar_folder.iterdir()))
        src_images = list(
            zip(sorted(v_src_folder.iterdir()), sorted(mask_folder.iterdir()))
        )

        if offset > 0:
            src_images = [(None, None)] * offset + src_images

        all_images = zip(tar_images, src_images)

        for counter, (tar, src) in enumerate(all_images):
            # print(f"{tar}")
            im_t = io.imread(tar)

            if src[0] is None:
                im_s = im_t.copy()
                im_mask = np.zeros(im_s.shape[:2], dtype=np.uint8)
            else:
                im_s = io.imread(src[0])
                im_mask = io.imread(src[1], as_gray=True)

            if src[0] is not None:
                # Convert mask and masked image
                max_bb, mask_orig_bb = utils.get_max_bb(im_mask)

                translate, scale, centroid =  utils.transform_mask(
                            max_bb, mask_orig_bb, im_mask
                )

                im_mask_new = utils.patch_transform(im_mask, mask_orig_bb, centroid,
                                                    translate, scale)

                im_mask_bool = im_mask > 0
                im_s_masked = im_mask_bool[..., None] * im_s

                im_s_new = utils.patch_transform(im_s_masked, mask_orig_bb,
                                                centroid, translate, scale)

                # get manipulated image
                im_mani = utils.splice(im_t, im_s_new, im_mask_new, do_blend=False)
            else:
                im_mani = im_t
                im_s_new = im_mask
                # im_mask_new = im_mask
                im_mask = None
                im_mask_new = None

            fname = this_write_dir / f"{counter}.jpg"
            io.imsave(fname, im_mani)

            fname2 = this_write_dir_gt_mask / f"{counter}.png"
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