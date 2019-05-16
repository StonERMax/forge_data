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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguemnt for DAVIS synthetic dataset")
    parser.add_argument(
        "--num", "-n", type=int, default=2, help="total manipulated video"
    )
    parser.add_argument("--seed", "-s", type=int, defaut=0,
                        help="random seed")

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
        v_src, v_tar = np.random.choice(all_sets, size=2, replace=False)
        this_write_dir = Path(f"./data/davis_tempered/vid/{i}")
        this_write_txt_file = Path(f"./data/davis_tempered/gt/{i}.txt")

        try:
            this_write_dir.mkdir(parents=True)
        except Exception as e:
            shutil.rmtree(str(this_write_dir))
            this_write_dir.mkdir(parents=True)

        this_write_txt_file.parent.mkdir(parents=True, exist_ok=True)

        gt_str = ""

        v_src_folder = im_root / v_src
        v_tar_folder = im_root / v_tar
        mask_folder = ann_path / v_src

        offset = 0

        tar_images = list(sorted(v_tar_folder.iterdir()))
        src_images = list(
            zip(sorted(v_src_folder.iterdir()), sorted(mask_folder.iterdir()))
        )

        if offset > 0:
            src_images = [(None, None)] * offset + src_images

        all_images = zip(tar_images, src_images)

        for counter, (tar, src) in enumerate(all_images):
            im_t = io.imread(tar)
            im_s = io.imread(src[0])
            im_mask = io.imread(src[1], as_gray=True)

            im_mani = utils.splice(im_t, im_s, im_mask, do_blend=True)

            fname = this_write_dir / f"{counter}.jpg"
            io.imsave(fname, im_mani)
            gt_str += f"{fname},{tar},{src[0]},{src[1]}\n"

        with this_write_txt_file.open("w") as fp:
            fp.write(gt_str)
