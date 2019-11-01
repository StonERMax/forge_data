
import os
import numpy as np
import cv2
from multiprocessing import Pool
import argparse
from pathlib import Path
from skimage import io
from tqdm import tqdm


def ToImg(flow, bound):
    '''
    this function scale the input pixels to 0-255 with bi-bound

    :param raw_flow: input raw pixel value (not in 0-255)
    :param bound: upper and lower bound (-bound, bound)
    :return: pixel value scale from 0 to 255
    '''
    flow[flow > bound] = bound
    flow[flow < -bound] = -bound
    flow += bound
    flow *= (255/float(2*bound))

    # return bgr
    return flow.astype(np.uint8)



def save_flows(flows, save_dir, name, bound):
    '''
    To save the optical flow images and raw images
    :param flows: contains flow_x and flow_y
    :param image: raw image
    :param save_dir: save_dir name (always equal to the video id)
    :param num: the save id, which belongs one of the extracted frames
    :param bound: set the bi-bound to flow images
    :return: return 0
    '''
    # rescale to 0~255 with the bound setting
    flow_x = ToImg(flows[..., 0], bound)
    flow_y = ToImg(flows[..., 1], bound)
    save_dir.mkdir(parents=True, exist_ok=True)

    # save the flows
    save_x = str(save_dir / 'flow_x_{}.png'.format(name))
    save_y = str(save_dir / 'flow_y_{}.png'.format(name))
    io.imsave(save_x, flow_x)
    io.imsave(save_y, flow_y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="youtube",
                        help="dataset to parse flow")

    args = parser.parse_args()
    data_root = Path("data/{}_tempered".format(args.dataset))

    im_root = data_root / "vid"
    flow_root = data_root / "flow"

    fn_read = lambda x: cv2.cvtColor(cv2.imread(str(x)),
                                    cv2.COLOR_BGR2GRAY)

    dtvl1 = cv2.createOptFlow_DualTVL1()

    for cnt, vid in enumerate(tqdm(im_root.iterdir())):
        print(cnt, " : ", vid)
        vid_name = vid.name
        this_flow_dir = flow_root / vid_name
        this_flow_dir.mkdir(exist_ok=True, parents=True)

        files = sorted(vid.iterdir(),
                       key=lambda x: int(x.stem))

        first_file = files[0]
        prev_im = fn_read(first_file)

        for i in range(1, len(files)):
            fp = files[i]
            img = fn_read(fp)
            flow = dtvl1.calc(prev_im, img, None)

            fname = fp.stem
            save_flows(flow, this_flow_dir, fname, 15)

            





