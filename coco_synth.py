from pycocotools.coco import COCO
import numpy as np
import cv2
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
from PIL import Image
from PIL import ImageFilter
import argparse
import sys
import pdb
from pathlib import Path


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="Number of images to work with")
    parser.add_argument(
        "--num", "-n", type=int, default=1000, help="total manipulated images"
    )
    parser.add_argument("--seed", "-s", type=str, default=0,
                        help="random seed")
    args = parser.parse_args()
    return args


args = parse_args()
print(args)

np.random.seed(args.seed)

dataDir = "./data/coco/"
dataType = "val2014"
annFile = f"{dataDir}/annotations/instances_{dataType}.json"

coco = COCO(annFile)
cats = coco.loadCats(coco.getCatIds())

maskDir = "./data/coco-tempered/mask"
Path(maskDir).mkdir(exist_ok=True)

maniDir = "./data/coco-tempered/manipulated"
Path(maniDir).mkdir(exist_ok=True)


for i in range(args.num):
    random_cat = np.random.choice(cats)
    catIds = random_cat["id"]
    imgIds = coco.getImgIds(catIds=catIds)
    img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]

    file_source = os.path.join(dataDir, dataType, img['file_name'])
    I_source = io.imread(file_source)
    annIds = coco.getAnnIds(imgIds=img["id"], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    ann = np.random.choice(anns)
    # coco.showAnns(anns)
    # bbx = anns[0]["bbox"]
    mask = np.array(coco.annToMask(ann))
    # print(np.shape(mask))
    # print(np.shape(I))
    # pdb.set_trace()
    masked_image = I_source * mask[..., None]
    rand = np.random.randint(100, size=1)[0]
    img1 = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
    b1 = io.imread(
        os.path.join(dataDir, dataType, "COCO_val2014_{:012d}.jpg".format(img1["id"]))
    )
    text_img = Image.new("RGBA", (np.shape(b1)[0], np.shape(b1)[1]), (0, 0, 0, 0))
    background = Image.fromarray(b1, "RGB")
    foreground = Image.fromarray(I1, "RGB").convert("RGBA")
    datas = foreground.getdata()
    # pdb.set_trace()
    newData = []
    for item in datas:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            newData.append((0, 0, 0, 0))
        else:
            newData.append(item)
    foreground.putdata(newData)
    foreground = foreground.resize(
        (background.size[0], background.size[1]), Image.ANTIALIAS
    )
    background.paste(foreground, (0, 0), mask=foreground.split()[3])
    if rand % 3 < 2:
        background = background.filter(ImageFilter.GaussianBlur(radius=1.5))
    # pdb.set_trace()

    filename = (
        "./data/filter_tamper/Tp_"
        + str(img["id"])
        + "_"
        + str(img1["id"])
        + "_"
        + str(bbx[0])
        + "_"
        + str(bbx[1])
        + "_"
        + str(bbx[0] + bbx[2])
        + "_"
        + str(bbx[1] + bbx[3])
        + "_"
        + cat["name"]
        + ".png"
    )

    if not os.path.isfile(filename):
        background.save(filename)

print("finished")
