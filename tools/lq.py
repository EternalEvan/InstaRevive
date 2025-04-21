from typing import Sequence, Dict, Union
import math
import time
import torch
import numpy as np
import cv2
import os
from PIL import Image
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

from utils.degradation import (
    random_mixed_kernels, random_add_gaussian_noise, random_add_jpg_compression
)

def blur(gt_path):
    for _ in range(3):
        try:
            pil_img = Image.open(gt_path).convert("RGB")
            success = True
            break
        except:
            time.sleep(1)
    assert success, f"failed to load image {gt_path}"
    pil_img_gt = np.array(pil_img)
    img_gt = (pil_img_gt[..., ::-1] / 255.0).astype(np.float32)
    h, w, _ = img_gt.shape
    kernel = random_mixed_kernels(
        ['iso', 'aniso'],
        [0.5, 0.5],
        41,
        [0.1, 10],
        [0.1, 10],
        [-math.pi, math.pi],
        noise_range=None
    )
    img_lq = cv2.filter2D(img_gt, -1, kernel)
    scale = np.random.uniform(2, 4)
    img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
    img_lq = random_add_gaussian_noise(img_lq, [0, 20])
    img_lq = random_add_jpg_compression(img_lq, [60, 100])
    img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
    img_lq = ((np.maximum(img_lq[...,::-1], 0) / img_lq[...,::-1].max())*255.0).astype(np.uint8)
    img_lq = cv2.cvtColor(img_lq, cv2.COLOR_RGB2BGR)
    return img_lq

def main(inputdir= '',outputdir=''):
    imglist = sorted(os.listdir(inputdir))
    os.makedirs(outputdir,exist_ok=True)
    for img in imglist:
        gt_path = os.path.join(inputdir,img)
        save_path = os.path.join(outputdir,img)
        lqlq = blur(gt_path)
        cv2.imwrite(save_path, lqlq)
    
if __name__ == "__main__":
    main('/home/zyx/PixArt-sigma/paper_input/imagenet','/home/zyx/PixArt-sigma/paper_input/imagenet-lq')
    # blur('/data1/zyx/celeba_512_validation_lq/00001267.png')