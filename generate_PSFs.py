import PIL
from PIL import Image
import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco, get_coco_kp

from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from engine import train_one_epoch, evaluate

import utils
import transforms as T

from torchvision.transforms import functional as F

from torch.utils.tensorboard import SummaryWriter
import random

import cv2 
import numpy as np 
import math

from motion_blur.generate_PSF import PSF
from motion_blur.generate_trajectory import Trajectory
from motion_blur.blur_image import BlurImageHandler
import sys
import argparse

if __name__ == "__main__":
  

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--slice_index', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--num_psfs', type=int, default=12000)

    args = parser.parse_args()
    
    
    output_path = args.output_path
    sliceIndex = args.slice_index
    sliceSize = int(args.num_psfs/args.num_workers)


    startIndex = sliceSize * sliceIndex 
    endIndex = startIndex + sliceSize


    np.random.seed(1337 * sliceIndex)
    random.seed(1337 * sliceIndex)

    useAxisAlignedBlur = False


    params = [0.005, 0.001, 0.00005]
    fractions = [1/18, 1/10, 1/5, 1/2, 1]
    
    for paramIndex, param in enumerate(params):
        for fractionIndex, fraction in enumerate(fractions):
            if not os.path.exists(path + "psfs/P" + str(paramIndex+1) + "E" + str(fractionIndex)):
                os.makedirs(path + "psfs/P" + str(paramIndex+1) + "E" + str(fractionIndex))

    startTime = time.perf_counter()
    for paramIndex, param in enumerate(params):
        for exposureIndex, exposure in enumerate(fractions): 
            folderPath = path + "psfs/P" + str(paramIndex+1) + "E" + str(exposureIndex)
            for index in range(startIndex, endIndex):
                
                trajectoryObj = Trajectory(canvas=256, max_len=96, expl=param).fit()
                trajectory = trajectoryObj.fit()

                psfObject = PSF(canvas=256, trajectory=trajectory, fraction = [exposure])

                psf = psfObject.fit()

                psfObject.centerPSF()

                psf = psfObject.PSFs[0]

                filePath = folderPath + "/I" + "{:06d}".format(index)
                with open(filePath, 'wb') as f:
                    np.save(f, psf.astype(np.float16))

                if index % 200 == 0:
                    elapsedTime = time.perf_counter() - startTime
                    averageImageTime = round((elapsedTime)*(1000/(index+1)), 2)

                    if elapsedTime > 86400:
                        elapsedTime = time.strftime('%d days %H:%M:%S', time.gmtime(elapsedTime - 86400))
                    else:
                        elapsedTime = time.strftime('%H:%M:%S', time.gmtime(elapsedTime - 86400))

                    print("Image Index " + str(paramIndex) + " " + str(exposureIndex) + " "  +  str(index)  + " Elapsed Time: " + elapsedTime + " Average Time/Image: " + str(averageImageTime) + "ms")  

        
