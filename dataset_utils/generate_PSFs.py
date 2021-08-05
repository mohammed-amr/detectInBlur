import sys
sys.path.append("/".join(sys.path[0].split("/")[:-1]))

import os
import time

import random

import numpy as np

from motion_blur.generate_PSF import PSF
from motion_blur.generate_trajectory import Trajectory

import argparse

def main(args):


    path = args.destination_path
    slice_index = args.worker_index

    slice_size = int(args.total_num_psfs/args.num_workers)


    start_index = slice_size * slice_index 
    endIndex = start_index + slice_size

    np.random.seed(1337 * slice_index)
    random.seed(1337 * slice_index)

    params = [0.005, 0.001, 0.00005]
    fractions = [1/18, 1/10, 1/5, 1/2, 1]
    
    for param_index, param in enumerate(params):
        for fraction_index, fraction in enumerate(fractions):
            if not os.path.exists(path + "psfs/P" + str(param_index+1) + "E" + str(fraction_index)):
                os.makedirs(path + "psfs/P" + str(param_index+1) + "E" + str(fraction_index))

    start_time = time.perf_counter()
    for param_index, param in enumerate(params):
        for exposure_index, exposure in enumerate(fractions): 

            folder_path = path + "psfs/P" + str(param_index+1) + "E" + str(exposure_index)

            for index in range(start_index, endIndex):
             
                trajectory_obj = Trajectory(canvas=256, max_len=96, expl=param).fit()
                trajectory = trajectory_obj.fit()

                psf_object = PSF(canvas=256, trajectory=trajectory, fraction = [exposure])

                psf = psf_object.fit()

                psf_object.centerPSF()

                psf = psf_object.PSFs[0]

                file_path = folder_path + "/I" + "{:06d}".format(index)
                with open(file_path, 'wb') as f:
                    np.save(f, psf.astype(np.float16))

                if index % 200 == 0:
                    elapsed_time = time.perf_counter() - start_time
                    average_image_time = round((elapsed_time)*(1000/(index+1)), 2)

                    if elapsed_time > 86400:
                        elapsed_time = time.strftime('%d days %H:%M:%S', time.gmtime(elapsed_time - 86400))
                    else:
                        elapsed_time = time.strftime('%H:%M:%S', time.gmtime(elapsed_time - 86400))

                    print("Image Index " + str(param_index) + " " + str(exposure_index) + " "  +  str(index)  + " Elapsed Time: " + elapsed_time + " Average Time/Image: " + str(average_image_time) + "ms")  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--destination_path', type = str, default="/media/mosayed/data_f_500/datasets/COCO/coco/")
    parser.add_argument('--worker_index', type = int, default=0)
    parser.add_argument('--num_workers', type = int, default=12)
    parser.add_argument('--total_num_psfs', type = int, default=12000)


    args = parser.parse_args()

    main(args)