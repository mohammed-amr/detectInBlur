import os 
import random
import numpy as np

for trialCount in range(10000):
    param_index = random.choice([3])
    fraction_index = random.choice([4])
    psfIndex = random.randint(0, 12000-1)

    file_path = "/media/mosayed/data_f_500/datasets/COCO/coco/psfs" + "/P" + str(param_index) + "E" + str(fraction_index) + "/I" + "{:06d}".format(psfIndex)

    with open(file_path, 'rb') as f:
        psf = [np.load(f)]

    non_zero_indices = np.nonzero(psf[0] > 0)
    coordY = non_zero_indices[0]
    coordX = non_zero_indices[1]

    if coordX.min() < (256/2) - 64 or coordY.min() < (256/2) - 64 or coordX.max() > (256/2) + 64 or coordY.max() > (256/2) + 64:   
        print(coordX.min(), coordY.min(), coordX.max(), coordY.max())
    else:
        print("np")