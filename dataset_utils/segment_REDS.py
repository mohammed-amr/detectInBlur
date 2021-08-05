import sys
sys.path.append("/".join(sys.path[0].split("/")[:-1]))


from math import e
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from skimage import io, transform
import os 
from pathlib import Path
from PIL import Image, ImageDraw

import pickle

import random

root_dir = "/media/mosayed/data_f_256/datasets/REDS"

split = "val" 
sharpImages = False 
blurredImages = True 
threshold = 0.4 

dataDirectory = os.path.join(root_dir, "val" + "/")
dataDirectory2 = os.path.join(root_dir, "train" + "/")


imagePaths = list(Path(dataDirectory).rglob("*.[pP][nN][gG]")) + list(Path(dataDirectory2).rglob("*.[pP][nN][gG]"))

if sharpImages:
    imagePaths = [path for path in imagePaths if "sharp/" in str(path)]
elif blurredImages:
    imagePaths = [path for path in imagePaths if "blur/" in str(path)]
else:
    raise Exception("Panicked, no viable image type option.")

random.shuffle(imagePaths)

annotationPaths = []
for imagePath in imagePaths:
    imagePathStr = str(imagePath).replace("blur", "sharp")
    annPathStr = imagePathStr.replace(".png", "_DORS.npy")
    annotationPaths.append(annPathStr)


allwedEmptyImageSize = 20
emptyCount = 0
culledImagePaths = []
culledAnnotationPaths = []
for imagePathIndex, imagePath in enumerate(imagePaths):
    
    annotationfile_path = annotationPaths[imagePathIndex]

    with open(annotationfile_path, 'rb') as f:
        annotations = np.load(f, allow_pickle=True)


    boxCount = 0
    for classIndex, classArray in enumerate(annotations):
        confidenceTrimmedBoxes = classArray[classArray[:, 4] > threshold, :]
        for index in range(0, confidenceTrimmedBoxes.shape[0]):
            boxCount+=1

    

    if boxCount == 0:
        emptyCount+=1

    if (boxCount == 0) and (emptyCount > allwedEmptyImageSize):
        #enough of that.
        continue

    culledImagePaths.append(imagePath)
    culledAnnotationPaths.append(annotationfile_path)

    if imagePathIndex % 200 == 0:
        print("Caching " + str(imagePathIndex))

culledImagePaths = culledImagePaths[0:5000]
culledAnnotationPaths = culledAnnotationPaths[0:5000]


splitfile_path = root_dir + "/"
splitfile_path += "blurry"

with open(splitfile_path + ".txt" , 'w') as trainFile:
    for imagePathIndex, imagePath in enumerate(culledImagePaths):
        trainFile.write(str(imagePath) + "," + str(culledAnnotationPaths[imagePathIndex]) + "\n")


sharpImagePaths = []
for imagePath in culledImagePaths:
    imagePathStr = str(imagePath).replace("blur", "sharp")
    sharpImagePaths.append(imagePathStr)


splitfile_path = root_dir + "/"
splitfile_path += "sharp"

with open(splitfile_path + ".txt" , 'w') as trainFile:
    for imagePathIndex, imagePath in enumerate(sharpImagePaths):
        trainFile.write(str(imagePath) + "," + str(culledAnnotationPaths[imagePathIndex]) + "\n")