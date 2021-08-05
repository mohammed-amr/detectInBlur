import sys
sys.path.append("/".join(sys.path[0].split("/")[:-1]))

import sys
import os
from PIL import Image
from pathlib import Path
import pickle
import numpy as np
import math

sharpDirectory = "/media/mosayed/Cache/datasets/GOPROSynth/estimatorDS/sharp"
blurryDirectory = "/media/mosayed/Cache/datasets/GOPROSynth/estimatorDS/blur"
auxBlurryDirectory = "/media/mosayed/Cache/datasets/GOPROSynth/estimatorDS/auxBlur"


sharpImagePaths = np.array(list(Path(sharpDirectory).rglob("*.[pP][nN][gG]")))
blurryImagePaths = np.array(list(Path(blurryDirectory).rglob("*.[pP][nN][gG]")))
auxBlurryImagePaths = np.array(list(Path(auxBlurryDirectory).rglob("*.[pP][nN][gG]")))

trainListFileName = "/media/mosayed/Cache/datasets/GOPROSynth/estimatorDS/trainSplit.txt"
valListFileName = "/media/mosayed/Cache/datasets/GOPROSynth/estimatorDS/valSplit.txt"

auxBlurAnnotations = pickle.load(open("GTBlurEstFiles/GOPROSynthLoadauxBlurTrueLEHETrue.npy", "rb"))
shuffledIndices = np.random.permutation(len(auxBlurAnnotations))
auxBlurAnnotations = np.array(auxBlurAnnotations)
shuffledAuxBlurredImagePaths = auxBlurryImagePaths[shuffledIndices]
shuffledAuxBlurAnnotations = auxBlurAnnotations[shuffledIndices]


blurAnnotations = pickle.load(open("GTBlurEstFiles/GOPROSynthLoadblurTrueLEHETrue.npy", "rb"))
shuffledIndices = np.random.permutation(len(blurAnnotations))
blurAnnotations = np.array(blurAnnotations)
shuffledBlurredImagePaths = blurryImagePaths[shuffledIndices]
shuffledBlurAnnotations = blurAnnotations[shuffledIndices]


sharpAnnotations = pickle.load(open("GTBlurEstFiles/GOPROSynthLoadblurFalseLEHETrue.npy", "rb"))
shuffledIndices = np.random.permutation(len(list(Path(sharpDirectory).rglob("*.[pP][nN][gG]"))))
shuffledSharpImagePaths = sharpImagePaths[shuffledIndices]
shuffledIndices = np.random.permutation(len(sharpAnnotations))
shuffledSharpAnnotations = np.array(sharpAnnotations)

trainSplit = 0.9
valSplit = 0.1

trainAuxBlurryImagesPaths = shuffledAuxBlurredImagePaths[:math.ceil(trainSplit*shuffledAuxBlurredImagePaths.shape[0])]
trainAuxBlurryAnnotations= shuffledAuxBlurAnnotations[:math.ceil(trainSplit*shuffledAuxBlurAnnotations.shape[0])]
testAuxBlurryImagesPaths = shuffledAuxBlurredImagePaths[math.ceil(trainSplit*shuffledAuxBlurredImagePaths.shape[0]):]
testAuxBlurryAnnotations= shuffledAuxBlurAnnotations[math.ceil(trainSplit*shuffledAuxBlurAnnotations.shape[0]):]

trainBlurryImagesPaths = shuffledBlurredImagePaths[:math.ceil(trainSplit*shuffledBlurredImagePaths.shape[0])]
trainBlurryAnnotations= shuffledBlurAnnotations[:math.ceil(trainSplit*shuffledBlurAnnotations.shape[0])]
testBlurryImagesPaths = shuffledBlurredImagePaths[math.ceil(trainSplit*shuffledBlurredImagePaths.shape[0]):]
testBlurryAnnotations = shuffledBlurAnnotations[math.ceil(trainSplit*shuffledBlurAnnotations.shape[0]):]


trainSharpImagesPaths = shuffledSharpImagePaths[:math.ceil(trainSplit*shuffledSharpImagePaths.shape[0])]
trainSharpAnnotations= shuffledSharpAnnotations[:math.ceil(trainSplit*shuffledSharpAnnotations.shape[0])]
testSharpImagesPaths = shuffledSharpImagePaths[math.ceil(trainSplit*shuffledSharpImagePaths.shape[0]):]
testSharpAnnotations= shuffledSharpAnnotations[math.ceil(trainSplit*shuffledSharpAnnotations.shape[0]):]


with open(trainListFileName, 'w') as trainFile:
    for imagePathIndex, imagePath in enumerate(trainAuxBlurryImagesPaths):
        trainFile.write(str(imagePath) + "," + str(trainAuxBlurryAnnotations[imagePathIndex]) + "\n")

    for imagePathIndex, imagePath in enumerate(trainBlurryImagesPaths):
        trainFile.write(str(imagePath) + "," + str(trainBlurryAnnotations[imagePathIndex]) + "\n")

    for imagePathIndex, imagePath in enumerate(trainSharpImagesPaths):
        trainFile.write(str(imagePath) + "," + str(trainSharpAnnotations[imagePathIndex]) + "\n")

with open(valListFileName, 'w') as valFile:
    for imagePathIndex, imagePath in enumerate(testAuxBlurryImagesPaths):
        valFile.write(str(imagePath) + "," + str(testAuxBlurryAnnotations[imagePathIndex]) + "\n")

    for imagePathIndex, imagePath in enumerate(testBlurryImagesPaths):
        valFile.write(str(imagePath) + "," + str(testBlurryAnnotations[imagePathIndex]) + "\n")

    for imagePathIndex, imagePath in enumerate(testSharpImagesPaths):
        valFile.write(str(imagePath) + "," + str(testSharpAnnotations[imagePathIndex]) + "\n")
        