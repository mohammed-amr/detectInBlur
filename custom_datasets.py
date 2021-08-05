from math import e
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from skimage import io, transform
import os 
from pathlib import Path
from PIL import Image, ImageDraw

from utils import coco80_to_coco91_class

import pickle

import random 

class GOPRO(Dataset):
    """GOPRO dataset as det."""

    def __init__(self, root_dir = "/media/mosayed/data_f_256/datasets/GOPRO", split = "test", sharpImages = False, blurredImages = True, threshold = 0.4, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.split = split

        self.blurredImages = blurredImages
        self.sharpImages = sharpImages


        self.dataDirectory = os.path.join(self.root_dir, "train" + "/")
        self.dataDirectory2 = os.path.join(self.root_dir, "test" + "/")

        
        self.imagePaths = list(Path(self.dataDirectory).rglob("*.[pP][nN][gG]")) + list(Path(self.dataDirectory2).rglob("*.[pP][nN][gG]"))
        if sharpImages:
            self.imagePaths = [path for path in self.imagePaths if "sharp/" in str(path)]
        elif blurredImages:
            self.imagePaths = [path for path in self.imagePaths if "blur/" in str(path)]
        else:
            raise Exception("Panicked, no viable image type option.")


        #self.imagePaths = self.imagePaths[:100]

        # self.annotationPaths = list(Path(self.dataDirectory).rglob("*.[nN][pP][yY]"))
        # self.annotationPaths = [path for path in self.annotationPaths if "sharp/" in str(path)]
        self.annotationPaths = []
        for imagePath in self.imagePaths:
            imagePathStr = str(imagePath).replace("blur", "sharp")
            annPathStr = imagePathStr.replace(".png", "_DORS.npy")
            self.annotationPaths.append(annPathStr)

        
        self.threshold = threshold

        self.transform = transform

        self.cocoConvertor = coco80_to_coco91_class()

        self.conversionMode = False

        self.boxCount = 0

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.imagePaths[idx]

        #image = io.imread(img_path)
        if not self.conversionMode:
            image = Image.open(img_path)
        else:
            image = None
            self.width = 1280
            self.height = 720


        annotationfile_path = self.annotationPaths[idx]

        with open(annotationfile_path, 'rb') as f:
            annotations = np.load(f, allow_pickle=True)
            
        image_id = idx
        boxes = np.array([[], [], [], []]).T
        labels = []
        iscrowd = []
        areas = []

        for classIndex, classArray in enumerate(annotations):
            confidenceTrimmedBoxes = classArray[classArray[:, 4] > self.threshold, :]
            boxes = np.vstack((boxes, confidenceTrimmedBoxes[:, 0:4]))
            for index in range(0, confidenceTrimmedBoxes.shape[0]):
                labels.append(self.cocoConvertor[classIndex])
                iscrowd.append(False)
                self.boxCount+=1
            
        
        labels = np.array(labels, dtype=int).T
        iscrowd = np.array(iscrowd, dtype=bool).T

        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        areas = np.array(areas).T

        targets = {}
        targets["image_id"] = torch.tensor(image_id)
        targets["boxes"] = torch.from_numpy(boxes)
        targets["labels"] = torch.from_numpy(labels)
        targets["iscrowd"] = torch.from_numpy(iscrowd)
        targets["area"] = torch.from_numpy(areas)

        blur_dict = {}

        
        if self.transform and not self.conversionMode:
            image, targets, blur_dict = self.transform(image, targets, blur_dict)
            
        return image, targets, blur_dict

class VidBlur(Dataset):
    """VidBlur dataset as det."""

    def __init__(self, root_dir = "/media/mosayed/data_f_256/datasets/DeepVideoDeblurring_Dataset/", split = "test", sharpImages = False, blurredImages = True, threshold = 0.4, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = "/media/mosayed/data_f_256/datasets/DeepVideoDeblurring_Dataset/quantitative_datasets/"#root_dir
        self.split = split

        self.blurredImages = blurredImages
        self.sharpImages = sharpImages


        self.dataDirectory = self.root_dir
        
        self.imagePaths = list(Path(self.dataDirectory).rglob("*.[jJ][pP][gG]"))

        if sharpImages:
            self.imagePaths = [path for path in self.imagePaths if "GT/" in str(path)]
        elif blurredImages:
            self.imagePaths = [path for path in self.imagePaths if "input/" in str(path)]
        else:
            raise Exception("Panicked, no viable image type option.")
                
        # self.annotationPaths = list(Path(self.dataDirectory).rglob("*.[nN][pP][yY]"))
        # self.annotationPaths = [path for path in self.annotationPaths if "sharp/" in str(path)]
        self.annotationPaths = []
        for imagePath in self.imagePaths:
            annPathStr = str(imagePath).replace(".jpg", "_DORS.npy")
            annPathStr = annPathStr.replace("input/", "GT/")
            self.annotationPaths.append(annPathStr)
        
        self.threshold = threshold

        self.transform = transform

        self.cocoConvertor = coco80_to_coco91_class()

        self.conversionMode = False

        self.boxCount = 0

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.imagePaths[idx]

        #image = io.imread(img_path)
        if not self.conversionMode:
            image = Image.open(img_path)
        else:
            image = None
            self.width = 1280
            self.height = 720


        annotationfile_path = self.annotationPaths[idx]

        with open(annotationfile_path, 'rb') as f:
            annotations = np.load(f, allow_pickle=True)

        image_id = idx
        boxes = np.array([[], [], [], []]).T
        labels = []
        iscrowd = []
        areas = []

        for classIndex, classArray in enumerate(annotations):
            confidenceTrimmedBoxes = classArray[classArray[:, 4] > self.threshold, :]
            boxes = np.vstack((boxes, confidenceTrimmedBoxes[:, 0:4]))
            for index in range(0, confidenceTrimmedBoxes.shape[0]):
                labels.append(self.cocoConvertor[classIndex])
                iscrowd.append(False)
                self.boxCount+=1
            
        
        labels = np.array(labels, dtype=int).T
        iscrowd = np.array(iscrowd, dtype=bool).T

        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        areas = np.array(areas).T

        targets = {}
        targets["image_id"] = torch.tensor(image_id)
        targets["boxes"] = torch.from_numpy(boxes)
        targets["labels"] = torch.from_numpy(labels)
        targets["iscrowd"] = torch.from_numpy(iscrowd)
        targets["area"] = torch.from_numpy(areas)

        blur_dict = {}

        
        if self.transform and not self.conversionMode:
            image, targets, blur_dict = self.transform(image, targets, blur_dict)
            
        return image, targets, blur_dict

def getGOPRO(root_dir = "/media/mosayed/data_f_256/datasets/GOPRO", split = "test", sharpImages = False, blurredImages = True, threshold = 0.4, transform=None):
    return GOPRO(root_dir, split, sharpImages, blurredImages, threshold, transform)


class RealBlur(Dataset):
    """RealBlur dataset as det."""

    def __init__(self, root_dir = "/media/mosayed/data_f_256/datasets/realBlur/", split = "test", sharpImages = False, blurredImages = True, threshold = 0.6, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.split = split

        self.blurredImages = blurredImages
        self.sharpImages = sharpImages


        self.image_indexfile_path = os.path.join(self.root_dir, "RealBlur_J_train_list.txt")
        self.image_indexFile = open(self.image_indexfile_path, "r")


        self.imagePaths = []
        self.annotationPaths = []
        for line in self.image_indexFile:
            gtPath = line.split(" ")[0]
            blurPath = line.split(" ")[1]

            if sharpImages:
                self.imagePaths.append(os.path.join(self.root_dir, gtPath))
            elif blurredImages:
                self.imagePaths.append(os.path.join(self.root_dir, blurPath))
            else:
                raise Exception("Panicked, no viable image type option.")
            
            annotationFileRelPath = gtPath.replace(".png", "_DORS.npy")
            self.annotationPaths.append(os.path.join(self.root_dir, annotationFileRelPath))

        self.image_indexfile_path = os.path.join(self.root_dir, "RealBlur_J_test_list.txt")
        self.image_indexFile = open(self.image_indexfile_path, "r")
        for line in self.image_indexFile:
            gtPath = line.split(" ")[0]
            blurPath = line.split(" ")[1]

            if sharpImages:
                self.imagePaths.append(os.path.join(self.root_dir, gtPath))
            elif blurredImages:
                self.imagePaths.append(os.path.join(self.root_dir, blurPath))
            else:
                raise Exception("Panicked, no viable image type option.")
            
            annotationFileRelPath = gtPath.replace(".png", "_DORS.npy")
            self.annotationPaths.append(os.path.join(self.root_dir, annotationFileRelPath))

        
        self.threshold = threshold

        self.transform = transform

        self.cocoConvertor = coco80_to_coco91_class()

        self.conversionMode = False

        self.boxCount = 0

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.imagePaths[idx]

        #image = io.imread(img_path)
        if not self.conversionMode:
            image = Image.open(img_path)
        else:
            image = None
            self.width = 1280
            self.height = 720


        annotationfile_path = self.annotationPaths[idx]

        with open(annotationfile_path, 'rb') as f:
            annotations = np.load(f, allow_pickle=True)

        image_id = idx
        boxes = np.array([[], [], [], []]).T
        labels = []
        iscrowd = []
        areas = []

        for classIndex, classArray in enumerate(annotations):
            confidenceTrimmedBoxes = classArray[classArray[:, 4] > self.threshold, :]
            boxes = np.vstack((boxes, confidenceTrimmedBoxes[:, 0:4]))
            for index in range(0, confidenceTrimmedBoxes.shape[0]):
                labels.append(self.cocoConvertor[classIndex])
                iscrowd.append(False)
                self.boxCount+=1
            
        
        labels = np.array(labels, dtype=int).T
        iscrowd = np.array(iscrowd, dtype=bool).T

        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        areas = np.array(areas).T

        targets = {}
        targets["image_id"] = torch.tensor(image_id)
        targets["boxes"] = torch.from_numpy(boxes)
        targets["labels"] = torch.from_numpy(labels)
        targets["iscrowd"] = torch.from_numpy(iscrowd)
        targets["area"] = torch.from_numpy(areas)

        blur_dict = {}

        
        if self.transform and not self.conversionMode:
            image, targets, blur_dict = self.transform(image, targets, blur_dict)
            
        return image, targets, blur_dict



class REDS(Dataset):
    """REDS dataset as det."""

    def __init__(self, root_dir = "/media/mosayed/data_f_256/datasets/REDS", split = "val", sharpImages = False, blurredImages = True, threshold = 0.4, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.split = split

        self.blurredImages = blurredImages
        self.sharpImages = sharpImages

        if blurredImages:
            self.dataListFile = root_dir + "/" + "blurry.txt"
        else:
            self.dataListFile = root_dir + "/" + "sharp.txt"


        with open(self.dataListFile) as fp:
            listContents = fp.readlines()

        self.imagePaths = []
        self.annotationPaths = []
        for line in listContents:
            splitLine = line.strip('\n').split(",")
            self.imagePaths.append(splitLine[0])
            self.annotationPaths.append(splitLine[1])


        # self.dataDirectory = os.path.join(self.root_dir, "val" + "/")
        # self.dataDirectory2 = os.path.join(self.root_dir, "train" + "/")

        
        # self.imagePaths = list(Path(self.dataDirectory).rglob("*.[pP][nN][gG]")) + list(Path(self.dataDirectory2).rglob("*.[pP][nN][gG]"))
    
        # if sharpImages:
        #     self.imagePaths = [path for path in self.imagePaths if "sharp/" in str(path)]
        # elif blurredImages:
        #     self.imagePaths = [path for path in self.imagePaths if "blur/" in str(path)]
        # else:
        #     raise Exception("Panicked, no viable image type option.")
        
        # self.imagePaths = random.sample(self.imagePaths, 5000)


        # # self.annotationPaths = list(Path(self.dataDirectory).rglob("*.[nN][pP][yY]"))
        # # self.annotationPaths = [path for path in self.annotationPaths if "sharp/" in str(path)]
        # self.annotationPaths = []
        # for imagePath in self.imagePaths:
        #     imagePathStr = str(imagePath).replace("blur", "sharp")
        #     annPathStr = imagePathStr.replace(".png", "_DORS.npy")
        #     self.annotationPaths.append(annPathStr)

        self.threshold = threshold

        self.transform = transform

        self.cocoConvertor = coco80_to_coco91_class()

        self.conversionMode = False

        self.boxCount = 0

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.imagePaths[idx]

        #image = io.imread(img_path)
        if not self.conversionMode:
            image = Image.open(img_path)
        else:
            image = None
            self.width = 1280
            self.height = 720


        annotationfile_path = self.annotationPaths[idx]

        with open(annotationfile_path, 'rb') as f:
            annotations = np.load(f, allow_pickle=True)

        image_id = idx
        boxes = np.array([[], [], [], []]).T
        labels = []
        iscrowd = []
        areas = []

        for classIndex, classArray in enumerate(annotations):
            confidenceTrimmedBoxes = classArray[classArray[:, 4] > self.threshold, :]
            boxes = np.vstack((boxes, confidenceTrimmedBoxes[:, 0:4]))
            for index in range(0, confidenceTrimmedBoxes.shape[0]):
                labels.append(self.cocoConvertor[classIndex])
                iscrowd.append(False)
                self.boxCount+=1
            
        
        labels = np.array(labels, dtype=int).T
        iscrowd = np.array(iscrowd, dtype=bool).T

        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        areas = np.array(areas).T

        targets = {}
        targets["image_id"] = torch.tensor(image_id)
        targets["boxes"] = torch.from_numpy(boxes)
        targets["labels"] = torch.from_numpy(labels)
        targets["iscrowd"] = torch.from_numpy(iscrowd)
        targets["area"] = torch.from_numpy(areas)

        blur_dict = {}

        
        if self.transform and not self.conversionMode:
            image, targets, blur_dict = self.transform(image, targets, blur_dict)
            
        return image, targets, blur_dict




def drawBoxes(im, boxes):
    im = im.copy()
    draw = ImageDraw.Draw(im)

    for boxIndex in range(boxes.shape[0]):
        tl = (boxes[boxIndex, 0], boxes[boxIndex, 1])
        br = (boxes[boxIndex, 2], boxes[boxIndex, 3])
        draw.rectangle([tl, br], fill=None, outline=None, width=1)
    
    return im

def getimage_indexInFolder(imagePath, folderDirectory):
    imageListInDiretory = list(Path(folderDirectory).rglob("*.[pP][nN][gG]"))
    imageListInDiretory = sorted(imageListInDiretory)
    for imIndex, imagePathInSubFolder in enumerate(imageListInDiretory):
        if str(imagePath) in str(imagePathInSubFolder):
            return imIndex, imageListInDiretory


class ImageInfo():
    def __init__(self, imageArrayIndex, imagePath, folderDirectory, imageDirectoryList):
        self.imageArrayIndex = imageArrayIndex
        self.cleanImagePath = imagePath

        self.cleanImage = Image.open(self.cleanImagePath)

        self.imageDirectoryList = imageDirectoryList
        self.folderDirectory = folderDirectory

        self.maxFolderIndex = len(imageDirectoryList) - 1

        self.blurredImageLoc = ""
        self.blurring = False

        self.expandedBoxesAlready = False
        self.blurredImageAlready = False

    def loadAnn(self, treshold, cocoConvertor):
        self.annPathStr  = getAnnPath(self.cleanImagePath)
        
        with open(self.annPathStr, 'rb') as f:
            annotations = np.load(f, allow_pickle=True)
            f.close()

        boxes = np.array([[], [], [], []]).T
        labels = []
        iscrowd = []
        areas = []
        confidences = []

        self.boxCount = 0

        for classIndex, classArray in enumerate(annotations):
            confidenceTrimmedBoxes = classArray[classArray[:, 4] > treshold, :]
            boxes = np.vstack((boxes, confidenceTrimmedBoxes[:, 0:4]))
            for index in range(0, confidenceTrimmedBoxes.shape[0]):
                labels.append(cocoConvertor[classIndex])
                confidences.append(confidenceTrimmedBoxes[index, 4])
                iscrowd.append(False)
                self.boxCount+=1
            
        
        labels = np.array(labels, dtype=int).T
        iscrowd = np.array(iscrowd, dtype=bool).T
        confidences = np.array(confidences).T

        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        areas = np.array(areas).T

        self.targets = {}
        self.targets["image_id"] = torch.tensor(self.image_id)
        self.targets["boxes"] = torch.from_numpy(boxes)
        self.targets["labels"] = torch.from_numpy(labels)
        self.targets["iscrowd"] = torch.from_numpy(iscrowd)
        self.targets["area"] = torch.from_numpy(areas)
        self.targets["confidences"] = torch.from_numpy(confidences)


        self.blur_dict = {}

        return

    def nearestNeighborInterp(self, array_in, xyLoc, width_in, height_in, width_out, height_out):
        indexingX = round(xyLoc[0])
        indexingX = max(indexingX, 0)
        indexingX = min(indexingX, width_in-1)

        indexingY = round(xyLoc[1])
        indexingY = max(indexingY, 0)
        indexingY = min(indexingY, height_in-1)

        xValue = array_in[0, 0, round(indexingY), indexingX]
        yValue = array_in[0, 1, round(indexingY), indexingX]

        return xValue, yValue

    def bilinearInterp(self, array_in, xyLoc, width_in, height_in, width_out, height_out):

        # this snippet borrowed from https://eng.aurelienpierre.com/2020/03/bilinear-interpolation-on-images-stored-as-python-numpy-ndarray/
        # Relative coordinates of the pixel in output space
        x_out = xyLoc[0] / width_out
        y_out = xyLoc[1] / height_out

        # Corresponding absolute coordinates of the pixel in input space
        x_in = (x_out * width_in)
        y_in = (y_out * height_in)

        # Nearest neighbours coordinates in input space
        x_prev = int(np.floor(x_in))
        x_next = x_prev + 1
        y_prev = int(np.floor(y_in))
        y_next = y_prev + 1

        # Sanitize bounds - no need to check for < 0
        x_prev = min(x_prev, width_in - 1)
        x_next = min(x_next, width_in - 1)
        y_prev = min(y_prev, height_in - 1)
        y_next = min(y_next, height_in - 1)
        
        # Distances between neighbour nodes in input space
        Dy_next = y_next - y_in
        Dy_prev = 1. - Dy_next; # because next - prev = 1
        Dx_next = x_next - x_in
        Dx_prev = 1. - Dx_next; # because next - prev = 1
        
        # Interpolate
        xValue = Dy_prev * (array_in[0, 0, y_next, x_prev] * Dx_next + array_in[0, 0, y_next, x_next] * Dx_prev) \
        + Dy_next * (array_in[0, 0, y_prev, x_prev] * Dx_next + array_in[0, 0, y_prev, x_next] * Dx_prev)

        yValue = Dy_prev * (array_in[0, 1, y_next, x_prev] * Dx_next + array_in[0, 1, y_next, x_next] * Dx_prev) \
        + Dy_next * (array_in[0, 1, y_prev, x_prev] * Dx_next + array_in[0, 1, y_prev, x_next] * Dx_prev)

        return xValue, yValue

    def fixIndices(self, coord):
        if coord[0] < 0:
            coord[0] = 1
        if coord[0] > 1279:
            coord[0] = 1279
        
        if coord[1] < 0:
            coord[1] = 1
        if coord[1] > 719:
            coord[1] = 719

        return coord

    def expandBoxes(self):
        if self.expandedBoxesAlready:
            return

        forwardFlows = []
        backwardFlows = []

        for image_index in range(self.imageArrayIndex, self.imageArrayIndex + self.windowSize):
            flowPath = str(self.imageDirectoryList[image_index]).replace(".png", "_flow.npy")
            with open(flowPath, 'rb') as f:
                flows = {}
                flows["cpuFlowFHR"] = np.load(f, allow_pickle=True)/2
                flows["cpuFlowBHR"] = np.load(f, allow_pickle=True)/2
                flows["cpuFlowFLR"] = np.load(f, allow_pickle=True)
                flows["cpuFlowBLR"] = np.load(f, allow_pickle=True)
                f.close()

            forwardFlows.append(flows)

        for image_index in reversed(range(self.imageArrayIndex - self.windowSize + 1, self.imageArrayIndex + 1)):
            flowPath = str(self.imageDirectoryList[image_index]).replace(".png", "_flow.npy")
            with open(flowPath, 'rb') as f:
                flows = {}
                flows["cpuFlowFHR"] = np.load(f, allow_pickle=True)/2
                flows["cpuFlowBHR"] = np.load(f, allow_pickle=True)/2
                flows["cpuFlowFLR"] = np.load(f, allow_pickle=True)
                flows["cpuFlowBLR"] = np.load(f, allow_pickle=True)
                f.close()

            backwardFlows.append(flows)

        for boxIndex in range(self.targets["boxes"].shape[0]):
            tl = self.targets["boxes"][boxIndex, [0, 1]]
            br = self.targets["boxes"][boxIndex, [2, 3]]

            tr = self.targets["boxes"][boxIndex, [2, 1]]
            bl = self.targets["boxes"][boxIndex, [0, 3]]

            listOfOriginalPoints = [tl, br, tr, bl]
            listOfPoints = torch.vstack((tl, br, tr, bl))

            for point in listOfOriginalPoints:
                shiftedPoint = self.walkThroughFlow(forwardFlows, point, False)
                listOfPoints = torch.vstack((listOfPoints, shiftedPoint))

                shiftedPoint = self.walkThroughFlow(backwardFlows, point, True)
                listOfPoints = torch.vstack((listOfPoints, shiftedPoint))

            newTL = torch.min(listOfPoints, dim = 0)[0]
            newBR = torch.max(listOfPoints, dim = 0)[0]

            newTL = self.fixIndices(newTL)
            newBR = self.fixIndices(newBR)

            self.targets["boxes"][boxIndex, [0, 1]] = newTL
            self.targets["boxes"][boxIndex, [2, 3]] = newBR


        self.targets["area"] = torch.from_numpy(np.array((self.targets["boxes"][:, 3] - self.targets["boxes"][:, 1]) * (self.targets["boxes"][:, 2] - self.targets["boxes"][:, 0])).T)

        #drawBoxes(self.blurredImage, self.targets["boxes"]).save("newSetDebugImages/testBlurredExpBoxes.png")

        self.expandedBoxesAlready = True
        return

    def walkThroughFlow(self, flows, startXYLoc, reverse):
        highRes = False

        if highRes:
            xyLoc = startXYLoc/2
        else:
            xyLoc = startXYLoc/8


        for flowDict in flows:

            if highRes:
                if reverse:
                    flow = flowDict["cpuFlowBHR"]
                else:
                    flow = flowDict["cpuFlowFHR"]
            else:
                if reverse:
                    flow = flowDict["cpuFlowBLR"]
                else:
                    flow = flowDict["cpuFlowFLR"]

            # x value
            #xChange, yChange = self.nearestNeighborInterp(flow, xyLoc, width_in=flow.shape[3], height_in=flow.shape[2], width_out=flow.shape[3], height_out=flow.shape[2])
            xChange, yChange = self.bilinearInterp(flow, xyLoc, width_in=flow.shape[3], height_in=flow.shape[2], width_out=flow.shape[3], height_out=flow.shape[2])


            xyLoc[0] += xChange
            xyLoc[1] += yChange


        if highRes:
            return 2 * xyLoc
        else:
            return 8 * xyLoc

    def blurImage(self):
        if self.blurredImageAlready:
            return
        
        overAllExposure = np.array(Image.open(self.cleanImagePath)).astype(np.float)
        #Image.open(self.cleanImagePath).save("newSetDebugImages/testClean.png")
        #drawBoxes(Image.open(self.cleanImagePath), self.targets["boxes"]).save("newSetDebugImages/testCleanBoxes.png")

        for image_index in range(self.imageArrayIndex, self.imageArrayIndex + self.windowSize + 1):
            #Image.open(self.imageDirectoryList[image_index]).save("newSetDebugImages/" + str(image_index) + ".png")
            overAllExposure += np.array(Image.open(self.imageDirectoryList[image_index]))

        for image_index in reversed(range(self.imageArrayIndex - self.windowSize, self.imageArrayIndex)):
            #Image.open(self.imageDirectoryList[image_index]).save("newSetDebugImages/" + str(image_index) + ".png")
            overAllExposure += np.array(Image.open(self.imageDirectoryList[image_index]))

        overAllExposure = overAllExposure/(2*self.windowSize + 1)
        overAllExposure = np.clip(overAllExposure, 0, 255)

        self.blurredImage = Image.fromarray(overAllExposure.astype(np.uint8))
        #drawBoxes(self.blurredImage, self.targets["boxes"]).save("newSetDebugImages/testBlurredBoxes.png")

        #self.blurredImage.save("newSetDebugImages/testBlurred.png")

        self.blurredImageAlready = True
        
        return
    def getTargetCopy(self):
        targets = {}

        targets["image_id"] = self.targets["image_id"].detach().clone()
        targets["boxes"] = self.targets["boxes"].detach().clone()
        targets["labels"] = self.targets["labels"].detach().clone()
        targets["iscrowd"] = self.targets["iscrowd"].detach().clone()
        targets["area"] = self.targets["area"].detach().clone()
        targets["confidences"] = self.targets["confidences"].detach().clone()

        return targets
def getAnnPath(imagePath):
    imagePathStr = str(imagePath).replace("blur", "sharp")
    annPathStr = imagePathStr.replace(".png", "_DORS.npy")
    return annPathStr

class GOPROSynth(Dataset):
    """GOPROSynth dataset as det."""

    def __init__(self, root_dir = "/media/mosayed/Cache/datasets/GOPRO_all", split = "test", sharpImages = False, blurredImages = True, expandBoxes = False, threshold = 0.4, transform=None, auxBlur = False):
        """
        Args:
            root_dir (string): Directory with all the images and annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.boxCount = 0
        self.threshold = threshold

        self.transform = transform

        self.cocoConvertor = coco80_to_coco91_class()

        self.conversionMode = False

        self.expandBoxes = expandBoxes


        self.root_dir = root_dir
        self.split = split

        self.blurredImages = blurredImages
        self.sharpImages = sharpImages


        self.dataDirectory = os.path.join(self.root_dir, "train" + "/")
        self.dataDirectory2 = os.path.join(self.root_dir, "test" + "/")

        
        self.imagePaths = list(Path(self.dataDirectory).rglob("*.[pP][nN][gG]")) + list(Path(self.dataDirectory2).rglob("*.[pP][nN][gG]")) #+ list(Path(self.dataDirectory).rglob("*.[pP][nN][gG]"))
        random.shuffle(self.imagePaths)

        if auxBlur:
            maxWindow = 6
            minWindow = 6
        else:
            maxWindow = 6
            minWindow = 3

        self.imageInfos = []

        #self.imagePaths = random.sample(self.imagePaths, 30000)

        allwedEmptyImageSize = 20
        emptyCount = 0
        for imagePathIndex, imagePath in enumerate(self.imagePaths):
            folderDirectory = "/".join(str(imagePath).split("/")[:-1])
            imageArrayIndex, folderImageList = getimage_indexInFolder(imagePath, folderDirectory)
            imageInfo = ImageInfo(imageArrayIndex, imagePath, folderDirectory, folderImageList)
            idealWindowSize = random.randint(minWindow,maxWindow)
            windowSize = min(idealWindowSize, imageInfo.maxFolderIndex-imageArrayIndex)
            if imageArrayIndex-windowSize < 0:
                windowSize = imageArrayIndex-0

            imageInfo.windowSize = windowSize
            self.imageInfos.append(imageInfo)
            imageInfo.image_id = len(self.imageInfos)-1
            imageInfo.loadAnn(self.threshold, self.cocoConvertor)

            imageInfo.targets["windowSize"] = imageInfo.windowSize

            if imageInfo.boxCount == 0:
                emptyCount+=1

            if (imageInfo.boxCount == 0) and (emptyCount > allwedEmptyImageSize):
                #enough of that.
                self.imageInfos = self.imageInfos[:-1]
                continue

            self.boxCount += imageInfo.boxCount

            if imagePathIndex % 200 == 0:
                print("Caching " + str(imagePathIndex))


        imagePathsToStore = []
        for imageInfo in self.imageInfos:
            imagePathsToStore.append(str(imageInfo.cleanImagePath) + "," + str(imageInfo.windowSize))
        
        with open("goproSynthStore.txt", "w") as outfile:
            outfile.write("\n".join(imagePathsToStore))
        # self.annotationPaths = list(Path(self.dataDirectory).rglob("*.[nN][pP][yY]"))
        # self.annotationPaths = [path for path in self.annotationPaths if "sharp/" in str(path)]
        
        print("Done Caching. " + str(len(self.imageInfos)))


    def __len__(self):
        return len(self.imageInfos)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imageInfo = self.imageInfos[idx]

        if self.blurredImages:
            imageInfo.blurImage()
            image = imageInfo.blurredImage.copy()
        else:
            image = imageInfo.cleanImage.copy()

        if self.expandBoxes:
            imageInfo.expandBoxes()

        targets = imageInfo.getTargetCopy()
        
        blur_dict = imageInfo.blur_dict

        if self.transform and not self.conversionMode:
            image, targets, blur_dict = self.transform(image, targets, blur_dict)
            
        return image, targets, blur_dict



class GOPROSynthLoad(Dataset):
    """GOPRO dataset as det."""

    def __init__(self, root_dir = "/media/mosayed/Cache/datasets/GOPROSynth/normalRat", split = "test", sharpImages = False, blurredImages = True, threshold = 0.6, transform=None, expandBoxes = False):
        """
        Args:
            root_dir (string): Directory with all the images and annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.split = split

        self.threshold = threshold

        self.blurredImages = blurredImages
        self.sharpImages = sharpImages

        self.expandBoxes = expandBoxes

        self.dataDirectory = self.root_dir

        
        self.imagePaths = list(Path(self.dataDirectory).rglob("*.[pP][nN][gG]"))
        if sharpImages:
            self.imagePaths = [path for path in self.imagePaths if "sharp/" in str(path)]
        elif blurredImages:
            self.imagePaths = [path for path in self.imagePaths if "blur/" in str(path)]
        else:
            raise Exception("Panicked, no viable image type option.")
                
        self.imagePaths = sorted(self.imagePaths)

        with open("goproSynthStore.txt", "r") as infile:
            imagePathInfos = infile.readlines()



        badDirectories = ["GOPR0384_11_02",
                        "GOPR0384_11_01",
                        "GOPR0384_11_04",
                        "GOPR0477_11_00",
                        "GOPR0477_11_03",
                        "GOPR0384_11_05",
                        "GOPR0385_11_01"]
        
        badDirectories = ["GOPR0871_11_01", 
                        "GOPR0868_11_02", 
                        "GOPR0868_11_01", 
                        "GOPR0857_11_00", 
                        "GOPR0374_11_03", 
                        "GOPR0374_11_02", 
                        "GOPR0374_11_01", 
                        "GOPR0374_11_00", 
                        "GOPR0884_11_00", 
                        "GOPR0881_11_00",
                        "GOPR0396_11_00",
                        "GOPR0862_11_00",
                        "GOPR0868_11_00",
                        "GOPR0869_11_00",
                        "GOPR0871_11_00",
                        "GOPR0881_11_01"]

        badDirectories = ["GOPR0374_11_03",
                        "GOPR0374_11_02",
                        "GOPR0374_11_01",
                        "GOPR0374_11_00",
                        "GOPR0857_11_00",
                        "GOPR0868_11_02",
                        "GOPR0396_11_00",
                        "GOPR0868_11_00",
                        "GOPR0871_11_00"]

        #badDirectories = []

        self.imagePathsTrimmed = []
        for image_index, imagePath in enumerate(self.imagePaths):
            imagePathInfo = imagePathInfos[image_index]
            goodImage = True


            for badDirectory in badDirectories:
                if badDirectory in imagePathInfo:
                    goodImage = False

            if goodImage:
                self.imagePathsTrimmed.append(imagePath)
        
        self.imagePaths = self.imagePathsTrimmed

        # self.annotationPaths = list(Path(self.dataDirectory).rglob("*.[nN][pP][yY]"))
        # self.annotationPaths = [path for path in self.annotationPaths if "sharp/" in str(path)]
        self.annotationPaths = []
        for imagePath in self.imagePaths:
            if expandBoxes:
                imagePathStr = str(imagePath)
            else: 
                imagePathStr = str(imagePath).replace("blur", "sharp")

            annPathStr = imagePathStr.replace(".png", ".dat")
            self.annotationPaths.append(annPathStr)

        self.transform = transform

        self.conversionMode = False

        self.boxCount = 0

        self.emptyCount = 0

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.imagePaths[idx]

        #image = io.imread(img_path)
        if not self.conversionMode:
            image = Image.open(img_path)
        else:
            image = None
            self.width = 1280
            self.height = 720


        annotationfile_path = self.annotationPaths[idx]

        with open(annotationfile_path, 'rb') as f:
            targets = pickle.load(f)

        indicesToTake = []

        for boxIndex in range(targets["confidences"].shape[0]):
            if targets["confidences"][boxIndex] > self.threshold:
                indicesToTake.append(boxIndex)

        targets["confidences"] = targets["confidences"][indicesToTake]
        targets["area"] = targets["area"][indicesToTake]
        targets["iscrowd"] = targets["iscrowd"][indicesToTake]
        targets["labels"] = targets["labels"][indicesToTake]
        targets["boxes"] = targets["boxes"][indicesToTake, :]

        if targets["boxes"].shape[0] == 0:
            self.emptyCount += 1

        self.boxCount+=targets["boxes"].shape[0]
            
        if "windowSize" not in targets:
            targets["windowSize"] = 0    
        targets["image_id"] = torch.tensor(idx)
        targets["windowSize"] = torch.tensor(targets["windowSize"])


        blur_dict = {}

        
        if self.transform and not self.conversionMode:
            image, targets, blur_dict = self.transform(image, targets, blur_dict)
            
        return image, targets, blur_dict

class ImageStruct():
    def __init__(self, imagePath, annPath, blurLabel):
        self.imagePath = imagePath
        self.annPath = annPath
        self.blurLabel = blurLabel
    



class GOPROBlurEst(Dataset):
    """GOPRO dataset as det."""

    def __init__(self, root_dir = "/media/mosayed/Cache/datasets/GOPROSynth/estimatorDS", split = "train", transform=None, expandBoxes = False):
        """
        Args:
            root_dir (string): Directory with all the images and annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """


        self.root_dir = root_dir
        self.split = split

        self.expandBoxes = expandBoxes

        self.dataDirectory = self.root_dir

        if "train" in split:
            self.dataListFile =   root_dir + "/trainSplit.txt"
            self.train = True
        elif "val" in split:
            self.dataListFile =   root_dir + "/valSplit.txt"
            self.train = False
        else:
            raise Exception("Unrecognized split.")


        self.numberOfClasses = 4
        self.labelLists = []
        for i in range(self.numberOfClasses):
            self.labelLists.append([])
        
        self.completeList = []
        self.completeListAnn = []


        with open(self.dataListFile) as fp:
            listContents = fp.readlines()

        for line in listContents:
            splitLine = line.split(",")
            imagePath = splitLine[0]
            blurLabel = int(splitLine[1])
            annPath = imagePath.replace(".png", ".dat")

            if self.train:
                if blurLabel == 0:
                    if random.random() < 0.65:
                        continue
                elif blurLabel == 1:
                    if random.random() < 0.1:
                        continue

            self.labelLists[blurLabel].append(ImageStruct(imagePath, annPath, blurLabel))

            self.completeList.append(ImageStruct(imagePath, annPath, blurLabel))

        self.classCounts = []
        for labelList in self.labelLists:
            self.classCounts.append(len(labelList))

        self.classRatios = np.array(self.classCounts)
        self.classRatios = self.classRatios/np.sum(self.classRatios)
        

        self.transform = transform

        self.conversionMode = False

        self.boxCount = 0

        self._epoch_number= -1

    def __len__(self):
        return len(self.completeList)

    def getItemVal(self, idx):
        imgStruct = self.completeList[idx]


        #image = io.imread(img_path)
        image = Image.open(imgStruct.imagePath)

        with open(imgStruct.annPath, 'rb') as f:
            targets = pickle.load(f)


        self.boxCount+=targets["boxes"].shape[0]
            

        targets["image_id"] = torch.tensor(idx)
        blur_dict = {}
    

        blur_dict["blur_est_label"] = imgStruct.blurLabel
        
        if "blur" in imgStruct.imagePath or "Blur" in imgStruct.imagePath:
            blur_dict["preBlurred"] = True
            blur_dict["windowSize"] = targets["windowSize"]
            del targets["windowSize"]
        else:
            blur_dict["preBlurred"] = True
            blur_dict["windowSize"] = 0

        

        if self.transform and not self.conversionMode:
            image, targets, blur_dict = self.transform(image, targets, blur_dict)

        return image, targets, blur_dict
                
    def getItemTrain(self, idx):

        imgStruct = self.completeList[idx]


        #image = io.imread(img_path)
        image = Image.open(imgStruct.imagePath)

        with open(imgStruct.annPath, 'rb') as f:
            targets = pickle.load(f)


        self.boxCount+=targets["boxes"].shape[0]
            

        targets["image_id"] = torch.tensor(idx)
        blur_dict = {}
    

        blur_dict["blur_est_label"] = imgStruct.blurLabel
        
        if "blur" in imgStruct.imagePath or "Blur" in imgStruct.imagePath:
            blur_dict["preBlurred"] = True
            blur_dict["windowSize"] = targets["windowSize"]
            del targets["windowSize"]
        else:
            blur_dict["preBlurred"] = True
            blur_dict["windowSize"] = 0

        

        if self.transform and not self.conversionMode:
            image, targets, blur_dict = self.transform(image, targets, blur_dict)

        return image, targets, blur_dict

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.train:
            return self.getItemTrain(idx)
        else: 
            return self.getItemVal(idx)
                
