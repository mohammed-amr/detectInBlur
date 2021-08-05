# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base augmentations operators."""



import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import torch
from torchvision import transforms
import cv2
# ImageNet code should change this value

def overlay_boxes_torch(image, predictions):
    """
    Adds the predicted boxes on top of the image
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """

    image = np.array(image)[:, :, [2, 1, 0]].copy()

    labels = predictions["labels"].cpu()
    boxes = predictions['boxes'].cpu()

    colors = compute_colors_for_labels(labels).tolist()
    for box, color in zip(boxes, colors):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(image, (int(top_left[0]), int(top_left[1])), (int(bottom_right[0]), int(bottom_right[1])), tuple(color), 5)

    return image

def compute_colors_for_labels(labels, palette=None):
    """
    Simple function that adds fixed colors depending on the class
    """
    if palette is None:
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors



def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
        level: Level of the operation that will be between [0, `PARAMETER_MAX`].
        maxval: Maximum value that the operation can have. This will be scaled to
            level/PARAMETER_MAX.
    Returns:
        An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.
    Args:
        level: Level of the operation that will be between [0, `PARAMETER_MAX`].
        maxval: Maximum value that the operation can have. This will be scaled to
            level/PARAMETER_MAX.
    Returns:
        A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.

def maskToPIL(masks):

    return masks

def maskToNumpy(masks):

    return masks


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level, target, modify_target_boxes):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees

    degreesRad = -(degrees/180)*np.pi
    rotMat = [[np.cos(degreesRad), -np.sin(degreesRad)],
            [np.sin(degreesRad), np.cos(degreesRad)]]
    
    #Image.fromarray(cv2.cvtColor(overlay_boxes_torch(pil_img, target), cv2.COLOR_BGR2RGB)).save("original.png")

    for index, box in enumerate(target["boxes"]):
        box = box.numpy()
        topLeft = np.array([box[0], box[1]])
        topRight = np.array([box[2], box[1]])

        bottomLeft = np.array([box[0], box[3]])
        bottomRight = np.array([box[2], box[3]])

        points = np.stack([topLeft, topRight, bottomLeft, bottomRight], axis = 1)

        points[0, :] = points[0, :] - pil_img.width/2
        points[1, :] = points[1, :] - pil_img.height/2

        rotatedPoints = np.matmul(rotMat, points)

        rotatedPoints[0, :] = rotatedPoints[0, :] + pil_img.width/2
        rotatedPoints[1, :] = rotatedPoints[1, :] + pil_img.height/2

        xmin = rotatedPoints[0, :].min()
        xmax = rotatedPoints[0, :].max()

        ymin = rotatedPoints[1, :].min()
        ymax = rotatedPoints[1, :].max()

        
        target["boxes"][index] = torch.FloatTensor([xmin, ymin, xmax, ymax])
    
    #Image.fromarray(cv2.cvtColor(overlay_boxes_torch(pil_img.rotate(degrees, resample=Image.BILINEAR), target), cv2.COLOR_BGR2RGB)).save("test.png")

    return pil_img.rotate(degrees, resample=Image.BILINEAR), target


def solarize(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level, target, modify_target_boxes):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level


    #level = level*5
    affineMat = [[1, -level],
            [0, 1]]
    
    #Image.fromarray(cv2.cvtColor(overlay_boxes_torch(pil_img, target), cv2.COLOR_BGR2RGB)).save("original.png")

    for index, box in enumerate(target["boxes"]):
        box = box.numpy()
        topLeft = np.array([box[0], box[1]])
        topRight = np.array([box[2], box[1]])

        bottomLeft = np.array([box[0], box[3]])
        bottomRight = np.array([box[2], box[3]])

        points = np.stack([topLeft, topRight, bottomLeft, bottomRight], axis = 1)

        rotatedPoints = np.matmul(affineMat, points)


        xmin = rotatedPoints[0, :].min()
        xmax = rotatedPoints[0, :].max()

        ymin = rotatedPoints[1, :].min()
        ymax = rotatedPoints[1, :].max()

        target["boxes"][index] = torch.FloatTensor([xmin, ymin, xmax, ymax])

    #Image.fromarray(cv2.cvtColor(overlay_boxes_torch(pil_img.transform(pil_img.size, Image.AFFINE, (1, level, 0, 0, 1, 0), resample=Image.BILINEAR), target), cv2.COLOR_BGR2RGB)).save("test.png")

    return pil_img.transform(pil_img.size,
                            Image.AFFINE, (1, level, 0, 0, 1, 0),
                            resample=Image.BILINEAR), target


def shear_y(pil_img, level, target, modify_target_boxes):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level

    #level = level*5
    affineMat = [[1, 0],
            [-level, 1]]
    
    #Image.fromarray(cv2.cvtColor(overlay_boxes_torch(pil_img, target), cv2.COLOR_BGR2RGB)).save("original.png")

    for index, box in enumerate(target["boxes"]):
        box = box.numpy()
        topLeft = np.array([box[0], box[1]])
        topRight = np.array([box[2], box[1]])

        bottomLeft = np.array([box[0], box[3]])
        bottomRight = np.array([box[2], box[3]])

        points = np.stack([topLeft, topRight, bottomLeft, bottomRight], axis = 1)

        rotatedPoints = np.matmul(affineMat, points)


        xmin = rotatedPoints[0, :].min()
        xmax = rotatedPoints[0, :].max()

        ymin = rotatedPoints[1, :].min()
        ymax = rotatedPoints[1, :].max()

        target["boxes"][index] = torch.FloatTensor([xmin, ymin, xmax, ymax])

    #Image.fromarray(cv2.cvtColor(overlay_boxes_torch(pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, level, 1, 0), resample=Image.BILINEAR), target), cv2.COLOR_BGR2RGB)).save("test.png")

    return pil_img.transform(pil_img.size,
                            Image.AFFINE, (1, 0, 0, level, 1, 0),
                            resample=Image.BILINEAR), target


def translate_x(pil_img, level, target, modify_target_boxes):
    level = int_parameter(sample_level(level), ((pil_img.size[0] + pil_img.size[1])/2) / 3)
    if np.random.random() > 0.5:
        level = -level
    
    #level = level*5
    #Image.fromarray(cv2.cvtColor(overlay_boxes_torch(pil_img, target), cv2.COLOR_BGR2RGB)).save("original.png")

    for index, box in enumerate(target["boxes"]):
        box = box.numpy()
        topLeft = np.array([box[0], box[1]])
        topRight = np.array([box[2], box[1]])

        bottomLeft = np.array([box[0], box[3]])
        bottomRight = np.array([box[2], box[3]])

        points = np.stack([topLeft, topRight, bottomLeft, bottomRight], axis = 1)

        points[0, :] = points[0, :] - level

        xmin = points[0, :].min()
        xmax = points[0, :].max()

        ymin = points[1, :].min()
        ymax = points[1, :].max()

        target["boxes"][index] = torch.FloatTensor([xmin, ymin, xmax, ymax])

    #Image.fromarray(cv2.cvtColor(overlay_boxes_torch(pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, level, 0, 1, 0), resample=Image.BILINEAR), target), cv2.COLOR_BGR2RGB)).save("test.png")


    return pil_img.transform(pil_img.size,
                            Image.AFFINE, (1, 0, level, 0, 1, 0),
                            resample=Image.BILINEAR), target


def translate_y(pil_img, level, target, modify_target_boxes):
    level = int_parameter(sample_level(level), ((pil_img.size[0] + pil_img.size[1])/2) / 3)
    if np.random.random() > 0.5:
        level = -level

    #Image.fromarray(cv2.cvtColor(overlay_boxes_torch(pil_img, target), cv2.COLOR_BGR2RGB)).save("original.png")

    for index, box in enumerate(target["boxes"]):
        box = box.numpy()
        topLeft = np.array([box[0], box[1]])
        topRight = np.array([box[2], box[1]])

        bottomLeft = np.array([box[0], box[3]])
        bottomRight = np.array([box[2], box[3]])

        points = np.stack([topLeft, topRight, bottomLeft, bottomRight], axis = 1)

        points[1, :] = points[1, :] - level

        xmin = points[0, :].min()
        xmax = points[0, :].max()

        ymin = points[1, :].min()
        ymax = points[1, :].max()

        target["boxes"][index] = torch.FloatTensor([xmin, ymin, xmax, ymax])

    #Image.fromarray(cv2.cvtColor(overlay_boxes_torch(pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, 0, 1, level), resample=Image.BILINEAR), target), cv2.COLOR_BGR2RGB)).save("test.png")

    return pil_img.transform(pil_img.size,
                            Image.AFFINE, (1, 0, 0, 0, 1, level),
                            resample=Image.BILINEAR), target


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
        level = float_parameter(sample_level(level), 1.8) + 0.1
        return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
        level = float_parameter(sample_level(level), 1.8) + 0.1
        return ImageEnhance.Contrast(pil_img).enhance(level)

# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
        level = float_parameter(sample_level(level), 1.8) + 0.1
        return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
        level = float_parameter(sample_level(level), 1.8) + 0.1
        return ImageEnhance.Sharpness(pil_img).enhance(level)


augmentations = [
        autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
        translate_x, translate_y
]

non_pos_augmentations = [
        autocontrast, equalize, posterize, solarize
]

augmentations_all = [
        autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
        translate_x, translate_y, color, contrast, brightness, sharpness
]