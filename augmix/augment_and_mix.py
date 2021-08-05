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
"""Reference implementation of AugMix's data augmentation method in numpy."""
import augmix.augmentations
import numpy as np
from PIL import Image
import copy
import cv2
# # CIFAR-10 constants
# MEAN = [0.4914, 0.4822, 0.4465]
# STD = [0.2023, 0.1994, 0.2010]

# COCO constants
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def normalize(image):
    """Normalize input image channel-wise to zero mean and unit variance."""
    image = image.transpose(2, 0, 1)    # Switch to channel-first
    mean, std = np.array(MEAN), np.array(STD)
    image = (image - mean[:, None, None]) / std[:, None, None]
    return image.transpose(1, 2, 0)

def denormalize(image):
    """Normalize input image channel-wise to zero mean and unit variance."""
    image = image.transpose(2, 0, 1)    # Switch to channel-first
    mean, std = np.array(MEAN), np.array(STD)
    image = image * std[:, None, None] + mean[:, None, None]
    return image.transpose(1, 2, 0)


def apply_op(image, op, severity):
    image = np.clip(image * 255., 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(image)    # Convert to PIL.Image
    pil_img = op(pil_img, severity)
    return np.asarray(pil_img) / 255.

def apply_pos_op(image, op, severity, target = None, modify_target_boxes = False):
    image = np.clip(image * 255., 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(image)    # Convert to PIL.Image
    pil_img, target = op(pil_img, severity, target, modify_target_boxes)
    return np.asarray(pil_img) / 255., target

def fix_bounding_box_squeeze(target, image_shape):
    #numpy image shape

    # do this first to catch any weird min max problems
    target["boxes"][target["boxes"][:, 0] > image_shape[1]-1, 0] = image_shape[1] - 1
    target["boxes"][target["boxes"][:, 1] > image_shape[0]-1, 1] = image_shape[0] - 1

    target["boxes"][target["boxes"][:, 2] > image_shape[1]-1, 2] = image_shape[1] - 1 
    target["boxes"][target["boxes"][:, 3] > image_shape[0]-1, 3] = image_shape[0] - 1


    target["boxes"][target["boxes"][:, 0] < 0, 0] = 0
    target["boxes"][target["boxes"][:, 1] < 0, 1] = 0

    target["boxes"][target["boxes"][:, 2] < 0, 2] = 0
    target["boxes"][target["boxes"][:, 3] < 0, 3] = 0

    #take care of bad apples.
    troubleSomeIndices = target["boxes"][:, 0] >= target["boxes"][:, 2]
    target["boxes"][troubleSomeIndices, 2] = target["boxes"][troubleSomeIndices, 2] + 1
    target["boxes"][troubleSomeIndices, 0] = target["boxes"][troubleSomeIndices, 0] - 1

    troubleSomeIndices = target["boxes"][:, 1] >= target["boxes"][:, 3]
    target["boxes"][troubleSomeIndices, 3] = target["boxes"][troubleSomeIndices, 3] + 1
    target["boxes"][troubleSomeIndices, 1] = target["boxes"][troubleSomeIndices, 1] - 1

    # do this again to fix problems left behind by the previous step.
    target["boxes"][target["boxes"][:, 0] > image_shape[1]-1, 0] = image_shape[1] - 1
    target["boxes"][target["boxes"][:, 1] > image_shape[0]-1, 1] = image_shape[0] - 1

    target["boxes"][target["boxes"][:, 2] > image_shape[1]-1, 2] = image_shape[1] - 1
    target["boxes"][target["boxes"][:, 3] > image_shape[0]-1, 3] = image_shape[0] - 1


    target["boxes"][target["boxes"][:, 0] < 0, 0] = 0
    target["boxes"][target["boxes"][:, 1] < 0, 1] = 0

    target["boxes"][target["boxes"][:, 2] < 0, 2] = 0
    target["boxes"][target["boxes"][:, 3] < 0, 3] = 0

    return target


def getPILMasks(masks):
    pilMasks = []
    for mask in masks:
        pilMasks.append(Image.fromarray(masks[0].numpy()))
    return pilMasks

def getPILImage(image):
    image = np.clip(image * 255., 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(image)

    return pil_img

def combine_targets(target1, target2):
    combinedTarget = copy.deepcopy(target1)
    for index, (box1, box2) in enumerate(zip(target1["boxes"], target2["boxes"])):
        combinedTarget["boxes"][index][0] = min(box1[0], box2[0])
        combinedTarget["boxes"][index][1] = min(box1[1], box2[1])

        combinedTarget["boxes"][index][2] = max(box1[2], box2[2])
        combinedTarget["boxes"][index][3] = max(box1[3], box2[3])
    
    return combinedTarget

def augment_and_mix(image, severity=-1, width=3, depth=-1, alpha=1., denormalize_image = False, positional_aug = False, target = None, modify_target_boxes = False):
    """Perform AugMix augmentations and compute mixture.
    Args:
        image: Raw input image as float32 np.ndarray of shape (h, w, c)
        severity: Severity of underlying augmentation operators (between 1 to 10).
        width: Width of augmentation chain
        depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
            from [1, 3]
        alpha: Probability coefficient for Beta and Dirichlet distributions.
    Returns:
        mixed: Augmented and mixed image.
    """
    ws = np.float32(
            np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    if severity == -1:
        severity_to_use = np.random.randint(1, 11)
        pos_severity_to_use = np.random.randint(1, 5)
    else:
        severity_to_use = severity
        pos_severity_to_use = severity

    mix = np.zeros_like(image)
    target_mix = None
    for i in range(width):
        image_aug = image.copy()
        target_aug = copy.deepcopy(target)
        d = depth if depth > 0 else np.random.randint(1, 4)

        for _ in range(d):
            if positional_aug:
                op = np.random.choice(augmix.augmentations.augmentations)
            else:
                op = np.random.choice(augmix.augmentations.non_pos_augmentations)

            #op = augmix.augmentations.translate_y

            if op in augmix.augmentations.non_pos_augmentations:
                image_aug = apply_op(image_aug, op, severity_to_use)
            else:
                image_aug, target_aug = apply_pos_op(image_aug, op, pos_severity_to_use, target_aug, modify_target_boxes)
        
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * normalize(image_aug)

        if target_mix is None:
            target_mix = target_aug
        else:
            #mix target
            target_mix = combine_targets(target_mix, target_aug)

    
    target_mix = fix_bounding_box_squeeze(target_mix, image.shape)

    mixed = (1 - m) * normalize(image) + m * mix

    if denormalize_image:
        mixed = denormalize(mixed)

    if modify_target_boxes:
        return mixed, target_mix
    else:
        return mixed, target