import time
import random
import torch
from PIL import Image
import PIL

from motion_blur.generate_PSF import PSF as PSF
from motion_blur.generate_trajectory import Trajectory as Trajectory
from motion_blur.blur_image import BlurImageHandler as BlurImageHandler
from motion_blur.blur_image import BlurImageHandlerSparseIndex as BlurImageHandlerSparseIndex

import numpy as np
import cv2
from torchvision.transforms import functional as F

import os
import math

import augmix.augment_and_mix
import matplotlib.pyplot as plt
import scipy
import copy


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, blur_dict = {}, epoch_number = None, dryRun = False):
        
        blur_dict = copy.deepcopy(blur_dict)
        blur_dict["epoch_number"] = epoch_number
        blur_dict["dryRun"] = dryRun

        for t in self.transforms:
            image, target, blur_dict = t(image, target, blur_dict)
        return image, target, blur_dict

class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target, blur_dict = {}):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target, blur_dict

class AugMix(object):
    def __init__(self, prob = 1.1, include_pos_aug_mix = False, modify_target_boxes = False):
        self.prob = prob
        self.include_pos_aug_mix = include_pos_aug_mix
        self.modify_target_boxes = modify_target_boxes

    def __call__(self, image, target, blur_dict = {}):
        if random.random() < self.prob:
            image, target = augmix.augment_and_mix.augment_and_mix(np.asarray(image)/255, denormalize_image=True, positional_aug=self.include_pos_aug_mix, target = target, modify_target_boxes = self.modify_target_boxes)
            image = Image.fromarray((image*255).astype(np.uint8))

        return image, target, blur_dict

def warpPoints(boxes, M, imageSize = None):
    points1 = np.concatenate((boxes[:, [0,1]], np.ones((boxes.shape[0], 1))), axis = 1)
    points3 = np.concatenate((boxes[:, [2,3]], np.ones((boxes.shape[0], 1))), axis = 1)
    points2 = np.concatenate((boxes[:, [2,1]], np.ones((boxes.shape[0], 1))), axis = 1)
    points4 = np.concatenate((boxes[:, [0,3]], np.ones((boxes.shape[0], 1))), axis = 1)

    points1 = np.matmul(points1, M.T)
    points1 = points1/np.tile(points1[:, [2]], [1,3])

    points3 = np.matmul(points3, M.T)
    points3 = points3/np.tile(points3[:, [2]], [1,3])

    points2 = np.matmul(points2, M.T)
    points2 = points2/np.tile(points2[:, [2]], [1,3])

    points4 = np.matmul(points4, M.T)
    points4 = points4/np.tile(points4[:, [2]], [1,3])

    allXs = np.concatenate((points1[:, [0]], points2[:, [0]], points3[:, [0]], points4[:, [0]]), axis = 1)
    allYs = np.concatenate((points1[:, [1]], points2[:, [1]], points3[:, [1]], points4[:, [1]]), axis = 1)

    topLefts = np.concatenate((np.expand_dims(allXs.min(axis = 1), axis = 1), np.expand_dims(allYs.min(axis = 1), axis = 1)), axis = 1)
    bottomRights = np.concatenate((np.expand_dims(allXs.max(axis = 1), axis = 1), np.expand_dims(allYs.max(axis = 1), axis = 1)), axis = 1)


    newBoxes = torch.from_numpy(np.concatenate((topLefts[:, [0]], topLefts[:, [1]], bottomRights[:, [0]], bottomRights[:, [1]]), axis = 1))

    if imageSize is not None:
        newBoxes[newBoxes[:, 0] > imageSize[0], 0] = imageSize[0] - 1
        newBoxes[newBoxes[:, 1] > imageSize[1], 1] = imageSize[1] - 1

        newBoxes[newBoxes[:, 2] > imageSize[0], 2] = imageSize[0] - 1 
        newBoxes[newBoxes[:, 3] > imageSize[1], 3] = imageSize[1] - 1


        newBoxes[newBoxes[:, 0] < 0, 0] = 1
        newBoxes[newBoxes[:, 1] < 0, 1] = 1

        newBoxes[newBoxes[:, 2] < 0, 2] = 1
        newBoxes[newBoxes[:, 3] < 0, 3] = 1

        badIndices = newBoxes[:, 1] > newBoxes[:, 3]
        temp = newBoxes[badIndices, 3]
        newBoxes[badIndices, 3] = newBoxes[badIndices, 1]
        newBoxes[badIndices, 1] = temp

        badIndices = newBoxes[:, 0] > newBoxes[:, 2]
        temp = newBoxes[badIndices, 2]
        newBoxes[badIndices, 2] = newBoxes[badIndices, 0]
        newBoxes[badIndices, 0] = temp

    return newBoxes


def warpMasksAndTarget(target, M):
    #warps masks then gets bounding boxes for them.
    masks = np.moveaxis(target["masks"].numpy(), 0, 2)
    warpedMasks = cv2.warpPerspective(masks, M, (masks.shape[1], masks.shape[0]), flags = cv2.INTER_LINEAR)

    if target["masks"].shape[0] == 1:
        warpedMasks = np.expand_dims(warpedMasks, 2)

    for mask_index in range(target["masks"].shape[0]):
        numpyMask = warpedMasks[:, :, mask_index]

        target["masks"][mask_index, :, :] = torch.from_numpy(numpyMask)

        non_zero_indices = np.nonzero(numpyMask > 0)
        coordY = non_zero_indices[0]
        coordX = non_zero_indices[1]

        if len(coordX) == 0 or len(coordY) == 0:
            print("Couldn't do anything about this mask, skipping")
            continue

        minX = coordX.min()
        maxX = coordX.max()

        minY = coordY.min()
        maxY = coordY.max()

        target["boxes"][mask_index, :] = torch.FloatTensor([minX, minY, maxX, maxY])

        del coordX
        del coordY
        del non_zero_indices

    del masks
    del warpedMasks

    return target

class ToTensor(object):
    def __call__(self, image, target, blur_dict = {}):
        image = F.to_tensor(image)
        return image, target, blur_dict


def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def gammaFunc(x, gammaVal):
  return math.pow(x, gammaVal)


class BlurImage(object):
    def __init__(self, 
        prob = 0.5, 
        blur_type = None, 
        blur_exposure = None, 
        use_stored_psfs = False, 
        stored_psf_directory = None, 
        blur_image_in_transform = True, 
        dont_center_psf = False, 
        low_exposure = False, 
        high_exposure = False, 
        dilate_psf = False, 
        LEHE_blur_seg = False):

        self.prob = prob
        self.blur_type = blur_type
        self.blur_exposure = blur_exposure

        self.use_stored_psf = use_stored_psfs
        self.stored_psf_directory = stored_psf_directory
        self.blur_image_in_transform = blur_image_in_transform
        self.dont_center_psf = dont_center_psf

        self.LEHE_blur_seg = LEHE_blur_seg

        self.low_exposure = low_exposure
        self.high_exposure = high_exposure

        self.dilate_psf = dilate_psf

        if self.blur_image_in_transform:
            print("Blurring internally in transform on CPU")
        else:
            print("Not blurring internally on CPU.")

        self.count = 0

    def __call__(self, image, target = None, blur_dict = {}):
        #epochNumber = None, dryRun = False, psf = None, inverseWarp = None, theta_rad = None, scale_factor_lambda1 = None, scale_factor_lambda2 = None
        if "preBlurred" in blur_dict and blur_dict["preBlurred"]:
            blur_dict["blurring"] = False 
            blur_dict["psf"] = [0]
            blur_dict["inverseWarp"] = None
            blur_dict["theta_rad"] = 0
            blur_dict["scale_factor_lambda1"] = 1
            blur_dict["scale_factor_lambda2"] = 1
            blur_dict["param_index"] = None
            blur_dict["fraction_index"] = None

            return image, target, blur_dict

        # equal among 4 classes for LEHE, so we need to make not blurring less likey.
        if self.LEHE_blur_seg:
            thresholdProb = 1-0.0625
        else:
            thresholdProb = self.prob

        #are we blurring? 
        if random.random() < thresholdProb:
            # params = [0.005, 0.001, 0.00005]
            # fractions = [1/18, 1/10, 1/5, 1/2, 1]

            params = [0.005, 0.001, 0.00005]
            fractions = [1/18, 1/10, 1/5, 1/2, 1]

            # check if we have a predetermined exposure value to use.
            if self.blur_exposure is not None:
                fraction = self.blur_exposure
            else:
                # random selection based on the range we need.
                if self.high_exposure:
                    fraction_index = random.choice(range(len(fractions[3:]))) + 3
                elif self.low_exposure:
                    fraction_index = random.choice(range(len(fractions[:3])))
                elif self.LEHE_blur_seg:
                    fraction_index = random.choices(range(len(fractions)), weights = [0.0625, 0.0625, 0.0625, 0.375, 0.375])[0]
                else:
                    fraction_index = random.choice(range(len(fractions)))

                fraction = fractions[fraction_index]

            # check if we have a predetermined blur type to use.
            if self.blur_type is not None:
                param = self.blur_type
            else:
                # random selection
                param_index = random.choice(range(len(params)))
                param = params[param_index]

            # should we be used stored PSFs and not generate our own?
            if self.use_stored_psf:
                # If we're loading a stored PSF, then we're limited by whatever types indices we have stored.
                # Sample an index and discard what we had previously.
                if self.blur_type is not None:
                    param_index = self.blur_type
                else:
                    param_index = random.choice([1,2,3])
                
                # do the same here too.
                if self.blur_exposure is not None:
                    fraction_index = self.blur_exposure
                else:
                    if self.high_exposure:
                        fraction_index = random.choice([3,4])
                    elif self.low_exposure:
                        fraction_index = random.choice([0,1,2])
                    elif self.LEHE_blur_seg:
                        fraction_index = random.choices([0,1,2,3,4], weights = [0.0625, 0.0625, 0.0625, 0.375, 0.375])[0]
                    else:
                        fraction_index = random.choice([0,1,2,3,4])

                # pick a random PSF index.
                psfIndex = random.randint(0, 12000-1)

                #load the pickled PSF file into memory.
                filePath =self.stored_psf_directory + "/P" + str(param_index) + "E" + str(fraction_index) + "/I" + "{:06d}".format(psfIndex)

                with open(filePath, 'rb') as f:
                    psf = [np.load(f)]

                    #psf generation produced 256 by 256 kernels. If that's the case then crop them down.
                    #this shouldn't clip the PSF itself inside the matrix.
                    if psf[0].shape[0] > 128:
                        psf[0] = psf[0][64:128+64,64:128+64]
            else:
                #not using a stored PSF, so we need to generate it ourselves.

                #start_time = time.perf_counter()
                
                #get a trajectory
                trajectory_obj = Trajectory(canvas=256, max_len=96, expl=param).fit()
                trajectory = trajectory_obj.fit()
                
                #rasterize the trajectory into a canvas. 
                psf_object = PSF(canvas=256, trajectory=trajectory, fraction = [fraction])
                psf = psf_object.fit()

                #if we're not centering the PSF, then we can't trim the 256 canvas down to 128 
                #since the PSF will spill outside the boundary.
                if self.dont_center_psf:
                    psf = psf_object.PSFs
                else:
                    # we can center. 
                    psf_object.centerPSF()

                    psf = psf_object.PSFs

                    # since we're centering, we can crop away excess black.
                    if psf[0].shape[0] > 128:
                        psf[0] = psf[0][64:128+64,64:128+64]
            
            # defocus
            if self.dilate_psf:
                # filter the PSF with a random Gaussian.
                sigma = np.random.uniform(low=0, high=3)
                psf[0] = scipy.ndimage.gaussian_filter(psf[0], sigma)
                psf[0] = psf[0]/psf[0].max()
                
            if self.blur_image_in_transform:
                # cpu blurring! 
                #start_time = time.perf_counter()

                # send to the CPU image blurer.
                blur_image_obj = BlurImageHandler(image_path = None, PSFs = [psf[0].astype(np.float32)], pillowImage = image)
                allGood = blur_image_obj.blur_image()

                #store result
                self.pilImageResult = blur_image_obj.pilImageResult
                if not allGood:
                    print("Error in blurring.")
                if blur_image_obj.pilImageResult is None:
                    print("Error in blurring.")

                outputImage = blur_image_obj.pilImageResult
            else:
                outputImage = image
            

            # extracting PSF principal components.

            nonZeroIndices = np.nonzero(psf[0] > 0)
            coordY = nonZeroIndices[0]
            coordX = nonZeroIndices[1]
            
            coordYP = (coordY - coordY.mean())
            coordXP = (coordX - coordX.mean())

            cov = (coordYP*coordXP).mean()

            varX = (coordXP*coordXP).mean()
            varY = (coordYP*coordYP).mean()


            lambda1 = (varX + varY)/2 + math.sqrt(math.pow((varX-varY)/2, 2) + math.pow(cov, 2))
            lambda2 = (varX + varY)/2 - math.sqrt(math.pow((varX-varY)/2, 2) + math.pow(cov, 2))

            scale_factor_lambda1 = 1 - (sigmoid(math.sqrt(lambda1)/10)-0.5)*0.6
            scale_factor_lambda2 = 1 - (sigmoid(math.sqrt(lambda2)/10)-0.5)*0.6

            theta_rad = -math.atan2(lambda1 - varX, -cov)

            if self.blur_image_in_transform:
                outputImage = blur_image_obj.pilImageResult
            else:
                outputImage = image

            #house keeping for memory leaks. 

            if self.blur_image_in_transform:
                del blur_image_obj
            if not self.use_stored_psf:
                del trajectory_obj
                del psf_object

            del coordYP
            del coordXP
            del coordX
            del coordY
            del nonZeroIndices

            
            # internal counter for debug
            self.count += 1

            del image

            blur_dict["blurring"] = True
            blur_dict["psf"] = psf[0]
            blur_dict["theta_rad"] = theta_rad
            blur_dict["scale_factor_lambda1"] = scale_factor_lambda1
            blur_dict["scale_factor_lambda2"] = scale_factor_lambda2

            if self.blur_type is not None:
                # This is a hacky way of getting a binned index for an input blur type
                # used for sweep blur evals.
                # need to update based on closest type index.
                blur_param_differences = np.abs(np.asarray(params) - self.blur_type)
                param_index = np.argmin(blur_param_differences)
                blur_dict["param_index"] = param_index

                # for storedPSFs, the index is off by one
                if self.use_stored_psf:
                    blur_dict["param_index"] = blur_dict["param_index"] - 1

            else:
                # we picked the index here internally so pass as is.
                blur_dict["param_index"] = param_index

                if self.use_stored_psf:
                    blur_dict["param_index"] = blur_dict["param_index"] - 1

            if self.blur_exposure is not None:
                # same as blur type
                blur_fraction_differences = np.abs(np.asarray(fractions) - self.blur_exposure)
                fraction_index = np.argmin(blur_fraction_differences)
                blur_dict["fraction_index"] = fraction_index

                # for stored PSFs we don't bother with exposures lower than 1/100. Not used for eval
                # anymore either, but left here for backwards compatibility.
                if self.blur_exposure < 1/90:
                    blur_dict["fraction_index"] = -1
            else:
                blur_dict["fraction_index"] = fraction_index
            
            # return everything including the blur_dict.
            return outputImage, target, blur_dict
            

        else:
            blur_dict["blurring"] = False 
            blur_dict["psf"] = [0]
            blur_dict["theta_rad"] = 0
            blur_dict["scale_factor_lambda1"] = 1
            blur_dict["scale_factor_lambda2"] = 1
            blur_dict["param_index"] = None
            blur_dict["fraction_index"] = None

            return image, target, blur_dict



def add_jpeg_artifact_to_image(image_GPU, jpeg_compressor, quality):
    image_GPU = image_GPU.unsqueeze(0)
    originalImgWidth = image_GPU.shape[3]
    originalImgHeight = image_GPU.shape[2]


    width_to_pad = 16-originalImgWidth%16
    height_to_pad = 16-originalImgHeight%16

    left_pad = math.floor(width_to_pad/2)
    right_pad = math.ceil(width_to_pad/2)
    top_pad = math.floor(height_to_pad/2)
    bottom_pad = math.ceil(height_to_pad/2)

    padded_img_tensor = torch.nn.functional.pad(image_GPU, (left_pad, right_pad, top_pad, bottom_pad), mode='reflect')

    padded_img_width = padded_img_tensor.shape[3]
    padded_img_height = padded_img_tensor.shape[2]

    jpeg_compressor.setQuality(quality)
    jpeg_compressor.setRes(padded_img_height, padded_img_width)

    img_tensor_comp = jpeg_compressor(padded_img_tensor.float())

    image_GPU = img_tensor_comp[:, :, top_pad:padded_img_height-bottom_pad, left_pad:padded_img_width-right_pad].cpu()

    return image_GPU.half().detach().squeeze()
