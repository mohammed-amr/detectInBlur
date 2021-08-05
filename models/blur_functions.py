import math
import torch 
import numpy as np

import utils

import time

import transforms

def manual_blur(image_GPU, psf_GPU, add_noise = False, noise_level = 0.001, add_block = False, add_jpeg_artifact = False, jpeg_compressor = None):
    # image is CxHxW. 
    # psf is kxk where k is either 128 or 256. psf should be normalized (all vals sum to one)
    # this is faster than using a standard conv with filter size 128 * 128

    # for filters of size 256
    if psf_GPU.shape[0] > 129:

        image_GPU = image_GPU.unsqueeze(0)
        image_width = image_GPU.shape[3]
        image_height = image_GPU.shape[2]

        width = 256
        height = 256
        p1d = (math.floor(width/2)-1, math.ceil(width/2), math.floor(height/2)-1, math.ceil(height/2))

        # if the image is smaller than the filter, then we can't use reflect padding.
        if image_GPU.shape[2] <= 128 or image_GPU.shape[3] <= 128:
            pad_mode = "replicate"
        else:
            pad_mode = 'replicate'

        image_GPU = torch.nn.functional.pad(image_GPU, p1d , mode=pad_mode)

        output = torch.zeros_like(image_GPU)

        non_zero_points = psf_GPU.nonzero(as_tuple=False)

        for coord_index in range(non_zero_points.shape[0]):
            output += torch.roll(image_GPU, shifts = (non_zero_points[coord_index, 0]-127, non_zero_points[coord_index, 1]-127), dims = (2,3)) * psf_GPU[non_zero_points[coord_index, 0], non_zero_points[coord_index, 1]]

        output =  output[:, :, 127: 127 + image_height, 127: 127 + image_width].squeeze()
    else:
        # for filters of size 128
        image_GPU = image_GPU.unsqueeze(0)
        image_width = image_GPU.shape[3]
        image_height = image_GPU.shape[2]


        width = 128
        height = 128
        p1d = (math.floor(width/2)-1, math.ceil(width/2), math.floor(height/2)-1, math.ceil(height/2))

        # if the image is smaller than the filter, then we can't use reflect padding.
        if image_GPU.shape[2] < 64 or image_GPU.shape[3] < 64:
            pad_mode = "constant"
        else:
            pad_mode = 'reflect'
        image_GPU = torch.nn.functional.pad(image_GPU, p1d , mode=pad_mode)

        output = torch.zeros_like(image_GPU)

        non_zero_points = psf_GPU.nonzero(as_tuple=False)
        
        # loop over PSF nonzero values and mult sum image to output
        for coord_index in range(non_zero_points.shape[0]):
            output += torch.roll(image_GPU, shifts = (non_zero_points[coord_index, 0]-63, non_zero_points[coord_index, 1]-63), dims = (2,3)) * psf_GPU[non_zero_points[coord_index, 0], non_zero_points[coord_index, 1]]

        output = output[:, :, 63: 63 + image_height, 63: 63 + image_width].squeeze()
    
    
    if add_noise:
        noise_var = np.random.uniform(0.00000001, noise_level)
        output = torch.clamp(output + (torch.randn_like(output) * math.sqrt(noise_var)), 0, 1)

    if add_block:
        if np.random.uniform(0,1) > 0.5:
            original_shape = output.shape
            scale_factor = np.random.uniform(0.6, 1)
            output = torch.nn.functional.interpolate(output.unsqueeze(axis = 0), scale_factor = (scale_factor, scale_factor), mode='nearest').squeeze()
            output = torch.nn.functional.interpolate(output.unsqueeze(axis = 0), size = original_shape[1:], mode='nearest').squeeze()


    if add_jpeg_artifact:
        if np.random.uniform(0,1) > 0.35:
            quality = np.random.uniform(20,90) 
            output = transforms.add_jpeg_artifact_to_image(output, jpeg_compressor, quality)

    return output


def blur_image_list(images_GPU, blur_dicts, psfs_GPU, add_noise = False, noise_level = 0.001, add_block = False, add_jpeg_artifact = False, jpeg_compressor = None):

    for image_index, (image_GPU, blur_dict, psf_GPU) in enumerate(zip(images_GPU, blur_dicts, psfs_GPU)):
        if not blur_dict["blurring"]:
            continue

        psf_GPU = psf_GPU/psf_GPU.sum()
        
        images_GPU[image_index] = manual_blur(image_GPU, psf_GPU, add_noise = add_noise, noise_level = noise_level, add_block = add_block, add_jpeg_artifact = add_jpeg_artifact, jpeg_compressor = jpeg_compressor)