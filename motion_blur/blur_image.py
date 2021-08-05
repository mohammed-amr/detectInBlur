import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import math
from scipy import signal
from scipy import misc
from matplotlib.pyplot import imread
from motion_blur.generate_PSF import PSF
from motion_blur.generate_trajectory import Trajectory
from PIL import Image
import PIL
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# This code borrowed from DeblurGAN, which is a direct python translation of the original
# Boracchi paper.


class BlurImageHandler(object):
    
    def __init__(self, image_path, PSFs=None, pillowImage = None, part=None, path__to_save=None, buffPadImage = True):
        """
        :param image_path: path to square, RGB image.
        :param PSFs: array of Kernels.
        :param part: int number of kernel to use.
        :param path__to_save: folder to save results.
        """

        self.path_to_save = path__to_save
        if PSFs is None:
            if self.path_to_save is None:
                self.PSFs = PSF(canvas=self.shape[0]).fit()
            else:
                self.PSFs = PSF(canvas=self.shape[0], path_to_save=os.path.join(self.path_to_save,
                                                                                'PSFs.png')).fit(save=True)
        else:
            self.PSFs = PSFs

        if pillowImage is None:
            if os.path.isfile(image_path):
                self.image_path = image_path
                self.original = Image.open(self.image_path)
            else:
                raise Exception('Not correct path to image.')
                return
        else:
            self.original = pillowImage
            self.originalPillowImage = self.original
        

        #handle resize before opencv conversion
        self.originalSize = self.original.size
        yN, xN = self.original.size
        key, kex = self.PSFs[0].shape
        deltaY = yN - key
        deltaX = xN - kex
        if deltaY < 0 or deltaX < 0:#'resolution of image should be higher than kernel'
            ratioY = key/yN
            ratioX = kex/xN
            if ratioX > ratioY:
                self.original = self.original.resize((math.ceil(ratioX*xN), math.ceil(ratioX*yN)), PIL.Image.BICUBIC)
            else:
                self.original = self.original.resize((math.ceil(ratioY*xN), math.ceil(ratioY*yN)), PIL.Image.BICUBIC)
        else:
            self.originalSize = None
        
        
        #to numpy array
        self.original = np.array(self.original)


        #now handle padding.
        self.buffPadImage = buffPadImage
        if buffPadImage:
            paddingR = round(self.PSFs[0].shape[0]/2)
            paddingC = round(self.PSFs[0].shape[1]/2)
            if len(self.original.shape) > 2:
                padding = ((paddingR, paddingR), (paddingC, paddingC), (0, 0))
            else:
                padding = ((paddingR, paddingR), (paddingC, paddingC))
            self.original = np.pad(self.original, pad_width=padding, mode='edge')


        self.shape = self.original.shape

        #copy across channels to make RGB if grayscale
        if len(self.shape) < 3:
            tmp = np.zeros((self.shape[0], self.shape[1], 3))
            tmp[:,:,0] = self.original
            tmp[:,:,1] = self.original
            tmp[:,:,2] = self.original
            self.original = tmp
            self.shape = self.original.shape

        

        self.part = part
        self.result = []

    def blur_image(self, save=False, show=False, oldDeltaPad = False):
        if self.part is None:
            psf = self.PSFs
        else:
            psf = [self.PSFs[self.part]]
        yN, xN, channel = self.shape
        key, kex = self.PSFs[0].shape
        deltaY = yN - key
        deltaX = xN - kex

        # pad psf to make compatible with image.
        psf = psf[0]
        if oldDeltaPad:
            tmp = np.pad(psf, deltaX // 2, 'constant')
        else:
            padRight = deltaX//2
            padLeft = math.ceil(deltaX/2)
            padTop = deltaY//2
            padBottom = math.ceil(deltaY/2)
            tmp = np.pad(psf, ((padTop, padBottom),(padLeft, padRight)), 'constant')

        result = []

        # normalize images then use fft convolve to blur the image.
        cv2.normalize(tmp, tmp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        blured = cv2.normalize(self.original, self.original, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                dtype=cv2.CV_32F)
        blured[:, :, 0] = np.array(signal.fftconvolve(blured[:, :, 0], tmp, 'same'))
        blured[:, :, 1] = np.array(signal.fftconvolve(blured[:, :, 1], tmp, 'same'))
        blured[:, :, 2] = np.array(signal.fftconvolve(blured[:, :, 2], tmp, 'same'))
        blured = cv2.normalize(blured, blured, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        #unpad
        if self.buffPadImage:
            paddingR = round(self.PSFs[0].shape[0]/2)
            paddingC = round(self.PSFs[0].shape[1]/2)
            blured = blured[paddingR:blured.shape[0]-paddingR, paddingC:blured.shape[1]-paddingC, :]

        if self.originalSize is not None:
            blured = cv2.resize(blured, self.originalSize, interpolation=cv2.INTER_LANCZOS4)

        result.append(np.abs(blured))

        self.pilImageResult = Image.fromarray((blured*255).astype(np.uint8))

        self.result = result
        
        if show or save:
            self.plot_canvas(show, save)
        
        return True

    def plot_canvas(self, show, save):
        if len(self.result) == 0:
            raise Exception('Please run blur_image() method first.')
        else:
            toSave = cv2.cvtColor(self.result[0], cv2.COLOR_RGB2BGR)
            if self.path_to_save is None:
                raise Exception('Please create Trajectory instance with path_to_save')
            cv2.imwrite(os.path.join(self.path_to_save, self.image_path.split('/')[-1]), toSave * 255)

class BlurImageHandlerSparseIndex(object):
    # same as BluImageHandler, but an attempt at using sparse non zero indices (CPU side) instead of fftconvolve. Ultimately slower.
    def __init__(self, image_path, PSFs=None, pillowImage = None, part=None, path__to_save=None, buffPadImage = True):
        """
        :param image_path: path to square, RGB image.
        :param PSFs: array of Kernels.
        :param part: int number of kernel to use.
        :param path__to_save: folder to save results.
        """

        self.path_to_save = path__to_save
        if PSFs is None:
            if self.path_to_save is None:
                self.PSFs = PSF(canvas=self.shape[0]).fit()
            else:
                self.PSFs = PSF(canvas=self.shape[0], path_to_save=os.path.join(self.path_to_save,
                                                                                'PSFs.png')).fit(save=True)
        else:
            self.PSFs = PSFs

        if pillowImage is None:
            if os.path.isfile(image_path):
                self.image_path = image_path
                self.original = Image.open(self.image_path)
            else:
                raise Exception('Not correct path to image.')
                return
        else:
            self.original = pillowImage
            self.originalPillowImage = self.original
        

        #handle resize before opencv conversion
        self.originalSize = self.original.size
        yN, xN = self.original.size
        key, kex = self.PSFs[0].shape
        deltaY = yN - key
        deltaX = xN - kex
        if deltaY < 0 or deltaX < 0:#'resolution of image should be higher than kernel'
            ratioY = key/yN
            ratioX = kex/xN
            if ratioX > ratioY:
                self.original = self.original.resize((math.ceil(ratioX*xN), math.ceil(ratioX*yN)), PIL.Image.BICUBIC)
            else:
                self.original = self.original.resize((math.ceil(ratioY*xN), math.ceil(ratioY*yN)), PIL.Image.BICUBIC)
        else:
            self.originalSize = None
        
        
        #to numpy array
        self.original = np.array(self.original)


        #now handle padding.
        self.buffPadImage = buffPadImage
        if buffPadImage:
            paddingR = round(self.PSFs[0].shape[0]/2)
            paddingC = round(self.PSFs[0].shape[1]/2)
            if len(self.original.shape) > 2:
                padding = ((paddingR, paddingR), (paddingC, paddingC), (0, 0))
            else:
                padding = ((paddingR, paddingR), (paddingC, paddingC))
            self.original = np.pad(self.original, pad_width=padding, mode='edge')


        self.shape = self.original.shape
        if len(self.shape) < 3:
            tmp = np.zeros((self.shape[0], self.shape[1], 3))
            tmp[:,:,0] = self.original
            tmp[:,:,1] = self.original
            tmp[:,:,2] = self.original
            self.original = tmp
            self.shape = self.original.shape

    
        self.part = part
        self.result = []

    def blur_image(self):
        if self.part is None:
            psf = self.PSFs
        else:
            psf = [self.PSFs[self.part]]
        yN, xN, channel = self.shape
        key, kex = self.PSFs[0].shape
        deltaY = yN - key
        deltaX = xN - kex

        psf = psf[0]

        psf = psf/psf.sum()

        width = 128
        height = 128
        

        non_zero_points = np.nonzero(psf > 0)

        normalized = cv2.normalize(self.original, self.original, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        blured = np.zeros_like(normalized)
        for coord_index in range(non_zero_points[0].shape[0]):
            blured += np.roll(normalized, shift = (non_zero_points[0][coord_index]-63, non_zero_points[1][coord_index]-63), axis = (0,1)) * psf[non_zero_points[0][coord_index], non_zero_points[1][coord_index]]

        #unpad
        if self.buffPadImage:
            paddingR = round(self.PSFs[0].shape[0]/2)
            paddingC = round(self.PSFs[0].shape[1]/2)
            blured = blured[paddingR:blured.shape[0]-paddingR, paddingC:blured.shape[1]-paddingC, :]

        if self.originalSize is not None:
            blured = cv2.resize(blured, self.originalSize, interpolation=cv2.INTER_LANCZOS4)


        self.pilImageResult = Image.fromarray((blured*255).astype(np.uint8))

        return True
