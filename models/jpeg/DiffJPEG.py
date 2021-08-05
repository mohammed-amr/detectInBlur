# Pytorch
import torch
import torch.nn as nn
# Local
from models.jpeg import compression, decompression
from models.jpeg.utils import diff_round, quality_to_factor

# copied from https://github.com/mlomnitz/DiffJPEG

class DiffJPEG(nn.Module):
    def __init__(self, height, width, differentiable=False, quality=80):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image hieght
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme. 
        '''
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round
        factor = quality_to_factor(quality)
        self.compress = compression.compress_jpeg(rounding=rounding, factor=factor)
        self.decompress = decompression.decompress_jpeg(height, width, rounding=rounding,
                                          factor=factor)

    def forward(self, x):
        '''
        '''
        y, cb, cr = self.compress(x)
        recovered = self.decompress(y, cb, cr)
        return recovered

    def setQuality(self, quality):
        newFactor = quality_to_factor(quality)
        self.compress.setFactor(newFactor)
        self.decompress.setFactor(newFactor)

    def setRes(self, new_height, new_width):
        self.decompress.setRes(new_height, new_width)

        