import torch
import torch.nn as nn

import time

class Warper(nn.Module):
    def __init__(self):
        

        super(Warper, self).__init__()
        return
        
    def forward(self, x, thetas, lambda1s, lambda2s):
        
        if len(x.shape) == 3:
            width = x.shape[2]
            height = x.shape[1]
        elif len(x.shape) == 4:
            width = x.shape[3]
            height = x.shape[2]

        zero_tensor = torch.zeros_like(lambda1s)
        one_tensor = torch.ones_like(lambda1s)

        first_row = torch.stack([lambda1s, zero_tensor, zero_tensor], dim = 1)
        second_row = torch.stack([zero_tensor, lambda2s, zero_tensor], dim = 1)
        third_row = torch.stack([zero_tensor, zero_tensor, one_tensor], dim = 1)
        scale_matrices = torch.stack([first_row, second_row, third_row], dim = 2)

        thetas = -thetas
        first_row = torch.stack([torch.cos(thetas), torch.sin(thetas), zero_tensor], dim = 1)
        second_row = torch.stack([-torch.sin(thetas), torch.cos(thetas), zero_tensor], dim = 1)
        rot_matrices = torch.stack([first_row, second_row, third_row], dim = 2)


        first_row = torch.stack([one_tensor, zero_tensor, one_tensor * width], dim = 1)
        second_row = torch.stack([zero_tensor, one_tensor, one_tensor * height], dim = 1)
        trans_matrices = torch.stack([first_row, second_row, third_row], dim = 2)


        forward_warp = torch.bmm(rot_matrices, trans_matrices)
        forward_warp_scaled = torch.bmm(scale_matrices, forward_warp)
        overall_transform = torch.bmm(torch.inverse(forward_warp.float()).half(), forward_warp_scaled)
        
        overall_transform = torch.inverse(overall_transform.float()).half()
        overall_transform = overall_transform[:, 0:2, :]

        affine_grid = torch.nn.functional.affine_grid(theta = overall_transform, size = x.shape, align_corners=False).float().half()
        warpedTensors = torch.nn.functional.grid_sample(x.half(), affine_grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        
        return warpedTensors.float()