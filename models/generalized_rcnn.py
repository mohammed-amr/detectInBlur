# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

from collections import OrderedDict
import torch
from torch import nn
import warnings
from torch.jit.annotations import Tuple, List, Dict, Optional
from torch import Tensor

import torchvision

import numpy as np
from PIL import Image
from matplotlib import cm

import models.warper

def saveImageTensors(imageTensors, suffix = "preWarp"):
    for index in range(imageTensors.shape[0]):
        torchvision.transforms.functional.to_pil_image(imageTensors[index,:,:,:].squeeze().cpu().float()).save("debugImages/GenRcnnFiles/" + str(index) + "GenRcnnImage" + suffix + ".png")

    return

def saveActivationTensors(activations, suffix = "preWarp"):
    for index in range(activations.shape[0]):
        act = activations[index,:,:,:].detach().cpu().squeeze()

        act = torch.mean(act, dim = 0).squeeze()
        act = act - torch.min(act)
        act = act/torch.max(act)
        act = act.numpy()

        img = Image.fromarray(np.uint8(cm.inferno(act)*255)).convert('RGB')
        
        img.save("debugImages/GenRcnnFiles/" + str(index) + "GenRcnnimageAct" + suffix + ".png")

    return


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform, warp_internally):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False

        self.warp_internally = warp_internally

        if warp_internally:
            self.warper = models.warper.Warper()

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None, thetas = None, lambda1s = None, lambda2s = None, killWarp = False, newMeans = None, newSTDs = None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets, newMeans, newSTDs)
        #images, targets = self.transform(images, targets)


        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenrate box
                    bb_idx = degenerate_boxes.any(dim=1).nonzero().view(-1)[0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invaid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        if self.warp_internally and not killWarp:
            #saveImageTensors(images.tensors, suffix = "PreSkew")
            warpedTensors = self.warper(images.tensors, thetas, lambda1s, lambda2s)
            #saveImageTensors(warpedTensors, suffix = "PostSkew")
            
            features = self.backbone(warpedTensors)

            for featureKey, feature in features.items():
                #saveActivationTensors(feature, suffix = "PreSkew" + featureKey)
                features[featureKey] = self.warper(feature, thetas, 1/lambda1s, 1/lambda2s)
                #saveActivationTensors(features[featureKey], suffix = "PostSkew" + featureKey)
        else:
            features = self.backbone(images.tensors)
            
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return (losses, detections)
        else:
            return self.eager_outputs(losses, detections)
