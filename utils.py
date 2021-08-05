from __future__ import print_function

from collections import defaultdict, deque
import datetime
import pickle
import time

import torch
import torch.distributed as dist
import torchvision

from torchvision import transforms as torch_transforms

import numpy as np

import errno
import os

import models

import colorsys

import cv2

def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x

###################################################### BN Methods ####################################################################
def convert_to_regular_batchnorm(model):
    
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = convert_to_regular_batchnorm(model=module)

        if type(module) == torchvision.ops.misc.FrozenBatchNorm2d:
            layer_old = module
            num_features = module.running_mean.shape[0]
            eps = module.eps
            layer_new = torch.nn.BatchNorm2d(num_features, eps=eps, momentum=0.1, affine=True, track_running_stats=True)

            layer_new.weight = torch.nn.Parameter(module.weight)
            layer_new.bias = torch.nn.Parameter(module.bias)
            layer_new.running_mean = module.running_mean.clone().detach()
            layer_new.running_var = module.running_var.clone().detach()

            model._modules[name] = layer_new

    return model

def convert_to_custom_batch_norm(model, batch_norm_to_use):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = convert_to_custom_batch_norm(module, batch_norm_to_use)



        if type(module) == torchvision.ops.misc.FrozenBatchNorm2d:
            num_features = module.running_mean.shape[0]
            layer_new = batch_norm_to_use(num_features, momentum=0.1, affine=True, track_running_stats=True)

            layer_new.weight = torch.nn.Parameter(module.weight)
            layer_new.bias = torch.nn.Parameter(module.bias)
            layer_new.running_mean = module.running_mean.clone().detach()
            layer_new.running_var = module.running_var.clone().detach()

            model._modules[name] = layer_new

    return model

def turn_batch_norm_on(model):
    if isinstance(model, torch.nn.modules.batchnorm._BatchNorm) or isinstance(model, models.batchnorm._BatchNorm):
        model.train()

    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = turn_batch_norm_on(model=module)

        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) or isinstance(module, models.batchnorm._BatchNorm):
            module.train()

    return model

def set_batch_momentum_zero(model):
    if isinstance(model, torch.nn.modules.batchnorm._BatchNorm) or isinstance(model, models.batchnorm._BatchNorm):
        model.momentum = None

    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = set_batch_momentum_zero(model=module)

        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) or isinstance(module, models.batchnorm._BatchNorm):
            module.momentum = None

    return model

def set_batch_norm_acclimation_mode(model, flag):
    if isinstance(model, torch.nn.modules.batchnorm._BatchNorm) or isinstance(model, models.batchnorm._BatchNorm):
        model.acclimation_mode = flag

    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = set_batch_norm_acclimation_mode(module, flag)

        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) or isinstance(module, models.batchnorm._BatchNorm):
            module.acclimation_mode = flag

    return model

def set_batch_norm_mode1(model, flag):
    if isinstance(model, torch.nn.modules.batchnorm._BatchNorm) or isinstance(model, models.batchnorm._BatchNorm):
        model.mode_one = flag

    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = set_batch_norm_mode1(module, flag)

        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) or isinstance(module, models.batchnorm._BatchNorm):
            module.mode_one = flag

    return model

def set_batch_momentum(model, momentum):
    if isinstance(model, torch.nn.modules.batchnorm._BatchNorm) or isinstance(model, models.batchnorm._BatchNorm):
        model.momentum = momentum

    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = set_batch_momentum(module, momentum)

        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) or isinstance(module, models.batchnorm._BatchNorm):
            module.momentum = momentum

    return model

def set_batch_norm_N(model, N):
    if isinstance(model, torch.nn.modules.batchnorm._BatchNorm) or isinstance(model, models.batchnorm._BatchNorm):
        model.num_batches_tracked = module.num_batches_tracked + N

    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = set_batch_norm_N(module, N)

        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) or isinstance(module, models.batchnorm._BatchNorm):
            module.num_batches_tracked = module.num_batches_tracked + N

    return model


def get_BN_handles(model, nameSoFar = ""):
    bnNames = []
    bnList = []

    for idx, (name, module) in enumerate(model._modules.items()):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) or isinstance(module, models.batchnorm._BatchNorm):
            bnNames.append(nameSoFar + "_" + name)
            bnList.append(module)

    for idx, (name, module) in enumerate(model._modules.items()):
        if len(list(module.children())) > 0:
            newBNList, newBNNames = get_BN_handles(module, nameSoFar + '_' + name)
            bnList = bnList + newBNList
            bnNames = bnNames + newBNNames

    return bnList, bnNames

def check_batch_norm_status(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = check_batch_norm_status(model=module)

        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) or isinstance(module, models.batchnorm._BatchNorm):
            print(module.training, module.momentum, module.num_batches_tracked)
            
    return model

def copy_BN_stats_to_old(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = copy_BN_stats_to_old(model=module)

        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) or isinstance(module, models.batchnorm._BatchNorm):
            module.old_running_mean = module.running_mean.clone()
            module.old_running_var = module.running_var.clone()
            module.old_num_batches_tracked = module.num_batches_tracked.clone()


    return model

def reset_batch_norm_values(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = reset_batch_norm_values(model=module)

        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) or isinstance(module, models.batchnorm._BatchNorm):
            module.reset_running_stats()


    return model

def get_norm_params(blur_dicts, use_custom_image_norm):
    #output from transform is one of 5 possible blur exopsures (0,1,2,3,4)
    canonicalMeansStats = [0.485, 0.456, 0.406]
    canonicalSTDStats = [0.229, 0.224, 0.225]

    meansP1 = np.asarray([[0.4695, 0.4461, 0.4068], [0.4694, 0.4461, 0.4067], [0.4696, 0.4462, 0.4069], [0.4696, 0.4462, 0.4069], [0.4693, 0.4460, 0.4067], [0.4692, 0.4459, 0.4066]]).T
    meansP2 = np.asarray([[0.4695, 0.4461, 0.4068], [0.4694, 0.4461, 0.4067], [0.4696, 0.4462, 0.4069], [0.4695, 0.4462, 0.4068], [0.4693, 0.4460, 0.4066], [0.4691, 0.4458, 0.4065]]).T
    meansP3 = np.asarray([[0.4695, 0.4461, 0.4068], [0.4694, 0.4461, 0.4067], [0.4696, 0.4462, 0.4069], [0.4695, 0.4462, 0.4068], [0.4693, 0.4460, 0.4066], [0.4693, 0.4460, 0.4066]]).T

    stdP1 = np.asarray([[0.2384, 0.2334, 0.2370], [0.2337, 0.2288, 0.2325], [0.2270, 0.2221, 0.2261], [0.2209, 0.2161, 0.2203], [0.2127, 0.2082, 0.2126], [0.2087, 0.2043, 0.2088]]).T
    stdP2 = np.asarray([[0.2384, 0.2334, 0.2370], [0.2337, 0.2287, 0.2325], [0.2267, 0.2218, 0.2258], [0.2184, 0.2137, 0.2180], [0.2048, 0.2006, 0.2051], [0.1950, 0.1911, 0.1957]]).T
    stdP3 = np.asarray([[0.2384, 0.2334, 0.2370], [0.2337, 0.2287, 0.2325], [0.2266, 0.2217, 0.2258], [0.2182, 0.2136, 0.2178], [0.2012, 0.1972, 0.2017], [0.1824, 0.1790, 0.1838]]).T

    stdP1 = (stdP1*0.229)/0.2384
    stdP2 = (stdP2*0.229)/0.2384
    stdP3 = (stdP3*0.229)/0.2384


    if blur_dicts is None:
        norm_means = np.zeros((1, 3))
        norm_stds = np.zeros((1, 3))
        norm_means[0, :] = canonicalMeansStats
        norm_stds[0, :] = canonicalSTDStats

        return norm_means, norm_stds


    norm_means = np.zeros((len(blur_dicts), 3))
    norm_stds = np.zeros((len(blur_dicts), 3))

    for blur_dict_index, blur_dict in enumerate(blur_dicts):
        
        if use_custom_image_norm and blur_dict["blurring"] and blur_dicts[blur_dict_index]["param_index"] is not None:
            param_index = blur_dicts[blur_dict_index]["param_index"]
            fraction_index = blur_dicts[blur_dict_index]["fraction_index"]

            if fraction_index == -1:
                norm_means[blur_dict_index, :] = canonicalMeansStats
                norm_stds[blur_dict_index, :] = canonicalSTDStats
                continue
            
            if param_index == 2:
                norm_means[blur_dict_index, :] = canonicalMeansStats#meansP3[:, fraction_index+1]
                norm_stds[blur_dict_index, :] = stdP3[:, fraction_index+1]
            if param_index == 1:
                norm_means[blur_dict_index, :] = canonicalMeansStats#meansP2[:, fraction_index+1]
                norm_stds[blur_dict_index, :] = stdP2[:, fraction_index+1]
            if param_index == 0:
                norm_means[blur_dict_index, :] = canonicalMeansStats#meansP1[:, fraction_index+1]
                norm_stds[blur_dict_index, :] = stdP1[:, fraction_index+1] 
        else:
            norm_means[blur_dict_index, :] = canonicalMeansStats
            norm_stds[blur_dict_index, :] = canonicalSTDStats

    return norm_means, norm_stds


###################################################### Viz ####################################################################


def create_unique_color_float(tag, hue_step=0.05):
    """Create a unique RGB color code for a given track id (tag).
    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.
    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).
    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]
    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v) 
    r = (1+((tag % 10)/1.5) ) * r
    g = (1+((tag % 10)/2) ) * g
    b = (1+((tag % 10)/3) ) * b
    return r*255, g*255, b*255

def compute_colors_for_labels(labels, palette=None):
    """
    Simple function that adds fixed colors depending on the class
    """
    if palette is None:
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        palette = torch.tensor([10 - 1, 7 - 1, 5 - 1])


    colors = []
    for label in labels[:, None]:
        colors.append(create_unique_color_float(label.cpu().numpy()[0]))

    colors = np.asarray(colors)

    colors = (colors % 255)

    return colors

def overlay_boxes_torch(image_GPU, predictions):
    """
    Adds the predicted boxes on top of the image
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """

    image = torch_transforms.ToPILImage()(image_GPU.float().cpu()).convert("RGB")
    image = np.array(image)[:, :, [2, 1, 0]].copy()

    labels = predictions["labels"].cpu()
    boxes = predictions['boxes'].cpu()

    if 'scores' in predictions:
        scores = predictions['scores'].cpu()

        colors = compute_colors_for_labels(labels).tolist()
        for box, color, score in zip(boxes, colors, scores):
            if score > 0.5:
                box = box.to(torch.int64)
                top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
                image = cv2.rectangle(image, (int(top_left[0]), int(top_left[1])), (int(bottom_right[0]), int(bottom_right[1])), tuple(color), 2)
    else:
        colors = compute_colors_for_labels(labels).tolist()
        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(image, (int(top_left[0]), int(top_left[1])), (int(bottom_right[0]), int(bottom_right[1])), tuple(color), 2)

    return image

###################################################### Bounding Box Utils ####################################################################
def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def expand_targets(targets_GPU, blur_dicts, psfs_GPU, images_GPU):

    for target_index, (target_GPU, blur_dict, psf_GPU, image_GPU) in enumerate(zip(targets_GPU, blur_dicts, psfs_GPU, images_GPU)):
        if not blur_dict["blurring"]:
            continue

        # all filters are 128 wide, except those that are non centered. 
        # If not centered they have no business here with expand.
        # This function is not flexible on purpose.
        if psf_GPU.shape[0] != 128:
            raise Exception("Trying to expand with filters that are not 128 wide!")

        psf_GPU = psf_GPU/psf_GPU.sum()

        non_zero_points = psf_GPU.nonzero(as_tuple=False)

        leftExpansion = non_zero_points[:, 1].min() - 63 #x
        rightExpansion = non_zero_points[:, 1].max() - 63 #x

        topExpansion = non_zero_points[:, 0].min() - 63 #y
        bottomExpansion = non_zero_points[:, 0].max() - 63 #y

        target_GPU["boxes"][:, 0] = target_GPU["boxes"][:, 0] + leftExpansion
        target_GPU["boxes"][:, 2] = target_GPU["boxes"][:, 2] + rightExpansion

        target_GPU["boxes"][:, 1] = target_GPU["boxes"][:, 1] + topExpansion
        target_GPU["boxes"][:, 3] = target_GPU["boxes"][:, 3] + bottomExpansion

        fix_bounding_box_squeeze(target_GPU, image_GPU.shape)

        targets_GPU[target_index] = target_GPU

    return targets_GPU


def fix_bounding_box_squeeze(target, image_shape):

    # do this first to catch any weird min max problems
    target["boxes"][target["boxes"][:, 0] > image_shape[2]-1, 0] = image_shape[2] - 1
    target["boxes"][target["boxes"][:, 1] > image_shape[1]-1, 1] = image_shape[1] - 1

    target["boxes"][target["boxes"][:, 2] > image_shape[2]-1, 2] = image_shape[2] - 1 
    target["boxes"][target["boxes"][:, 3] > image_shape[1]-1, 3] = image_shape[1] - 1


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
    target["boxes"][target["boxes"][:, 0] > image_shape[2]-1, 0] = image_shape[2] - 1
    target["boxes"][target["boxes"][:, 1] > image_shape[1]-1, 1] = image_shape[1] - 1

    target["boxes"][target["boxes"][:, 2] > image_shape[2]-1, 2] = image_shape[2] - 1
    target["boxes"][target["boxes"][:, 3] > image_shape[1]-1, 3] = image_shape[1] - 1


    target["boxes"][target["boxes"][:, 0] < 0, 0] = 0
    target["boxes"][target["boxes"][:, 1] < 0, 1] = 0

    target["boxes"][target["boxes"][:, 2] < 0, 2] = 0
    target["boxes"][target["boxes"][:, 3] < 0, 3] = 0

    return target
    
def update_boxes_from_masks(target):
    #takes masks then gets bounding boxes for them.

    for mask_index in range(target["masks"].shape[0]):


        non_zero_indices = torch.nonzero(target["masks"][mask_index] > 0.1, as_tuple = True)
        coordY = non_zero_indices[0]
        coordX = non_zero_indices[1]

        if len(coordX) == 0 or len(coordY) == 0:
            print("Couldn't do anything about this mask, skipping")
            continue

        minX = coordX.min()
        maxX = coordX.max()

        minY = coordY.min()
        maxY = coordY.max()

        target["boxes"][mask_index, :] = torch.stack((minX, minY, maxX, maxY))

    return target

###################################################### Misc ####################################################################


def get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def collate_fn(batch):
    return tuple(zip(*batch))


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
