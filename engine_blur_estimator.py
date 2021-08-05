import math
from pickle import FALSE
import sys
import time
import torch
import os
import numpy as np
import cv2

import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils


from torchvision import transforms
import transforms as ourTransforms

import models.warper 
import models.net_transforms
import models.jpeg.DiffJPEG




def manual_blur(image_GPU, psf_GPU, resize_images = False):
    image_GPU = image_GPU.unsqueeze(0)
    image_width = image_GPU.shape[3]
    image_height = image_GPU.shape[2]

    if resize_images:
        if image_height > image_width:
            # transpose, old width and height swapped
            image_GPU = image_GPU.permute(0,1,3,2)
            new_height = 800
            new_width = int(new_height * image_height/image_width)
        else:
            new_height = 800
            new_width = int(new_height * image_width/image_height)

        image_GPU = torch.nn.functional.interpolate(image_GPU, size = (new_height, new_width), mode = "bilinear")
  

    width = 128
    height = 128
    p1d = (math.floor(width/2)-1, math.ceil(width/2), math.floor(height/2)-1, math.ceil(height/2))
    if image_GPU.shape[2] < 64 or image_GPU.shape[3] < 64:
        pad_mode = "constant"
    else:
        pad_mode = 'reflect'
    image_GPU = torch.nn.functional.pad(image_GPU, p1d , mode=pad_mode)

    output = torch.zeros_like(image_GPU)

    non_zero_points = psf_GPU.nonzero(as_tuple=False)

    for coord_index in range(non_zero_points.shape[0]):
        output += torch.roll(image_GPU, shifts = (non_zero_points[coord_index, 0]-63, non_zero_points[coord_index, 1]-63), dims = (2,3)) * psf_GPU[non_zero_points[coord_index, 0], non_zero_points[coord_index, 1]]

    output = output[:, :, 63: 63 + image_height, 63: 63 + image_width]

    if resize_images:
        if image_height > image_width:
            # transpose, old width and height swapped
            image_GPU = image_GPU.permute(0,1,3,2)

        output = torch.nn.functional.interpolate(output, size = (image_height, image_width), mode = "bilinear")
        
    return output.squeeze()

def blur_image_list(images_GPU, blur_dicts, psfs_GPU, resize_images = False):
    for image_index, (image_GPU, blur_dict, psf_GPU) in enumerate(zip(images_GPU, blur_dicts, psfs_GPU)):
        if not blur_dict["blurring"]:
            continue

        psf_GPU = psf_GPU/psf_GPU.sum()
        
        images_GPU[image_index] = manual_blur(image_GPU, psf_GPU, resize_images)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_target_from_blur_dict(blur_dicts, target):
    for index, blur_dict in enumerate(blur_dicts):
        if blur_dict["blurring"]:
            target[index] = torch.Tensor([(blur_dict["param_index"])*5 + blur_dict["fraction_index"] + 1]).long()
        else:
            target[index] = torch.Tensor([0]).long()

    return target

def get_target_from_blur_dict_LEHE(blur_dicts, target):

    
    for index, blur_dict in enumerate(blur_dicts):
        if "blur_est_label" in blur_dict:
            target[index] = torch.Tensor([blur_dict["blur_est_label"]]).long()
        else:
            if blur_dict["blurring"]:
                if blur_dict["fraction_index"] < 3:
                    target[index] = torch.Tensor([0]).long()
                    continue

                elif blur_dict["param_index"] == 0:
                    target[index] = torch.Tensor([1]).long()
                elif blur_dict["param_index"] == 1:
                    target[index] = torch.Tensor([2]).long()
                elif blur_dict["param_index"] == 2:
                    target[index] = torch.Tensor([3]).long()

            else:
                target[index] = torch.Tensor([0]).long()

    return target


def train_one_epoch(model, 
                    optimizer, 
                    criterion, 
                    data_loader, 
                    device, 
                    print_freq = 500, 
                    epoch = 0, 
                    distributed_mode = False, 
                    writer = None, 
                    gpu_blur = False,
                    LEHE_blur_seg = False, 
                    resize_images = False,
                    quantize_image = False,
                    crop_images = False, 
                    add_noise = False, 
                    noise_level = 0.001,
                    add_block = False,
                    add_jpeg_artifact = False,
                    early_stop = None,
                    blur_train = False):

    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]

    transform_and_batcher = models.net_transforms.GeneralizedRCNNTransform(800, 1333, image_mean, image_std, crop_images = crop_images)


    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if add_jpeg_artifact:
        if distributed_mode:
            jpeg_compressor = models.jpeg.DiffJPEG.DiffJPEG(height=100, width=100, differentiable=False, quality = 10).to(device)
        else:
            jpeg_compressor = models.jpeg.DiffJPEG.DiffJPEG(height=100, width=100, differentiable=False, quality = 10).cuda()
    else:
        jpeg_compressor = None

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    warper = None

    iteration_count = 0
    for images_CPU, targets, blur_dicts in metric_logger.log_every(data_loader, print_freq, header):

        if distributed_mode:
            images_GPU = list(image.half().to(device) for image in images_CPU)
            targets_GPU = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            if blur_train:
                psfs_GPU = [torch.HalfTensor(blur_dict["psf"]).to(device) for blur_dict in blur_dicts]
    
        else:
            images_GPU = list(image.half().cuda() for image in images_CPU)
            targets_GPU = [{k: v.cuda() for k, v in t.items()} for t in targets]

            if blur_train:
                psfs_GPU = [torch.HalfTensor(blur_dict["psf"]).cuda() for blur_dict in blur_dicts]
        
        if gpu_blur and blur_train:
            blur_image_list(images_GPU, blur_dicts, psfs_GPU, resize_images)
        
        for image_GPU_index, image_GPU in enumerate(images_GPU):
            if add_noise:
                noise_var = np.random.uniform(0.0001, noise_level)
                images_GPU[image_GPU_index] = torch.clamp(image_GPU + (torch.randn_like(image_GPU) * math.sqrt(noise_var)), 0, 1)

            if add_block:
                if np.random.uniform(0,1) > 0.3:
                    original_shape = image_GPU.shape
                    scale_factor = np.random.uniform(0.6, 1)
                    images_GPU[image_GPU_index] = torch.nn.functional.interpolate(images_GPU[image_GPU_index].unsqueeze(axis = 0), scale_factor = (scale_factor, scale_factor), mode='nearest').squeeze()
                    images_GPU[image_GPU_index] = torch.nn.functional.interpolate(images_GPU[image_GPU_index].unsqueeze(axis = 0), size = original_shape[1:], mode='nearest').squeeze()

            if add_jpeg_artifact:
                if np.random.uniform(0,1) > 0.35:
                    quality = np.random.uniform(20,90) 
                    images_GPU[image_GPU_index] = ourTransforms.add_jpeg_artifact_to_image(images_GPU[image_GPU_index], jpeg_compressor, quality)

            if quantize_image:
                images_GPU[image_GPU_index] = (images_GPU[image_GPU_index] * 255).type(torch.uint8).type(torch.half)/255

 


        if distributed_mode: 
            images_GPU = list(image.float().to(device) for image in images_GPU)
        else:
            images_GPU = list(image.float().cuda() for image in images_GPU)


        images_batched = transform_and_batcher(images_GPU, targets_GPU)


        if not LEHE_blur_seg:
            if distributed_mode:
                target = torch.zeros(len(blur_dicts), requires_grad = False).uniform_(0, 15).long().to(device)
            else:
                target = torch.zeros(len(blur_dicts), requires_grad = False).uniform_(0, 15).long().cuda()

            target = get_target_from_blur_dict(blur_dicts, target)
        else:
            if distributed_mode:
                target = torch.zeros(len(blur_dicts), requires_grad = False).uniform_(0, 3).long().to(device)
            else:
                target = torch.zeros(len(blur_dicts), requires_grad = False).uniform_(0, 3).long().cuda()
            target = get_target_from_blur_dict_LEHE(blur_dicts, target)

        output = model(images_batched[0].tensors)

        torch.cuda.synchronize()

        loss_dict = {}
        loss_dict["loss"] = criterion(output, target)

        torch.cuda.synchronize()

        del images_CPU
        del images_GPU
        del targets
        del targets_GPU
        del blur_dicts

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        
        if (not distributed_mode or (distributed_mode and torch.distributed.get_rank() == 0)) and iteration_count % print_freq == 0:
            for key, item_loss_value in loss_dict_reduced.items():
                writer.add_scalar('losses/' + key, item_loss_value, iteration_count + (epoch*len(data_loader)))

            writer.add_scalar('losses/overallLoss', loss_value, iteration_count + (epoch*len(data_loader)))
            writer.add_scalar('learningRate', optimizer.param_groups[0]["lr"], iteration_count + (epoch*len(data_loader)))
        

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        
        optimizer.zero_grad()

        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        iteration_count += 1

        if early_stop is not None:
            if iteration_count > early_stop:
                break


@torch.no_grad()
def evaluate(model, 
            data_loader, 
            device, 
            distributed_mode = False,
            blurring_images = False, 
            gpu_blur = False, 
            LEHE_blur_seg = False, 
            send_back_preds_targets = False, 
            add_jpeg_artifact = False,
            resize_images = False,
            quantize_image = False, 
            add_noise = False, 
            noise_level = 0.001,
            add_block = False,
            early_stop = None):

    n_threads = torch.get_num_threads()
    
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    transform_and_batcher = models.net_transforms.GeneralizedRCNNTransform(800, 1333, image_mean, image_std)

    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    if add_jpeg_artifact:
        if distributed_mode:
            jpeg_compressor = models.jpeg.DiffJPEG.DiffJPEG(height=100, width=100, differentiable=False, quality = 10).to(device)
        else:
            jpeg_compressor = models.jpeg.DiffJPEG.DiffJPEG(height=100, width=100, differentiable=False, quality = 10).cuda()
    else:
        jpeg_compressor = None

    count = 0

    total = 0
    correct = 0
    correctCounts = [0,0]
    targetsAll = []
    predsAll = []

    for images_CPU, targetsCPU, blur_dicts in metric_logger.log_every(data_loader, 100, header):

        torch.cuda.synchronize()
        model_time = time.time()

        if distributed_mode:
            images_GPU = list(image.half().to(device) for image in images_CPU)
            targets_GPU = [{k: v.to(device) for k, v in t.items()} for t in targetsCPU]
            
            if blurring_images:
                psfs_GPU = [torch.HalfTensor(blur_dict["psf"]).to(device) for blur_dict in blur_dicts]
        else:
            images_GPU = list(image.half().cuda() for image in images_CPU)
            targets_GPU = [{k: v.cuda() for k, v in t.items()} for t in targetsCPU]

            if blurring_images:
                psfs_GPU = [torch.HalfTensor(blur_dict["psf"]).cuda() for blur_dict in blur_dicts]


        if gpu_blur:
            blur_image_list(images_GPU, blur_dicts, psfs_GPU, resize_images)


        for image_GPU_index, image_GPU in enumerate(images_GPU):
            if add_noise:
                noise_var = np.random.uniform(0.0001, noise_level)
                images_GPU[image_GPU_index] = torch.clamp(image_GPU + (torch.randn_like(image_GPU) * math.sqrt(noise_var)), 0, 1)
            if add_block:
                if np.random.uniform(0,1) > 0.3:
                    original_shape = image_GPU.shape
                    scale_factor = np.random.uniform(0.6, 1)
                    images_GPU[image_GPU_index] = torch.nn.functional.interpolate(images_GPU[image_GPU_index].unsqueeze(axis = 0), scale_factor = (scale_factor, scale_factor), mode='nearest').squeeze()
                    images_GPU[image_GPU_index] = torch.nn.functional.interpolate(images_GPU[image_GPU_index].unsqueeze(axis = 0), size = original_shape[1:], mode='nearest').squeeze()
            
            if add_jpeg_artifact:
                if np.random.uniform(0,1) > 0.35:
                    quality = np.random.uniform(20,90) 
                    images_GPU[image_GPU_index] = transforms.add_jpeg_artifact_to_image(images_GPU[image_GPU_index], jpeg_compressor, quality)

            if quantize_image:
                images_GPU[image_GPU_index] = (images_GPU[image_GPU_index] * 255).type(torch.uint8).type(torch.half)/255



        if distributed_mode:
            images_GPU = list(image.float().to(device) for image in images_GPU)
        else:
            images_GPU = list(image.float().cuda() for image in images_GPU)


        images_GPU = transform_and_batcher(images_GPU)

        outputs = model(images_GPU[0].tensors)

        model_time = time.time() - model_time

        evaluator_time = time.time()
        
        with torch.no_grad():
            topk = (1,2)

            if not LEHE_blur_seg:
                if distributed_mode:
                    target = torch.zeros(len(blur_dicts), requires_grad = False).uniform_(0, 15).long().to(device)
                else:
                    target = torch.zeros(len(blur_dicts), requires_grad = False).uniform_(0, 15).long().cuda()

                target = get_target_from_blur_dict(blur_dicts, target)
            else:
                if distributed_mode:
                    target = torch.zeros(len(blur_dicts), requires_grad = False).uniform_(0, 3).long().to(device)
                else:
                    target = torch.zeros(len(blur_dicts), requires_grad = False).uniform_(0, 3).long().cuda()

                target = get_target_from_blur_dict_LEHE(blur_dicts, target)


            total += target.size(0)
        
            maxk = max(topk)

            _, pred = outputs.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            
            targetsAll.append(target[0])
            predsAll.append(pred[0])
            
            for kIndex, k in enumerate(topk):
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                correctCounts[kIndex] += correct_k

        evaluator_time = time.time() - evaluator_time

        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        count += 1

        del images_CPU
        del images_GPU
        del targetsCPU
        del targets_GPU
        del blur_dicts

        if early_stop is not None:
            if count > early_stop:
                break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    # accumulate predictions from all images
    accuracies = [100*(correctCounts[0].item()/total), 100*(correctCounts[1].item()/total)]
    print( 'Top 1 Accuracy: {0:.2f}%'.format(accuracies[0]) )
    print( 'Top 2 Accuracy: {0:.2f}%'.format(accuracies[1]) )

    mergedPreds = torch.stack(predsAll).squeeze()
    mergedTargets = torch.stack(targetsAll).squeeze()

    totalAcc = 0
    classAccs = []
    valid_class_count = 0
    for classInd in range(4):
        class_count = int((mergedTargets == classInd).sum())

        if class_count == 0:
            continue

        valid_class_count += 1

        classAcc = int(torch.logical_and(mergedTargets == classInd, mergedPreds == mergedTargets).sum())/int((mergedTargets == classInd).sum())

        classAccs.append(classAcc)

        totalAcc += classAcc

    totalAcc = totalAcc/valid_class_count

    print( 'Top 1 Mean Acc: {0:.2f}%'.format(totalAcc*100) )

    torch.set_num_threads(n_threads)

    if send_back_preds_targets: 
        return accuracies, targetsAll, predsAll
    else:
        return accuracies
