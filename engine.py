import math
import sys
import time
import torch
import os
import numpy as np
import cv2
import PIL

import colorsys
import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils


from torchvision import transforms as torch_transforms
import transforms as transforms

import models.warper 
import models.blur_functions 
import models.batchnorm
import models.net_transforms
import models.jpeg.DiffJPEG

from PIL import Image, ImageDraw
        

def train_one_epoch(model, 
                    optimizer,
                    data_loader, 
                    device, 
                    epoch = 0, 
                    print_freq = 200, 
                    writer = None, 
                    distributed_mode = False,
                    blur_train = False,
                    early_stop = False,
                    gpu_blur = False, 
                    expand_target_boxes = False, 
                    use_custom_image_norm = False, 
                    add_noise = False, 
                    noise_level = 0.001,
                    add_block = False,
                    add_jpeg_artifact = False):

    if writer is None:
        print("Warning! No tensorboard logger.")

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



    iteration_count = 0
    for images_CPU, targets, blur_dicts in metric_logger.log_every(data_loader, print_freq, header):
        thetas_GPU = None
        lambdas1_GPU = None
        lambdas2_GPU = None
        
        if distributed_mode:
            images_GPU = list(image.half().to(device) for image in images_CPU)
            targets_GPU = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            if blur_train:
                psfs_GPU = [torch.HalfTensor(blur_dict["psf"]).to(device) for blur_dict in blur_dicts]
                thetas_GPU = [torch.HalfTensor([blur_dict["theta_rad"]]).to(device) for blur_dict in blur_dicts]
                lambdas1_GPU = [torch.HalfTensor([blur_dict["scale_factor_lambda1"]]).to(device) for blur_dict in blur_dicts]
                lambdas2_GPU = [torch.HalfTensor([blur_dict["scale_factor_lambda2"]]).to(device) for blur_dict in blur_dicts]
                images_Info = list(torch.HalfTensor([image.shape])[0] for image in images_CPU)
        else:
            images_GPU = list(image.half().cuda() for image in images_CPU)
            targets_GPU = [{k: v.cuda() for k, v in t.items()} for t in targets]

            if blur_train:
                psfs_GPU = [torch.HalfTensor(blur_dict["psf"]).cuda() for blur_dict in blur_dicts]
                thetas_GPU = [torch.HalfTensor([blur_dict["theta_rad"]]).cuda() for blur_dict in blur_dicts]
                lambdas1_GPU = [torch.HalfTensor([blur_dict["scale_factor_lambda1"]]).cuda() for blur_dict in blur_dicts]
                lambdas2_GPU = [torch.HalfTensor([blur_dict["scale_factor_lambda2"]]).cuda() for blur_dict in blur_dicts]
                images_Info = list(torch.HalfTensor([image.shape])[0] for image in images_CPU)
        

        if gpu_blur and blur_train:
            models.blur_functions.blur_image_list(images_GPU, blur_dicts, psfs_GPU = psfs_GPU, add_noise = add_noise, noise_level = noise_level, add_block=add_block, add_jpeg_artifact=add_jpeg_artifact, jpeg_compressor = jpeg_compressor)
        
        if expand_target_boxes:
            targets_GPU = utils.expand_targets(targets_GPU, blur_dicts, psfs_GPU, images_GPU)

        if distributed_mode:
            images_GPU = list(image.float().to(device) for image in images_GPU)
        else:
            images_GPU = list(image.float().cuda() for image in images_GPU)

        
        norm_means, norm_stds = utils.get_norm_params(blur_dicts, use_custom_image_norm)

        if gpu_blur and thetas_GPU is not None:
            thetas_GPU = torch.stack(thetas_GPU).squeeze()
            lambdas1_GPU = torch.stack(lambdas1_GPU).squeeze()
            lambdas2_GPU = torch.stack(lambdas2_GPU).squeeze()

        loss_dict = model(images_GPU, targets_GPU, thetas = thetas_GPU, lambda1s = lambdas1_GPU, lambda2s = lambdas2_GPU, newMeans = norm_means, newSTDs = norm_stds)

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

        if iteration_count % 500 == 0 and writer is not None:
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
        
        if early_stop is not None:
            if iteration_count > early_stop:
                break

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        iteration_count += 1
    


def get_network_index_to_use_oracle(blur_dicts, model_indices):
    # incoming params are one less than 1 based.
    # P1 = 0, P2 = 1, so on

    for blur_dict_index, blur_dict in enumerate(blur_dicts):
        
        if blur_dict["blurring"] and blur_dicts[blur_dict_index]["param_index"] is not None:
            param_index = blur_dicts[blur_dict_index]["param_index"]
            fraction_index = blur_dicts[blur_dict_index]["fraction_index"]

            if fraction_index == -1:
                return model_indices[0]
            if param_index == 2:
                return model_indices[3]

            if param_index == 1:
                return model_indices[2]

            if param_index == 0:
                return model_indices[1]
        else:
            return model_indices[0]

def get_network_index_to_use_blur_estimator(blur_estimation, model_indices):

    maxIndex = blur_estimation.argmax()

    if maxIndex in [1,2,3,4,5]:
        return model_indices[1]
    elif maxIndex in [6,7,8,9,10]:
        return model_indices[2]
    elif maxIndex in [11,12,13,14,15]:
        return model_indices[3]
    else:
        return model_indices[0]

def get_network_index_to_use_blur_estimator_LEHE(blur_estimation, model_indices):

    maxIndex = blur_estimation.argmax()

    if maxIndex in [1]:
        return model_indices[1]
    elif maxIndex in [2]:
        return model_indices[2]
    elif maxIndex in [3]:
        return model_indices[3]
    else:
        return model_indices[0]

@torch.no_grad()
def evaluate(model, 
            data_loader, 
            device,
            distributed_mode = False,
            early_stop = None,
            vanilla_eval = False,
            blurring_images = False, 
            gpu_blur = False, 
            expand_target_boxes = False,
            deblur_first = False,
            deblurer = None, 
            use_custom_image_norm = False, 
            use_ensemble = False, 
            ensemble_models = None, 
            blur_estimator = None,
            add_noise = False,
            noise_level = 0.001,
            add_block = False,
            add_jpeg_artifact = False,
            image_output_folder = None,
            LEHE = False):
            
    n_threads = torch.get_num_threads()

    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    
    if add_jpeg_artifact:
        if distributed_mode:
            jpeg_compressor = models.jpeg.DiffJPEG.DiffJPEG(height=100, width=100, differentiable=False, quality = 10).to(device)
        else:
            jpeg_compressor = models.jpeg.DiffJPEG.DiffJPEG(height=100, width=100, differentiable=False, quality = 10).cuda()
    else:
        jpeg_compressor = None

    if use_ensemble:
        transform_and_batcher = None
        for ensemble_model in ensemble_models:
            ensemble_model.eval()

        if blur_estimator is not None:
            image_mean = [0.485, 0.456, 0.406]
            image_std = [0.229, 0.224, 0.225]
            transform_and_batcher = models.net_transforms.GeneralizedRCNNTransform(800, 1333, image_mean, image_std, crop_images = True)
            blur_estimator.eval()
    else:
        model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = utils.get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    count = 0
    faultyBoxes = 0
    totalNumberOfBoxes = 0
    for images_CPU, targetsCPU, blur_dicts in metric_logger.log_every(data_loader, 100, header):

        torch.cuda.synchronize()
        model_time = time.time()

        if distributed_mode:
            images_GPU = list(image.half().to(device) for image in images_CPU)
            targets_GPU = [{k: v.to(device) for k, v in t.items()} for t in targetsCPU]
            
            if blurring_images:
                psfs_GPU = [torch.HalfTensor(blur_dict["psf"]).to(device) for blur_dict in blur_dicts]
                thetas_GPU = [torch.HalfTensor([blur_dict["theta_rad"]]).to(device) for blur_dict in blur_dicts]
                lambdas1_GPU = [torch.HalfTensor([blur_dict["scale_factor_lambda1"]]).to(device) for blur_dict in blur_dicts]
                lambdas2_GPU = [torch.HalfTensor([blur_dict["scale_factor_lambda2"]]).to(device) for blur_dict in blur_dicts]
                images_Info = list(torch.HalfTensor([image.shape])[0] for image in images_CPU)
        else:
            images_GPU = list(image.half().cuda() for image in images_CPU)
            targets_GPU = [{k: v.cuda() for k, v in t.items()} for t in targetsCPU]

            if blurring_images:
                psfs_GPU = [torch.HalfTensor(blur_dict["psf"]).cuda() for blur_dict in blur_dicts]
                thetas_GPU = [torch.HalfTensor([blur_dict["theta_rad"]]).cuda() for blur_dict in blur_dicts]
                lambdas1_GPU = [torch.HalfTensor([blur_dict["scale_factor_lambda1"]]).cuda() for blur_dict in blur_dicts]
                lambdas2_GPU = [torch.HalfTensor([blur_dict["scale_factor_lambda2"]]).cuda() for blur_dict in blur_dicts]
                images_Info = list(torch.HalfTensor([image.shape])[0] for image in images_CPU)


        if gpu_blur:
            models.blur_functions.blur_image_list(images_GPU, 
                            blur_dicts, 
                            psfs_GPU = psfs_GPU, 
                            add_noise = add_noise, 
                            noise_level = noise_level, 
                            add_block=add_block, 
                            add_jpeg_artifact=add_jpeg_artifact, 
                            jpeg_compressor = jpeg_compressor)

        if expand_target_boxes:
            targets_GPU = utils.expand_targets(targets_GPU, blur_dicts, psfs_GPU, images_GPU)

        if (deblur_first and blurring_images) or (vanilla_eval and deblur_first):
            imageForDeblur = (images_GPU[0]*255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            deblurredImage = deblurer.deblurImage(imageForDeblur).add_(0.5).clamp_(0, 255)/255
            images_GPU[0] = deblurredImage.cuda().squeeze()


        if expand_target_boxes:
            for target_index, target in enumerate(targets_GPU):
                boxes = utils.convert_to_xywh(target["boxes"])
                imageID = target["image_id"].item()
                cocoAnns = coco_evaluator.coco_gt.imgToAnns[imageID]
                for annIndex, cocoAnn in enumerate(cocoAnns):
                    totalNumberOfBoxes += 1
                    try:
                        cocoAnn["bbox"] = boxes[annIndex, :].cpu().numpy().tolist()
                        # cocoAnn["area"] = 1
                        # target["area"] = 1
                    except:
                        if len(cocoAnns) == annIndex+1:
                            faultyBoxes+=1
                        else: 
                            diff = len(cocoAnns) - annIndex
                            
                            print("Faulty " + str(diff) + " times over.")


        if distributed_mode:
            images_GPU = list(image.float().to(device) for image in images_GPU)
        else:
            images_GPU = list(image.float().cuda() for image in images_GPU)


        norm_means, norm_stds = utils.get_norm_params(blur_dicts, use_custom_image_norm)

        # ensemble 
        if use_ensemble:
            if blur_estimator is None:
                modelIndex = get_network_index_to_use_oracle(blur_dicts, model_indices = list(range(len(ensemble_models))))
                model = ensemble_models[modelIndex]
            else:
                images_batched, _ = transform_and_batcher(images_GPU, None)
                blur_estimation = blur_estimator(images_batched.tensors)
                if LEHE:
                    modelIndex = get_network_index_to_use_blur_estimator_LEHE(blur_estimation, list(range(len(ensemble_models))))
                else:
                    modelIndex = get_network_index_to_use_blur_estimator(blur_estimation, model_indices = list(range(len(ensemble_models))))

                model = ensemble_models[modelIndex]

        if blurring_images:
            if thetas_GPU is not None:
                thetas_GPU = torch.reshape(thetas_GPU[0], (1,))
                lambdas1_GPU = torch.reshape(lambdas1_GPU[0], (1,))
                lambdas2_GPU = torch.reshape(lambdas2_GPU[0], (1,))
            
            outputs = model(images_GPU, thetas = thetas_GPU, lambda1s = lambdas1_GPU, lambda2s = lambdas2_GPU, newMeans = norm_means, newSTDs = norm_stds)
        else:
            outputs = model(images_GPU, killWarp = True, newMeans = norm_means, newSTDs = norm_stds)


        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        

        if image_output_folder is not None:
            Image.fromarray(cv2.cvtColor(utils.overlay_boxes_torch(images_GPU[0], outputs[0]), cv2.COLOR_BGR2RGB)).save(image_output_folder + "/img" + str(count) + ".png")

        model_time = time.time() - model_time


        res = {target["image_id"].item(): output for target, output in zip(targets_GPU, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
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
    print("Averaged stats:", metric_logger)
    print("Number of Faulty boxes: " + str(faultyBoxes) + " Total number of boxes: " + str(totalNumberOfBoxes))
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


