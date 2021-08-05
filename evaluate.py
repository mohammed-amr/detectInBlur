import datetime
import os
import time
import numpy as np

import torch
import torch.utils.data
from torch import nn
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco, get_coco_kp

from engine import evaluate
import engine

import utils
import transforms as T

from pathlib import Path
import shutil
from torch.utils.tensorboard import SummaryWriter
import random
import gc

import models.faster_rcnn
import models.versatile_backbone_models

import models.deblur.deblurInterface as debInt

import custom_datasets

def get_dataset(name, image_set, transform, data_path, sharp = True, expand_synth_boxes = False):
    if "coco" in name:
        paths = {
            "coco": (data_path, get_coco, 91),
            "coco_kp": (data_path, get_coco_kp, 2)
        }
        p, ds_fn, num_classes = paths[name]

        ds = ds_fn(p, image_set=image_set, transforms=transform)

    elif "GOPROSynthLoad" in name:
        if "train" in image_set:
            split = "train"
        else:
            split = "test"
        ds = custom_datasets.GOPROSynthLoad(root_dir = data_path, split = split, sharpImages = sharp, blurredImages = not sharp, transform = transform, expandBoxes = expand_synth_boxes)

        num_classes = 91

    elif "GOPROSynth" in name:
        if "train" in image_set:
            split = "train"
        else:
            split = "test"
        ds = custom_datasets.GOPROSynth(root_dir = data_path, split = split, sharpImages = sharp, blurredImages = not sharp, transform = transform, expandBoxes = expand_synth_boxes)

        num_classes = 91

    elif "GOPRO" in name:
        if "train" in image_set:
            split = "train"
        else:
            split = "test"
        ds = custom_datasets.GOPRO(root_dir = data_path, split = split, sharpImages = sharp, blurredImages = not sharp, transform = transform)

        num_classes = 91

    elif "VidBlur" in name:
        if "train" in image_set:
            split = "train"
        else:
            split = "test"
        ds = custom_datasets.VidBlur(root_dir = data_path, split = split, sharpImages = sharp, blurredImages = not sharp, transform = transform)

        num_classes = 91

    elif "realblur" in name:
        if "train" in image_set:
            split = "train"
        else:
            split = "test"
        ds = custom_datasets.RealBlur(root_dir = data_path, split = split, sharpImages = sharp, blurredImages = not sharp, transform = transform)

        num_classes = 91

    elif "REDS" in name:
        if "train" in image_set:
            split = "train"
        else:
            split = "val"
        ds = custom_datasets.REDS(root_dir = data_path, split = split, sharpImages = sharp, blurredImages = not sharp, transform = transform)
        print("DS has " + str(len(ds)) + " images." )
        num_classes = 91


    return ds, num_classes

def get_transform(train, 
                blur_ratio = 1, 
                blur_exposure=None, 
                blur_type=None, 
                blur = False,  
                use_stored_psfs = False, 
                cpu_blur = False, 
                stored_psf_directory = None, 
                dont_center_psf = False, 
                dilate_psf = False):

    transforms = []
    #transforms.append(T.BlurImage(prob = blur_ratio, curriculum = curriculum, totalNumberofEpochs = totalNumberofEpochs))
    #transforms.append(T.NonUnifromScale(prob = scaleRatio))
    if blur:
        transforms.append(T.BlurImage(prob = blur_ratio, 
                                            blur_type = blur_type, 
                                            blur_exposure = blur_exposure,
                                            use_stored_psfs = use_stored_psfs, 
                                            stored_psf_directory = stored_psf_directory, 
                                            blur_image_in_transform = cpu_blur,
                                            dont_center_psf = dont_center_psf,
                                            dilate_psf = dilate_psf))
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def main(args):
    utils.init_distributed_mode(args)
    print(args)

    if not args.distributed:
        np.random.seed(1337)
        random.seed(1337)
        torch.manual_seed(1337)
    else:
        np.random.seed(torch.distributed.get_rank() * 1337)
        random.seed(torch.distributed.get_rank() * 1337)
        torch.manual_seed(torch.distributed.get_rank() * 1337)

    if not args.distributed or (args.distributed and torch.distributed.get_rank() == 0):
        path_to_use = args.tensorboard_path
        if Path(path_to_use).exists() and Path(path_to_use).is_dir():
            shutil.rmtree(Path(path_to_use))

        if Path(path_to_use).exists() and Path(path_to_use).is_dir():
            print("Opting not to create a writer.")
        else:
            print("Creating a writer!")
            writer = SummaryWriter(path_to_use)
    else:
        writer = None

    device = torch.device(args.device)

    ensemble_models = None
    blur_estimator = None
    if args.use_ensemble:
        model = None

        ensemble_models = []
        for model_path in args.ensemble_model_paths[0].split():

            print("Creating ensemble_model from " + model_path) 
            if "fasterrcnn_resnet50_fpn" in args.model:
                ensemble_model = models.faster_rcnn.fasterrcnn_resnet50_fpn(num_classes=91,
                                                                        pretrained=args.pretrained, warp_internally = args.warp_in_model)
            else:
                print("Unrecognized model type.")
                return

            ensemble_model.to(device)

            ensemble_model_without_ddp = ensemble_model
            if args.distributed:
                ensemble_model = torch.nn.parallel.DistributedDataParallel(ensemble_model, device_ids=[args.gpu])
                ensemble_model_without_ddp = ensemble_model.module

            print("Loading from " + model_path)
            checkpoint = torch.load(model_path, map_location='cpu')
            ensemble_model_without_ddp.load_state_dict(checkpoint['model'])

            ensemble_models.append(ensemble_model)
        
        if args.blur_estimator_path is not None:
            # blur estimator
            blur_estimator = torchvision.models.resnet18(pretrained = True)    
            num_ftrs = blur_estimator.fc.in_features

            if args.LEHE:
                blur_estimator.fc = nn.Linear(num_ftrs, 4)
            else:
                blur_estimator.fc = nn.Linear(num_ftrs, 16)

            blur_estimator.to(device)

            blur_estimator_without_ddp = blur_estimator
            if args.distributed:
                blur_estimator = torch.nn.parallel.DistributedDataParallel(blur_estimator, device_ids=[args.gpu])
                blur_estimator_without_ddp = blur_estimator.module

            print("Loading blur estimator from " + args.blur_estimator_path)
            checkpoint = torch.load(args.blur_estimator_path, map_location='cpu')
            blur_estimator_without_ddp.load_state_dict(checkpoint['model'])


    else:
        
        print("Creating model")
        if "fasterrcnn_resnet50_fpn" in args.model:
            model = models.faster_rcnn.fasterrcnn_resnet50_fpn(num_classes=91,
                                                                    pretrained=args.pretrained, warp_internally = args.warp_in_model)
        elif "mobile_net" in args.model:
            model = models.versatile_backbone_models.create_model(num_classes=91,
                                                                    pretrained=args.pretrained, backbone="mobile_net", warp_internally = args.warp_in_model)
        else:
            print("Unrecognized model type.")
            return

        model.to(device)

        model_without_ddp = model
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module


        if args.resume:
            print("Resuming training from from " + args.resume)
            checkpoint = torch.load(args.resume, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])

        if args.mode_one_norm:
            model_without_ddp = utils.convert_to_custom_batch_norm(model_without_ddp, batch_norm_to_use = models.batchnorm.BatchNorm2d)
            model_without_ddp = utils.set_batch_norm_N(model_without_ddp, 16)
            model_without_ddp = utils.set_batch_norm_mode1(model_without_ddp, True)
        

    if args.deblur_first:
        deblurer = debInt.Deblurer(model_location = args.deblurer_model_location)
    else:
        deblurer = None

    # natively blurred datasets are loaded as vanilla. 
    native_blur_dataset = "coco" not in args.dataset

    print(native_blur_dataset)

    if args.vanilla_eval or native_blur_dataset:
        dataset_test, _ = get_dataset(args.dataset, "val", get_transform(train=False, blur = False), args.data_path, sharp = not args.blurred_dataset, expand_synth_boxes=args.expand_synth_boxes)

        if args.distributed:
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
        else:
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)
            
        data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

        cocoEvaluator = evaluate(model = model, 
                                data_loader = data_loader_test, 
                                device = device,
                                use_ensemble = args.use_ensemble,
                                ensemble_models = ensemble_models,
                                blur_estimator = blur_estimator,
                                LEHE = args.LEHE,
                                vanilla_eval = True,
                                deblur_first = args.deblur_first,
                                deblurer = deblurer,
                                use_custom_image_norm = args.use_custom_image_norm,
                                early_stop = args.early_stop)

        writer.add_scalar("Clean/AccuraciesSweep", cocoEvaluator.coco_eval["bbox"].stats[0], 0)
        writer.add_scalar("Clean/Accuracies", cocoEvaluator.coco_eval["bbox"].stats[1], 0)
        writer.add_scalar("Clean/AccuraciesSmall", cocoEvaluator.coco_eval["bbox"].stats[3], 0)
        writer.add_scalar("Clean/AccuraciesMedium", cocoEvaluator.coco_eval["bbox"].stats[4], 0)
        writer.add_scalar("Clean/AccuraciesLarge", cocoEvaluator.coco_eval["bbox"].stats[5], 0)

        writer.add_scalar("Clean/recallSmall", cocoEvaluator.coco_eval["bbox"].stats[9], 0)
        writer.add_scalar("Clean/recallMedium", cocoEvaluator.coco_eval["bbox"].stats[10], 0)
        writer.add_scalar("Clean/recallLarge", cocoEvaluator.coco_eval["bbox"].stats[11], 0)
        writer.add_scalar("Clean/recall", cocoEvaluator.coco_eval["bbox"].stats[12], 0)

        writer.close()
        time.sleep(2)

        return 


    print("Start eval.")
    start_time = time.time()
    

    gc.collect()

    params = [0.01, 0.005, 0.001, 0.00005]
    fractions = [1/100, 1/25, 1/10, 1/5, 1/2, 1]

    for param_index, param in enumerate(params):
        # param 0 is not used, and is here due to legacy code.
        if param_index in [0]:
            continue
    
        for fraction_index, fraction in enumerate(fractions):
            # fraction 0 is not used, and is here due to legacy code.
            if fraction_index in [0]:
                continue

            gc.collect()
            print("################################## P" + str(param_index) + " and E" + str(fraction_index) + " ###################################")
            # Data loading code
            print("Loading data")

            dataset_test_blurred, _ = get_dataset(args.dataset, 
                                                "val", 
                                                get_transform(train=False, 
                                                            blur_exposure=fraction, 
                                                            blur_type=param, 
                                                            blur_ratio = 1, 
                                                            blur = True,
                                                            use_stored_psfs = args.use_stored_psfs, 
                                                            cpu_blur = args.cpu_blur, 
                                                            stored_psf_directory = args.stored_psf_directory, 
                                                            dont_center_psf = args.dont_center_psf,
                                                            dilate_psf = args.dilate_psf), args.data_path)
            

            if args.distributed:
                test_sampler_blurred = torch.utils.data.distributed.DistributedSampler(dataset_test_blurred)
            else:
                test_sampler_blurred = torch.utils.data.SequentialSampler(dataset_test_blurred)

            data_loader_test_blurred = torch.utils.data.DataLoader(
                dataset_test_blurred, batch_size=1,
                sampler=test_sampler_blurred, num_workers=args.workers,
                collate_fn=utils.collate_fn)

            cocoEvaluator = evaluate(model = model, 
                                data_loader = data_loader_test_blurred, 
                                device = device,
                                blurring_images = True, 
                                gpu_blur = args.gpu_blur, 
                                deblurer = deblurer,
                                deblur_first = args.deblur_first,
                                expand_target_boxes = args.expand_target_boxes,
                                use_custom_image_norm = args.use_custom_image_norm,
                                use_ensemble = args.use_ensemble,
                                ensemble_models = ensemble_models,
                                blur_estimator = blur_estimator,
                                LEHE = args.LEHE,
                                add_noise = args.add_noise,
                                noise_level=args.noise_level,
                                add_block = args.add_block,
                                image_output_folder=args.image_output_dir,
                                early_stop = args.early_stop)

            if not args.distributed or (torch.distributed.get_rank() == 0):
                writer.add_scalar("P" + str(param_index) + "/AccuraciesSweep", cocoEvaluator.coco_eval["bbox"].stats[0], fraction_index)
                writer.add_scalar("P" + str(param_index) + "/Accuracies", cocoEvaluator.coco_eval["bbox"].stats[1], fraction_index)
                writer.add_scalar("P" + str(param_index) + "/AccuraciesSmall", cocoEvaluator.coco_eval["bbox"].stats[3], fraction_index)
                writer.add_scalar("P" + str(param_index) + "/AccuraciesMedium", cocoEvaluator.coco_eval["bbox"].stats[4], fraction_index)
                writer.add_scalar("P" + str(param_index) + "/AccuraciesLarge", cocoEvaluator.coco_eval["bbox"].stats[5], fraction_index)

                writer.add_scalar("P" + str(param_index) + "/recallSmall", cocoEvaluator.coco_eval["bbox"].stats[9], fraction_index)
                writer.add_scalar("P" + str(param_index) + "/recallMedium", cocoEvaluator.coco_eval["bbox"].stats[10], fraction_index)
                writer.add_scalar("P" + str(param_index) + "/recallLarge", cocoEvaluator.coco_eval["bbox"].stats[11], fraction_index)
                writer.add_scalar("P" + str(param_index) + "/recall", cocoEvaluator.coco_eval["bbox"].stats[12], fraction_index)
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    time.sleep(5)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    # data
    parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--data_path', default='/media/mosayed/data_f_256/datasets/COCO/coco/', help='dataset')

    parser.add_argument("--use_stored_psfs", dest="use_stored_psfs", help="Use stored PSFs for when blurring in the train data_loader.", action="store_true")
    parser.add_argument('--stored_psf_directory', default='/media/mosayed/data_f_256/datasets/COCO/coco/psfs', help='Stored PSFs path.') 

    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 0)')

    # real blur datasets
    parser.add_argument("--blurred_dataset", dest="blurred_dataset", help="When using real blurred datasets, use this flag to specify the blurred set.", action="store_true")
    parser.add_argument("--expand_synth_boxes", dest="expand_synth_boxes", help="When using the GOPRO expand set, specify that labels should be expanded.", action="store_true")


    # model 
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--trainable_backbone_blocks', default=3, type=int, help='Resnet backbone blocks to train.')
    parser.add_argument("--pretrained", dest="pretrained", help="Use pre-trained models from the modelzoo", action="store_true")

    parser.add_argument('--resume', help='resume from checkpoint')

    parser.add_argument("--use_ensemble", dest="use_ensemble", help="Use blur network system ensemble.", action="store_true")
    parser.add_argument('--ensemble_model_paths',  nargs='+', help='Ensemble model paths.')

    parser.add_argument('--blur_estimator_path', help='Blur estimator model path.')

    # Eval
    parser.add_argument("--vanilla_eval", dest="vanilla_eval", help="Vanilla eval on clean COCO images.", action="store_true")
    parser.add_argument('--early_stop', type = int, help='early stop for eval')
    parser.add_argument('--device', default='cuda', help='device')

    # logging
    parser.add_argument('--tensorboard_path', default="debug", help='device')
    parser.add_argument('--output_dir', default='debug', help='Output directory for weights.')
    parser.add_argument('--image_output_dir', default='debug', help='Output directory for images.')


    # blur
    parser.add_argument("--blur_eval", dest="blur_eval", help="Blur during evaluation.", action="store_true")
    parser.add_argument("--cpu_blur", dest="cpu_blur", help="CPU blurring in fourier domain, happens on the CPU in the dataloader's worker processes.", action="store_true")
    parser.add_argument("--gpu_blur", dest="gpu_blur", help="GPU blurring, happens on the GPU in the GPU thread.", action="store_true")

    parser.add_argument('--param_index', default=None, help='Type of blur to use use for training blur. Options are 1, 2, and 3.')

    parser.add_argument("--high_exposure", dest="high_exposure", help="Train and evaluate with high exposure blur.", action="store_true")
    parser.add_argument("--low_exposure", dest="low_exposure", help="Train and evaluate with low exposure blur.", action="store_true")

    parser.add_argument("--LEHE", dest="LEHE", help="System with low and high exposure networks.", action="store_true")

    parser.add_argument("--expand_target_boxes", dest="expand_target_boxes", help="Expand target boxes during training and evaluating according to blur kernel shifts.", action="store_true")
    parser.add_argument("--dont_center_psf", dest="dont_center_psf", help="Don't center PSFs. You will need to generate psf on the fly for this. Cannot use stored psfs.", action="store_true")

    parser.add_argument("--add_noise", dest="add_noise", help="Add noise after blurring.", action="store_true")
    parser.add_argument('--noise_level', default=0.001, type=float, help='Noise level.')
    parser.add_argument("--add_block", dest="add_block", help="Add block artifacts after blurring.", action="store_true")
    parser.add_argument("--add_jpeg_artefacts", dest="add_jpeg_artefacts", help="Add jpeg compression artifacts.", action="store_true")

    parser.add_argument("--dilate_psf", dest="dilate_psf", help="Dialate PSF to simulate defocus with motion blur.", action="store_true")


    # Squint, blur needs to be enabled for this
    parser.add_argument("--warp_in_model", dest="warp_in_model", help="Warp and dewarp images before and after backbone.", action="store_true")

    # deblur model
    parser.add_argument("--deblur_first", dest="deblur_first", help="deblur_first.", action="store_true")
    parser.add_argument('--deblurer_model_location', default='/home/mosayed/code/DeepDeblur-PyTorch/experiment/GOPRO_L1', help='deblurer_model_location')

    # Augmix
    parser.add_argument("--non_pos_aug_mix", dest="non_pos_aug_mix", help="Non positional augmix.", action="store_true")
    parser.add_argument("--include_pos_aug_mix", dest="include_pos_aug_mix", help="Include positional augmentations in augmix.", action="store_true")
    parser.add_argument("--aug_mix_target_expand", dest="aug_mix_target_expand", help="Expand target boxes for AugMix according to positional shifts from spatial augmentations..", action="store_true")

    # batch norm
    parser.add_argument("--use_custom_image_norm", dest="use_custom_image_norm", help="Use blur specific normalization on input to network.", action="store_true")
    parser.add_argument("--unfrozen_batch_norm", dest="unfrozen_batch_norm", help="Use unfrozen batch normalization.", action="store_true")


    # batch norm robustness
    parser.add_argument("--mode_one_norm", dest="mode_one_norm", help="Batch normalization remedy by using test time statistics.", action="store_true")


    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)
