r"""Borrowed from the original sample on PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

"""
import datetime
import os
import time
from datetime import datetime
import torch
import torch.utils.data


import numpy as np
 
from coco_utils import get_coco, get_coco_kp

from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from engine import train_one_epoch, evaluate

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

def get_dataset(name, image_set, transform, data_path):
    paths = {
        "coco": (data_path, get_coco, 91),
        "coco_kp": (data_path, get_coco_kp, 2)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)

    return ds, num_classes

def get_transform(train, 
                blur = False, 
                cpu_blur = False,
                blur_ratio = 1,
                blur_exposure=None, 
                blur_type=None, 
                high_exposure = False,
                low_exposure = False,
                dont_center_psf = False,
                use_stored_psfs = False, 
                stored_psf_directory = None,
                non_pos_aug_mix = False, 
                include_pos_aug_mix = False, 
                aug_mix_target_expand = False):

    transforms = []

    if non_pos_aug_mix:
        transforms.append(T.AugMix(include_pos_aug_mix = include_pos_aug_mix, modify_target_boxes=aug_mix_target_expand))

    if blur:
        transforms.append(T.BlurImage(prob = blur_ratio, 
                                        blur_type = blur_type, 
                                        blur_exposure = blur_exposure,
                                        use_stored_psfs = use_stored_psfs, 
                                        stored_psf_directory = stored_psf_directory, 
                                        blur_image_in_transform = cpu_blur,
                                        dont_center_psf = dont_center_psf,
                                        high_exposure = high_exposure,
                                        low_exposure = low_exposure))
    if blur_type is None and blur:
        print("Blur type: " + "all types.")
    else:
        print("Blur type: " + str(blur_type))

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
        torch.cuda.manual_seed_all(1337)
        torch.cuda.manual_seed(1337)
    else:
        np.random.seed(torch.distributed.get_rank() * 1337)
        random.seed(torch.distributed.get_rank() * 1337)
        torch.manual_seed(torch.distributed.get_rank() * 1337)
        torch.cuda.manual_seed_all(1337)
        torch.cuda.manual_seed(1337)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    # Data loading
    print("Creating datasets...")

    if args.use_stored_psfs:
        if args.param_index is None:
            train_blur_type = None
        else:
            train_blur_type = int(args.param_index)
    else:
        params = [0.01, 0.005, 0.001, 0.00005]
        if args.param_index is None:
            train_blur_type = None
        else:
            train_blur_type = params[int(args.param_index)]

    if args.low_exposure:
        train_blur_ratio = 0.75
    elif args.high_exposure:
        train_blur_ratio = 1
    else:
        train_blur_ratio = 0.9

    # blurred and clean train dataset
    dataset, _ = get_dataset(args.dataset, "train", get_transform(train=True, 
                                                                    blur_type = train_blur_type, 
                                                                    blur_ratio = train_blur_ratio, 
                                                                    blur = args.blur_train,
                                                                    use_stored_psfs = args.use_stored_psfs, 
                                                                    cpu_blur = args.cpu_blur, 
                                                                    stored_psf_directory = args.stored_psf_directory,
                                                                    non_pos_aug_mix = args.non_pos_aug_mix, 
                                                                    include_pos_aug_mix = args.include_pos_aug_mix, 
                                                                    aug_mix_target_expand = args.aug_mix_target_expand,
                                                                    dont_center_psf = args.dont_center_psf,
                                                                    high_exposure = args.high_exposure,
                                                                    low_exposure = args.low_exposure), args.data_path)

    # clean validaton set
    dataset_test, _ = get_dataset(args.dataset, "val", get_transform(train=False, blur = False), args.data_path)

    if args.low_exposure:
        evalblur_type = None
    elif args.high_exposure:
        evalblur_type = train_blur_type
    else:
        evalblur_type = None

    # blurred validaton set
    dataset_test_blurred, _ = get_dataset(args.dataset, "val", get_transform(train=False,
                                                                            blur_ratio = 1,
                                                                            blur_type = evalblur_type, 
                                                                            blur = True,
                                                                            use_stored_psfs = args.use_stored_psfs, 
                                                                            cpu_blur = args.cpu_blur, 
                                                                            stored_psf_directory = args.stored_psf_directory, 
                                                                            dont_center_psf = args.dont_center_psf,
                                                                            high_exposure = args.high_exposure,
                                                                            low_exposure = args.low_exposure), args.data_path)

    print("Creating data loaders...")

    # in case running across multiple processes
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
        test_sampler_blurred = torch.utils.data.distributed.DistributedSampler(dataset_test_blurred)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        test_sampler_blurred = torch.utils.data.SequentialSampler(dataset_test_blurred)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True)


    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    data_loader_test_blurred = torch.utils.data.DataLoader(
        dataset_test_blurred, batch_size=1,
        sampler=test_sampler_blurred, num_workers=args.workers,
        collate_fn=utils.collate_fn)


    print("Creating model...")
    if "fasterrcnn_resnet50_fpn" in args.model:
        model = models.faster_rcnn.fasterrcnn_resnet50_fpn(num_classes=91,
                                                                pretrained=args.pretrained, warp_internally = args.warp_in_model, trainable_backbone_layers = args.trainable_backbone_blocks)
    elif "mobile_net" in args.model:
        model = models.versatile_backbone_models.create_model(num_classes=91,
                                                                pretrained=args.pretrained, backbone="mobile_net")#, warp_internally = args.warp_in_model)
    elif "resnet_50" in args.model:
        model = models.versatile_backbone_models.create_model(num_classes=91,
                                                                pretrained=args.pretrained, backbone="resnet_50")#, warp_internally = args.warp_in_model)
    else:
        print("Unrecognized model type! Quitting.")
        return

    # by default, using fixed batchnorm layers.
    if args.unfrozen_batch_norm:
        model = utils.convert_to_regular_batchnorm(model)

    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    # flags for resuming training
    if args.resume:
        print("Resuming training from from " + args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    # if you just want to resume from weights alone, without training state
    if args.start_from_weights:
        print("Using model weights from " + args.start_from_weights)
        checkpoint = torch.load(args.start_from_weights, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
    

    ########################################### Setup done, now for training ########################################


    print("Starting training.")

    start_time = time.time()

    if args.eval_first:
        # with blurred images
        cocoEvaluator = evaluate(model = model, 
                                data_loader = data_loader_test_blurred, 
                                device = device, 
                                epoch_number = None,
                                blurring_images = True,
                                gpu_blur = args.gpu_blur,
                                expand_target_boxes = args.expand_target_boxes, 
                                add_noise = args.add_noise,
                                noise_level = args.noise_level,
                                add_jpeg_artifact = args.add_jpeg_artefacts)
        
        # vanilla clean dataset
        cocoEvaluator = evaluate(model = model, 
                                data_loader = data_loader_test, 
                                device = device, 
                                epoch_number = None)

    
    gc.collect()
    for epoch in range(args.start_epoch, args.epochs):

        dataset.dataset._epoch_number = epoch
        if args.distributed:
            train_sampler.set_epoch(epoch)

        gc.collect()

        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print("Epoch " + str(epoch) + ", start time: ", dt_string)	

        train_one_epoch(model = model, 
                        optimizer = optimizer, 
                        data_loader = data_loader, 
                        device = device, 
                        epoch = epoch,  
                        distributed_mode = args.distributed,
                        blur_train = args.blur_train,
                        early_stop = args.early_stop,
                        print_freq = args.print_freq,
                        writer = writer,
                        gpu_blur = args.gpu_blur,
                        expand_target_boxes = args.expand_target_boxes,
                        use_custom_image_norm = args.use_custom_image_norm, 
                        add_noise = args.add_noise, 
                        noise_level = args.noise_level,
                        add_block = args.add_block,
                        add_jpeg_artifact = args.add_jpeg_artefacts)

        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print("Epoch " + str(epoch) + ", end time: ", dt_string)

        gc.collect()

        start_time = time.perf_counter()
        lr_scheduler.step()
        if args.output_dir:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch},
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
        elapsed_time = (time.perf_counter() - start_time)
        print("Saving weights took: " + str(round(elapsed_time, 2)) + "s")

        # evaluate after every epoch

        # clean images
        cocoEvaluator = evaluate(model = model, 
                                data_loader = data_loader_test, 
                                device = device, 
                                image_output_folder = None)
        if not args.distributed or (args.distributed and torch.distributed.get_rank() == 0):
            writer.add_scalar('Normal/AccuraciesSweep', cocoEvaluator.coco_eval["bbox"].stats[0], epoch)
            writer.add_scalar('Normal/Accuracies', cocoEvaluator.coco_eval["bbox"].stats[1], epoch)
            writer.add_scalar('Normal/AccuraciesSmall', cocoEvaluator.coco_eval["bbox"].stats[3], epoch)
            writer.add_scalar('Normal/AccuraciesMedium', cocoEvaluator.coco_eval["bbox"].stats[4], epoch)
            writer.add_scalar('Normal/AccuraciesLarge', cocoEvaluator.coco_eval["bbox"].stats[5], epoch)
            
            writer.add_scalar('Normal/recallSmall', cocoEvaluator.coco_eval["bbox"].stats[9], epoch)
            writer.add_scalar('Normal/recallMedium', cocoEvaluator.coco_eval["bbox"].stats[10], epoch)
            writer.add_scalar('Normal/recallLarge', cocoEvaluator.coco_eval["bbox"].stats[11], epoch)
            writer.add_scalar('Normal/recall', cocoEvaluator.coco_eval["bbox"].stats[12], epoch)

        # blurred images
        cocoEvaluator = evaluate(model = model, 
            data_loader = data_loader_test_blurred, 
            device = device,
            early_stop = args.early_stop,
            distributed_mode = args.distributed,
            blurring_images = True, 
            gpu_blur = args.gpu_blur, 
            expand_target_boxes = args.expand_target_boxes,
            use_custom_image_norm = args.use_custom_image_norm, 
            add_noise = args.add_noise, 
            noise_level = args.noise_level,
            add_block = args.add_block,
            add_jpeg_artifact = args.add_jpeg_artefacts)

        if not args.distributed or (args.distributed and torch.distributed.get_rank() == 0):
            writer.add_scalar('Blurred/AccuraciesSweep', cocoEvaluator.coco_eval["bbox"].stats[0], epoch)
            writer.add_scalar('Blurred/Accuracies', cocoEvaluator.coco_eval["bbox"].stats[1], epoch)
            writer.add_scalar('Blurred/AccuraciesSmall', cocoEvaluator.coco_eval["bbox"].stats[3], epoch)
            writer.add_scalar('Blurred/AccuraciesMedium', cocoEvaluator.coco_eval["bbox"].stats[4], epoch)
            writer.add_scalar('Blurred/AccuraciesLarge', cocoEvaluator.coco_eval["bbox"].stats[5], epoch)
            
            writer.add_scalar('Blurred/recallSmall', cocoEvaluator.coco_eval["bbox"].stats[9], epoch)
            writer.add_scalar('Blurred/recallMedium', cocoEvaluator.coco_eval["bbox"].stats[10], epoch)
            writer.add_scalar('Blurred/recallLarge', cocoEvaluator.coco_eval["bbox"].stats[11], epoch)
            writer.add_scalar('Blurred/recall', cocoEvaluator.coco_eval["bbox"].stats[12], epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    # data
    parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--data_path', default='/media/mosayed/data_f_256/datasets/COCO/coco/', help='dataset')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)

    parser.add_argument("--use_stored_psfs", dest="use_stored_psfs", help="Use stored PSFs for when blurring in the train data_loader.", action="store_true")
    parser.add_argument('--stored_psf_directory', default='/media/mosayed/data_f_256/datasets/COCO/coco/psfs', help='Stored PSFs path.') 


    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 0)')

    # model 
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--trainable_backbone_blocks', default=3, type=int, help='Resnet backbone blocks to train.')
    parser.add_argument("--pretrained", dest="pretrained", help="Use pre-trained models from the modelzoo", action="store_true")


    # training particulars
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch_size', default=8, type=int, help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--lr', default=0.04, type=float, help='initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--epochs', default=37, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')

    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--start_from_weights', help='start training from provided weights pth.tar')
    parser.add_argument('--start_epoch', default=0, type=int, help='Custom start epoch.')


    parser.add_argument('--early_stop', type = int, help='early stop for eval')

    parser.add_argument("--eval_first", dest="eval_first", help="Evaluate first before training.", action="store_true")

    # logging
    parser.add_argument('--tensorboard_path', default="debug", help='device')
    parser.add_argument('--output_dir', default='debug', help='Output directory for weights.')
    parser.add_argument('--image_output_dir', default='debug', help='Output directory for images.')
    parser.add_argument('--print_freq', default=20, type=int, help='print frequency')



    # blur
    parser.add_argument("--blur_train", dest="blur_train", help="Blur during training.", action="store_true")
    parser.add_argument("--cpu_blur", dest="cpu_blur", help="CPU blurring in fourier domain, happens on the CPU in the dataloader's worker processes.", action="store_true")
    parser.add_argument("--gpu_blur", dest="gpu_blur", help="GPU blurring, happens on the GPU in the GPU thread.", action="store_true")

    parser.add_argument('--param_index', default=None, help='Type of blur to use use for training blur. Options are 1, 2, and 3.')

    parser.add_argument("--high_exposure", dest="high_exposure", help="Train and evaluate with high exposure blur.", action="store_true")
    parser.add_argument("--low_exposure", dest="low_exposure", help="Train and evaluate with low exposure blur.", action="store_true")

    parser.add_argument("--expand_target_boxes", dest="expand_target_boxes", help="Expand target boxes during training and evaluating according to blur kernel shifts.", action="store_true")

    parser.add_argument("--dont_center_psf", dest="dont_center_psf", help="Don't center PSFs. You will need to generate psf on the fly for this. Cannot use stored psfs.", action="store_true")

    parser.add_argument("--add_noise", dest="add_noise", help="Add noise after blurring.", action="store_true")
    parser.add_argument('--noise_level', default=0.001, type=float, help='Noise level.')
    parser.add_argument("--add_block", dest="add_block", help="Add block artifacts after blurring.", action="store_true")
    parser.add_argument("--add_jpeg_artefacts", dest="add_jpeg_artefacts", help="Add jpeg compression artifacts.", action="store_true")


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

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)
