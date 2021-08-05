r"""Borrowed from the original sample on PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

"""
import datetime
import os
import time
import itertools
from numpy.core.fromnumeric import resize
import numpy as np

import torch
import torch.utils.data
from torch import nn
from torch.utils.data import distributed
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco, get_coco_kp

from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from engine_blur_estimator import train_one_epoch, evaluate

import utils
import transforms as T

from pathlib import Path
import shutil
from torch.utils.tensorboard import SummaryWriter
import gc


import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import numpy as np

import random

def get_dataset(name, image_set, transform, data_path):
    if "coco" in name:
        paths = {
            "coco": (data_path, get_coco, 91),
            "coco_kp": (data_path, get_coco_kp, 2)
        }

        p, ds_fn, num_classes = paths[name]

        ds = ds_fn(p, image_set=image_set, transforms=transform)

    return ds, num_classes

def get_transform(train, 
                blur_ratio = 1, 
                blur_exposure=None, 
                blur_type=None, 
                blur = False, 
                use_stored_psfs = False, 
                cpu_blur = False, 
                stored_psf_directory = None,
                non_pos_aug_mix = False, 
                include_pos_aug_mix = False, 
                aug_mix_target_expand = False,
                LEHE_blur_seg = True):



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
                                            LEHE_blur_seg = LEHE_blur_seg))
    if blur_type is None and blur:
        print("Blur type: " + "all types.")
    else:
        print("Blur type: " + str(blur_type))

    transforms.append(T.ToTensor())

    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

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

    # Data loading code
    print("Loading data")

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

    if not args.test_only:
        dataset, num_classes = get_dataset(args.dataset, "train", get_transform(train=True, 
                                                                                blur_type = train_blur_type, 
                                                                                blur_ratio = 0.9, 
                                                                                blur = args.blur_train, 
                                                                                use_stored_psfs = args.use_stored_psfs, 
                                                                                cpu_blur = args.cpu_blur, 
                                                                                stored_psf_directory = args.stored_psf_directory, 
                                                                                totalNumberofEpochs = args.epochs, 
                                                                                non_pos_aug_mix = args.non_pos_aug_mix, 
                                                                                include_pos_aug_mix = args.include_pos_aug_mix, 
                                                                                aug_mix_target_expand = args.aug_mix_target_expand,
                                                                                LEHE_blur_seg=args.LEHE_blur_seg), args.data_path)


    dataset_test_blurred, _ = get_dataset(args.dataset, "val", get_transform(train=False, 
                                                                            blur_ratio = 0.9, 
                                                                            blur = True, 
                                                                            use_stored_psfs = args.use_stored_psfs, 
                                                                            cpu_blur = args.cpu_blur, 
                                                                            stored_psf_directory = args.stored_psf_directory, 
                                                                            totalNumberofEpochs = args.epochs,
                                                                            LEHE_blur_seg=args.LEHE_blur_seg), args.data_path)

    print("Creating data loaders")
    if args.distributed:
        if not args.test_only:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler_blurred = torch.utils.data.distributed.DistributedSampler(dataset_test_blurred)
    else:
        if not args.test_only:
            train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler_blurred = torch.utils.data.SequentialSampler(dataset_test_blurred)

    if not args.test_only:
        if args.aspect_ratio_group_factor >= 0:
            group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
            train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
        else:
            train_batch_sampler = torch.utils.data.BatchSampler(
                train_sampler, args.batch_size, drop_last=True)

    if not args.test_only:
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
            collate_fn=utils.collate_fn)

    data_loader_test_blurred = torch.utils.data.DataLoader(
        dataset_test_blurred, batch_size=1,
        sampler=test_sampler_blurred, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    print("Creating model")

    model = torchvision.models.resnet18(pretrained = args.pretrained)    
    num_ftrs = model.fc.in_features
    if not args.LEHE_blur_seg:
        model.fc = nn.Linear(num_ftrs, 16)
    else:
        model.fc = nn.Linear(num_ftrs, 4)

    if args.test_only:
        criterion = None

    else:
        criterion = nn.CrossEntropyLoss()
        criterion.to(device)

    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    if args.resume:
        print("Resuming training from " + args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.start_from_weights:
        print("Using model weights from " + args.start_from_weights)
        checkpoint = torch.load(args.start_from_weights, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])


    if args.test_only:
        accuracies, targetsAll, predsAll = evaluate(model = model, 
                            data_loader = data_loader_test_blurred, 
                            device = device, 
                            distributed_mode=args.distributed,
                            blurring_images = True, 
                            gpu_blur = args.gpu_blur,
                            LEHE_blur_seg = args.LEHE_blur_seg,
                            add_jpeg_artifact = args.add_jpeg_artefacts,
                            resize_images=args.resize_images,
                            quantize_image = args.quantize_image,
                            add_noise = args.add_noise,
                            noise_level=args.noise_level,
                            add_block = args.add_block,
                            send_back_preds_targets = True,
                            early_stop = args.early_stop)

        targetsAll = torch.stack(targetsAll)
        predsAll = torch.stack(predsAll)


        if not args.LEHE_blur_seg:
            confMat = confusion_matrix(predsAll.squeeze().view(-1).cpu(), targetsAll.view(-1).cpu())

            fig = plt.figure(figsize=(9, 9))
            plt.imshow(confMat, interpolation='nearest', cmap=plt.cm.Blues)
            plt.show()

            plt.xticks(np.arange(16), ("NB", 'P1E1', 'P1E2', 'P1E3', 'P1E4', 'P1E5', 'P2E1', 'P2E2', 'P2E3', 'P2E4', 'P2E5', 'P3E1', 'P3E2', 'P3E3', 'P3E4', 'P3E5'))
            plt.yticks(np.arange(16), ("NB", 'P1E1', 'P1E2', 'P1E3', 'P1E4', 'P1E5', 'P2E1', 'P2E2', 'P2E3', 'P2E4', 'P2E5', 'P3E1', 'P3E2', 'P3E3', 'P3E4', 'P3E5'))

            thresh = confMat.max() / 2.
            normalize = False
            fmt = '.2f' if normalize else 'd'
            thresh = confMat.max() / 2.
            for i, j in itertools.product(range(confMat.shape[0]), range(confMat.shape[1])):
                plt.text(j, i, format(confMat[i, j], fmt), horizontalalignment="center", color="white" if confMat[i, j] > thresh else "black")

            plt.ylabel('True Blur')
            plt.xlabel('Predicted Blur')

            fig.savefig('confMat16.png')


        predsAll = predsAll.squeeze()

        predsAll4 = predsAll.clone()
        targetsAll4 = targetsAll.clone()

        if not args.LEHE_blur_seg:
            for predIndex, pred in enumerate(predsAll4):
                if pred.cpu() in [1,2,3,4,5]:
                    predsAll4[predIndex] = torch.tensor([1]).cuda()
                elif pred.cpu() in [6,7,8,9,10]:
                    predsAll4[predIndex] = torch.tensor([2]).cuda()
                elif pred.cpu() in [11,12,13,14,15]:
                    predsAll4[predIndex] = torch.tensor([3]).cuda()
                else:
                    predsAll4[predIndex] = torch.tensor([0]).cuda()

            for target_index, target in enumerate(targetsAll4):
                if target.cpu() in [1,2,3,4,5]:
                    targetsAll4[target_index] = torch.tensor([1]).cuda()
                elif target.cpu() in [6,7,8,9,10]:
                    targetsAll4[target_index] = torch.tensor([2]).cuda()
                elif target.cpu() in [11,12,13,14,15]:
                    targetsAll4[target_index] = torch.tensor([3]).cuda()
                else:
                    targetsAll4[target_index] = torch.tensor([0]).cuda()

            confMat = confusion_matrix(predsAll4.view(-1).cpu(), targetsAll4.view(-1).cpu())

            fig = plt.figure(figsize=(9, 9))
            plt.imshow(confMat, interpolation='nearest', cmap=plt.cm.Blues)
            plt.show()

            plt.xticks(np.arange(4), ("NB", 'P1', 'P2', 'P3'))
            plt.yticks(np.arange(4), ("NB", 'P1', 'P2', 'P3'))

            thresh = confMat.max() / 2.
            normalize = False
            fmt = '.2f' if normalize else 'd'
            thresh = confMat.max() / 2.
            for i, j in itertools.product(range(confMat.shape[0]), range(confMat.shape[1])):
                plt.text(j, i, format(confMat[i, j], fmt), horizontalalignment="center", color="white" if confMat[i, j] > thresh else "black")

            plt.ylabel('True Blur')
            plt.xlabel('Predicted Blur')

            fig.savefig('confMat4.png')


        predsAll4LEHE = predsAll.clone()
        targetsAll4LEHE = targetsAll.clone()

        if not args.LEHE_blur_seg:
            for predIndex, pred in enumerate(predsAll4LEHE):
                if pred.cpu() in [4,5]:
                    predsAll4LEHE[predIndex] = torch.tensor([1]).cuda()
                elif pred.cpu() in [9,10]:
                    predsAll4LEHE[predIndex] = torch.tensor([2]).cuda()
                elif pred.cpu() in [14,15]:
                    predsAll4LEHE[predIndex] = torch.tensor([3]).cuda()
                else:
                    predsAll4LEHE[predIndex] = torch.tensor([0]).cuda()

            for target_index, target in enumerate(targetsAll4LEHE):
                if target.cpu() in [4,5]:
                    targetsAll4LEHE[target_index] = torch.tensor([1]).cuda()
                elif target.cpu() in [9,10]:
                    targetsAll4LEHE[target_index] = torch.tensor([2]).cuda()
                elif target.cpu() in [14,15]:
                    targetsAll4LEHE[target_index] = torch.tensor([3]).cuda()
                else:
                    targetsAll4LEHE[target_index] = torch.tensor([0]).cuda()
        else:
            for predIndex, pred in enumerate(predsAll4LEHE):
                if pred.cpu() in [1]:
                    predsAll4LEHE[predIndex] = torch.tensor([1]).cuda()
                elif pred.cpu() in [2]:
                    predsAll4LEHE[predIndex] = torch.tensor([2]).cuda()
                elif pred.cpu() in [3]:
                    predsAll4LEHE[predIndex] = torch.tensor([3]).cuda()
                else:
                    predsAll4LEHE[predIndex] = torch.tensor([0]).cuda()

            for target_index, target in enumerate(targetsAll4LEHE):
                if target.cpu() in [1]:
                    targetsAll4LEHE[target_index] = torch.tensor([1]).cuda()
                elif target.cpu() in [2]:
                    targetsAll4LEHE[target_index] = torch.tensor([2]).cuda()
                elif target.cpu() in [3]:
                    targetsAll4LEHE[target_index] = torch.tensor([3]).cuda()
                else:
                    targetsAll4LEHE[target_index] = torch.tensor([0]).cuda()

        confMat = confusion_matrix(predsAll4LEHE.view(-1).cpu(), targetsAll4LEHE.view(-1).cpu())

        fig = plt.figure(figsize=(9, 9))
        plt.imshow(confMat, interpolation='nearest', cmap=plt.cm.Blues)
        plt.show()

        plt.xticks(np.arange(4), ("LE", 'P1HE', 'P2HE', 'P3HE'))
        plt.yticks(np.arange(4), ("LE", 'P1HE', 'P2HE', 'P3HE'))

        thresh = confMat.max() / 2.
        normalize = False
        fmt = '.2f' if normalize else 'd'
        thresh = confMat.max() / 2.
        for i, j in itertools.product(range(confMat.shape[0]), range(confMat.shape[1])):
            plt.text(j, i, format(confMat[i, j], fmt), horizontalalignment="center", color="white" if confMat[i, j] > thresh else "black")

        plt.ylabel('True Blur')
        plt.xlabel('Predicted Blur')

        fig.savefig('confMat4LEHE.png')


        return


    print("Start training")
    start_time = time.time()

    if args.eval_first:
        accuracies = evaluate(model = model, 
                            data_loader = data_loader_test_blurred, 
                            device = device, 
                            distributed_mode=args.distributed,
                            blurring_images = True, 
                            gpu_blur = args.gpu_blur,
                            LEHE_blur_seg = args.LEHE_blur_seg,
                            add_jpeg_artifact = args.add_jpeg_artefacts,
                            resize_images=args.resize_images,
                            quantize_image = args.quantize_image,
                            add_noise = args.add_noise,
                            noise_level=args.noise_level,
                            add_block = args.add_block,
                            early_stop = args.early_stop)


    gc.collect()
    for epoch in range(args.start_epoch, args.epochs):
        if "coco" in args.dataset: 
            dataset.dataset._epoch_number = epoch
        else:
            dataset._epoch_number = epoch

        if args.distributed:
            train_sampler.set_epoch(epoch)

        gc.collect()
        train_one_epoch(model = model, 
                        optimizer = optimizer, 
                        criterion = criterion,
                        data_loader = data_loader, 
                        device = device, 
                        epoch = epoch,
                        distributed_mode = args.distributed,
                        blur_train = args.blur_train,
                        writer = writer, 
                        gpu_blur = args.gpu_blur,
                        crop_images = args.crop_images,
                        LEHE_blur_seg = args.LEHE_blur_seg,
                        add_jpeg_artifact = args.add_jpeg_artefacts,
                        resize_images=args.resize_images,
                        quantize_image = args.quantize_image,
                        add_noise = args.add_noise,
                        noise_level=args.noise_level,
                        add_block = args.add_block,
                        early_stop = args.early_stop)

        gc.collect()

        lr_scheduler.step()
        if args.output_dir:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch},
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))


        accuracies = evaluate(model = model, 
                            data_loader = data_loader_test_blurred, 
                            device = device, 
                            distributed_mode=args.distributed,
                            blurring_images = True, 
                            gpu_blur = args.gpu_blur,
                            LEHE_blur_seg = args.LEHE_blur_seg,
                            add_jpeg_artifact = args.add_jpeg_artefacts,
                            resize_images=args.resize_images,
                            quantize_image = args.quantize_image,
                            add_noise = args.add_noise,
                            noise_level=args.noise_level,
                            add_block = args.add_block,
                            early_stop = args.early_stop)


        if not args.distributed or (args.distributed and torch.distributed.get_rank() == 0) :
            writer.add_scalar('Blurred/Top1Accuracy', accuracies[0], epoch)
            writer.add_scalar('Blurred/Top2Accuracy', accuracies[1], epoch)


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

    parser.add_argument("--crop_images", dest="crop_images", help="Crop images when batching.", action="store_true")
    parser.add_argument("--resize_images", dest="resize_images", help="resize_images", action="store_true")
    parser.add_argument("--quantize_image", dest="quantize_image", help="quantize_image", action="store_true")

    # model 
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--trainable_backbone_blocks', default=3, type=int, help='Resnet backbone blocks to train.')
    parser.add_argument("--pretrained", dest="pretrained", help="Use pre-trained models from the modelzoo", action="store_true")


    # training particulars
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch_size', default=8, type=int, help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 0)')
    parser.add_argument('--lr', default=0.04, type=float, help='initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--epochs', default=37, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')

    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--start_from_weights', help='start training from provided weights pth.tar')
    parser.add_argument('--start_epoch', default=0, type=int, help='Custom start epoch.')


    parser.add_argument('--early_stop', type = int, help='early stop for eval')

    parser.add_argument("--eval_first", dest="eval_first", help="Evaluate first before training.", action="store_true")
    parser.add_argument("--test_only", dest="test_only", help="Only perform test.", action="store_true")

    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')

    # logging
    parser.add_argument('--tensorboard_path', default="debug", help='device')
    parser.add_argument('--output_dir', default='debug', help='Output directory for weights.')
    parser.add_argument('--image_output_dir', default='debug', help='Output directory for images.')


    # blur
    parser.add_argument("--blur_train", dest="blur_train", help="Blur during training.", action="store_true")
    parser.add_argument("--cpu_blur", dest="cpu_blur", help="CPU blurring in fourier domain, happens on the CPU in the dataloader's worker processes.", action="store_true")
    parser.add_argument("--gpu_blur", dest="gpu_blur", help="GPU blurring, happens on the GPU in the GPU thread.", action="store_true")

    parser.add_argument('--param_index', default=None, help='Type of blur to use use for training blur. Options are 1, 2, and 3.')

    parser.add_argument("--LEHE_blur_seg", dest="LEHE_blur_seg", help="LEHE_blur_seg", action="store_true")
    parser.add_argument("--high_exposure", dest="high_exposure", help="Train and evaluate with high exposure blur.", action="store_true")
    parser.add_argument("--low_exposure", dest="low_exposure", help="Train and evaluate with low exposure blur.", action="store_true")

    parser.add_argument("--expand_target_boxes", dest="expand_target_boxes", help="Expand target boxes during training and evaluating according to blur kernel shifts.", action="store_true")

    parser.add_argument("--dont_center_psf", dest="dont_center_psf", help="Don't center PSFs. You will need to generate psf on the fly for this. Cannot use stored psfs.", action="store_true")

    parser.add_argument("--add_noise", dest="add_noise", help="Add noise after blurring.", action="store_true")
    parser.add_argument('--noise_level', default=0.001, type=float, help='Noise level.')
    parser.add_argument("--add_block", dest="add_block", help="Add block artifacts after blurring.", action="store_true")
    parser.add_argument("--add_jpeg_artefacts", dest="add_jpeg_artefacts", help="Add jpeg compression artifacts.", action="store_true")

    # Augmix
    parser.add_argument("--non_pos_aug_mix", dest="non_pos_aug_mix", help="Non positional augmix.", action="store_true")
    parser.add_argument("--include_pos_aug_mix", dest="include_pos_aug_mix", help="Include positional augmentations in augmix.", action="store_true")
    parser.add_argument("--aug_mix_target_expand", dest="aug_mix_target_expand", help="Expand target boxes for AugMix according to positional shifts from spatial augmentations..", action="store_true")

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)
