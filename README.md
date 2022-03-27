# detectInBlur
Code for the CVPR 2021 paper "Improved Handling of Motion Blur in Online Object Detection"

[[Project Page](http://visual.cs.ucl.ac.uk/pubs/handlingMotionBlur/)] [[Paper](http://visual.cs.ucl.ac.uk/pubs/handlingMotionBlur/Improved_Handling_of_Motion_Blur_in_Online_Object_Detection_CVPR2021.pdf)] [[Video](https://www.youtube.com/watch?v=NhH012avygI&t=31s&ab_channel=MohamedSayed)]

Most of this repo is based on the detection reference code from `torchvision`, found [here](https://github.com/pytorch/vision/tree/master/references/detection).

## Motion Blur
Blur kernel generation is explained in the paper and the supplemental. To specify that blurring should take place for both training and evaluation, use the `--blur_train` and `--blur_eval` flags respectively. This alone isn't enough, you must also specify how you want images to be blurred. Either on the GPU, `--gpu_blur`, or CPU, `--cpu_blur`. CPU blurring happens in the fourier domain while GPU blurring is done by a basic sparse correlation loop. We recommend the latter as it's faster to perform on the GPU and prevents the data_loader from being choked.

For evaluation, a transform will generate blur kernels on the fly. For training, we recommend you first create and store blur kernels using `generate_PSFs.py`, and then load them for each image to speed up training. To use stored PSFs, use `--use_stored_psfs`.

You can also specify the type of blur kernel via `--param_index` with available types: 1, 2, 3. You can specify the exposure range of the blur kernel via `--high_exposure` (4,5) and `--low_exposure` (1,2,3).

## Evaluating Models

To evaluate models, first download model weights from [https://drive.google.com/drive/folders/1_W40yar1wsKacrM0DPYS2kkTEfsynMTD?usp=sharing] [here]. 

`evaluate.py` will by default run through all ranges of exposure and blur type. You'll need to specify `--blur_eval` and hardware `--gpu_blur` or `--cpu_blur`. 

Any remedies that need specific flags are explained in the args section.

An eval command for evaluating a model looks like:

```python evaluate.py  -j 3 --tensorboard_path evals/test --blur_eval --gpu_blur --data_path /mnt/data_f2/mosayed/COCO/coco/ --resume weights/resnet50FPNBlur.pth```

If the model needs to be evaluated with expanded bounding boxes or the evaluation is on expanded bounding boxes, use:

```python evaluate.py  -j 3 --tensorboard_path evals/test --blur_eval --gpu_blur --data_path /mnt/data_f2/mosayed/COCO/coco/ --resume weights/resnet50FPNBlurExpand.pth --expand_target_boxes```

For ensemble models, you'll need to specifiy the blur estimator weights path along with four ensemble model weights paths. Make sure to specify the type of ensemble by using the 
`--LEHE` flag. For example:

```python evaluate.py  -j 3 --tensorboard_path evals/test --blur_eval --gpu_blur --data_path /mnt/data_f2/mosayed/COCO/coco/ --expand_target_boxes --use_ensemble --LEHE --blur_estimator_path weights/SpecByExpEstimator.pth --ensemble_model_paths "weights/resnet50FPNBlurLEExpand.pth weights/resnet50FPNBlurP1HEExpand.pth weights/resnet50FPNBlurP2HEExpand.pth weights/resnet50FPNBlurP3HEExpand.pth"```

For datasets with natural blur, you'll need to specify the dataset's name and location. You'll also have to specify if you want to evaluate sharp or blurry images via `--blurred_dataset`. For example, to evaluate GOPRO's blurred set with a pretrained model, you'd do: 

`python evaluate.py -j 3 --tensorboard_path evals/test --pretrained --dataset GOPRO --data_path /media/mosayed/data_f_256/datasets/GOPRO --blurred_dataset`

For the set with expanded labels, use: 

`python evaluate.py -j 3 --tensorboard_path evals/test --pretrained --dataset GOPROSynthLoad --data_path /media/mosayed/data_f_256/datasets/GOPROSynth --blurred_dataset --expand_target_boxes`


## Training Models
`train.py` will by default run through all ranges of exposure and blur type. You'll need to specify `--blur_eval` and hardware `--gpu_blur` or `--cpu_blur`. 

A base command for launching training on two GPUs with 8 images in a batch per GPU, blurring with pregenerated random kernels on the GPU, and using stored PSFs looks something like this: 

```CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2  --use_env train.py  -j 3 -b 8 --lr 0.04 --epochs 35 --lr-steps 16 21 --aspect-ratio-group-factor 3 --model fasterrcnn_resnet50_fpn --tensorboard_path runs/test --output_dir weights/test --pretrained --blur_train --gpu_blur --data_path /mnt/data_f2/mosayed/COCO/coco/ --stored_psf_directory /mnt/data_f2/mosayed/COCO/coco/psfs --use_stored_psfs```

If say you wanted to train with high exposure kernels and only type P3, you'd instead do: 

```CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2  --use_env train.py  -j 3 -b 8 --lr 0.04 --epochs 35 --lr-steps 16 21 --aspect-ratio-group-factor 3 --model fasterrcnn_resnet50_fpn --tensorboard_path runs/test --output_dir weights/test --pretrained --blur_train --gpu_blur --data_path /mnt/data_f2/mosayed/COCO/coco/ --stored_psf_directory /mnt/data_f2/mosayed/COCO/coco/psfs --use_stored_psfs --param_index 3 --high_exposure```

If you wanted expanded bounding boxes, you'd do: 

```CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2  --use_env train.py  -j 3 -b 8 --lr 0.04 --epochs 35 --lr-steps 16 21 --aspect-ratio-group-factor 3 --model fasterrcnn_resnet50_fpn --tensorboard_path runs/test --output_dir weights/test --pretrained --blur_train --gpu_blur --data_path /mnt/data_f2/mosayed/COCO/coco/ --stored_psf_directory /mnt/data_f2/mosayed/COCO/coco/psfs --use_stored_psfs --param_index 3 --high_exposure --expand_target_boxes```



## Blur Estimators
`train_blur_estimator.py` will allow you to train a small classifier for inferring the type of blur in an image. Blur flags are similar to `train.py`. We use this classifier for our final proposed model. There are two flavors of estimator. One that can detect among 16 different classes, and one that has coarser prediction over just four classes. To switch between them, use the `--LEHE` flag.

