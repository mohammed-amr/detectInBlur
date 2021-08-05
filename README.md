# detectInBlur
Code for the CVPR 2021 paper "Improved Handling of Motion Blur in Online Object Detection"

[[Project Page](http://visual.cs.ucl.ac.uk/pubs/handlingMotionBlur/)] [[Paper](http://visual.cs.ucl.ac.uk/pubs/handlingMotionBlur/Improved_Handling_of_Motion_Blur_in_Online_Object_Detection_CVPR2021.pdf)] [[Video](https://www.youtube.com/watch?v=NhH012avygI&t=31s&ab_channel=MohamedSayed)]

Most of this repo is based on the detection reference code from TorchVision, found [here](https://github.com/pytorch/vision/tree/master/references/detection).

## Motion Blur
Blur kernel generation is explained in the paper and the supplemental. To specify that blurring should take place for both training and evaluation, use the `--blur_train` and `--blur_eval` flags respectively. This alone isn't enough, you must also specify how you want images to be blurred. Either on the GPU, `--gpu_blur`, or CPU, `--cpu_blur`. CPU blurring happens in the fourier domain while GPU blurring is done by a basic sparse correlation loop. We recommend the latter as it's faster to perform on the GPU and prevents the data_loader from being choked.

For evaluation, a transform will generate blur kernels on the fly. For training, we recommend you first create and store blur kernels using , and then load them for each image to speed up training. To use stored PSFs, use `--use_stored_psfs`.

You can also specify the type of blur kernel via `--param_index` with available types: 1, 2, 3. You can specify the exposure range of the blur kernel via `--high_exposure` (4,5) and `--low_exposure` (1,2,3).

## Evaluating Models

To evaluate models, first download model weights from here: [Link](none). 

`evaluate.py` will by default run through all ranges of exposure and blur type. You'll need to specify `--blur_eval` and hardware `--gpu_blur` or `--cpu_blur`. 

Any remedies that need specific flags are explained in the args section.


## Training Models
`train.py` will by default run through all ranges of exposure and blur type. You'll need to specify `--blur_eval` and hardware `--gpu_blur` or `--cpu_blur`. 


