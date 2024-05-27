#!/bin/bash


source activate light8
echo 'Starting Shell Script'

model='WOLNET' # with new windowing
loss_type="WFTTOPK" #"WDCTOPK" version1 # 'FOCALDSC'  'CATEGORICAL' # loss_type='COMBINED' # inital runs without TOPK, if multi consider using it...
tag='WOLNET'
# data='RADCURE' #Dataset being used
# loss_type="TAL" # "WFTTOPK" #"WDCTOPK" version1 # 'FOCALDSC'  'CATEGORICAL' # loss_type='COMBINED' # inital runs without TOPK, if multi consider using it...
optim='RADAM' #'SGD' # 'RADAM' #'ADAM'
dce_version=1
deform=True
clip_min=-500
clip_max=1000 # clip_min=-300 # clip_max=200
epoch=20 # 500 # 100 # number of epochs
weight_decay=0.00001 # .000001 # decay rate for optimizer
batch=2 # batch size # unet3D can use 2
aug_p=0.9
scheduler_type='pleateau' # 0.5 at 75 epochs for the training step...
shuffle=True
classes=20 # 19 # number of classes (to test on), PAN HNSCC GTV/CTV... (Do we need/want that?)
norm='standard' # 'linear' # 'standard'
scale_by=2
window=56 # default is 5
crop_factor=176 # 192 # 448 # 384 # default is 512
crop_as='3D'
fmaps=48
spacing='3mm' # spacing between slices...
filter=True
path="./train.py" 
config_path='configs/example_config.json'
print_outputs_to=$model'_'$tag'_'$(date "+%b_%d_%Y_%T").txt

echo 'Started python script.'

python3 $path --model $model --config-path $config_path \
        --scheduler-type $scheduler_type --aug-prob $aug_p \
        --spacing $spacing --f-maps $fmaps \
        --n-classes $classes --shuffle-data $shuffle \
        --n-epochs $epoch \
        --decay $weight_decay --batch-size $batch \
        --loss $loss_type --optim $optim --norm $norm --crop-factor $crop_factor \
        --scale-factor $scale_by --crop-as $crop_as --clip-min $clip_min\
        --clip-max $clip_max --window $window --filter $filter \
        --tag $tag > $print_outputs_to

echo 'Python script finished.'
