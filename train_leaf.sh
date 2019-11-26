#!/bin/bash

phase="train"
dist_branch=True
include_bg=True
embedding_dim=16

train_dir="./tfrecords/CVPPP2017/train"
validation=False
# val_dir="./tfrecords/CVPPP2017_val/val"
image_depth="uint8"
image_channels=3
model_dir="./model_CVPPP2017"

lr=0.0001
batch_size=4
training_epoches=300


cd /work/scratch/chen/instance_segmentation_with_pixel_embeddings

/home/staff/chen/miniconda3/envs/tf/bin/python /work/scratch/chen/instance_segmentation_with_pixel_embeddings/main.py \
			--phase="$phase" \
			--dist_branch="$dist_branch" \
			--include_bg="$include_bg" \
			--embedding_dim="$embedding_dim" \
			--train_dir="$train_dir" \
			--validation="$validation" \
			--val_dir="$val_dir"\
			--image_depth="$image_depth" \
			--image_channels="$image_channels" \
			--model_dir="$model_dir" \
			--lr="$lr" \
			--batch_size="$batch_size" \
			--training_epoches="$training_epoches" 