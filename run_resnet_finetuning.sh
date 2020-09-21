#!/bin/sh
#############################################################################
# TODO: Initialize anything you need for the forward pass
#############################################################################
python -u train_resnet_finetuning.py \
    --model resnet_soft_B \
    --epochs 30 \
    --weight-decay 0.001 \
    --momentum 0.0 \
    --batch-size 128 \
    --lr 0.01| tee resnet_finetuning.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################