#!/bin/sh
#############################################################################
# TODO: Initialize anything you need for the forward pass
#############################################################################
python -u train_resconv.py \
    --model resnet_conv \
    --kernel-size 3 \
    --hidden-dim 150 \
    --epochs 20 \
    --weight-decay 0.0005 \
    --momentum 0.9 \
    --batch-size 128 \
    --lr 0.05 | tee resnet_conv.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
