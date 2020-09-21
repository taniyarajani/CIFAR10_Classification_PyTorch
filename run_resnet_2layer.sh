#!/bin/sh
#############################################################################
# TODO: Initialize anything you need for the forward pass
#############################################################################
python -u train_resnet_2layer.py \
    --model resnet_2layer \
    --hidden-dim 100 \
    --epochs 6 \
    --weight-decay 0.001 \
    --momentum 0.7 \
    --batch-size 128 \
    --lr 0.03| tee resnet_2layer.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
