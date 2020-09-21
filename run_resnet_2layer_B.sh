#!/bin/sh
#############################################################################
# TODO: Initialize anything you need for the forward pass
#############################################################################
python -u train_resnet_2layer_B.py \
    --model resnet_2layer_B \
    --hidden-dim 150 \
    --epochs 30 \
    --weight-decay 0.0 \
    --momentum 0.99 \
    --batch-size 128 \
    --lr 0.001 | tee resnet_2layer_B.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
