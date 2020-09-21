#!/bin/sh
#############################################################################
# TODO: Initialize anything you need for the forward pass
#############################################################################
python -u train_resnet_mymodel.py \
    --model resnet_mymodel \
    --kernel-size 3 \
    --hidden-dim 128 \
    --epochs 50 \
    --weight-decay 0.0001 \
    --momentum 0.85 \
    --batch-size 128 \
    --lr 0.04 | tee resnet_mymodel.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
