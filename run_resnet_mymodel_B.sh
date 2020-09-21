#!/bin/sh
#############################################################################
# TODO: Initialize anything you need for the forward pass
#############################################################################
python -u train_resnet_mymodel_B.py \
    --model resnet_mymodel_B \
    --kernel-size 3 \
    --hidden-dim 128 \
    --epochs 50 \
    --weight-decay 0.0001 \
    --momentum 0.7 \
    --batch-size 128 \
    --lr 0.04 | tee resnet_mymodel_B.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
