#!/bin/sh
#############################################################################
# TODO: Initialize anything you need for the forward pass
#############################################################################
python -u train_resnet_soft.py \
    --model resnet_softmax \
    --epochs 6 \
    --weight-decay 0.0001 \
    --momentum 0.99 \
    --batch-size 128 \
    --lr 0.001 | tee resnet_softmax.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
