#!/bin/sh
#############################################################################
# TODO: Initialize anything you need for the forward pass
#############################################################################
python -u train_resconv_B.py \
    --model resnet_conv_B \
    --kernel-size 3 \
    --hidden-dim 64 \
    --epochs 45 \
    --weight-decay 0.001 \
    --momentum 0.98 \
    --batch-size 128 \
    --lr 0.001 | tee resnet_conv_B.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
