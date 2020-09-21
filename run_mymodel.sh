#!/bin/sh
#############################################################################
# TODO: Initialize anything you need for the forward pass
#############################################################################
python -u train.py \
    --model mymodel \
    --kernel-size 5 \
    --hidden-dim 64 \
    --epochs 35 \
    --weight-decay 0.001 \
    --momentum 0.98 \
    --batch-size 128 \
    --lr 0.001| tee mymodel.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################