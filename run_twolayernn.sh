#!/bin/sh
#############################################################################
# TODO: Initialize anything you need for the forward pass
#############################################################################
#For SGD
# python -u train_tarun.py \
#     --model twolayernn \
#     --hidden-dim 100 \
#     --epochs 10 \
#     --weight-decay 0.01 \
#     --momentum 0.7 \
#     --batch-size 512 \
#     --lr 0.01 | tee twolayernn.log


#For Adam
python -u train.py \
    --model twolayernn \
    --hidden-dim 250 \
    --epochs 20 \
    --weight-decay 0.001 \
    --batch-size 256 \
    --lr 0.0001 | tee twolayernn.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
