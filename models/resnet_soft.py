import torchvision.models as models

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional


class MyModel(nn.Module):
    def __init__(self, im_size, n_classes):
        '''
        Extra credit model

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################

        self.model1 = models.resnet18(pretrained = True)

        for param in self.model1.parameters():
        	print("True")
        	param.requires_grad = False

        self.fc = nn.Linear(512*4, n_classes)


        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################



    def forward(self, images):
        '''
        Take a batch of images and run them through the model to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        scores = None
        #############################################################################
        # TODO: Implement the forward pass.
        #############################################################################

        out1 = functional.relu(self.model1(images))
        print("Shape of Out1", out1.shape)

        out2 = functional.softmax(self.fc(out1))
        print("Shape of Out2", out2.shape)

        scores = out2

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

