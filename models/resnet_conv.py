import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.nn.functional as F

import torchvision.models as models

class CNN(nn.Module):
    def __init__(self, hidden_dim, kernel_size, n_classes):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(CNN, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        
        self.conv1 = nn.Conv2d(256, hidden_dim*3, kernel_size, stride = 1, padding = 1)
        final_image_dim1 = self.compute_image_size((2,2), (kernel_size, kernel_size), 1, 1)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        final_image_dim2 = self.compute_image_size(final_image_dim1, (2,2), 2, 0)

     
        self.fc1 = torch.nn.Linear(hidden_dim*3*final_image_dim2[0]*final_image_dim2[1], n_classes)



        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################


    def compute_image_size(self, image_shape, filter_shape, stride, padding):
        f_height=1+(image_shape[0]+2*padding-filter_shape[0])//stride
        f_width=1+(image_shape[1]+2*padding-filter_shape[1])//stride
        return (f_height,f_width)


    def forward(self, images):
        '''
        Take a batch of images and run them through the CNN to
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
        # TODO: Implement the forward pass. This should take few lines of code.
        #############################################################################
        out1 = functional.relu(self.conv1(images))

             
        out2 = self.pool(out1)

             
        out2 = out2.view(out2.shape[0], -1)

              
        scores = functional.softmax(self.fc1(out2))

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores


class Combine(nn.Module): 
    def __init__(self, hidden_dim, kernel_size, n_classes):
        super(Combine, self).__init__()

        self.model_resnet = models.resnet18(pretrained=True)
        self.model1= nn.Sequential(*list(self.model_resnet.children())[:-3])

        self.cnn = CNN(hidden_dim, kernel_size, n_classes)


    def forward(self, images):

        N, C, H, W = images.size()
        c_in = images.view(N, C, H, W)

        with torch.no_grad():
            out = self.model1(c_in)

        out1 = F.pad(out, (0,1,0,1), mode="constant", value=0)
        
        
        c_out = self.cnn(out1)
        
        return c_out

