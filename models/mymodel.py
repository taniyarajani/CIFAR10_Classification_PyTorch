import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional


class MyModel(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
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
        self.conv1 = nn.Conv2d(im_size[0], hidden_dim, kernel_size, stride = 1, padding = 1)
        final_image_dim1 = self.compute_image_size(im_size[1:], (kernel_size, kernel_size), 1, 1)

        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        final_image_dim2 = self.compute_image_size(final_image_dim1, (2,2), 2, 0)

        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size, stride = 1, padding = 1)
        final_image_dim3 = self.compute_image_size(final_image_dim2, (kernel_size, kernel_size), 1, 1)

        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        final_image_dim4 = self.compute_image_size(final_image_dim3, (2,2), 2, 0)

        self.conv3 = nn.Conv2d(hidden_dim*2, hidden_dim*2*2, kernel_size, stride = 1, padding = 1)
        final_image_dim5 = self.compute_image_size(final_image_dim4, (kernel_size, kernel_size), 1, 1)

        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        final_image_dim6 = self.compute_image_size(final_image_dim5, (2,2), 2, 0)
     
        self.fc1 = torch.nn.Linear(hidden_dim*2*2*final_image_dim6[0]*final_image_dim6[1], hidden_dim*5)

        self.drop_layer1 = nn.Dropout(p=0.2)

        self.fc2 = torch.nn.Linear(hidden_dim*5, 50)

        self.drop_layer2 = nn.Dropout(p=0.2)

        self.fc3 = torch.nn.Linear(50, n_classes)


        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def compute_image_size(self, image_shape, filter_shape, stride, padding):
        f_height=1+(image_shape[0]+2*padding-filter_shape[0])//stride
        f_width=1+(image_shape[1]+2*padding-filter_shape[1])//stride
        return (f_height,f_width)


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
        out1 = functional.relu(self.conv1(images))
     
        out2 = self.pool1(out1)

        out3 = functional.relu(self.conv2(out2))

        out4 = self.pool2(out3)

        out5 = functional.relu(self.conv3(out4))

        out6 = self.pool3(out5)
     
        out6 = out6.view(out6.shape[0], -1)
      
        out7 = functional.relu(self.fc1(out6))

        out7 = self.drop_layer1(out7)

        out8 = functional.relu(self.fc2(out7))

        out8 = self.drop_layer2(out8)

        scores = functional.softmax(self.fc3(out8))

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

