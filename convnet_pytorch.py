"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
import torch
import torch.nn as nn
class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes ):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODO:
    Implement initialization of the network.
    """
    super(ConvNet, self).__init__()

    self.conv1 = nn.Sequential(
      nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU()
    )
    self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    self.conv2 = nn.Sequential(
      nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU()
    )
    self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    self.conv3_a = nn.Sequential(
      nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU()
    )
    self.conv3_b = nn.Sequential(
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU()
    )
    self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    self.conv4_a = nn.Sequential(
      nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU()
    )
    self.conv4_b = nn.Sequential(
      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU()
    )
    self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    self.conv5_a = nn.Sequential(
      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU()
    )
    self.conv5_b = nn.Sequential(
      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU()
    )
    self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    self.fc = nn.Sequential(
      nn.Linear(512 * 1 * 1, 512),
      nn.ReLU(),
      nn.Linear(512, n_classes)
    )
    # ########################
    # # PUT YOUR CODE HERE  #
    # #######################
    # raise NotImplementedError
    # ########################
    # # END OF YOUR CODE    #
    # #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """
    x = self.conv1(x)
    x = self.maxpool1(x)
    
    x = self.conv2(x)
    x = self.maxpool2(x)
    
    x = self.conv3_a(x)
    x = self.conv3_b(x)
    x = self.maxpool3(x)
    
    x = self.conv4_a(x)
    x = self.conv4_b(x)
    x = self.maxpool4(x)
    
    x = self.conv5_a(x)
    x = self.conv5_b(x)
    x = self.maxpool5(x)
    
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    
    return x
    # # ########################
    # # # PUT YOUR CODE HERE  #
    # # #######################
    # # raise NotImplementedError
    # # ########################
    # # # END OF YOUR CODE    #
    # # #######################

    # return out
