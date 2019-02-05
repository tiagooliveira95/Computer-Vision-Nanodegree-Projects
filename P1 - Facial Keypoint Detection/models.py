## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
       
    
        # Handy Formulas
        # padding = (filter_size - 1) / 2
        # output_size =  (input_size - filer_size + 1(padding) / stride) + 1
    
        #BatchNorm
        self.batchNorm1 =nn.BatchNorm2d(32)
        self.batchNorm2 =nn.BatchNorm2d(64)
        self.batchNorm3 =nn.BatchNorm2d(128)
        self.batchNorm4 =nn.BatchNorm2d(256)
    
        #Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 2)
        
        #MaxPool Layer
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
            
        #DropOuts
        self.drop2 = nn.Dropout(p=0.2)
        self.drop5 = nn.Dropout(p=0.5)

        #Fully Connected Layers
        self.fc1 = nn.Linear(in_features = 36864, out_features = 1000)
        self.fc2 = nn.Linear(in_features = 1000, out_features = 500)
        self.fc3 = nn.Linear(in_features = 500, out_features = 136)

        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.batchNorm1(x)
                   
        x = self.pool(F.relu(self.conv2(x)))
        x = self.batchNorm2(x)

        x = self.pool(F.relu(self.conv3(x)))
        x = self.batchNorm3(x)
                
        x = self.pool(F.relu(self.conv4(x)))
        x = self.batchNorm4(x)
        
        
        x = x.view(x.size(0), -1)
      
        x = self.drop2(self.fc1(x))
       

        x = self.drop5(self.fc2(x))

        x = self.fc3(x)
     
              
        return x
