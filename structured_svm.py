import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from GLOBALS import * 

class SeeInDark_Structured_SVM(nn.Module):
    def __init__(self):
        super(SeeInDark_Structured_SVM, self).__init__()
        
        self.conv1_1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        
        self.upv6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        self.upv7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        self.upv8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.upv9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        
        self.conv10_1 = nn.Conv2d(32, 12, kernel_size=1, stride=1)
        
        self.outty = nn.Conv2d(1, 256, kernel_size=3, stride=1, padding=1)
        
        self.lin_concat = nn.Conv2d(512*3, 3, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
#         print('Forward pass')
        conv1 = self.lrelu(self.conv1_1(x))
        conv1 = self.lrelu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)
        
        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)
        
        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)
        
        conv4 = self.lrelu(self.conv4_1(pool3))
        conv4 = self.lrelu(self.conv4_2(conv4))
        pool4 = self.pool1(conv4)
        
        conv5 = self.lrelu(self.conv5_1(pool4))
        conv5 = self.lrelu(self.conv5_2(conv5))
        
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        conv6 = self.lrelu(self.conv6_2(conv6))
        
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        conv7 = self.lrelu(self.conv7_2(conv7))
        
        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.lrelu(self.conv8_1(up8))
        conv8 = self.lrelu(self.conv8_2(conv8))
        
        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.lrelu(self.conv9_1(up9))
        conv9 = self.lrelu(self.conv9_2(conv9))
        conv10= self.conv10_1(conv9)
        
        out = nn.functional.pixel_shuffle(conv10, 2)
        
        # (unary_potentials, rightNeighbor0, rightNeighbor1, rightNeighbor2, 
        #  downNeighbor0, downNeighbor1, downNeighbor2)
        experimental = self.experimental(out)

        return experimental

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

    def lrelu(self, x):
        outt = torch.max(0.2*x, x)
        return outt
    
    def experimental(self, out):
#         print('Shape of out: ' + str(out.shape))
        rgb = torch.split(out, 1, dim = 1)
        rgb = list(rgb)
#         print('Splitted in to: ' + str(len(rgb)) + ' partitions')
#         print('The first 0th partition has the shape: ' + str(rgb[0].shape))
        
        # Pass through convolution layer
        color0 = self.outty(rgb[0])
        color1 = self.outty(rgb[1])
        color2 = self.outty(rgb[2])
#         print('Convolution Layer of 0th has the shape: ' + str(color0.shape))
        
        # Take the softmax
        color0 = F.softmax(color0, dim = 1)
        color1 = F.softmax(color1, dim = 1)
        color2 = F.softmax(color2, dim = 1)
        #print('After softmax, Convolution Layer of 0th has the shape: ' + str(color0.shape))
        
        # Take the max
        u1, _ = torch.max(color0, dim = 1)
        u2, _ = torch.max(color1, dim = 1)
        u3, _ = torch.max(color2, dim = 1)
        result = torch.cat((u1, u2, u3), 0)
        unary_potentials = result.unsqueeze(0).float()
        
        # Concatenating Neighboring pixels of same color (To the right)
        edge_width = 2*ps - 1

        # Color 0
        currentPixelR0 = color0[0, :, :edge_width, :]
        rightPixel0 = color0[0, :, 1:, :]
        rightNeighbor0 = torch.cat((currentPixelR0, rightPixel0))
       
        # Color 1
        currentPixelR1 =  color1[0, :, :edge_width, :]
        rightPixel1 = color1[0, :, 1:, :]
        rightNeighbor1 = torch.cat((currentPixelR1, rightPixel1), 0)

        # Color 2
        currentPixelR2 = color2[0, :, :edge_width, :]
        rightPixel2 = color2[0, :, 1:, :]
        rightNeighbor2 = torch.cat((currentPixelR2, rightPixel2), 0)

        # Concatenating Neighboring pixels of same color (To below)
        # Color 0
        currentPixelD0 = color0[0, :, :, :edge_width]
        downPixel0 = color0[0, :, :, 1:]
        downNeighbor0 = torch.cat((currentPixelD0, downPixel0), 0)
        
        # Color 1
        currentPixelD1 = color1[0, :, :, :edge_width]
        downPixel1 = color1[0, :, :, 1:]
        downNeighbor1 = torch.cat((currentPixelD1, downPixel1), 0)
        
        # Color 2
        currentPixelD2 = color2[0, :, :, :edge_width]
        downPixel2 = color2[0, :, :, 1:]
        downNeighbor2 = torch.cat((currentPixelD2, downPixel2), 0)
        
        right_neighbors = torch.cat((rightNeighbor0, rightNeighbor1, rightNeighbor2)).unsqueeze(0)
        right_neighbors = self.lin_concat(right_neighbors).squeeze()
        
        down_neighbors = torch.cat((downNeighbor0, downNeighbor1, downNeighbor2)).unsqueeze(0)
        down_neighbors = self.lin_concat(down_neighbors).squeeze()
        return (unary_potentials, right_neighbors, down_neighbors)
                