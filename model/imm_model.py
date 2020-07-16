import torch
import torch.nn as nn
from torchsummary import summary


class Encoder(nn.Module):
    def __init__(self,num_filter):
        super(Encoder, self).__init__()
        self.filter = num_filter
        self.layers = []
        self.conv1 = conv_block(ni=3,nf = self.filter,kernal=7,stride=1,pad=(3,3))
        self.conv2 = conv_block(ni= self.filter,nf=self.filter,kernal=3,stride=1,pad=(1,1))
        self.filter1 = self.filter*2
        self.conv3 = conv_block(ni=self.filter,nf=self.filter1,kernal=3,stride=2,pad=(1,1))
        self.conv4 = conv_block(ni=self.filter1,nf=self.filter1,kernal=3,stride=1,pad=(1,1))
        self.filter2 = self.filter1*2
        self.conv5 = conv_block(ni=self.filter1,nf=self.filter2,kernal=3,stride=2,pad=(1,1))
        self.conv6 = conv_block(ni=self.filter2,nf=self.filter2,kernal=3,stride=1,pad=(1,1))
        self.filter3 = self.filter2*2
        self.conv7 = conv_block(ni=self.filter2,nf=self.filter3,kernal=3,stride=2,pad=(1,1))
        self.conv8 = conv_block(ni=self.filter3,nf=self.filter3,kernal=3,stride=1,pad=(1,1))
        self.layers = [self.conv1,self.conv2,self.conv3,self.conv4,self.conv5,self.conv6,self.conv7,self.conv8]
    
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x 


def encoder():
    print("WIP")

def conv_block (ni,nf,kernal,stride=1,pad=(3,3),batch_norm=True,activation='ReLu'):
    conv_block =  nn.Sequential(
    nn.Conv2d(in_channels=ni,out_channels=nf,kernel_size=kernal,stride=stride,padding=pad),
    nn.BatchNorm2d(nf),
    nn.ReLU()
    )
    return conv_block

device = torch.device("cpu")

#with torch.no_grad():
#    y= Encoder(torch.zeros(3,128,128, device=device))
#print(y.shape)
print(Encoder(32))
summary(Encoder(32), input_size=(3, 128, 128))
