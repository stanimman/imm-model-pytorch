import torch
import torch.nn as nn
from torchsummary import summary


class ImageEncoder(nn.Module):
    def __init__(self,num_filter):
        super(ImageEncoder, self).__init__()
        self.layers = encoder_block(num_filter)
        self.mod_layers = nn.ModuleList(self.layers)
    def forward(self,x):
        for layer in self.mod_layers:
            x = layer(x)
        return x 

class PoseEncoder(nn.Module):
    def __init__(self,num_filter,nmaps=1):
        super(PoseEncoder, self).__init__()
        self.layers = encoder_block(num_filter)
        self.mod_layers = nn.ModuleList(self.layers)
        fin_shape = self.layers[-1][0].weight.shape[0]
        self.convf = nn.Conv2d(fin_shape,nmaps,kernel_size=1,stride=1)
        #self.convf = nn.conv2d()
    def forward(self,x):
        for layer in self.mod_layers:
            x = layer(x)
        x = self.convf(x)
        return x 


def conv_block (ni,nf,kernal,stride=1,pad=(3,3),batch_norm=True,activation='ReLu'):
    conv_block =  nn.Sequential(
    nn.Conv2d(in_channels=ni,out_channels=nf,kernel_size=kernal,stride=stride,padding=pad),
    nn.BatchNorm2d(nf),
    nn.ReLU()
    )
    return conv_block

def encoder_block(num_filter):
    filter = num_filter
    layers = []
    conv1 = conv_block(ni=3,nf =filter,kernal=7,stride=1,pad=(3,3))
    conv2 = conv_block(ni=filter,nf=filter,kernal=3,stride=1,pad=(1,1))
    filter1 = filter*2
    conv3 = conv_block(ni=filter,nf=filter1,kernal=3,stride=2,pad=(1,1))
    conv4 = conv_block(ni=filter1,nf=filter1,kernal=3,stride=1,pad=(1,1))
    filter2 = filter1*2
    conv5 = conv_block(ni=filter1,nf=filter2,kernal=3,stride=2,pad=(1,1))
    conv6 = conv_block(ni=filter2,nf=filter2,kernal=3,stride=1,pad=(1,1))
    filter3 = filter2*2
    conv7 = conv_block(ni=filter2,nf=filter3,kernal=3,stride=2,pad=(1,1))
    conv8 = conv_block(ni=filter3,nf=filter3,kernal=3,stride=1,pad=(1,1))
    layers = [conv1,conv2,conv3,conv4,conv5,conv6,conv7,conv8]
    return layers
device = torch.device("cpu")

#with torch.no_grad():
#    y= Encoder(torch.zeros(3,128,128, device=device))
#print(y.shape)
print(PoseEncoder(32,10))
summary(PoseEncoder(32,10), input_size=(3, 128, 128))
