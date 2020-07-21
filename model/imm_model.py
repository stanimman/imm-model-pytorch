import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self,num_filter,nmaps=1,map_sizes=None,gauss_mode='gaus',):
        super(PoseEncoder, self).__init__()
        self.map_sizes = map_sizes
        self.gauss_mode = gauss_mode
        self.layers = encoder_block(num_filter)
        self.mod_layers = nn.ModuleList(self.layers)
        fin_shape = self.layers[-1][0].weight.shape[0]
        self.convf = nn.Conv2d(fin_shape,nmaps,kernel_size=1,stride=1)
    def forward(self,x):
        for layer in self.mod_layers:
            x = layer(x)
        x = self.convf(x)
        gauss_y, gauss_y_prob = self.get_coord(x,2, x.shape[1])  # B,NMAP
        gauss_x, gauss_x_prob = self.get_coord(x,1, x.shape[2])  # B,NMAP
        gauss_mu = torch.stack([gauss_y, gauss_x], axis=2)
        gauss_xy = []
        for map_size in self.map_sizes:
            gauss_xy_ = get_gaussian_maps(gauss_mu, [map_size, map_size],
                                      1.0 / 10,
                                      mode=self.gauss_mode)
        gauss_xy.append(gauss_xy_)
        return  gauss_mu,gauss_xy
    def get_coord(self,x,other_axis, axis_size):
        # get "x-y" coordinates:
        g_c_prob = torch.mean(x, axis=other_axis)  # B,W,NMAP
        g_c_prob = F.softmax(g_c_prob, dim=1)  # B,W,NMAP
        coord_pt = torch.linspace(-1.0, 1.0, axis_size) # W
        coord_pt = coord_pt.view([1, axis_size, 1])
        g_c = torch.sum(g_c_prob * coord_pt, axis=1)
        return g_c, g_c_prob



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
def get_gaussian_maps(mu, shape_hw, inv_std, mode='gaus'):
  
    mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]

    y = torch.linspace(-1.0, 1.0, shape_hw[0])

    x = torch.linspace(-1.0, 1.0, shape_hw[1])

    if (mode in ['rot', 'flat']):
        mu_y, mu_x = torch.unsqueeze(mu_y, -1), torch.unsqueeze(mu_x, -1)

        y = y.reshape([1, 1, shape_hw[0], 1])
        x = x.reshape([1, 1, 1, shape_hw[1]])

        g_y = torch.square(y - mu_y)
        g_x = torch.square(x - mu_x)
        dist = (g_y + g_x) * inv_std**2

        if mode == 'rot':
            g_yx = torch.exp(-dist)
        else:
            g_yx = torch.exp(-torch.pow(dist + 1e-5, 0.25))

    if mode == 'gaus':
        y = y.reshape([1, 1, shape_hw[0]])
        x = x.reshape([1, 1, shape_hw[1]])
        g_y = torch.exp(-torch.sqrt(1e-4+torch.abs((mu_y-y)*inv_std)))
        g_x = torch.exp(-torch.sqrt(1e-4+torch.abs((mu_x-x)*inv_std)))

        g_y = torch.unsqueeze(g_y, dim=3)
        g_x = torch.unsqueeze(g_x, dim=2)
        g_yx = torch.matmul(g_y, g_x)  # [B, NMAPS, H, W]

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    g_yx = g_yx.permute([0, 2, 3, 1])
    return g_yx

#print(PoseEncoder(32,10,[128,128]))
#summary(PoseEncoder(32,10,[128,128]), input_size=(3, 128, 128))
#PoseEncoder(32,10,[128,128]
#torch.randn((1,3, 5, 5))
#in = 
inpu = torch.randn((2,3, 128, 128))
Net = ImageEncoder(32)
out  = Net(inpu)
out.shape