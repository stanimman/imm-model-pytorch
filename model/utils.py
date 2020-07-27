
import torch
from torchvision import models
import torch.nn.functional as F 


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = vgg_pretrained_features[:3] #conv1_2
        self.slice2 = vgg_pretrained_features[3:7] #conv2_2
        self.slice3 = vgg_pretrained_features[7:10] #conv3_2
        self.slice4 = vgg_pretrained_features[10:12] #conv4_2
        self.slice5 = vgg_pretrained_features[12:14] #conv5_2
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_conv1_2 = h
        h = self.slice2(h)
        h_conv2_2 = h
        h = self.slice3(h)
        h_relu3_2 = h
        h = self.slice4(h)
        h_relu4_2 = h
        h = self.slice5(h)
        h_relu5_2 = h
        
        out = [X,h_conv1_2, h_conv2_2, h_relu3_2, h_relu4_2, h_relu5_2]
        return out

def perceptual_loss(fut_im,gen_im,mse=True):
  fu_Vgg19_Net  = Vgg19()
  gen_Vgg19_Net  = Vgg19()
  fut_im_Vgg19,gen_im_Vgg19  = fu_Vgg19_Net(fut_im),gen_Vgg19_Net(gen_im)
  #https://openaccess.thecvf.com/content_ICCV_2017/papers/Chen_Photographic_Image_Synthesis_ICCV_2017_paper.pdf
  ws = [4.9,10.4,5.2,1.3,2.6,2.6]
  batch_size = gen_im_Vgg19[0].shape[0]
  loss = 0
  for i in range(len(gen_im_Vgg19)):
    if mse:
      l = F.mse_loss(fut_im_Vgg19[i],gen_im_Vgg19[i])
    else:
      l = torch.abs(fut_im_Vgg19[i]-gen_im_Vgg19[i])
    #print(l)
    l = (ws[i]*l) / batch_size
    l = torch.sum(l)
    loss = loss + l
  return loss