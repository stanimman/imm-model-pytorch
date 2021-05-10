from model.imm_model import lmm_model
from model.utils import perceptual_loss
from data.data_util import CelabDataset,BatchTransform
import torch 
import torch.nn as nn
import torch.optim as optim
import copy
from torch.optim import lr_scheduler
import time
import datetime as dt
from tqdm import tqdm
from logger import Logger
# Celab  = CelabDataset(datapath = 'drive/My Drive/lmm_Model/img_align_celeba_sample',csv_file_path = 'drive/My Drive/lmm_Model/',csv_filename = 'list_landmarks_align_celeba_sample.csv')
# fin_model = lmm_model(num_filter=32,final_channel_size=3,inv_std=10,nmaps=2,map_sizes=[16],gauss_mode='gaus')
## This function just evaluate the loss / optimize  and returns model and the weight of the epoch which has highest accuracy


def train_model(model,dsts,dataloaders,optimizer, scheduler, num_epochs=5,data_type='image',checkpoint=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_sizes = {"train": len(dsts['train']),"val":len(dsts['val'])}
    print(dataset_sizes)
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000000.0
    log_dir = 'drive/My Drive/lmm_Model/'
    if checkpoint is not None:
        start_epoch, train_it = Logger.load_cpk(checkpoint,model)
        
    else:
        start_epoch = 0
        train_it = 0
        
    with Logger(log_dir=log_dir) as logger:
      for epoch in range(start_epoch, start_epoch+num_epochs):
          print('Epoch {}/{}'.format(epoch, start_epoch+num_epochs - 1))
          print('-' * 10)
          # Each epoch has a training and validation phase
          # In train phase they are settting 2 variable in model class - train() and schedular  = step()
          # In Validation phase setting the model class - eval()
          for phase in ['train', 'val']:
              if phase == 'train':
                  #scheduler.step()
                  model.train()  # Set model to training mode
              else:
                  model.eval()   # Set model to evaluate mode
              running_loss = 0.0
              running_corrects = 0
              # Iterate over data.# phase - train or validation
              for inputs, labels in tqdm(dataloaders[phase]):
                  inputs = inputs.to(device)
                  labels = labels.to(device)
                  if device == "cpu":
                      inputs = inputs.type(torch.FloatTensor).permute([0,3,1,2]) # after permute shape is [B,C,H,W]
                      if data_type == 'image': 
                          labels = labels.type(torch.FloatTensor) 
                      if data_type == 'video': 
                          labels = labels.type(torch.FloatTensor).permute([0,3,1,2])
                  else:
                      inputs = inputs.type(torch.cuda.FloatTensor).permute([0,3,1,2]) # after permute shape is [B,C,H,W]
                      if data_type == 'image': 
                          labels = labels.type(torch.cuda.FloatTensor)
                      if data_type == 'video':
                          labels = labels.type(torch.FloatTensor).permute([0,3,1,2])
                  if data_type == 'image': 
                      deformed_batchc = BatchTransform()
                      deformed_batch = deformed_batchc.exe(inputs, landmarks=labels)
                      im, future_im, mask = deformed_batch['image'], deformed_batch['future_image'], deformed_batch['mask'] # shape is [B,C,H,W] 
                  if data_type == 'video': 
                      im, future_im = inputs,labels
                  optimizer.zero_grad()
                  with torch.set_grad_enabled(phase == 'train'):
                      gen_img = model(im, future_im)
                      loss = perceptual_loss(gen_img, future_im)
                      if phase == 'train':
                          loss.backward()
                          optimizer.step()
                  running_loss += loss
              epoch_loss = running_loss / dataset_sizes[phase]
              scheduler.step()
              print('{} Loss: {:.4f}'.format(
                  phase, epoch_loss,))
              #print(preds[1:10],labels.data[1:10])
              # deep copy the model
              if phase == 'train':
                logger.log_iter(train_it,names="train_perceptual_loss",values= epoch_loss.detach().cpu().numpy(), inp=future_im, out=gen_img)
                train_it += 1 
                logger.log_epoch(epoch,{'lmm_model':model})
                
              if phase == 'val' and epoch_loss < best_loss:
                  best_loss = epoch_loss
                  logger.log_best(epoch,{'lmm_model':model})
          #print()
      time_elapsed = time.time() - since
      print('Training complete in {:.0f}m {:.0f}s'.format(
          time_elapsed // 60, time_elapsed % 60))
      print('Best Loss : {:4f}'.format(best_loss))
      # load best model weights
      model.load_state_dict(best_model_wts)
      return model