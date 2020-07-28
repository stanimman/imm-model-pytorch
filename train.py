from model.imm_model import fin_model
from model.utils import perceptual_loss
from data.data_util import CelabDataset,BatchTransform
import torch 
import torch.nn as nn
import torch.optim as optim
import copy
from torch.optim import lr_scheduler
import time
import datetime as dt
# Celab  = CelabDataset(datapath = 'drive/My Drive/lmm_Model/img_align_celeba_sample',csv_file_path = 'drive/My Drive/lmm_Model/',csv_filename = 'list_landmarks_align_celeba_sample.csv')
# fin_model = lmm_model(num_filter=32,final_channel_size=3,inv_std=10,nmaps=2,map_sizes=[16],gauss_mode='gaus')
## This function just evaluate the loss / optimize  and returns model and the weight of the epoch which has highest accuracy


def train_model(model,dsts,dataloaders,optimizer, scheduler, num_epochs=5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_sizes = {"train": len(dsts['train']),"val":len(dsts['val'])}
    print(dataset_sizes)
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        # In train phase they are settting 2 variable in model class - train() and schedular  = step()
        # In Validation phase setting the model class - eval()
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.# phase - train or validation
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                inputs = inputs.type(torch.FloatTensor).permute([0,3,1,2]) # after permute shape is [B,C,H,W]
                labels = labels.type(torch.FloatTensor) 
                deformed_batchc = BatchTransform()
                deformed_batch = deformed_batchc.exe(inputs, landmarks=labels)
                im, future_im, mask = deformed_batch['image'], deformed_batch['future_image'], deformed_batch['mask'] # shape is [B,C,H,W] 
                
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    gen_img = model(im, future_im)
                    #print(outputs.shape)
                    
                    loss = perceptual_loss(gen_img, future_im)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss
                #print(phase,preds,labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss,))
            #print(preds[1:10],labels.data[1:10])
            # deep copy the model
            if phase == 'valid' and epoch_loss > best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        #print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Loss : {:4f}'.format(best_loss))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



