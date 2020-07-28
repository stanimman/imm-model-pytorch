import yaml
from argparse import ArgumentParser
from model.imm_model import lmm_model
from data.data_util import CelabDataset,BatchTransform
from train import train_model
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch.optim import lr_scheduler


parser = ArgumentParser()
parser.add_argument("--config", required=True, help="path to config")
opt = parser.parse_args()
with open(opt.config) as f:
    config = yaml.load(f)

# Model Parameters

model_param  = config['model']
num_filter = model_param['n_filters']
final_channel_size = model_param['final_channel_size']
inv_std = model_param['inv_std']
n_maps = model_param['n_maps']
map_sizes = model_param['map_sizes']
gauss_mode = model_param['gauss_mode']

# Data parameters

data_param = config['training']['train_dset_params']
train_data_path = data_param['train_datadir']
train_csv_file_path = data_param['train_datalabeldir']
train_csv_filename = data_param['train_datalabelcsv']

valid_data_path = data_param['valid_datadir']
valid_csv_file_path = data_param['valid_datalabeldir']
valid_csv_filename = data_param['valid_datalabelcsv']

train_celeb_ds  = CelabDataset(datapath = train_data_path,
                csv_file_path = train_csv_file_path,
                csv_filename = train_csv_filename)
valid_celeb_ds  = CelabDataset(datapath = valid_data_path,
                    csv_file_path = valid_csv_file_path,
                    csv_filename = valid_csv_filename)

batch = config['training']['batch']


train_dl = torch.utils.data.DataLoader(train_celeb_ds,batch_size=batch, shuffle=True)
valid_dl = torch.utils.data.DataLoader(valid_celeb_ds,batch_size=batch, shuffle=True)
#for image in train_dl:
#    print(image.size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dsts  = {"train": train_celeb_ds, "val": valid_celeb_ds}
dataloaders = {"train": train_dl, "val": valid_dl}
    
model = lmm_model(num_filter=num_filter,final_channel_size=final_channel_size,inv_std=inv_std,nmaps=n_maps,map_sizes=map_sizes,gauss_mode=gauss_mode)
# Neural Net Parameters

n_epoch = config['training']['n_epoch']
lr = config['training']['lr']['start_val']
wts_decay = config['training']['lr']['decay']
step_sz = config['training']['lr']['step']

optimizer_ft = optim.Adam(list(filter(lambda p: p.requires_grad, model.parameters())), 
                lr = lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=wts_decay, amsgrad=False)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_sz, gamma=0.1)
model_ft = train_model(model, dsts,dataloaders,optimizer_ft, exp_lr_scheduler,
                       num_epochs=n_epoch)
