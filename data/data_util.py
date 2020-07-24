import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import pandas as pd
from PIL import Image
from skimage import io, transform


torch.manual_seed(0)

class CelabDataset(Dataset):
  def __init__(self,datapath,csv_file_path,csv_filename,data_type=None,transforms=None): #datatype is train or valid  
    filenames = os.listdir(datapath)
    self.full_filenames = [os.path.join(datapath, f) for f in filenames]
    lm_label = pd.read_csv(os.path.join(csv_file_path,csv_filename))
    self.label = np.array(lm.iloc[:,2:], dtype=np.float32)
    self.transforms = transforms

  def __len__(self):
      # return size of dataset
      return len(self.full_filenames)

  def __getitem__(self, idx):
      # returns transformed image with label
      #image = Image.open(self.full_filenames[idx])  # PIL image
      image = io.imread(self.full_filenames[idx])
      #print(image)
      if self.transforms:
            image = self.transforms(image)
      #image = self.transforms(image)
      return image, self.label[idx].reshape(-1, 2)

