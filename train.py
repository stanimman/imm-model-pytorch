from model.imm_model import fin_model
from model.utils import perceptual_loss
from data.data_util import CelabDataset,BatchTransform

# Celab  = CelabDataset(datapath = 'drive/My Drive/lmm_Model/img_align_celeba_sample',csv_file_path = 'drive/My Drive/lmm_Model/',csv_filename = 'list_landmarks_align_celeba_sample.csv')
# fin_model = lmm_model(num_filter=32,final_channel_size=3,inv_std=10,nmaps=2,map_sizes=[16],gauss_mode='gaus')
def train():
    x = 2
    print(x)
    encoder()
    loss()



