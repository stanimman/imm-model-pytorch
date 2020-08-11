import os
import math
import numpy as np
import torch
import imageio
from skimage.draw import circle
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, log_dir, log_file_name='log.txt',batch_size=4,log_freq_iter=2, cpk_freq_epoch=2,
                 zfill_num=8, visualizer_params=None):

        self.loss_list = []
        self.cpk_dir = log_dir
        self.visualizations_dir = os.path.join(log_dir, 'train-vis')
        if not os.path.exists(self.visualizations_dir):
            os.makedirs(self.visualizations_dir)
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.batch_size = batch_size
        self.log_freq = log_freq_iter
        self.cpk_freq = cpk_freq_epoch
        self.zfill_num = zfill_num
        self.output_dir = 'drive/My Drive/lmm_Model/'
        self.visualizer = Visualizer()

        self.epoch = 0
        self.it = 0

    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)
        loss_string = loss_names+'-'+str(loss_mean)
        loss_string = str(self.it).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file)
        self.loss_list = []
        self.log_file.flush()

    def visualize_rec(self, inp, out):
        fut_image,gen_image = self.visualizer.visualize_reconstruction(inp, out)
        self.save_image(fut_image,'fut_img',self.it,self.output_dir,self.batch_size, 128,  padding=2)
        self.save_image(gen_image,'gen_img',self.it,self.output_dir,self.batch_size, 128,  padding=2)
        #imageio.mimsave(os.path.join(self.visualizations_dir, "%s-fut-image.png" % str(self.it).zfill(self.zfill_num)), fut_image)
        #imageio.mimsave(os.path.join(self.visualizations_dir, "%s-gen-image.png" % str(self.it).zfill(self.zfill_num)), gen_image)

    def save_cpk(self):
        cpk = {k: v.state_dict() for k, v in self.models.items()}
        cpk['epoch'] = self.epoch
        cpk['it'] = self.it
        torch.save(cpk, os.path.join(self.cpk_dir, '%s-checkpoint.pth.tar' % str(self.epoch).zfill(self.zfill_num)))

    @staticmethod
    def load_cpk(checkpoint_path, model=None):
        checkpoint = torch.load(checkpoint_path)
        if model is not None:
            model.load_state_dict(checkpoint['lmm_model'])
        return checkpoint['epoch'], checkpoint['it']

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'models' in self.__dict__:
            self.save_cpk()
        self.log_file.close()

    def log_iter(self, it, names, values, inp, out):
        self.it = it
        self.names = names
        self.loss_list.append(values)
        if it % self.log_freq == 0:
            self.log_scores(self.names)
            self.visualize_rec(inp, out)

    def log_epoch(self, epoch, models):
        self.epoch = epoch
        self.models = models
        if epoch % self.cpk_freq == 0:
            self.save_cpk()

    def log_best(self, epoch, models):
        self.epoch = epoch
        self.best_models = models
        if epoch % self.cpk_freq == 0:
            self.save_best_cpk()

    def save_best_cpk(self):
        cpk = {k: v.state_dict() for k, v in self.best_models.items()}
        cpk['epoch'] = self.epoch
        cpk['it'] = self.it
        torch.save(cpk, os.path.join(self.cpk_dir, '%s-best_model_checkpoint.pth.tar' % str(self.epoch).zfill(self.zfill_num)))
  
    def save_image(self,data,image_name, it, output_dir,batch_size=4,image_size=128, padding=2):
        """ save image """
        #data = data.detach().cpu().numpy()
        datanp = np.clip(
            (data - np.min(data))*(255.0/(np.max(data) - np.min(data))), 0, 255).astype(np.uint8)
        x_dim = min(8, batch_size)
        y_dim = int(math.ceil(float(batch_size) / x_dim))
        height, width = int(image_size + padding), int(image_size + padding)
        grid = np.zeros((height * y_dim + 1 + padding // 2, width *
                        x_dim + 1 + padding // 2, 3), dtype=np.uint8)
        k = 0
        for y in range(y_dim):
            for x in range(x_dim):
                if k >= batch_size:
                    break
                start_y = y * height + 1 + padding // 2
                end_y = start_y + height - padding
                start_x = x * width + 1 + padding // 2
                end_x = start_x + width - padding
                np.copyto(grid[start_y:end_y, start_x:end_x, :], datanp[k])
                k += 1
        imageio.imwrite(
            '{}/epoch_{}_{}.png'.format(output_dir, it,image_name), grid) 

class Visualizer:
    def __init__(self, kp_size=2, draw_border=False, colormap='gist_rainbow'):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)


    def visualize_reconstruction(self, inp, out):
        out = out.detach().cpu().numpy().transpose((0, 2, 3, 1))
        inp = inp.detach().cpu().numpy().transpose((0, 2, 3, 1))
        
        return inp,out