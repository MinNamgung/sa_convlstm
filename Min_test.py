
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import optim
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import random
from torch.utils.data import Subset

# from data_loader import MovingMNIST

import utils.utils as utils
from skimage.metrics import structural_similarity as ssim
from data_loader.data_loader_MovingMNIST import MovingMNIST
from model.Encode2Decode import Encode2Decode
from model.seq2seq import EncoderDecoderConvLSTM


def split_train_val(dataset):
    idx = [i for i in range(len(dataset))]
#     print(f'idx: {len(dataset)}')
#     idx = [i for i in range(30)]

    random.seed(1234)
    random.shuffle(idx)

    num_train = int(0.8 * len(idx))
    num_val = int(0.2 * len(idx))
#     num_test = int(0.2 * len(idx))
#     num_val = int(0.2 * num_train)

    train_idx = idx[:num_train]
#     test_idx = idx[num_train: (num_train + num_test)]
    val_idx = idx[num_train:]

    train_dataset = Subset(dataset, train_idx)
#     test_dataset = Subset(dataset, test_idx)
    val_dataset = Subset(dataset, val_idx)

    return train_dataset, val_dataset


def reshape_patch(images, patch_size):
    bs = images.size(0) 
    nc = images.size(1) 
    height = images.size(2) 
    weight = images.size(3)    
    x = images.reshape(bs, nc, int(height / patch_size), patch_size, int(weight / patch_size), patch_size)
    
    x = x.transpose(2, 5)
    x = x.transpose(4, 5)
    x = x.reshape(bs, nc * patch_size * patch_size, int(height / patch_size), int(weight / patch_size))

    return x

class Model():
    def __init__(self, params, loading_path = None, set_device=4):
        if params['model_cell'] == 'sa_convlstm':
            self.model = Encode2Decode(params).to(params['device'])
        else: # params['model_cell'] == 'convlstm':
            self.model = EncoderDecoderConvLSTM(params).to(params['device'])
        self.loss = params['loss']
        if self.loss == 'SSIM':
            # self.criterion = SSIM().to(device)
            self.criterion = nn.MSELoss()
        elif self.loss == 'L2':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.L1Loss()
        self.output = params['output_dim']
        self.device = params['device']
        self.optim = optim.Adam(self.model.parameters(), lr=params['lr'])
    
    def evaluate(self, train_dataset, path):
        prediction = []
        test_groundtruth = []
        
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
#         self.model.load_state_dict(torch.load(path))
        
        train_pred, train_gt = [], []
        
        with torch.no_grad():
            self.model.eval()
            for i, data in enumerate(train_dataset):
                x, y = data
                x = x.to(self.device)
                y = y.to(self.device) * 255.0
                pred = self.model(x, y) * 255.0
                train_pred.append(pred.cpu().data.numpy())
                train_gt.append(y.cpu().data.numpy())
        
        train_pred = np.concatenate(train_pred)
        train_gt = np.concatenate(train_gt)
        
        path_pred = './npy_file_save/train_pred_Feb20.npy'
        path_gt = './npy_file_save/train_gt_Feb20.npy'

        np.save(path_pred, train_pred)
        np.save(path_gt, train_gt)
        
        return prediction
    
    
    def evaluate_test(self, test_dataset, path):
        prediction = []
        test_groundtruth = []
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
#         self.model.load_state_dict(torch.load(path))
        

        test_pred, test_gt = [], []
    
        with torch.no_grad():
            self.model.eval()
            for i, data in enumerate(test_dataset):
                x, y = data
                x = x.to(self.device)
                y = y.to(self.device) 
                pred = self.model(x, y, teacher_forcing_rate = 0) 
                test_pred.append(pred.cpu().data.numpy())
                test_gt.append(y.cpu().data.numpy())
        
        test_pred = np.concatenate(test_pred)
        test_gt = np.concatenate(test_gt)

        mse = utils.mse(test_gt, test_pred)
        print('TEST Data loader - MSE = {:.6f}'.format(mse))
        
        # Frame-wise comparison in MSE and SSIM 

        overall_mse = 0
        overall_ssim = 0
        frame_mse = np.zeros(test_gt.shape[1])
        frame_ssim = np.zeros(test_gt.shape[1])

        for i in range(test_gt.shape[1]):
            for j in range(test_gt.shape[0]):
                
                mse_ = np.square(test_gt[j,i] - test_pred[j,i]).sum()
                test_gt_img = np.squeeze(test_gt[j,i])
                test_pred_img = np.squeeze(test_pred[j,i])
                ssim_ = ssim(test_gt_img, test_pred_img)

                overall_mse += mse_
                overall_ssim += ssim_ 
                frame_mse[i] += mse_
                frame_ssim[i] += ssim_
                
        overall_mse /= 10
        overall_ssim /= 10
        frame_mse /= 1000
        frame_ssim /= 1000
        print(f'overall_mse.shape {overall_mse}')
        print(f'overall_ssim.shape {overall_ssim}')
        print(f'frame_mse.shape {frame_mse}')
        print(f'frame_ssim.shape {frame_ssim}')
        
        
        
        path_pred = './results/npy_file_save/saconvlstm_test_pred_speedpt5.npy'
        path_gt = './results/npy_file_save/saconvlstm_test_gt_speedpt5.npy'

        np.save(path_pred, test_pred)
        np.save(path_gt, test_gt)
        
        return prediction
    


if __name__ == '__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    root = '/panfs/roc/groups/14/yaoyi/namgu007/weather4cast-master/sa-convlstm-movingmnist'
    dataset = MovingMNIST(root, train=False) # For test dataset
    
    dataset2 = MovingMNIST(root, train=True) # For train/val dataset
    train_dataset, val_dataset = split_train_val(dataset2) 
    

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    valid_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0) 
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0) # Shuffle = False
    
    

    torch.manual_seed(42)
    BATCH_SIZE = 8
    img_size = (16, 16)
    new_size = (16, 16)
    strides = img_size
    input_window_size, output = 10, 10
    epochs = 1
    lr = 1e-3
    hid_dim = 16 # 16
    loss = 'L2' 
    att_hid_dim = 64
    n_layers = 4
    bias = True
    
    # 1) sa_convlstm
    # 2) convlstm (else for now)
    model_name = 'sa_convlstm'

    params = {'input_dim': 1, 'batch_size': BATCH_SIZE, 'padding': 1, 'lr': lr, 'device': device,
              'att_hidden_dim': att_hid_dim, 'kernel_size': 3, 'img_size': img_size, 'hidden_dim': hid_dim,
              'n_layers': n_layers, 'output_dim': output, 'input_window_size': input_window_size, 'loss': loss,
              'model_cell': model_name, 'bias': bias}
    
    print(f'Moving Mnist Image size (64 to 64) by processing reducing image to {img_size}')
    print('data has been loaded!')
    print('This is Test.py')
    print(f'Model name: {model_name}')
    
    model = Model(params)

#     path = 'weather4cast-master/sa-convlstm-movingmnist/reconstruction/results/model_save/model_sa_convlstm_10to10_BS8_1000epochs_4layers_64atthid_L2loss_64hid.pt'
    
#     path = 'checkpoint.pt' # SA-Convlstm
    path = 'model_sa_convlstm_10to10_BS8_200epochs_4layers_64atthid_L2loss_16hid.pt'
    prediction_test = model.evaluate_test(test_dataloader, path)