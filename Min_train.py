import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import optim
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import random
from torch.utils.data import Subset

from utils.earlystopping import EarlyStopping
from data_loader.data_loader_MovingMNIST import MovingMNIST
from model.Encode2Decode import Encode2Decode
from model.seq2seq import EncoderDecoderConvLSTM


def split_train_val(dataset):
    idx = [i for i in range(len(dataset))]

    random.seed(1234)
    random.shuffle(idx)

    num_train = int(0.8 * len(idx))
    num_val = int(0.2 * len(idx))

    train_idx = idx[:num_train]
    val_idx = idx[num_train:]

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    
    print(f'train index: {len(train_idx)}')
    print(f'val index: {len(val_idx)}')

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


def reshape_patch_back(images, patch_size):
    bs = images.size(0)
    nc = int(images.size(1) / (patch_size * patch_size))
    height = images.size(2)
    weight = images.size(3)
    x = images.reshape(bs, nc, patch_size, patch_size, height, weight)
    x = x.transpose(4, 5)
    x = x.transpose(2, 5)
    x = x.reshape(bs, nc, height * patch_size, weight * patch_size)

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

    def train(self, train_dataset, valid_dataset, epochs, path):
        min_loss = 1e9
        n_total_steps = len(train_dataset)
        train_losses = []


        early_stopping = EarlyStopping(patience=20, verbose=True)
        avg_train_losses = []
        avg_valid_losses = []

        for i in range(epochs):
            losses, val_losses = [], []
            self.model.train()
            epoch_loss = 0.0
            val_epoch_loss = 0.0

            for ite, data in enumerate(train_dataset):
                x, y = data
                
                x_train = x.to(self.device)
                y_train = y.to(self.device)
                self.optim.zero_grad()
                pred_train = self.model(x_train, y_train)
                loss = self.criterion(pred_train, y_train)
                loss.backward()
                self.optim.step()

                train_losses.append(loss.item())
                print(f'epoch {i + 1} / {epochs}, step {ite + 1}/{n_total_steps}, encode + decode loss = {loss.item():.4f}')
                
                epoch_loss += loss.item()

            with torch.no_grad():
                self.model.eval()
                for _, data in enumerate(valid_dataset):
                    x, y = data
                    x_val = x.to(self.device)
                    y_val = y.to(self.device)
                    pred_val = self.model(x_val, y_val, teacher_forcing_rate = 0)
                    loss = self.criterion(pred_val, y_val)
                    val_losses.append(loss.item())
                    val_epoch_loss += loss.item()

            train_loss = np.average(train_losses)
            valid_loss = np.average(val_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            
            print('{}th epochs train loss {}, valid loss {}'.format(i, np.mean(train_loss), np.mean(valid_loss)))
            
            torch.save(self.model.state_dict(), path)
    
#             Uncomment for applying early stopping
            early_stopping(valid_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            plt.plot(avg_train_losses, '-o')
            plt.plot(avg_valid_losses, '-o')
            plt.xlabel('epoch')
            plt.ylabel('losses')
            plt.legend(['Train', 'Validation'])
            plt.title('(MSE) Avg Train vs Validation Losses')
            plt.savefig('./results/npy_file_save/test_new_trainer.png')
            plt.clf()


if __name__ == '__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    root = '/panfs/roc/groups/14/yaoyi/namgu007/weather4cast-master/sa-convlstm-movingmnist'
    dataset = MovingMNIST(root, train=True)

    train_dataset, val_dataset = split_train_val(dataset)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    valid_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0) 


    torch.manual_seed(42)
    BATCH_SIZE = 8
    img_size = (16, 16)
    new_size = (16, 16)
    strides = img_size
    input_window_size, output = 10, 10
    epochs = 2
    lr = 1e-3
    hid_dim = 64 # 16
    loss = 'L2' 
    att_hid_dim = 64
    n_layers = 4
    bias = True
    
    # 1) sa-convlstm
    # 2) convlstm (else for now)
    model_name = 'sa_convlstm'

    params = {'input_dim': 1, 'batch_size': BATCH_SIZE, 'padding': 1, 'lr': lr, 'device': device,
              'att_hidden_dim': att_hid_dim, 'kernel_size': 3, 'img_size': img_size, 'hidden_dim': hid_dim,
              'n_layers': n_layers, 'output_dim': output, 'input_window_size': input_window_size, 'loss': loss,
              'model_cell': model_name, 'bias': bias}
    
    print(f'Moving Mnist Image size (64 to 64) by processing reducing image to {img_size}')
    print('data has been loaded!')
    print('This is Train.py')
    print(f'Model name: {model_name}')
    
    model = Model(params)
    
    model.train(train_dataloader, valid_dataloader, epochs = epochs, path = './results/model_save/model_{}_{}to{}_BS{}_{}epochs_{}layers_{}atthid_{}loss_{}hid.pt'.format(model_name, input_window_size, output, BATCH_SIZE, epochs, n_layers, att_hid_dim, loss, hid_dim))

