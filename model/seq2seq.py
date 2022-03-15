import torch
import torch.nn as nn
import random

from model.convLSTM_cell import ConvLSTMCell


###
#
# Author: Min Namgung
# Contact: namgu007@umn.edu
#
# ###

class EncoderDecoderConvLSTM(nn.Module):
    def __init__(self, params):
        super(EncoderDecoderConvLSTM, self).__init__()

        self.in_chan = params['hidden_dim']
        self.h_chan = params['hidden_dim']
        self.out_chan = params['hidden_dim']
        self.input_window_size = params['input_window_size']
        self.n_layers = params['n_layers']
        self.img_size = params['img_size']
        self.batch_size = params['batch_size']
        self.device = params['device']
        self.cells = []
        self.h_t, self.c_t = [], []

        for i in range(self.n_layers):
            self.cells.append(ConvLSTMCell(in_channels=self.in_chan,
                                           h_channels=self.h_chan,
                                           kernel_size=(3, 3),
                                           bias=True))
            # self.bns.append(nn.LayerNorm((params['hidden_dim'], 16, 16)))  # Use layernorm
        
        self.cells = nn.ModuleList(self.cells)
        
        self.img_encode = nn.Sequential(
            nn.Conv2d(in_channels=params['input_dim'], kernel_size=1, stride=1, padding=0,
                      out_channels=params['hidden_dim']),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=params['hidden_dim'], kernel_size=3, stride=2, padding=1,
                      out_channels=params['hidden_dim']),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=params['hidden_dim'], kernel_size=3, stride=2, padding=1,
                      out_channels=params['hidden_dim']),
            nn.LeakyReLU(0.1)
        )

        # Prediction layer
        self.img_decode = nn.Sequential(
            nn.ConvTranspose2d(in_channels=params['hidden_dim'], kernel_size=3, stride=2, padding=1, output_padding=1,
                               out_channels=params['hidden_dim']),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(in_channels=params['hidden_dim'], kernel_size=3, stride=2, padding=1, output_padding=1,
                               out_channels=params['hidden_dim']),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=params['hidden_dim'], kernel_size=1, stride=1, padding=0,
                      out_channels=params['input_dim'])
        )

        # Linear
        self.decoder_predict = nn.Conv2d(in_channels=params['hidden_dim'],
                                         out_channels=params['hidden_dim'],
                                         kernel_size=(1, 1),
                                         padding=(0, 0))

    def forward(self, x, y, teacher_forcing_rate=0.5, hidden=None):
        # Init hidden weight
        if hidden == None:
            hidden = self.init_hidden(batch_size=self.batch_size, img_size=self.img_size)


        b, seq_len, x_c, h, w = x.size()
        _, future_seq, y_c, _, _ = y.size()

        frames = torch.cat([x, y], dim=1)
        predict_temp_de = []

        # Seq2seq
        for t in range(19):

            if t < self.input_window_size or random.random() < teacher_forcing_rate:
                x = frames[:, t, :, :, :]
            else:
                x = out
            
            x = self.img_encode(x)

            for i, cell in enumerate(self.cells):

                if i == 0:
                    # hid = cell(input_tensor=x, cur_state=[hid[0], hid[1]])
                    hidden[i] = cell(input_tensor=x, cur_state=hidden[i])
                    out = self.decoder_predict(hidden[i][0])

                else:
                    hidden[i] = cell(input_tensor=x, cur_state=hidden[i])
                    out = self.decoder_predict(hidden[i][0])

            out = self.img_decode(out)
            predict_temp_de.append(out)

        predict_temp_de = torch.stack(predict_temp_de, dim=1)
        final = predict_temp_de[:, 9:, :, :, :]

        return final


    def init_hidden(self, batch_size, img_size):
        states = []
        for i in range(self.n_layers):
            states.append(self.cells[i].init_hidden(batch_size, img_size))

        return states

