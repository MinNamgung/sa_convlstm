from model.sa_convLSTM_cell import SA_Convlstm_cell

import torch
import torch.nn as nn
import random

###
#
# Author: Min Namgung
# Contact: namgu007@umn.edu
#
# ###

class Encode2Decode(nn.Module):  # self-attention convlstm for spatiotemporal prediction model
    def __init__(self, params):
        super(Encode2Decode, self).__init__()
        # hyperparams
        self.batch_size = params['batch_size']
        self.img_size = params['img_size']
        self.cells, self.bns, self.decoderCells = [], [], []
        self.n_layers = params['n_layers']
        self.input_window_size = params['input_window_size']
        self.output_window_size = params['output_dim']

        # Written By Min
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


        for i in range(params['n_layers']):
            params['input_dim'] == params['hidden_dim'] if i == 0 else params['hidden_dim']
            params['hidden_dim'] == params['hidden_dim']
            self.cells.append(SA_Convlstm_cell(params))
            self.bns.append(nn.LayerNorm((params['hidden_dim'], 16, 16)))  # Use layernorm



        self.cells = nn.ModuleList(self.cells)

        self.bns = nn.ModuleList(self.bns)
        self.decoderCells = nn.ModuleList(self.decoderCells)

        # Linear
        self.decoder_predict = nn.Conv2d(in_channels=params['hidden_dim'],
                                         out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0))

    def forward(self, x, y, teacher_forcing_rate=0.5, hidden=None):
        if hidden == None:
            hidden = self.init_hidden(batch_size=self.batch_size, img_size=self.img_size)

        b, seq_len, x_c, h, w = x.size()
        _, horizon, y_c, h, w = y.size()

        predict_temp_de = []

        in_x = min(x_c, y_c)
        # lag_y = torch.cat([x[:, -1:, :in_x, :, :], y[:, :-1, :in_x, :, :]], dim=1)

        frames = torch.cat([x, y], dim=1)


        for t in range(19):

            if t < self.input_window_size or random.random() < teacher_forcing_rate:
                x = frames[:, t, :, :, :]
            else:
                x = out

            x = self.img_encode(x)

            for i, cell in enumerate(self.cells):

                if i == 0:
                    out, hidden[i] = cell(x, hidden[i])
                    out = self.bns[i](out)

                else:
                    out, hidden[i] = cell(x, hidden[i])
                    out = self.bns[i](out)

            # out = self.decoder_predict(out)
            out = self.img_decode(out)
            predict_temp_de.append(out)

        predict_temp_de = torch.stack(predict_temp_de, dim=1)

        predict_temp_de = predict_temp_de[:, 9:, :, :, :]

        return predict_temp_de


    def init_hidden(self, batch_size, img_size):
        states = []
        for i in range(self.n_layers):
            states.append(self.cells[i].init_hidden(batch_size, img_size))

        return states