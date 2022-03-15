import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):

    def __init__(self, in_channels, h_channels, kernel_size, bias=True):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.h_channels = h_channels
        padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.conv = nn.Conv2d(in_channels=in_channels + h_channels,
                              out_channels=4 * h_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)
        
    def forward(self, input_tensor, cur_state):

        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.h_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        
        return (torch.zeros(batch_size, self.h_channels, height, width).to(self.device),
                torch.zeros(batch_size, self.h_channels, height, width).to(self.device))
        
#         return (torch.zeros(batch_size, self.h_channels, height, width, device=self.conv.weight.device),
#                 torch.zeros(batch_size, self.h_channels, height, width, device=self.conv.weight.device))

