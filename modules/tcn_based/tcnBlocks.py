import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import math
import torch.nn.functional as F

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # print(x[0][0])
        # print(x[:, :, :-self.chomp_size].contiguous()[0][0])
        # print('-------------------')
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 kernel_size,
                 stride,
                 dilation,
                 padding,
                 activation,
                 dropout=0.2):
        super(TemporalBlock, self).__init__()
        if activation =='relu':
            self.act = nn.ReLU()
        elif activation =='gelu':
            self.act = nn.GELU()
        elif activation =='silu':
            self.act = nn.SiLU()
        elif activation =='elu':
            self.act = nn.ELU()
        elif activation =='leak_relu':
            self.act = nn.LeakyReLU()
        elif activation =='swish':
            self.act = nn.Hardswish()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.dropout2 = nn.Dropout(dropout)

        if kernel_size > 1:
            self.net = nn.Sequential(self.conv1, self.chomp1, self.act, self.dropout1,
                                     self.conv2, self.chomp2, self.act, self.dropout2)

            self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
            self.init_weights()
        elif kernel_size == 1:
            self.net = nn.Sequential(self.conv1, self.act, self.dropout1,
                                     self.conv2, self.act, self.dropout2)

            self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
            self.init_weights()



    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):

        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.act(out + res)
        # return self.relu(out + res)


class InceptionTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout,dilation, activation):
        super(InceptionTCNBlock, self).__init__()

        self.block1=TemporalBlock(in_channels, out_channels, 1, stride=1, dilation=1,
                                     padding=(1 - 1) * 1, dropout=dropout,
                                     activation=activation)
        self.block2=TemporalBlock(in_channels, out_channels, 3, stride=1, dilation=dilation,
                                     padding=(3 - 1) * dilation, dropout=dropout,
                                     activation=activation)
        self.block3 = TemporalBlock(in_channels, out_channels, 5, stride=1, dilation=dilation,
                                    padding=(5 - 1) * dilation, dropout=dropout,
                                    activation=activation)
    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x)
        x3 = self.block3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        # print('inception output' + str(x))
        return x