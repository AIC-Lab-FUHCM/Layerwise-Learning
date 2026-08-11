import torch
import torch.nn as nn
import math

class FixedReadout(nn.Module):
    def __init__(self, readout_matrix, readout_type):
        super(FixedReadout, self).__init__()
        self.readout_type = readout_type
        self.register_buffer('readout_matrix', readout_matrix)

    def forward(self, x):
        # x is expected to be of shape (batch_size, input_dim)
        if self.readout_type == 1:
            #print(x.shape, self.readout_matrix.shape)
            return torch.matmul(x, self.readout_matrix)

        elif self.readout_type == 2:
            return torch.matmul(self.readout_matrix, x)

if __name__ == '__main__':
    pass
