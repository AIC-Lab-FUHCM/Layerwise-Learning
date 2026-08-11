import torch
import torch.nn as nn
from modules.tcn_based.tcnBlocks import InceptionTCNBlock, TemporalBlock
from modules.constantDecoder import FixedReadout

from modules.tcn_based.tcnBlock2 import TemporalBlockBTF


class Local_TCNNet(nn.Module):
    def __init__(self, num_inputs,
                 num_channels,
                 dropout,
                 activation,
                 window_size,
                 predicted_length,
                 tcn_type):
        super(Local_TCNNet, self).__init__()

        # bien layer la 1 list of modules
        self.net = nn.ModuleList()

        #num_channels la 1 list of channels tai moi layer
        self.num_channels = num_channels

        #num_level la so luong layer
        num_levels = len(num_channels)

        self.window_size =window_size
        self.predicted_length = predicted_length

        #khoi tao truc doc
        for i in range(num_levels):
            dilation_size = 2 ** i

            if tcn_type =='in_tcn':
                in_channels = num_inputs if i == 0 else num_channels[i - 1]*3
                out_channels = num_channels[i]*3
                print(in_channels, out_channels)

                #generate a random fix redout matrix
                readout_matrix1 = torch.randn(num_inputs, num_channels[i]*3)
                readout_matrix2 = torch.randn(window_size, predicted_length)
                # print(readout_matrix1.shape)
                # print(readout_matrix2.shape)

                layer = nn.ModuleList([
                    InceptionTCNBlock(in_channels=in_channels, out_channels=out_channels,
                                      dropout=dropout, dilation=dilation_size, activation=activation),
                    FixedReadout(readout_matrix=readout_matrix1, readout_type=1),
                    FixedReadout(readout_matrix=readout_matrix2, readout_type=2)
                ])

                self.net.append(layer)

            if tcn_type =='b_tcn':
                in_channels = num_inputs if i == 0 else num_channels[i - 1]
                out_channels = num_channels[i]
                print(in_channels, out_channels)

                #generate a random fix redout matrix
                readout_matrix1 = torch.ones(num_channels[i],num_inputs)
                # readout_matrix2 = torch.randn(predicted_length, window_size)
                # print(readout_matrix1.shape)
                # print(readout_matrix2.shape)

                layer = nn.ModuleList([
                    # TemporalBlock(n_inputs=in_channels,n_outputs=out_channels,
                    #               kernel_size=3,stride=1,dilation=dilation_size,padding=2,
                    #               activation=activation,dropout=dropout),

                    TemporalBlockBTF(in_features=in_channels,out_features=out_channels,
                                     kernel_size=3,dilation=2),
                    FixedReadout(readout_matrix=readout_matrix1, readout_type=1)
                    # FixedReadout(readout_matrix=readout_matrix2)
                ])
                if predicted_length != window_size:
                    readout_matrix2 = torch.randn(predicted_length, window_size)
                    layer.append(FixedReadout(readout_matrix=readout_matrix2,
                                              readout_type=2)
                                 )

                self.net.append(layer)

    def forwardAtLayer(self, x, layerid):
        x = self.net[layerid][0](x)
        print('output inception: ' + str(x.shape))
        aux = self.net[layerid][1](x)
        if self.predicted_length != self.window_size:
            aux = self.net[layerid][2](aux)
        return x, aux

    def forwardToLayer(self, x, layerid):

        aux = None
        for i in range(layerid + 1):
            x = self.net[i][0](x)
            aux = self.net[i][1](x)

            if self.predicted_length != self.window_size:
                aux = self.net[i][2](aux)

        return x, aux

    def forward(self, x):
        aux = None

        for layer in self.net:
            x = layer[0](x)
            #print(x.shape)
            aux = layer[1](x)

            if self.predicted_length != self.window_size:
                aux = layer[2](aux)

        return x,aux


if __name__ == '__main__':
    x = torch.rand(5, 16, 2)  # batch , length, dim
    print(x.shape)

    block = TemporalBlockBTF(in_features=2,
                                   out_features=32,
                                   kernel_size=3,
                                   dilation=2)
    # out = block(x)
    # print(out.shape)

    model = Local_TCNNet(num_inputs=2,
                         num_channels=[32],
                         dropout=0.,
                         activation='relu',
                         window_size=16,
                         predicted_length=16,
                         tcn_type='b_tcn'
                         )
    out1, out2= model(x)

    print(out1.shape, out2.shape)

    pass