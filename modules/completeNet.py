import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import math
import torch.nn.functional as F
from modules.tcn_based.tcnNet import Local_TCNNet


def build_CompleteNet(args):
    if args.encoder =='b_tcn' or args.encoder =='in_tcn':
        return Local_TCNNet(num_inputs=args.dimension,
                             num_channels=args.tcn_channels,
                             dropout=args.dropout,
                             activation=args.activation,
                             window_size=args.w,
                             predicted_length=args.predicted_length,
                            tcn_type= args.encoder)


    else:
        pass





