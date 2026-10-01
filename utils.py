import numpy as np
import random
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

data_source = './dataset/gesture/gesture/'
checkpoint_loc ='./checkpoints/'