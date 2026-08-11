import numpy as np
import random
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

data_source = './dataset/pd/pd/'
checkpoint_loc ='./checkpoints/'