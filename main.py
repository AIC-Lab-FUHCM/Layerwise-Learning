import argparse
import numpy as np
import random
import torch
from utils import set_seed, data_source, checkpoint_loc
from modules.completeNet import build_CompleteNet
from data_processing.ds_reader import DatasetLoader
from train_test.model_training import model_training
from train_test.model_testing import model_testing

def main_app(args):
    ds_loader = DatasetLoader(
        data_path=args.data_source,
        window_size=args.w,
        ts_num=args.ts_num,
        dataset=args.dataset
    )

    model = build_CompleteNet(args)
    model_name = [
        args.dataset + '_' + args.encoder
    ]

    if args.mode == 'train':
        model_training(
            ds_loader=ds_loader,
            net=model,
            model_name=model_name,
            params=args
        )

    elif args.mode == 'test':
        model_testing(
            ds_loader=ds_loader,
            net=model,
            model_name=model_name,
            params=args
        )

if __name__ == '__main__':
    set_seed()
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', default='test' , help='mode : train or test')
   
    parser.add_argument('--dimension', default=2, type=int, help='number of dimension of dataset')
    parser.add_argument('--dataset', default='gesture', help='dataset: ecg, gesture, pd')
    parser.add_argument('--ts_num', default=0)
    parser.add_argument('--data_source', default=data_source)
    parser.add_argument('--checkpoint_loc', default=checkpoint_loc)

    parser.add_argument('-encoder', default='b_tcn', type=str, help='model type: b_tcn, in_tcn')
    parser.add_argument('-tcn_channels', default=[32,32,32,32], help='a list of number of channels')
    

    parser.add_argument('-activation', default='gelu', type=str,
                        help='activation: relu, elu, gelu')

    parser.add_argument('--save_path', default=checkpoint_loc, type=str)

    parser.add_argument('-batch_size', default=256, help='number of samples in each batch')
    parser.add_argument('-dropout', default= 0.2, help='drop-out ratio')
    parser.add_argument('-w', default=16, type=int, help='window size of each sample')  
    parser.add_argument('-n_epochs', default=300, type=int, help='number of training epochs')
    parser.add_argument('-lr', default=0.0001, type=float, help='learning rate ')
    parser.add_argument('-scheduling', default=True, type=bool, help='learning rate scheduling')
    parser.add_argument('-kernel_size', default=3, type=int, help='kernel size')
    parser.add_argument('-loss_type', default='mse', type=str, help='reconstruction loss')
    parser.add_argument('-masked_value', default='0.0', type=float, help='masking value ')
    parser.add_argument('-predicted_length', default='16', type=int, help='prediction length')
    parser.add_argument('-regularization', default='None', help='regularization')
    parser.add_argument('-test_type', default='normal', type=str, help='test type: normal')
    args = parser.parse_args()

    main_app(args)



