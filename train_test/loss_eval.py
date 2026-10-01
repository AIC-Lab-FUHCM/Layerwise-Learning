import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

class ReconstructionLoss:
    def __init__(self, loss_type='mse', reduction='mean'):
        if loss_type not in ['mse', 'mae']:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Unsupported reduction type: {reduction}")
        self.loss_type = loss_type
        self.reduction = reduction

    def __call__(self, x, x_reconstructed):
        if self.loss_type == 'mse':
            return F.mse_loss(x_reconstructed, x, reduction=self.reduction)
        elif self.loss_type == 'mae':
            return F.l1_loss(x_reconstructed, x, reduction=self.reduction)



def F1_PA(groundtrue, predicted,delay):
    start_anomalies=[]
    end_anomalies=[]
    predicted = np.array(predicted)

    #find anomaly range
    for i in range (len(groundtrue)-1):
        if groundtrue[i+1] ==1. and groundtrue[i]==0:
            start_anomalies.append(i+1)
        if groundtrue[i] ==1. and groundtrue[i+1] ==0.:
            end_anomalies.append(i+1)

    print(start_anomalies)
    print(end_anomalies)

    if delay ==None:

        for k in range (len(start_anomalies)):
            for j in range(start_anomalies[k], end_anomalies[k] + 1):
                if predicted[j] == 1.:
                    print('here')
                    np.put(predicted, np.arange(start_anomalies[k], end_anomalies[k] + 1, 1), 1.)
                    break
    else:
        for k in range (len(start_anomalies)):
            for j in range(start_anomalies[k], start_anomalies[k]+delay):
                if predicted[j] == 1.:
                    np.put(predicted, np.arange(start_anomalies[k], end_anomalies[k] + 1, 1), 1.)
                    break

    score, precision, recall = F1_score(predicted, groundtrue)
    return score,precision, recall

