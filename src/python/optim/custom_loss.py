import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        p = nn.functional.softmax(input, dim=1)
        if self.alpha is not None:
            alpha = self.alpha[target]
        else:
            alpha = 1
        loss = - alpha * (1 - p.gather(1, target.unsqueeze(1))) ** self.gamma * p.gather(1, target.unsqueeze(1)).log()
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss