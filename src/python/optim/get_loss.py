import torch.nn as nn
from .custom_loss import FocalLoss

def get_loss(loss_type):
    loss = None
    if loss_type == "CE":
        loss = nn.CrossEntropyLoss()
    elif loss_type == "KL":
        loss = nn.KLDivLoss()
    elif loss_type == "NLL":
        loss = nn.NLLLoss()
    elif loss_type == "Focal":
        loss = FocalLoss()
    return loss