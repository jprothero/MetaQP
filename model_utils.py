import torch.optim as optim
import torch

from models import Q, P
import config

from IPython.core.debugger import set_trace
from collections import OrderedDict

import os

def setup_Q_optim(Q):
    q_optim = optim.SGD(Q.parameters(),
        lr=config.LR,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY)
    
    return q_optim

def setup_P_optim(P):
    p_optim = optim.SGD(P.parameters(),
        lr=config.LR,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY)
    
    return p_optim

def load_Q():
    try:
        return torch.load('checkpoints/models/Q_best.t7')
    except:
        print('Initialize new Network Weights for Q')
        return Q()

def load_P():
    try:
        return torch.load('checkpoints/models/P_best.t7')
    except:
        print('Initialize new Network Weights for P')
        return P()

def save_Q(Q):
    print("Saving best Q")
    torch.save(Q, "checkpoints/models/Q_best.t7")

def save_P(P):
    print("Saving best P")
    torch.save(P, "checkpoints/models/P_best.t7")

def save_temp(model, name):
    os.makedirs("temp", exist_ok=True)
    torch.save(model, "temp/{}_temp.t7".format(name))

def load_temp(name):
    return torch.load("temp/{}_temp.t7".format(name))

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
