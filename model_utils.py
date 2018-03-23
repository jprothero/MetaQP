import torch.optim as optim
import torch

from models import QP
import config

def setup_optims(qp):
    if config.OPTIM.lower() == "adam":
        q_optim = optim.Adam(qp.parameters(),
                            lr=config.LR,
                            weight_decay=config.WEIGHT_DECAY)

        p_optim = optim.Adam(qp.parameters(),
                            lr=config.LR,
                            weight_decay=config.WEIGHT_DECAY)
    else:
        q_optim = optim.SGD(qp.parameters(),
                            lr=config.LR,
                            momentum=config.MOMENTUM,                            
                            weight_decay=config.WEIGHT_DECAY)

        p_optim = optim.SGD(qp.parameters(),
                            lr=config.LR,
                            momentum=config.MOMENTUM,
                            weight_decay=config.WEIGHT_DECAY)

    return q_optim, p_optim


def load_model(name="qp"):
    try:
        return torch.load('checkpoints/models/%s_best.t7' % name)
    except:
        print('Initialize new Network Weights for %s_best' % name)
        return QP()


def save_model(qp, name="qp"):
    print("Saving best model")
    torch.save(qp, "checkpoints/models/%s_best.t7" % name)
