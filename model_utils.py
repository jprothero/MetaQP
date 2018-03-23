import torch.optim as optim
import torch

from models import QP
import config

from IPython.core.debugger import set_trace
from collections import OrderedDict


def setup_optims(qp):
    if config.OPTIM.lower() == "adam":
        params = qp.state_dict()

        state_params = qp.StateModule.parameters()
        q_params = qp.Q.parameters()
        p_params = qp.P.parameters()

        q_optim = optim.Adam([
            {"params": state_params}, {"params": q_params}
        ],
            lr=config.LR,
            weight_decay=config.WEIGHT_DECAY)

        p_optim = optim.Adam([
            {"params": state_params}, {"params": p_params}
        ],
            lr=config.LR,
            weight_decay=config.WEIGHT_DECAY)
    else:
        q_optim = optim.SGD([
            {"params": state_params}, {"params": q_params}
        ],
            lr=config.LR,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY)

        p_optim = optim.SGD([
            {"params": state_params}, {"params": p_params}
        ],
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
