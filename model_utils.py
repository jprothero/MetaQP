import torch.optim as optim
import torch

from models import QP
import config


def setup_optims(self):
    q_optim = optim.Adam(self.qp.q.parameters(),
                         lr=config.LR,
                         weight_decay=config.WEIGHT_DECAY)

    p_optim = optim.Adam(self.qp.p.parameters(),
                         lr=config.LR,
                         weight_decay=config.WEIGHT_DECAY)

    return q_optim, p_optim


def load_model(self, name="qp"):
    try:
        return torch.load('checkpoints/models/%s_best.t7' % name)
    except:
        print('Initialize new Network Weights for %s_best' % name)
        return QP()


def save_model(self, name="qp"):
    print("Saving best model")
    torch.save(self.mcts, "checkpoints/models/%s_best.t7" % name)
