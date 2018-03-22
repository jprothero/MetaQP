import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import config

from IPython.core.debugger import set_trace


def conv_layer(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1)


class MetaNet(nn.Module):
    def __init__(self):
        super(MetaNet, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # Create state res blocks
        self.state_res_blocks = [
            self._make_layer(config.CH,
                             config.NUM_STATE_RES_FILTERS,
                             ResBlock)
        ]

        for _ in range(config.NUM_STATE_RES_BLOCKS - 1):
            self.state_res_blocks.extend([self._make_layer(config.NUM_STATE_RES_FILTERS,
                                                           config.NUM_STATE_RES_FILTERS,
                                                           ResBlock)])

        # Create Q res blocks
        self.q_res_blocks = [
            self._make_layer(config.CH,
                             config.NUM_Q_RES_FILTERS,
                             ResBlock)
        ]

        for _ in range(config.NUM_Q_RES_BLOCKS - 2):
            self.state_res_blocks.extend([self._make_layer(config.NUM_Q_RES_FILTERS,
                                                           config.NUM_Q_RES_FILTERS,
                                                           ResBlock)])

        self.q_res_blocks.extend([self._make_layer(config.NUM_Q_RES_FILTERS,
                                                   config.NUM_Q_RES_FILTERS,
                                                   QHead,
                                                   1)])

        # Create P res blocks
        self.p_res_blocks = [
            self._make_layer(config.CH, config.NUM_P_RES_FILTERS, ResBlock)
        ]

        for _ in range(config.NUM_P_RES_BLOCKS - 2):
            self.state_res_blocks.extend([self._make_layer(config.NUM_P_RES_FILTERS,
                                                           config.NUM_P_RES_FILTERS,
                                                           ResBlock)])

        self.p_res_blocks.extend([self._make_layer(config.NUM_P_RES_FILTERS,
                                                   config.NUM_P_RES_FILTERS,
                                                   PolicyHead,
                                                   config.R * config.C)])

    def _make_layer(self, in_dims, h_dims, block, out_dims=None):
        layers = []
        layers.append(block(in_dims, h_dims, out_dims))

        return nn.Sequential(*layers)

    def forward(self, state, policy=None, percent_random=None):
        batch_size = state.shape[0].data.numpy()[0]

        x = self.state_res_blocks[0](state)
        for res in self.state_res_blocks[:1]:
            x = res(x)

        x = self.q_res_blocks[0](x)


        if policy is None:
            policy = self.policy(x)

        # might need this view
        # policy_view = policy.view(1, config.BATCH_SIZE, config.R, config.C)
        # Q_input = torch.cat((x.view(self.num_filters, config.BATCH_SIZE, config.R, config.C),
        #     policy_view), axis=0)
        # Q = self.Q(Q_input.view(config.BATCH_SIZE, -1))

        policy_view = policy.view(batch_size, 1, config.R, config.C)

        Q_input = torch.cat((x, policy_view), axis=1)

        Q = self.Q(Q_input)

        if percent_random is not None:
            set_trace()
            noise = np.random.dirichlet([1] * 42, size=(batch_size))
            policy = policy * (1 - percent_random) + noise * percent_random

        return Q, policy

class ResBlock(nn.Module):
    def __init__(self, in_dims, h_dims, out_dims=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv_layer(in_dims, h_dims)
        self.bn1 = nn.BatchNorm2d(h_dims)
        self.relu = nn.ReLU()
        if out_dims is None:
            out_dims = h_dims
        self.conv2 = conv_layer(h_dims, out_dims)
        self.bn2 = nn.BatchNorm2d(out_dims)

    def forward(self, x):
        residual = x.squeeze()

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class QHead(nn.Module):
    def __init__(self, in_dims, h_dims, out_dims=1):
        super(QHead, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.conv1 = nn.Conv2d(in_dims, 1, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm1d(1)
        self.lin1 = nn.Linear(1, 32)

        self.scalar = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.lin1(x)
        x = self.relu(x)

        x = self.scalar(x)
        Q = self.tanh(x)

        return Q

class PolicyHead(nn.Module):
    def __init__(self, in_dims, h_dims, out_dims=1):
        super(PolicyHead, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.conv1 = nn.Conv2d(in_dims, 2, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm1d(2)

        self.policy = nn.Linear(2, config.R*config.C)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        policy = self.policy(x.view(-1, config.R*config.C))

        return policy
