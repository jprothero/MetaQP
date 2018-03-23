import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import config

from IPython.core.debugger import set_trace


def conv_layer(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1)

def make_layer(in_dims, h_dims, block, out_dims=None):
    layers = []
    layers.append(block(in_dims, h_dims, out_dims))

    return nn.Sequential(*layers)

class StateModule(nn.Module):
    def __init__(self):
        super(StateModule, self).__init__()                
        self.state_res_inp = make_layer(config.CH,
                                              config.NUM_STATE_RES_FILTERS,
                                              ResBlock)
        self.state_res1 = make_layer(config.NUM_STATE_RES_FILTERS,
                                           config.NUM_STATE_RES_FILTERS,
                                           ResBlock)
        self.state_res2 = make_layer(config.NUM_STATE_RES_FILTERS,
                                           config.NUM_STATE_RES_FILTERS,
                                           ResBlock)
        self.state_res_out = make_layer(config.NUM_STATE_RES_FILTERS,
                                              config.NUM_STATE_RES_FILTERS,
                                              ResBlock)

    def forward(self, state):
        s = self.state_res_inp(state)
        s = self.state_res1(s)
        s = self.state_res2(s)
        state_out = self.state_res_out(s)

        return state_out

class QModule(nn.Module):
    def __init__(self):
        super(QModule, self).__init__()        
        self.q_res_inp = make_layer(config.NUM_Q_RES_FILTERS + 1,
                                          config.NUM_Q_RES_FILTERS,
                                          ResBlock)
        self.q_res1 = make_layer(config.NUM_Q_RES_FILTERS,
                                       config.NUM_Q_RES_FILTERS,
                                       ResBlock)
        self.q_res_out = make_layer(config.NUM_Q_RES_FILTERS,
                                          config.NUM_Q_RES_FILTERS,
                                          QHead,
                                          1)

    def forward(self, q_input):
        q = self.q_res_inp(q_input)
        q = self.q_res1(q)
        Q = self.q_res_out(q)

        return Q

class PolicyModule(nn.Module):
    def __init__(self):
        super(PolicyModule, self).__init__()       

        self.p_res_inp = make_layer(
            config.NUM_STATE_RES_FILTERS,
            config.NUM_P_RES_FILTERS,
            ResBlock)
        self.p_res1 = make_layer(
            config.NUM_P_RES_FILTERS,
            config.NUM_P_RES_FILTERS,
            ResBlock)
        self.p_res_out = make_layer(
            config.NUM_P_RES_FILTERS,
            config.NUM_P_RES_FILTERS,
            PolicyHead,
            config.R * config.C)

    def forward(self, state_out, percent_random):
        p = self.p_res_inp(state_out)
        p = self.p_res1(p)
        policy_out = self.p_res_out(p)

        if percent_random is not None:
            noise = Variable(torch.from_numpy(np.random.dirichlet(
                [1] * config.R * config.C, size=(config.BATCH_SIZE,)).astype("float32")))
            if config.CUDA:
                noise = noise.cuda()
            policy_out = policy_out * \
                (1 - percent_random) + noise * percent_random

        policy = policy_out

        return policy

class QP(nn.Module):
    def __init__(self):
        super(QP, self).__init__()
        self.StateModule = StateModule()
        self.Q = QModule()
        self.P = PolicyModule()

    def forward(self, state, policy=None, percent_random=None):
        state_out = self.StateModule(state)

        if policy is None:
            policy = self.P(state_out, percent_random)
            
        policy_view = policy.view(1, config.BATCH_SIZE, config.R, config.C)
        state_out = state_out.permute(1, 0, 2, 3)

        q_input = torch.cat((state_out, policy_view), axis=1)        

        q_input = q_input.permute(1, 0, 2, 3)

        Q = self.Q(q_input)

        # might need this view
        # policy_view = policy.view(1, config.BATCH_SIZE, config.R, config.C)
        # Q_input = torch.cat((x.view(self.num_filters, config.BATCH_SIZE, config.R, config.C),
        #     policy_view), axis=0)
        # Q = self.Q(Q_input.view(config.BATCH_SIZE, -1))

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

        if out.shape == residual.shape:
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
        self.lin1 = nn.Linear(config.R*config.C, 32)

        self.scalar = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = x.view(config.BATCH_SIZE, -1)

        x = self.lin1(x)
        x = self.relu(x)

        x = self.scalar(x)
        Q = self.tanh(x)

        return Q


class PolicyHead(nn.Module):
    def __init__(self, in_dims, h_dims, out_dims=config.R*config.C):
        super(PolicyHead, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.conv1 = nn.Conv2d(in_dims, config.POLICY_HEAD_FILTERS, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm1d(config.POLICY_HEAD_FILTERS)

        self.lin = nn.Linear(config.POLICY_HEAD_FILTERS*config.R*config.C, out_dims)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        logits = self.lin(x.view(config.BATCH_SIZE, -1))

        policy = F.softmax(logits, dim=1)

        return policy
