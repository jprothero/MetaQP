import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import config

from IPython.core.debugger import set_trace


def conv_layer(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1)


def make_layer(in_dims, h_dims, block, out_dims=None, head="normal"):
    layers = []
    layers.append(block(in_dims, h_dims, out_dims, head))

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
        self.state_res3 = make_layer(config.NUM_STATE_RES_FILTERS,
                                     config.NUM_STATE_RES_FILTERS,
                                     ResBlock)
        self.state_res4 = make_layer(config.NUM_STATE_RES_FILTERS,
                                     config.NUM_STATE_RES_FILTERS,
                                     ResBlock)
        self.state_res5 = make_layer(config.NUM_STATE_RES_FILTERS,
                                     config.NUM_STATE_RES_FILTERS,
                                     ResBlock)
        self.state_res6 = make_layer(config.NUM_STATE_RES_FILTERS,
                                     config.NUM_STATE_RES_FILTERS,
                                     ResBlock)
        self.state_res7 = make_layer(config.NUM_STATE_RES_FILTERS,
                                     config.NUM_STATE_RES_FILTERS,
                                     ResBlock)

        self.state_res_out = make_layer(config.NUM_STATE_RES_FILTERS,
                                        config.NUM_STATE_RES_FILTERS,
                                        ResBlock)

    def forward(self, state):
        s = self.state_res_inp(state)
        s = self.state_res1(s)
        s = self.state_res2(s)
        s = self.state_res3(s)
        s = self.state_res4(s)
        # s = self.state_res5(s)
        # s = self.state_res6(s)
        # s = self.state_res7(s)

        state_out = self.state_res_out(s)

        return state_out

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

        policy_view = policy.view(state.size()[0], 1, config.R, config.C)
        q_input = torch.cat((state_out, policy_view), dim=1)
        Q = self.Q(q_input)
        return Q, policy

class P(nn.Module):
    def __init__(self):
        super(P, self).__init__()        
        self.StateModule = StateModule()
        self.P = PolicyModule()

    def forward(self, state, percent_random=None):
        state_out = self.StateModule(state)

        policy = self.P(state_out, percent_random)

        return policy

class Q(nn.Module):
    def __init__(self):
        super(Q, self).__init__()        
        self.StateModule = StateModule()
        self.Q = QModule()

    def forward(self, state, policy):
        state_out = self.StateModule(state)

        policy_view = policy.view(state.size()[0], 1, config.R, config.C)

        q_input = torch.cat((state_out, policy_view), dim=1)

        Q = self.Q(q_input)

        return Q

class QModule(nn.Module):
    def __init__(self):
        super(QModule, self).__init__()
        self.q_res_inp = make_layer(config.NUM_Q_RES_FILTERS + 1,
                                    config.NUM_Q_RES_FILTERS,
                                    ResBlock)
        self.q_res1 = make_layer(config.NUM_Q_RES_FILTERS,
                                 config.NUM_Q_RES_FILTERS,
                                 ResBlock)
        self.q_res2 = make_layer(config.NUM_Q_RES_FILTERS,
                                 config.NUM_Q_RES_FILTERS,
                                 ResBlock)
        self.q_res3 = make_layer(config.NUM_Q_RES_FILTERS,
                                 config.NUM_Q_RES_FILTERS,
                                 ResBlock)
        self.q_res4 = make_layer(config.NUM_Q_RES_FILTERS,
            config.NUM_Q_RES_FILTERS,
            ResBlock)
        self.q_res_out = make_layer(config.NUM_Q_RES_FILTERS,
                                    config.NUM_Q_RES_FILTERS,
                                    QHead,
                                    1)

    def forward(self, q_input):
        q = self.q_res_inp(q_input)
        q = self.q_res1(q)
        q = self.q_res2(q)
        # q = self.q_res3(q)
        # q = self.q_res4(q)
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
        self.p_res2 = make_layer(
            config.NUM_P_RES_FILTERS,
            config.NUM_P_RES_FILTERS,
            ResBlock)
        self.p_res3 = make_layer(
            config.NUM_P_RES_FILTERS,
            config.NUM_P_RES_FILTERS,
            ResBlock)

        self.p_res4 = make_layer(
            config.NUM_P_RES_FILTERS,
            config.NUM_P_RES_FILTERS,
            ResBlock)

        self.p_res5 = make_layer(
            config.NUM_P_RES_FILTERS,
            config.NUM_P_RES_FILTERS,
            ResBlock)

        self.p_res_out = make_layer(
            config.NUM_P_RES_FILTERS,
            config.NUM_P_RES_FILTERS,
            PolicyHead,
            config.R * config.C)

        # self.noise_attention_head = make_layer(
        #     config.NUM_P_RES_FILTERS,
        #     config.NUM_P_RES_FILTERS,
        #     PolicyHead,
        #     config.R * config.C)

        # self.noise_gating_head = make_layer(
        #     config.NUM_P_RES_FILTERS,
        #     config.NUM_P_RES_FILTERS,
        #     PolicyHead,
        #     config.R * config.C,
        #     head="relu_tanh")

    def forward(self, state_out, percent_random):
        p = self.p_res_inp(state_out)
        p = self.p_res1(p)
        p = self.p_res2(p)
        p = self.p_res3(p)
        # p = self.p_res4(p)
        # p = self.p_res5(p)
        policy_out = self.p_res_out(p)
        # this will in effect choose one of the policies to be random.
        # this might be ideal since we are mixing policies
        # consider adding in a relu(tanh()) head to control the magnitude of the mixing
        # why not add it now.
        # noise_attention = self.noise_attention_head(p)

        # noise = Variable(torch.from_numpy(np.random.uniform(size=(state_out.size()[0], config.R*config.C)).astype('float32')))
        # if config.CUDA:
        #     noise = noise.cuda()

        # consider removing one and seeing how it effects performance

        # so let me see. the attention sums to one, basically we are choosing what percentage
        # of noise vs what percentage of real to keep, seems fine.
        # noise = noise_attention*noise
        # policy_out = policy_out * (1 - noise_attention) + noise*noise_attention

        # if percent_random is not None:
        #     #so the other idea was learning a head that would taking uniform noise, gate how much to let through,
        #     #and then choose how much to mix that noise with the policy
        #     #it might be good, but for now lets keep it simple
        #     noise = Variable(torch.from_numpy(np.random.dirichlet(
        #         [1] * config.R * config.C, size=(state_out.size()[0],)).astype("float32")))
        #     if config.CUDA:
        #         noise = noise.cuda()
        #     policy_out = policy_out * \
        #         (1 - percent_random) + noise * percent_random

        if percent_random is not None:
            noise = Variable(torch.from_numpy(np.random.dirichlet(
                [1] * config.R * config.C, size=(state_out.size()[0],)).astype("float32")))
            if config.CUDA:
                noise = noise.cuda()
            policy_out = policy_out * \
                (1 - percent_random) + noise * percent_random

        policy = policy_out

        return policy

class ResBlock(nn.Module):
    def __init__(self, in_dims, h_dims, out_dims=None, head="normal"):
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
    def __init__(self, in_dims, h_dims, out_dims=1, head="normal"):
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

        x = x.view(x.size()[0], -1)

        x = self.lin1(x)
        x = self.relu(x)

        x = self.scalar(x)
        Q = self.tanh(x)

        return Q


class PolicyHead(nn.Module):
    def __init__(self, in_dims, h_dims, out_dims=config.R*config.C, head="normal"):
        super(PolicyHead, self).__init__()
        self.head = head
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.conv1 = nn.Conv2d(
            in_dims, config.POLICY_HEAD_FILTERS, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm1d(config.POLICY_HEAD_FILTERS)

        self.lin = nn.Linear(config.POLICY_HEAD_FILTERS *
                             config.R*config.C, out_dims)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        logits = self.lin(x.view(x.size()[0], -1))
        if self.head is "relu-tanh":
            # policy
            out = self.relu(self.tanh(logits))
        else:
            # relu-gate
            out = F.softmax(logits, dim=1)

        return out
