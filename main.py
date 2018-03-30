from MetaQP import MetaQP
from Connect4 import Connect4
import config
import numpy as np
import pickle
from IPython.core.debugger import set_trace
import torch

connect4 = Connect4()
actions = connect4.actions
get_legal_actions = connect4.get_legal_actions
transition_and_evaluate = connect4.transition_and_evaluate

root_state = np.zeros(shape=(3, 6, 7), dtype="float32")
iteration = 0

metaqp = MetaQP(actions=actions, get_legal_actions=get_legal_actions,
    transition_and_evaluate=transition_and_evaluate, cuda=False)

metaqp.find_lrs()

while True:
    metaqp.train()
    metaqp.run_episode(root_state)

    iteration += 1
    print("Iteration Number "+str(iteration))