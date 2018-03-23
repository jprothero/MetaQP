from MetaQP import MetaQP
from Connect4 import Connect4
import config
import numpy as np
import pickle
from IPython.core.debugger import set_trace
import torch

connect4 = Connect4()
actions = connect4.actions
calculate_reward = connect4.calculate_reward
get_legal_actions = connect4.get_legal_actions
transition = connect4.transition

root_state = np.zeros(shape=(3, 6, 7), dtype="float32")
iteration = 0

metaqp = MetaQP(actions=actions, calculate_reward=calculate_reward,
    get_legal_actions=get_legal_actions, transition=transition)

while True:
    metaqp.train_memories()
    metaqp.meta_self_play(root_state)

    iteration += 1
    print("Iteration Number "+str(iteration))