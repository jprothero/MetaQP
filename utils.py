import os
import pickle
import torch.optim as optim

import config


def create_folders():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + 'models'):
        os.makedirs('checkpoints/' + 'models')


def load_history():
    print("Loading History...")
    try:
        history = pickle.load(open("checkpoints/history.p", "rb"))
        print("Successfully loaded history.")
    except FileNotFoundError:
        print("Loss history not found, starting new history.")

        history = {
            "Q": [], "P": []
        }

    return history


def load_memories():
    print("Loading memories...")
    try:
        memories = pickle.load(
            open("checkpoints/memories.p", "rb"))
        print("Number of memories: " + str(len(memories)))
    except FileNotFoundError:
        print("Memories not found, making new memories.")
        memories = []

    return memories
