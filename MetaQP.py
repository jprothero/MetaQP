from models import QP
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from copy import copy
from random import shuffle, sample
import numpy as np
from IPython.core.debugger import set_trace

import config
import utils

np.seterr(all="raise")


class MetaQP:
    def __init__(self,
                 actions,
                 calculate_reward,
                 get_legal_actions,
                 transition,
                 cuda=torch.cuda.is_available(),
                 best=False):
        utils.create_folders()

        self.version = version
        self.cuda = cuda
        self.qp = load_model()
        if self.cuda:
            self.qp = self.qp.cuda()

        self.actions = actions
        self.get_legal_actions = get_legal_actions
        self.calculate_reward = calculate_reward
        self.transition = transition

        if not best:
            self.q_optim, self.p_optim = utils.setup_optims()
            self.history = utils.setup_history()
            self.memories = utils.load_memories()

            self.best_qp = load_model()

    def self_play(self, root_state):
        results = {
            "new": 0, "best": 0, "draw": 0
        }
        self.qp.eval()
        self.best_qp.eval()

        for _ in tqdm(range(config.EPISODES)):
            state = np.copy(root_state)
            # game_over = False
            # turn = 0
            # episode_memories = []
            # legal_actions = self.get_legal_actions(state[:2])

            while not game_over:
                curr_player = turn % 2
                turn += 1
                if curr_player != best_player:
                    state, memory = self.select_action(
                            state, turn, legal_actions, curr_player)
                else:
                    state, memory = self.best_mcts.select_action(
                        state, turn, legal_actions, curr_player)
                else:

                legal_actions = self.get_legal_actions(state[:2])

                reward, game_over = self.calculate_reward(state[:2])
                episode_memories.extend([memory])

                if len(legal_actions) == 0:
                    game_over = True
                    results["draw"] += 1
                elif game_over:
                    if best_player == curr_player:
                        results["best"] += 1
                    else:
                        results["new"] += 1

            for memory in episode_memories:
                if memory["curr_player"] == curr_player:
                    memory["result"] = -1 * reward
                else:
                    memory["result"] = reward
            memories.extend(episode_memories)

        print("Results: ", results)
        if results["new"] > results["best"] * config.SCORING_THRESHOLD:
            self.save_best_model()
            print("Loading new best_mcts")
            self.best_mcts.mcts = self.load_model("best")
            if self.cuda:
                self.best_mcts.mcts = self.best_mcts.mcts.cuda()
        elif results["best"] > results["new"] * config.SCORING_THRESHOLD:
            print("Reverting to previous best")
            self.mcts = self.load_model("best")
            if self.cuda:
                self.mcts = self.mcts.cuda()
            self.optim = optim.SGD(self.mcts.parameters(),
                                    lr=config.LR,
                                    weight_decay=config.WEIGHT_DECAY,
                                    momentum=config.MOMENTUM)
        return memories

    # def select_action(self, state, turn, legal_actions, curr_player):

    def correct_policy(self, policy, state):
        legal_actions = self.get_legal_actions(state[:2])

        mask = np.zeros((config.BATCH_SIZE, len(self.actions)))
        mask[legal_actions] = 1

        policy = policy * mask

        pol_sum = (np.sum(policy * 1.0))

        if pol_sum == 0:
            pass
        else:
            policy = policy / pol_sum

        return policy

    def correct_policies(self, policies, state)
        for policy in policies:
            policy = self.correct_policy(policy)
        return policies

    def get_improved_task_policies_list(self, policies):
        improved_policies = []
        for i in range(config.EPISODE_BATCH_SIZE // config.N_WAY):
            improved_policy = policies[i:i+config.N_WAY].sum()
            improved_policies.extend([improved_policy])

        return improved_policies

    def wrap_to_variable(tensor, volatile=False):
        var = Variable(torch.from_numpy(tensor), volatile=volatile)
        if self.cuda:
            var = var.cuda()
        return var

    def create_task_tensor(self, state):
        # a task is going to be from the perspective of a certain player
        # so we want to

        # np.array to make a fast copy
        state = np.array(np.expand_dims(state, 0))
        n_way_state = np.repeat(state, config.N_WAY, axis=0)
        n_way_state_tensor = torch.from_numpy(n_way_state)

        return n_way_state_tensor

    def update_task_memories(tasks, corrected_final_policies, improved_task_policies):
        for i, task in enumerate(tasks):
            task["improved_policy"] = improved_task_policies[i]
            for policy in corrected_final_policies[i:i+config.N_WAY]:
                task["memories"].extend({"policy": policy})

        return task

    #so what are the targets going to be
    #the Q will get a set of states and policies (maybe a mix of the initial policy, and
    #the corrected_policy). It's goal to generalize Q values for that state

    #the policy net will get one example will get a small aux loss driving it to
    #the corrected policy, and maybe we will have a meta policy prediction later
    #if we do the first idea

    # so we can have it where the training net always goes first, then the best net
    # always goes second. we randomly choose the starting player for the input states,
    # and the new and best nets should get an even number of games for player 1 / 2

    #so let me think about this some more, all I really need are states, some slightly
    #different policies, and the results.

    def mix_task_policies(self, improved_task_policies, policies, perc=0):
        for improved_policy in improved_task_policies:
            for policy in policies[i:i+config.N_WAY]:
                policy = policy(1-perc) + improved_policy*perc

        return policies

    def transition_batch_task_tensor(self, batch_task_tensor, 
            corrected_final_policies, is_done):
        for i, (state, policy) in enumerate(zip(batch_task_tensor, 
                corrected_final_policies)):
            if not is_done[i]:
                action = np.random.choice(self.actions, p=policy)
                state = self.transition(state[:2], action)

        return batch_task_tensor

    def check_finished_games(self, batch_task_tensor, is_done, tasks):
        idx = 0
        for j in range(config.EPISODE_BATCH_SIZE//config.N_WAY):
            for i, state in enumerate(batch_task_tensor[j:j+config.N_WAY]):
                if not is_done[idx]:
                    legal_actions = self.get_legal_actions(state[:2])
                    if len(legal_actions) == 0:
                        is_done[idx] = True
                        tasks[j][idx]["result"] = reward
                    else:
                        reward, game_over = self.calculate_reward(state[:2])
                        
                        if game_over:
                            is_done[idx] = True
                            curr_player = state[2][0]
                            if tasks[j]["starting_player"] != curr_player:
                                reward *= -1
                            tasks[j][idx]["result"] = reward

                idx += 1

        return is_done, tasks

    def meta_self_play(self, state):
        # fast copy it
        state = np.array(state)
        self.qp.eval()
        self.best_qp.eval()
        tasks = []
        batch_task_tensor = torch.zeros(config.EPISODE_BATCH_SIZE,
            config.CH, config.R, config.C)

        for i in range(config.EPISODE_BATCH_SIZE // config.N_WAY):
            # starting player chosen randomly
            starting_player = np.random.choice(1)
            state[2] = starting_player
            task_tensor = self.create_task_tensor(state)
            batch_task_tensor[i * config.N_WAY] = task_tensor

            task = {
                "state": task_tensor, "starting_player": starting_player, "memories": []
            }

        batch_task_variable = Variable(batch_task_tensor)

        best_start = np.random.choice(1)

        if best_start == 1:
            qp = self.best_qp
        else:
            qp = self.qp

        qs, policies = qp(batch_task_variable, percent_random=.2)

        # scales from -1 to 1 to 0 to 1
        scaled_qs = (qs + 1) / 2

        weighted_policies = policies*scaled_qs

        improved_task_policies = self.get_improved_task_policies_list(weighted_policies)

        #well in theory since the orig policies are partially random and 
        #the final policy is random, using only the improved policy might be fine
        #will set it like that for now. it will lead to a bit less
        #diversity in the policies that the Q sees, which is kind of bad.
        #but then again, we will get more of a true Q value for that policy.
        #we can try it out for now. ill set to .8 so some difference happens
        final_policies = self.mix_task_policies(improved_corrected_task_policies, 
            policies, perc_improved=1)

        corrected_final_policies = self.correct_policies(policies, state)

        tasks = self.update_task_memories(tasks, corrected_final_policies, improved_task_policies)
        
        is_done = []
        for i in range(config.EPISODE_BATCH_SIZE):
            is_done.extend([False])


        #sooo let me think. the new_net and best_net will continually trade off batch
        #evaluations. basically the new_net chooses some initial_moves, and 
        #then it alternates until all the games are done. #this will bias that the new
        #net always makes the first move, which can be significant
        #so now it's random start. so the opposing moves for each turn will be chosen
        #by the opposite net
        num_done = 0
        if best_start == 1:
            best_turn = 0
        else:
            best_turn = 1

        results = {
            "new": 0
            , "best": 0
            , "draw": 0
        }

        while num_done < config.EPISODE_BATCH_SIZE:
            batch_task_tensor = self.transition_batch_task_tensor(batch_task_tensor, 
                corrected_final_policies, is_done)

            is_done, tasks, results = self.check_finished_games(batch_task_tensor, is_done, tasks) 

            batch_task_variable = Variable(batch_task_tensor)

            if best_turn == 1:
                qp = self.best_qp
            else:
                qp = self.qp

            _, policies = self.qp(batch_task_variable)

            policies = self.correct_policies(policies)

        self.memories.extend(tasks)

        return tasks, results

    def train_memories(self, memories):
        self.meta_net.train()

        # so memories are a list of lists containing memories
        if len(memories) < config.NUM_TASKS:
            print("Need {} tasks, have {}".format(
                config.NUM_TASKS, len(memories)))
            return

        for _ in range(config.TRAINING_LOOPS):
            tasks = sample(memories, config.NUM_TASKS)
            self.train_tasks(tasks)

            # config.SAMPLE_SIZE - config.SAMPLE_SIZE%config.BATCH_SIZE)
        # minibatches = [
        #     data[x:x + config.BATCH_SIZE]
        #     for x in range(0, len(data), config.BATCH_SIZE)
        # ]

        # self.train_minibatches(minibatches)

    def train_tasks(self, tasks):
        # for training we want to sample a set of examples from a task(a state)
        # i.e. from 128 samples of one state, sample 8 and do 8-way learning

        Q_loss = 0
        inferred_loss = 0
        optimal_loss = 0

        for task in tasks:
            states = np.zeros(
                shape=(config.N_way, config.CH, config.R, config.C))
            random_policies = np.zeros(
                shape=(config.N_way, config.R * config.C))
            results = np.zeros(shape=(config.N_way))
            improved_policies = np.zeros(
                shape=(config.N_way, config.R * config.C))

            results = []
            for i, memory in enumerate(task):
                states[i] = memory["state"]
                random_policies[i] = memory["rand_pol"]
                results[i] = memory["result"]
                improved_policies[i] = memory["improved_policy"]

            states = self.wrap_to_variable(states, volatile=True)
            random_policies = self.wrap_to_variable(
                random_policies, volatile=True)
            results = self.wrap_to_variable(results, volatile=True)
            improved_policies = self.wrap_to_variable(
                improved_policies, volatile=True)

            Qs, _, inferred_policies, optimal_policies = \
                self.meta_net(states, random_policies, weighted=False)

            Q_loss += F.mse_loss(Qs, results)
            inferred_loss += torch.mm(inferred_policies.t(),
                                      torch.log(random_policies))
            optimal_loss += torch.mm(improved_policies[0].t(),
                                     torch.log(optimal_policies[0]))

        total_loss = Q_loss + inferred_loss + optimal_loss
        total_loss.backward()