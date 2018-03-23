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
import model_utils

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

        self.cuda = cuda
        self.qp = model_utils.load_model()
        if self.cuda:
            self.qp = self.qp.cuda()

        self.actions = actions
        self.get_legal_actions = get_legal_actions
        self.calculate_reward = calculate_reward
        self.transition = transition

        if not best:
            self.q_optim, self.p_optim = model_utils.setup_optims(self.qp)
            self.best_qp = model_utils.load_model()

            if self.cuda:
                self.best_qp = self.best_qp.cuda()

            self.history = utils.load_history()
            self.memories = utils.load_memories()

    def correct_policy(self, policy, state, mask=True):
        if mask:
            legal_actions = self.get_legal_actions(state[:2])

            mask = np.zeros((len(self.actions),))
            mask[legal_actions] = 1

            policy = policy * mask

        pol_sum = (np.sum(policy * 1.0))

        if pol_sum == 0:
            pass
        else:
            policy = policy / pol_sum

        return policy

    def correct_policies(self, policies, states):
        for i, (policy, state) in enumerate(zip(policies, states)):
            policies[i] = self.correct_policy(policy, state)
        return policies

    def get_improved_task_policies_list(self, policies):
        improved_policies = []
        idx = 0
        for _ in range(config.EPISODE_BATCH_SIZE // config.N_WAY):
            improved_policy = 0
            for _ in range(config.N_WAY):
                improved_policy += policies[idx]
            improved_policies.extend([improved_policy])

        return improved_policies

    def wrap_to_variable(self, tensor, volatile=False):
        var = Variable(torch.from_numpy(
            tensor.astype("float32")), volatile=volatile)
        if self.cuda:
            var = var.cuda()
        return var

    # def update_task_memories(self, tasks, corrected_final_policies, improved_task_policies):
    #     for i, task in enumerate(tasks):
    #         task["improved_policy"] = improved_task_policies[i]
    #         for policy in corrected_final_policies[i:i + config.N_WAY]:
    #             task["memories"].extend([{"policy": policy}])

    #     return tasks

    # so what are the targets going to be
    # the Q will get a set of states and policies (maybe a mix of the initial policy, and
    # the corrected_policy). It's goal to generalize Q values for that state

    # the policy net will get one example will get a small aux loss driving it to
    # the corrected policy, and maybe we will have a meta policy prediction later
    # if we do the first idea

    # so we can have it where the training net always goes first, then the best net
    # always goes second. we randomly choose the starting player for the input states,
    # and the new and best nets should get an even number of games for player 1 / 2

    # so let me think about this some more, all I really need are states, some slightly
    # different policies, and the results.

    # def mix_task_policies(self, improved_task_policies, policies, perc_improved=0):
    #     for i, improved_policy in enumerate(improved_task_policies):
    #         for policy in policies[i:i + config.N_WAY]:
    #             policy = policy * (1 - perc_improved) + \
    #                 improved_policy * perc_improved

    #     return policies

    def transition_batch_task_tensor(self, batch_task_tensor,
                                     corrected_final_policies, is_done):
        for i, (state, policy) in enumerate(zip(batch_task_tensor,
                                                corrected_final_policies)):
            if not is_done[i]:
                batch_task_tensor[i] = np.array(state)
                state = batch_task_tensor[i]
                action = np.random.choice(self.actions, p=policy)
                curr_player = int(state[2][0][0])
                state[curr_player] = self.transition(
                    np.array(state[curr_player]), action)
                state[2] = (curr_player + 1) % 2

        return batch_task_tensor

    def check_finished_games(self, batch_task_tensor, is_done, tasks, num_done, results, best_turn):
        idx = 0
        for i in range(config.EPISODE_BATCH_SIZE // config.N_WAY):
            for j in range(config.N_WAY):
                if not is_done[idx]:
                    state = batch_task_tensor[idx]
                    legal_actions = self.get_legal_actions(state[:2])
                    # I think draws are bugged, need to inspect
                    if len(legal_actions) == 0:
                        is_done[idx] = True
                        num_done += 1
                        tasks[i][idx]["result"] = 0
                        results["draw"] += 1
                    else:
                        reward, game_over = self.calculate_reward(state[:2])

                        if game_over:
                            is_done[idx] = True
                            curr_player = int(state[2][0][0])
                            if tasks[i]["starting_player"] != curr_player:
                                reward *= -1

                            tasks[i]["memories"][j]["result"] = reward
                            if best_turn == 1:
                                key = "best"
                                other = "new"
                            else:
                                key = "new"
                                other = "best"
                            if reward == 1:
                                results[key] += reward
                            else:
                                results[other] += -reward

                            num_done += 1

                idx += 1

        return is_done, tasks, results, num_done

    def meta_self_play(self, state):
        np.set_printoptions(precision=3)
        self.qp.eval()
        self.best_qp.eval()
        tasks = []
        batch_task_tensor = np.zeros((config.EPISODE_BATCH_SIZE,
                                      config.CH, config.R, config.C))

        idx = 0
        for _ in range(config.EPISODE_BATCH_SIZE // config.N_WAY):
            # starting player chosen randomly
            starting_player = np.random.choice(2)
            state[2] = starting_player

            task = {
                "state": np.array(state), "starting_player": starting_player, "memories": []
            }

            tasks.extend([task])

            for _ in range(config.N_WAY):
                batch_task_tensor[idx] = np.array(state)

        batch_task_variable = self.wrap_to_variable(batch_task_tensor)
        best_start = np.random.choice(2)

        if best_start == 1:
            qp = self.best_qp
        else:
            qp = self.qp

        _, policies = qp(batch_task_variable, percent_random=.2)

        # qs = qs.detach().numpy()
        policies = policies.detach().numpy()

        policies = self.correct_policies(policies, batch_task_tensor)

        policies_input = self.wrap_to_variable(policies)

        qs, _ = qp(batch_task_variable, policies_input)

        qs = qs.detach().numpy()

        scaled_qs = (qs + 1) / 2
        weighted_policies = policies * scaled_qs

        idx = 0
        for i in range(config.EPISODE_BATCH_SIZE // config.N_WAY):
            summed_policy = 0
            for j in range(config.N_WAY):
                summed_policy += weighted_policies[idx]
                idx += 1
            idx -= config.N_WAY

            corrected_policy = self.correct_policy(
                summed_policy, batch_task_tensor[idx], mask=True)

            for j in range(config.N_WAY):
                weighted_policies[idx] = corrected_policy
                tasks[i]["memories"].extend([{"policy": corrected_policy}])
                idx += 1

        corrected_policies = weighted_policies

        # The goal is average over the 5 semi random policies weighted by the Q's
        # so at first we make a new uncorrected policy, then we pass it through the net
        # just for the Q value

        # scales from -1 to 1 to 0 to 1
        # scaled_qs = (qs + 1) / 2
        # weighted_policies = policies * scaled_qs

        # improved_task_policies = self.get_improved_task_policies_list(
        #     weighted_policies)

        #***Idea***
        # since the initial policy will be for the first state, we could average
        # the whole batch and argmax to pick the next initial state,
        # effect following a very strong trajectory, and maybe biasing play
        # towards better results?
        # Although we are naturally seeing early states a lot more since those are the seeds
        # for trajectories. So, the policy should be especially good for those

        # well in theory since the orig policies are partially random and
        # the final policy is random, using only the improved policy might be fine
        # will set it like that for now. it will lead to a bit less
        # diversity in the policies that the Q sees, which is kind of bad.
        # but then again, we will get more of a true Q value for that policy.
        # we can try it out for now. ill set to .8 so some difference happens
        # final_policies = self.mix_task_policies(improved_task_policies,
        #                                         policies, perc_improved=1)

        # corrected_final_policies = self.correct_policies(
        #     final_policies, batch_task_tensor)

        # qs, _ = qp(batch_task_variable, corrected_final_policies)

        # tasks = self.update_task_memories(
        #     tasks, corrected_final_policies, improved_task_policies)

        is_done = []
        for _ in range(config.EPISODE_BATCH_SIZE):
            is_done.extend([False])

        # sooo let me think. the new_net and best_net will continually trade off batch
        # evaluations. basically the new_net chooses some initial_moves, and
        # then it alternates until all the games are done. #this will bias that the new
        # net always makes the first move, which can be significant
        # so now it's random start. so the opposing moves for each turn will be chosen
        # by the opposite net
        num_done = 0
        if best_start == 1:
            best_turn = 0
        else:
            best_turn = 1

        results = {
            "new": 0, "best": 0, "draw": 0
        }

        while num_done < config.EPISODE_BATCH_SIZE:
            batch_task_tensor = self.transition_batch_task_tensor(np.array(batch_task_tensor),
                                                                  corrected_policies, is_done)

            is_done, tasks, results, num_done = self.check_finished_games(batch_task_tensor, is_done,
                                                                          tasks, num_done, results, best_turn)

            batch_task_variable = self.wrap_to_variable(batch_task_tensor)

            if best_turn == 1:
                qp = self.best_qp
            else:
                qp = self.qp

            _, policies = self.qp(batch_task_variable)

            policies = policies.detach().numpy()

            policies = self.correct_policies(policies, batch_task_tensor)
            print("{} of {} done".format(num_done, config.TRAINING_BATCH_SHAPE))

        self.memories.extend(tasks)
        if len(self.memories) > config.MAX_TASK_MEMORIES:
            self.memories[-config.MAX_TASK_MEMORIES:]
        utils.save_memories(self.memories)

        print("Results: ", results)
        if results["new"] > results["best"] * config.SCORING_THRESHOLD:
            model_utils.save_model(self.qp)
            print("Loading new best model")
            self.best_qp = model_utils.load_model()
            if self.cuda:
                self.best_qp = self.best_qp.cuda()
        elif results["best"] > results["new"] * config.SCORING_THRESHOLD:
            print("Reverting to previous best")
            self.qp = model_utils.load_model()
            if self.cuda:
                self.qp = self.qp.cuda()
            self.q_optim, self.p_optim = model_utils.setup_optims(self.qp)

    def train_memories(self):
        self.qp.train()

        # so memories are a list of lists containing memories
        if len(self.memories) < config.MIN_TASK_MEMORIES:
            print("Need {} tasks, have {}".format(
                config.MIN_TASK_MEMORIES, len(self.memories)))
            return

        for _ in tqdm(range(config.TRAINING_LOOPS)):
            tasks = sample(self.memories, config.SAMPLE_SIZE)

            BATCH_SIZE = config.TRAINING_BATCH_SIZE // config.N_WAY
            extra = config.SAMPLE_SIZE % BATCH_SIZE
            minibatches = [
                tasks[x:x + BATCH_SIZE]
                for x in range(0, len(tasks) - extra, BATCH_SIZE)
            ]
            self.train_tasks(minibatches)

        # self.train_minibatches(minibatches)

    def train_tasks(self, minibatches_of_tasks):
        batch_task_tensor = np.zeros((config.TRAINING_BATCH_SIZE,
                                      config.CH, config.R, config.C))

        result_tensor = np.zeros((config.TRAINING_BATCH_SIZE, 1))

        policies_tensor = np.zeros((
            config.TRAINING_BATCH_SIZE, config.R * config.C))

        optimal_value_tensor = np.ones((config.TRAINING_BATCH_SIZE, 1))

        for mb in minibatches_of_tasks:
            self.q_optim.zero_grad()
            self.p_optim.zero_grad()

            Q_loss = 0
            policy_loss = 0

            idx = 0
            for i, task in enumerate(mb):
                state = task["state"]

                for memory in task["memories"]:
                    result_tensor[idx] = memory["result"]

                    policies_tensor[idx] = memory["policy"]
                    batch_task_tensor[idx] = state
                    idx += 1

            state_input = self.wrap_to_variable(batch_task_tensor)
            corrected_policies_input = self.wrap_to_variable(policies_tensor)
            result_target = self.wrap_to_variable(result_tensor)

            Qs, _ = self.qp(state_input, corrected_policies_input)

            Q_loss += F.mse_loss(Qs, result_target)

            Q_loss.backward()

            self.q_optim.step()

            self.q_optim.zero_grad()
            self.p_optim.zero_grad()

            Qs, policies = self.qp(state_input)

            for correct_policy, policy in zip(corrected_policies_input, policies):
                correct_policy = correct_policy.unsqueeze(0)
                policy = policy.unsqueeze(-1)
                policy_loss += -torch.mm(correct_policy,
                                         torch.log(policy)).squeeze()

            optimal_value_var = self.wrap_to_variable(optimal_value_tensor)            

            for _ in range(config.TRAINING_BATCH_SIZE):
                Qs, policies = self.qp(state_input)
                policy_loss += F.mse_loss(Qs, optimal_value_var)

            policy_loss.backward()

            self.p_optim.step()
