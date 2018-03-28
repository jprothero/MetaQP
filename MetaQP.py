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
from copy import deepcopy

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
                action = np.random.choice(self.actions, p=policy)
                curr_player = int(batch_task_tensor[i][2][0][0])
                batch_task_tensor[i][curr_player] = self.transition(
                    batch_task_tensor[i][curr_player], action)
                batch_task_tensor[i][2] = (curr_player + 1) % 2

        return batch_task_tensor

    def check_finished_games(self, batch_task_tensor, is_done, tasks, num_done, results, bests_turn, best_starts):
        idx = 0
        for i in range(config.EPISODE_BATCH_SIZE // config.N_WAY):
            for j in range(config.N_WAY):
                if not is_done[idx]:
                    state = batch_task_tensor[idx]
                    legal_actions = self.get_legal_actions(state[:2])
                    if len(legal_actions) == 0:
                        is_done[idx] = True
                        num_done += 1
                        tasks[i]["memories"][j]["result"] = 0
                        if results is not None:
                            results["draw"] += 1
                    else:
                        reward, game_over = self.calculate_reward(state[:2])

                        if game_over:
                            is_done[idx] = True
                            num_done += 1

                            if results is not None:
                                for k in range(config.N_WAY-j):
                                    if k == 0:
                                        pass
                                    elif not is_done[idx+k]:
                                        is_done[idx+k] = True
                                        is_done[idx] = False
                                        batch_task_tensor[idx] = np.array(
                                            batch_task_tensor[idx+k])
                                        break
                                if bests_turn == best_starts:
                                    key = "best"
                                    other = "new"
                                else:
                                    key = "new"
                                    other = "best"

                                if reward == 1:
                                    results[key] += 1
                                else:
                                    results[other] += 1
                            else:
                                starting_player = tasks[i]["starting_player"]
                                curr_player = int(state[2][0][0])
                                if starting_player != curr_player:
                                    reward *= -1

                                tasks[i]["memories"][j]["result"] = reward
                idx += 1

        return is_done, tasks, results, num_done, batch_task_tensor

    def get_states_from_next_tensor(self, next_batch_task_tensor):
        states = []
        for i, state in enumerate(next_batch_task_tensor):
            states.extend([np.array(state)])

        return states

    def setup_tasks(self, states, starting_player_list, episode_is_done):
        tasks = []
        batch_task_tensor = np.zeros((config.EPISODE_BATCH_SIZE,
                                      config.CH, config.R, config.C))
        idx = 0
        for i in range(config.EPISODE_BATCH_SIZE // config.N_WAY):
            if not episode_is_done[i]:
                task = {
                "state": states[i],
                "starting_player": starting_player_list[i],
                "memories": []
                }
                tasks.extend([task])
            else:
                tasks.extend([None])

            for _ in range(config.N_WAY):
                batch_task_tensor[idx] = states[i]
                idx += 1

        return batch_task_tensor, tasks

    def run_episode(self, orig_states):
        np.set_printoptions(precision=3)
        results = {
            "new": 0, "best": 0, "draw": 0
        }
        states = np.array(orig_states)
        episode_is_done = []
        for _ in range(config.EPISODE_BATCH_SIZE):
            episode_is_done.extend([False])

        episode_num_done = 0

        best_starts = np.random.choice(2)
        # so I think I just need one indicator telling if the best player started or not
        # according to that if best_turn == best_started the rewards are given
        # I think this should maybe be renamed, it's confusing
        # So I want to determine if the best_starts or not
        # if best_starts==1, best_qp is used
        # #if turn_is_best == 1,
        # starting_player_list = []
        # for _ in range(config.EPISODE_BATCH_SHAPE//config.N_WAY):
        #     if best_turn:
        #         starting_player_list.extend([])

        starting_player_list = [np.random.choice(2) for _ in range(
            config.EPISODE_BATCH_SIZE//config.N_WAY)]

        if len(states) != config.CH:
            for i, state in enumerate(states):
                states[i] = np.copy(state)
                states[i][2] = starting_player_list[i]
        else:
            new_states = []
            for starting_player in starting_player_list:
                new_state = np.copy(states)
                new_state[2] = starting_player
                new_states.extend([new_state])
            states = new_states

        bests_turn = best_starts
        while episode_num_done < config.EPISODE_BATCH_SIZE:
            print("Num done {}".format(episode_num_done))
            states, episode_is_done, episode_num_done, results = self.meta_self_play(states,
                                                                                     episode_is_done,
                                                                                     episode_num_done,
                                                                                     bests_turn,
                                                                                     results,
                                                                                     best_starts,
                                                                                     starting_player_list)
            bests_turn = (bests_turn+1) % 2

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

    def meta_self_play(self, states, episode_is_done, episode_num_done, bests_turn,
                       results, best_starts, starting_player_list):
        self.qp.eval()
        self.best_qp.eval()
        batch_task_tensor, tasks = self.setup_tasks(
            states, starting_player_list, episode_is_done)

        batch_task_variable = self.wrap_to_variable(batch_task_tensor)

        if bests_turn == 1:
            qp = self.best_qp
        else:
            qp = self.qp

        _, policies = qp(batch_task_variable, percent_random=.2)

        policies = policies.detach().data.numpy()

        policies = self.correct_policies(policies, batch_task_tensor)

        policies_copy = np.array(policies)

        policies_input = self.wrap_to_variable(policies)

        qs, _ = qp(batch_task_variable, policies_input)

        qs = qs.detach().data.numpy()

        idx = 0
        for i in range(config.EPISODE_BATCH_SIZE // config.N_WAY):
            for _ in range(config.N_WAY):
                if tasks[i] is not None:
                    tasks[i]["memories"].extend([{"policy": policies[idx]}])
                idx += 1

        scaled_qs = (qs + 1) / 2
        weighted_policies = policies * scaled_qs

        idx = 0
        for i in range(config.EPISODE_BATCH_SIZE // config.N_WAY):
            summed_policy = 0
            for _ in range(config.N_WAY):
                summed_policy += weighted_policies[idx]
                idx += 1
            idx -= config.N_WAY

            corrected_policy = self.correct_policy(
                summed_policy, batch_task_tensor[idx], mask=True)

            if tasks[i] is not None:
                tasks[i]["improved_policy"] = np.array(corrected_policy)
            for _ in range(config.N_WAY):
                weighted_policies[idx] = corrected_policy
                idx += 1

        is_done = deepcopy(episode_is_done)

        corrected_policies = weighted_policies

        next_batch_task_tensor = self.transition_batch_task_tensor(np.copy(batch_task_tensor),
                                                                   corrected_policies, episode_is_done)
        
        

        bests_turn = (bests_turn+1) % 2

        episode_is_done, _, _, episode_num_done, next_batch_task_tensor = self.check_finished_games(np.copy(next_batch_task_tensor), is_done=episode_is_done,
                                                                                                    tasks=tasks, num_done=episode_num_done,
                                                                                                    results=results, bests_turn=bests_turn, best_starts=best_starts)

        next_states = self.get_states_from_next_tensor(next_batch_task_tensor)

        # revert back to orig turn now that we are done
        bests_turn = (bests_turn+1) % 2

        # switch back to regular policies so that we can test them.
        # corrected_policies = policies_copy

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

        # sooo let me think. the new_net and best_net will continually trade off batch
        # evaluations. basically the new_net chooses some initial_moves, and
        # then it alternates until all the games are done. #this will bias that the new
        # net always makes the first move, which can be significant
        # so now it's random start. so the opposing moves for each turn will be chosen
        # by the opposite net
        num_done = episode_num_done

        # policies = corrected_policies
        policies = policies_copy

        while num_done < config.EPISODE_BATCH_SIZE:
            batch_task_tensor = self.transition_batch_task_tensor(np.array(batch_task_tensor),
                                                                  policies, is_done)
            bests_turn = (bests_turn+1) % 2
            if episode_num_done > 1:
                set_trace()
            is_done, tasks, _, num_done, _ = self.check_finished_games(batch_task_tensor, is_done,
                                                                    tasks, num_done, None, bests_turn, best_starts)

            batch_task_variable = self.wrap_to_variable(batch_task_tensor)

            if bests_turn == 1:
                qp = self.best_qp
            else:
                qp = self.qp

            _, policies = self.qp(batch_task_variable)

            policies = policies.detach().data.numpy()

            policies = self.correct_policies(policies, batch_task_tensor)
            # print("Miniround: {} of {} done".format(num_done, config.TRAINING_BATCH_SHAPE))

        fixed_tasks = []
        for i, task in enumerate(tasks):
            if task is not None:
                fixed_tasks.extend([task])

        self.memories.extend(fixed_tasks)
        if len(self.memories) > config.MAX_TASK_MEMORIES:
            self.memories[-config.MAX_TASK_MEMORIES:]
        utils.save_memories(self.memories)

        return next_states, episode_is_done, episode_num_done, results

    def train_memories(self):
        self.qp.train()

        # so memories are a list of lists containing memories
        if len(self.memories) < config.MIN_TASK_MEMORIES:
            print("Need {} tasks, have {}".format(
                config.MIN_TASK_MEMORIES, len(self.memories)))
            return

        for _ in tqdm(range(config.TRAINING_LOOPS)):
            # tasks = sample(self.memories, config.SAMPLE_SIZE)
            minibatch = sample(self.memories, config.TRAINING_BATCH_SIZE)

            # BATCH_SIZE = config.TRAINING_BATCH_SIZE // config.N_WAY
            # extra = config.SAMPLE_SIZE % BATCH_SIZE
            # minibatches = [
            #     tasks[x:x + BATCH_SIZE]
            #     for x in range(0, len(tasks) - extra, BATCH_SIZE)
            # ]
            self.train_tasks(minibatch)

        utils.save_history(self.history)

        # self.train_minibatches(minibatches)

    def train_tasks(self, minibatch):
        batch_task_tensor = np.zeros((config.TRAINING_BATCH_SIZE,
                                      config.CH, config.R, config.C))

        policies_view = []
        for i in range(config.TRAINING_BATCH_SIZE):
            if i % config.N_WAY == 0:
                policies_view.extend([i])

        result_tensor = np.zeros((config.TRAINING_BATCH_SIZE, 1))

        policies_tensor = np.zeros((
            config.TRAINING_BATCH_SIZE, config.R * config.C))

        improved_policies_tensor = np.zeros((
            config.TRAINING_BATCH_SIZE//config.N_WAY, config.R * config.C))

        optimal_value_tensor = np.ones(
            (config.TRAINING_BATCH_SIZE//config.N_WAY, 1))

        idx = 0
        for i, task in enumerate(minibatch):
            state = task["state"]
            improved_policies_tensor[i] = task["improved_policy"]

            for memory in task["memories"]:
                result_tensor[idx] = memory["result"]

                policies_tensor[idx] = memory["policy"]
                batch_task_tensor[idx] = state
                idx += 1
        state_input = self.wrap_to_variable(batch_task_tensor)
        policies_input = self.wrap_to_variable(policies_tensor)
        improved_policies_target = self.wrap_to_variable(
            improved_policies_tensor)
        result_target = self.wrap_to_variable(result_tensor)
        
        optimal_value_var = self.wrap_to_variable(optimal_value_tensor)

        for e in range(config.EPOCHS):
            self.q_optim.zero_grad()
            self.p_optim.zero_grad()

            Q_loss = 0
            policy_loss = 0

            Qs, _ = self.qp(state_input, policies_input)

            Q_loss += F.mse_loss(Qs, result_target)

            Q_loss.backward()

            self.q_optim.step()
            
            self.q_optim.zero_grad()
            # self.p_optim.zero_grad() #should be redundant

            Qs, policies = self.qp(state_input)

            # corrected_policy_loss = 0
            # for corrected_policy, policy in zip(policies_input, policies):
            #     corrected_policy = corrected_policy.unsqueeze(0)
            #     policy = policy.unsqueeze(-1)
            #     corrected_policy_loss += -torch.mm(corrected_policy,
            #                                         torch.log(policy))
            # corrected_policy_loss /= 3*len(policies_input)

            policies_smaller = policies[policies_view]

            improved_policy_loss = 0
            for improved_policy, policy in zip(improved_policies_target, policies_smaller):
                improved_policy = improved_policy.unsqueeze(0)
                policy = policy.unsqueeze(-1)
                improved_policy_loss += -torch.mm(improved_policy,
                                                    torch.log(policy))

            improved_policy_loss /= len(policies_smaller)

            Qs_smaller = Qs[policies_view]

            # policy_loss = corrected_policy_loss +
            policy_loss = improved_policy_loss + \
                F.mse_loss(Qs_smaller, optimal_value_var)

            #/ and * 2 to balance improved policies matching and regression

            # for _ in range(config.TRAINING_BATCH_SIZE):
            # Qs, policies = self.qp(state_input)
            # policy_loss += F.mse_loss(Qs, optimal_value_var)

            policy_loss.backward()
            # policies.grad
            # set_trace()

            self.p_optim.step()
            p_loss = policy_loss.data.numpy()[0]
            q_loss = Q_loss.data.numpy()[0]
            self.history["q_loss"].extend([q_loss])
            self.history["p_loss"].extend([p_loss])

            if e == (config.EPOCHS-1):
                print("Policy loss {}".format(policy_loss.data.numpy()[0]))
                print("Q loss: {}".format(Q_loss.data.numpy()[0]))
