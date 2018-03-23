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

        qs, policies = self.qp(batch_task_variable, percent_random=.2)

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

        num_done = 0
        while num_done < config.EPISODE_BATCH_SIZE:
            batch_task_tensor = self.transition_batch_task_tensor(batch_task_tensor, 
                corrected_final_policies, is_done)

            is_done, tasks = self.check_finished_games(batch_task_tensor, is_done, tasks) 

            batch_task_variable = Variable(batch_task_tensor)

            _, policies = self.qp(batch_task_variable)

        return memory

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

    def do_move(self, state, turn, legal_actions, curr_player):
        memory = dict()
        memory["curr_player"] = curr_player

        half_board = np.random.choice([0, 1], size=state[0].shape).flatten()
        std = np.std(half_board, axis=0)
        mean = np.mean(half_board, axis=0)

        inputs = torch.from_numpy(
            np.zeros(shape=config.BATCHED_SHAPE, dtype="float32"))
        for i in range(config.BATCH_SIZE):
            noise = np.random.normal(loc=mean, scale=std,
            size=(1,) + state[0].shape)
            noisy_state = np.concatenate((state, noise), axis=0)
            noisy_state = noisy_state.astype("float32")
            noisy_state = torch.from_numpy(noisy_state)

            if i == 0:
                memory["noisy_state"] = noisy_state

            inputs[i] = noisy_state

        inputs = Variable(inputs)
        if self.cuda:
            inputs = inputs.cuda()

        is_real_list = [np.random.uniform() > .5 for _ in range(len(inputs))]
        self.mcts.eval()
        _, _, policies, _, _ = self.mcts(inputs, is_real_list)
        memory["is_real_list"] = is_real_list

        policy_sum = 0
        for policy in policies:
            policy_sum += policy

        policy_sum = policy_sum.data.numpy()
        policy_sum += np.abs(policy_sum[np.argmin(policy_sum)] * (1 + 1e-7))
        policy = policy_sum

        if turn == 1 or turn == 2:
            nu = np.random.dirichlet([config.ALPHA] * policy.shape[0])
            policy = policy * (1 - config.EPSILON) + nu * config.EPSILON

        mask = np.zeros(policy.shape)
        mask[legal_actions] = 1
        policy *= mask
        pol_sum = (np.sum(policy) * 1.0)
        if pol_sum == 0:
            pass
        else:
            policy = policy / pol_sum

        memory["policy"] = policy

        if turn < config.TURNS_UNTIL_TAU0:
            action = np.random.choice(self.actions, p=policy)
        else:
            action = np.argmax(policy)

        state[curr_player] = self.transition(state[curr_player], action)

        curr_player += 1
        curr_player = curr_player % 2

        state[2] = curr_player

        return state, memory

    def train_memories(self, memories):
        self.mcts.train()

        data = sample(memories,
            config.SAMPLE_SIZE - config.SAMPLE_SIZE % config.BATCH_SIZE)
        minibatches = [
            data[x:x + config.BATCH_SIZE]
            for x in range(0, len(data), config.BATCH_SIZE)
        ]

        self.train_minibatches(minibatches)

    def is_valid(self, state):
        state = state.detach().numpy()
        state[np.where(state >= .5)] = 1
        state[np.where(state < .5)] = 0
        return self.test_valid(state)

    def train_minibatches(self, minibatches):
        for e in range(config.EPOCHS):
            if e > 0:
                shuffle(minibatches)

            for mb in minibatches:
                self.optim.zero_grad()

                valid_loss = 0
                value_loss = 0
                policy_loss = 0
                weights = [1, 1, 1]

                valids_list = []
                invalids_list = []
                is_real_list = []

                results = []

                mb_tensor = torch.FloatTensor(torch.from_numpy(np.zeros((config.BATCH_SIZE,) +
                    mb[0]["noisy_state"].shape, dtype="float32")))

                for i, memory in enumerate(mb):
                    results.extend([memory["result"]])
                    mb_tensor[i] = memory["noisy_state"]
                    is_real_list = memory["is_real_list"]

                mb_tensor = Variable(mb_tensor, requires_grad=True)
                if self.cuda:
                    mb_tensor = mb_tensor.cuda()

                valids, values, policies, imagined_states, _ = self.mcts(
                    mb_tensor, is_real_list)
                del mb_tensor
                for i, state in enumerate(imagined_states):
                    test_result = self.is_valid(state)
                    if test_result == -1:
                        invalids_list.extend([test_result])
                    else:
                        valids_list.extend([test_result])
                set_trace()

                # so the gan loss is the log probability for the discriminator
                # gan stuff:
                # https://paper.dropbox.com/doc/Wasserstein-GAN-GvU0p2V9ThzdwY3BbhoP7


def WGAN_GP(lmbda=10, n_critic=5, alpha=0.0001, beta_1=0, beta_2=0.9, m=config.BATCH_SIZE):
    w_0 = np.random.uniform()
    theta_0 = np.random.uniform()

    while True:
        for t in range(n_critic):
            for i in range(m):
                real_x = state
                # how big should the noise be. I feel like the plane of all ones and zeros
                # is way too hard to replicate. We can probably make a head or something
                # that outputs whether it is player one with the state or not.
                # the player is very important for this type of game, so
                # it needs something. I guess a head which will make a
                # so it sounds like latent variables dont need to match the image shape
                # necessarily
                # latent_variable = np.random.uniform(size=state.shape[0])
                # or do it for the batch_size

                # For what I want I need the input state
                # Because the validity depends on the input state
                # so the net will have a state + a latent variable for input (concated)
                # it will use this to make a generator create an imagined state
                # then the discriminator will take as input
                # x (internal representation), input_state (WITHOUT noise) and
                # x, imagined_state
                # I might need to remove the x since it has information, but since the
                # generator and discriminator are separate heads it might be okay.

                # so the network will output the generator, and the discriminator for the
                # real and the fake

                # the gen cost will be -torch.mean(disc_fake)
                # the disc cost will be torch.mean(disc_fake) - torch.mean(disc_real)

                # plus all of the other wgan stuff
                # so basically the gan is going to learn a representation
                # that allows for easy creation of a valid state
                # i.e. enforcing knowledge about what a valid state is.
                # in theory this should allow the value and policy to be more informed

                # perhaps we can take this a step further and have a loss which
                # creates policies or transitions that are wins or close to optimal
                # an issue with that is we need an optimal learning signal, which wouldnt work
                # maybe just produce a weighted policy by it's value function
                # so if we can learn a good value function we can scale how important the policy
                # is.

                # so what do we want. we want the network to produce a policy which is optimal
                # i.e. we want the maximum value for that state.
                # so maybe we show some low value states and high value states and produce a
                # valid high value state?
                # ideally we would like to infer the loss function since it is harder to
                # engineer one. meta learning encapsulates the loss function in a way
                # in other words it says "here are some examples, use them to determine
                # the true value for some test examples"
                # so basically with some examples of states, their policies, and their values
                # (under that policy), we want the network to infer the policy for
                # a value as close to 1 (max) as possible.
                # so basically we want to network to learn a policy improvement function
                # given a set of states (maybe a batch), and a value function output for each
                # of them, output the state

                # so maybe we want to fill in the gaps, i.e. we are given a bunch of states
                # with their policy and value (under that policy, I think a Q function)
                # we want to produce the policy for a given value and state (or as close)
                # to it as possible

                # so we have a meta learning problems, 5-way between 5 states w/ policy and value
                # concated. where did those come from? we want a net that takes a state and
                # outputs a policy and a value based on that state and policy, or perhaps
                # the representation.

                # the meta learner takes those 5 examples and guesses the policy for
                # a state. i.e. we are shown examples of policies, values, and states,
                # and we are asking the net to use those to produce a policy which has a
                # value as close to one as possible

                # so lets sketch it out
                # net inputs: state
                # net internal creates representation (res blocks)
                # net produces a policy

                # in here we could have an imagined valid future state generator which both
                # helps train the representation and would allow to peak at future valid states
                # so at test time it takes a latent variable and produces a valid transition
                # with it. that valid transition is then input to the policy and both to the head
                # that could be a way to use multiple examples from the same "game domain"

                # net also produces a value based on the representation and the policy (Q?)
                # the final outputs of the net are the policy and the value
                # those will be trained with the meta learning system
                # the meta learner takes a cat(state, policy, value, representation?),
                # does an internal representation (res blocks)
                # outputs the original state, with value of 1, and the policy going along with it
                # basically it is finding the mapping for a state and optimal value to
                # a policy. We need a wide array of policies (i.e. dirichlet noise or latent)
                # variables, to enable the network to not merely learn a uniform mapping
                # and again what will the loss be. we are trying to find a policy, so
                # we want the policy to be what we guess. so maybe for the examples
                # we have a bunch of states and values, and we're trying to predict a policy
                # for that.

                # so basically given a bunch of state, (result-value), predict the policy
                # associated with that. one issue is that there may be many policies with
                # similar values, and it may cause overfitting to the existing policy.

                # the loss in meta learning is

                # so for the meta learning problem we w

                # so my understanding of the meta learning process is that we
                # use a series of tasks, which is a 5-way examples, where the
                # so basically

                # so basically task Ti is a sample from some domain.
                # for example Ti could be sampled from one game
                # i.e. the distribution is the memories from all games
                # and the Ti is the distribution for one game
                # game=episode
                # so the goal of the network is to minimize the loss over
                # many different tasks, i.e. for examples from
                # one game, we want to minimize loss

                # so we run through a series of tasks, i.e. random moves from a game
                # perhaps we wise to enforce that half are wins and half are losses, but
                # it should even out.

                # so we sample from the same game.
                # so for example given the state, and policy for one

                # could we do meta learning with a gan?
                # so we want the gan loss for each new game
                # so for a new game we want to predict a transitioned state which
                # is indistinguishable from a valid configuration (i.e. randomly)
                # swapped with the input state
                # such that the transitioned state has a maximum value
                # i.e. we want to learn the optimal transition
                # so we have a wgan-gp that discriminates between invalid and valid states
                # this gives us valid future predictions to the future (or might give)
                # the original state if it misbehaves)
                # but assuming it doesnt, these future states can provide

                # well lets examine what we have:
                # we have transitions, states, values, rewards, and policies
                # what do we ideally want:
                # value is less efficient to compute but arguably the easier to do
                # transition is the most general and can allow for skipping steps or generalizing
                # the problem from reinforcement learning to other domains
                # policy allows for learning a stochastic policy and is more efficient than value
                # rewards are used to train the value function

                # so we always want to learn a value function to learn it somehow
                # we may also want to learn a q function, where based on a given policy
                # we estimate the value for that

                # capturing the power of gan's would be nice, since it is a powerful learner

                # what we ultimately want in this case is a policy.
                # we want a net that does {state -> net -> policy*}
                # how can we use meta learning to accomodate this.
                # we want to use different task domains
                # different domains could be different games, different policies?
                # different values?, not sure
                # the most

                # we want the input data to be self-play data from the latest net,
                # i.e. we want to focus on a input space? which is better and better players
                # we could use different games, such as given the value function
                # for a policy

                # so lets assume the task domain is games
                # we have a bunch of states whose losses are the difference between the
                # value and the reward (based on the policy, or maybe it's assumed)
                # so the meta learning task is to find the loss for a

                # so basically we have a series of

                # so we have a gan that discriminates between
                # again describe what I want and work backwards

                # I want a meta learning system which ideally utilizes gan training
                # to generate an optimal policy.

                # meta learning consists of using an internal loss function over a distribution
                # of tasks, i.e. tasks=episodes and there might be a few (state, result) labels
                # for each task.

                # so based on the state, result labels we want to produce a policy that
                # will perform will in an arbitrary domain (game)

                # so under that system we could create the inner loss to be minimizing
                # mse(value, result)
                #-sum(log(pi(a|theta)))*(result-value) for all a
                # so the inner loss function would be REINFORCE
                # and the outer loss function would be minimizing that loss over the different domains

                # meta learning models are a bit specialized in it, so it may be worth
                # looking into meta-sgd and such

                # so that is option one

                # the other ideas I have are this:
                # we want to have an emsemble of policy predictions where the weight of
                # each prediction is determined by the value function
                # in that case the value needs to be of a future state
                # which is trained to be a valid continuation (that hopefully doesnt cheat)
                # so basically the idea is have an internal generator generate an imagined
                # valid future state, and use the value function of that to determine
                # the weight that the policy prediction will get.
                # I think there is still a disconnect of how do we get the policy to point
                # towards the imagined future state

                # could we imagine a policy which is added (or concated) to the original state
                # and is used as in a q function?
                # so basically the generator imagines a policy and
                # then the net predicts a value based on the state and policy.

                # and the idea is that based on the value of the predicted policy
                # we want to weigh how much it's policy is weighted in the estimate of
                # the ideal policy. so we do some number of simulations about imagining
                # a policy and estimating a value based on it and the state

                # then we average the simulations based on the values, so that in theory
                # the higher value policies will have more weight

                # so a gan, or maybe just a net imagines a random policy, and
                # uses it to estimate a Q function

                # during the training phase the Q function is compared to the value of the
                # result, and the policy is compared to the averaged policy

                # as such high value policies should be encouraged

                # so basically the net idea is:
                # input: state, latent_variable
                # for _ in range(num_simulations):
                    # compute random policy
                    # compute Q of (state, policy)
                # wrong
                # average together policy*Q
                # output policy, Q

                ##

                # we just want the net to output the random policy and the Q
                # we can do a batch of however many estimates the want
                # then we compare the random policy to the averaged policy
                # one issue is how will we make it a non random policy?
                # what we could maybe do instead is just do a deterministic policy
                #+ dirichlet noise, and compute the values for those.
                # or we can have the net receive the latent_variable later on
                # and learn a gate for how much randomness to let in

                # training is:
                # mse(Q, result)
                # alpha zero loss between random Q and averaged Q

                # what I have now guesses a transition, which allows for it to be invalid
                # that makes the necessity for the gan which produces more likely to be valid
                # imagined states. and would allow the V function instead, where we learn
                # the value of two states
                # so the transition makes it quite a bit more complicated
                # we want the gan to imagine a number of random future valid states
                # and we want to use the V() of those to estimate the V for the original
                # state.

                # how about a filling in idea.
                # so we give a state, a Q value and we want to guess a policy to go along with it
                # we will feed the network with random policies since we want a

                # we dont want to use random policies. we want to use an ever improving policy
                # from a neural net.

                # so based on a semi-random policy and a Q function based on that, infer a policy
                # that would be optimal. i.e. given a state and a desired Q of 1, predict
                # an optimal policy that is as close to Q=1 as possible.

                # can kind of see Q as a critic and pi as a generator
                # we want WGAN-GP to

                # we will give a real policy which is a true observed policy
                # we will have the policy generator generate a fake policy
                # we will minimize the dif

                # so the generator cost is the -average pred over the fake data
                # so on average if the generator produces data the discriminator thinks
                # is real it will have a lower loss (the -)
                # the discriminator is the average of the fake - average of the true +
                # kind of complicated wgan thing

                # so how could you have a half fake half real thing with this
                # you could have half neural net policies and half random ones
                # and you want to learn the difference?
                # doesnt make sense
                # think about the goal
                # so the critic gets more updates, that's one take away
                # i.e. the critic will
                # I think that might be a wgan thing where we want the value to be really accurate
                # I need to make everything fit or nothing imo

                # can this issue be described as a gan?

                # traditionally a gan is discriminating between real and fake
                # if we had an optimal policy that would great, but we dont

                # how about we are trying to produce Q functions that are indistinguishable from
                # real? i.e. we try to make a perfect Q function.
                # so for example we generate a policy and a Q for it, and we randomly swap
                # the Q will a label
                #

                # oh okay maybe so we have a tried policy and a Q function.
                # we randomly swap the Q and the label
                # we try to have a discriminator discriminate between the Q and the label
                # this will force Q's to go closer to 1 or -1, but it doesn't really get us
                # maybe the value function.

                # idk last remarks:
                # consider having the net try to guess an optimal policy when it's given a reward, a q function and a state
                # give inputs to a meta net, i.e. from a domain
                # have it receive a state, a q function and a
                #
                # so basically have the net receive a state, a q function, a reward, and infer the policy
                # then we ask the net to predict the policy given the state, q function of 1 and reward of 1
                #
                # so the net gets a state
                # outputs a value function
                # outputs a policy function
                # and a q function based on the state and the policy

                # and we have a second net, or a divergence in it,
                # that takes a value, a q, and a state, and predicts the optimal policy
                # so the second part would take the q, v, state and predict the policy
                # and then using that at each turn we predict the policy of state, q=max(1)

                # improved wgan and implementation
                # https://github.com/igul222/improved_wgan_training/blob/master/gan_64x64.py
                # https://arxiv.org/pdf/1704.00028.pdf

                rand = np.random.uniform()
                latent_variables = np.random.uniform(size=state[:, 0].shape)
                input = torch.cat((states, latent_variables), axis=1)
                fake_x = G(input)
                mixed_x = rand * x + (1 - rand) * fake_x
                loss = D(fake_x) - D(real_x) + lmbda * (D(mixed_x) - 1)**2)
