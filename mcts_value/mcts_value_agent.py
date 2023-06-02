import argparse
import multiprocessing
import numpy as np
import time

import cpommerman
import pommerman
from pommerman.agents import BaseAgent, SimpleAgent
from pommerman import constants

import tensorflow as tf
import keras.backend as K
from keras.models import load_model

NUM_AGENTS = 4
NUM_ACTIONS = len(constants.Action)
NUM_CHANNELS = 18


def argmax_tiebreaking_axis1(Q):
    # find the best action with random tie-breaking
    mask = (Q == np.max(Q, axis=1, keepdims=True))
    return np.array([np.random.choice(np.flatnonzero(m)) for m in mask], dtype=np.uint8)


def argmax_tiebreaking(Q):
    # find the best action with random tie-breaking
    idx = np.flatnonzero(Q == np.max(Q))
    assert len(idx) > 0, str(Q)
    return np.random.choice(idx)


class MCTSNode(object):
    def __init__(self, p):
        # values for 6 actions
        self.Q = np.zeros((NUM_AGENTS, NUM_ACTIONS))
        self.W = np.zeros((NUM_AGENTS, NUM_ACTIONS))
        self.N = np.zeros((NUM_AGENTS, NUM_ACTIONS), dtype=np.uint32)
        assert p.shape == (NUM_AGENTS, NUM_ACTIONS)
        self.P = p

    def actions(self):
        U = args.mcts_c_puct * self.P * np.sqrt(np.sum(self.N, axis=1, keepdims=True)) / (1 + self.N)
        return argmax_tiebreaking_axis1(self.Q + U)

    def update(self, actions, rewards):
        assert len(actions) == len(rewards)
        self.W[range(NUM_AGENTS), actions] += rewards
        self.N[range(NUM_AGENTS), actions] += 1
        self.Q[range(NUM_AGENTS), actions] = self.W[range(NUM_AGENTS), actions] / self.N[range(NUM_AGENTS), actions]

    def probs(self, temperature=1):
        if temperature == 0:
            p = np.zeros(self.N.shape)
            idx = argmax_tiebreaking_axis1(self.N)
            p[range(NUM_AGENTS), idx] = 1
            return p
        else:
            Nt = self.N ** (1.0 / temperature)
            return Nt / np.sum(Nt, axis=1, keepdims=True)


class MCTSAgent(BaseAgent):
    def __init__(self, model_file, agent_id=0):
        super().__init__()
        self.model = load_model(model_file)
        self.agent_id = agent_id
        self.env = cpommerman.make()
        self.reset_tree()

    def reset_tree(self):
        self.tree = {}

    def search(self, root, num_iters, temperature=1):
        # remember current game state
        self.env.set_json_info(root)
        root = self.env.get_state()

        for i in range(num_iters):
            # restore game state to root node
            self.env.set_state(root)
            # serialize game state
            state = root

            trace = []
            done = False
            # fetch rewards so we know which agents are alive
            rewards = self.env.get_rewards()
            while not done:
                if state in self.tree:
                    node = self.tree[state]
                    # choose actions based on Q + U
                    actions = node.actions()
                    # use Stop action for all dead agents to reduce tree size
                    actions[rewards != 0] = constants.Action.Stop.value
                else:
                    # initialize action probabilities with policy network
                    feats = self.env.get_features()
                    probs, values = self.model.predict(feats)

                    # add new node to the tree
                    self.tree[state] = MCTSNode(probs)

                    # for alive agents use state value, for others use returned reward
                    rewards[rewards == 0] = values[rewards == 0, 0]

                    # stop at leaf node
                    break

                # step environment forward
                self.env.step(actions)
                rewards = self.env.get_rewards()
                done = self.env.get_done()
                trace.append((node, actions, rewards))

                # fetch next state
                state = self.env.get_state()

            # update tree nodes with rollout results
            for node, actions, rews in reversed(trace):
                # use the reward of the last timestep where it was non-null
                rewards[rews != 0] = rews[rews != 0]
                node.update(actions, rewards)
                rewards *= args.discount

        # return action probabilities
        return self.tree[root].probs(temperature)

    def rollout(self, env):
        # reset search tree in the beginning of each rollout
        self.reset_tree()

        # guarantees that we are not called recursively
        # and episode ends when this agent dies
        env.training_agent = self.agent_id
        obs = env.reset()

        length = 0
        done = False
        while not done:
            if args.render:
                env.render()

            root = env.get_json_info()
            # do Monte-Carlo tree search
            pi = self.search(root, args.mcts_iters, args.temperature)
            pi = pi[self.agent_id]
            # sample action from probabilities
            action = np.random.choice(NUM_ACTIONS, p=pi)

            # ensure we are not called recursively
            assert env.training_agent == self.agent_id
            # make other agents act
            actions = env.act(obs)
            # add my action to list of actions
            actions.insert(self.agent_id, action)
            # step environment
            obs, rewards, done, info = env.step(actions)
            assert self == env._agents[self.agent_id]
            length += 1
            print("Agent:", self.agent_id, "Step:", length, "Actions:", [constants.Action(a).name for a in actions], "Probs:", [round(p, 2) for p in pi], "Rewards:", rewards, "Done:", done)

        reward = rewards[self.agent_id]
        return length, reward, rewards

    def act(self, obs, action_space):
        # TODO
        assert False


def init_tensorflow():
    # make sure TF does not allocate all memory
    # NB! this needs to be done also in subprocesses!
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.compat.v1.Session(config=config))


def runner(id, num_episodes, fifo, _args):
    # make args accessible to MCTSAgent
    global args
    args = _args

    # make sure TF does not allocate all memory
    init_tensorflow()

    # make sure agents play at all positions
    agent_id = id % NUM_AGENTS
    agent = MCTSAgent(args.model_file, agent_id=agent_id)

    # create environment with three SimpleAgents
    agents = [
        SimpleAgent(),
        SimpleAgent(),
        SimpleAgent(),
    ]
    agent_id = id % NUM_AGENTS
    agents.insert(agent_id, agent)

    env = pommerman.make('PommeFFACompetition-v0', agents)

    for i in range(num_episodes):
        # do rollout
        start_time = time.time()
        length, reward, rewards = agent.rollout(env)
        elapsed = time.time() - start_time
        # add data samples to log
        fifo.put((length, reward, rewards, agent_id, elapsed))


def profile_runner(id, num_episodes, fifo, _args):
    import cProfile
    command = """runner(id, num_episodes, fifo, _args)"""
    cProfile.runctx(command, globals(), locals(), filename=_args.profile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file')
    parser.add_argument('--profile')
    parser.add_argument('--render', action="store_true", default=False)
    # runner params
    parser.add_argument('--num_episodes', type=int, default=400)
    parser.add_argument('--num_runners', type=int, default=4)
    # MCTS params
    parser.add_argument('--mcts_iters', type=int, default=100)
    parser.add_argument('--mcts_c_puct', type=float, default=1.0)
    # RL params
    parser.add_argument('--discount', type=float, default=0.9)
    parser.add_argument('--temperature', type=float, default=0.4)
    args = parser.parse_args()

    assert args.num_episodes % args.num_runners == 0, "The number of episodes should be divisible by number of runners"

    # use spawn method for starting subprocesses
    ctx = multiprocessing.get_context('spawn')

    # create fifos and processes for all runners
    fifo = ctx.Queue()
    for i in range(args.num_runners):
        process = ctx.Process(target=profile_runner if args.profile else runner, args=(i, args.num_episodes // args.num_runners, fifo, args))
        process.start()

    # do logging in the main process
    all_rewards = []
    all_lengths = []
    all_elapsed = []
    for i in range(args.num_episodes):
        # wait for a new trajectory
        length, reward, rewards, agent_id, elapsed = fifo.get()

        print("Episode:", i, "Reward:", reward, "Length:", length, "Rewards:", rewards, "Agent:", agent_id, "Time per step:", elapsed / length)
        all_rewards.append(reward)
        all_lengths.append(length)
        all_elapsed.append(elapsed)

    print("Average reward:", np.mean(all_rewards))
    print("Average length:", np.mean(all_lengths))
    print("Time per timestep:", np.sum(all_elapsed) / np.sum(all_lengths))
