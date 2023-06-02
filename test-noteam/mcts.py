import argparse
#argparse 是 python 标准库里面用来处理命令行参数的库
import multiprocessing
from queue import Empty
import numpy as np
import time
import csv

import pommerman
from pommerman.agents import BaseAgent, SimpleAgent
from pommerman import constants



NUM_AGENTS = 4
NUM_ACTIONS = len(constants.Action) #6
#Python len () 方法返回对象（字符、列表、元组等）长度或项目个数。
NUM_CHANNELS = 18
#类型均为int



def argmax_tiebreaking(Q): #返回获利最大的操作的标记
    # find the best action with random tie-breaking
    idx = np.flatnonzero(Q == np.max(Q))
    #np.flatnonzero：输入一个矩阵，返回了其中非零元素的位置.
    #np.max 接受一个参数，返回一个最大值
    assert len(idx) > 0, str(Q)
    #assert 的作用是现计算表达式 expression ，如果其值为假（即为 0），
    #那么它先向 stderr 打印一条出错信息，然后通过调用 abort 来终止程序运行。
    #使用 assert 的缺点是，频繁的调用会极大的影响程序的性能，增加额外的开销。
    return np.random.choice(idx)
    #numpy.random.choice(a, size=None, replace=True, p=None)
    #从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
    #replace:True表示可以取相同数字，False表示不可以取相同数字
    #数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同

class MCTSNode(object):
    def __init__(self, p):
        # values for 6 actions
        self.Q = np.zeros(NUM_ACTIONS)#平均行动价值
        #print(np.zeros(NUM_ACTIONS)  [0. 0. 0. 0. 0. 0.]

        self.W = np.zeros(NUM_ACTIONS)#总行动价值
        self.N = np.zeros(NUM_ACTIONS, dtype=np.uint32)#访问节点的次数
        #typedef unsigned int   uint32_t;  
        #用法：zeros (shape, dtype=float, order='C')
        #返回：返回来一个给定形状和类型的用 0 填充的数组；
        #参数：shape: 形状
        #dtype: 数据类型，可选参数，默认 numpy.float64
        assert p.shape == (NUM_ACTIONS,)
        self.P = p#先验概率

    def action(self):
        U = args.mcts_c_puct * self.P * np.sqrt(np.sum(self.N)) / (1 + self.N)
        ##U (s, a) = c_puct × 概率 P (s, a) × np.sqrt (父节点访问次数 N) / ( 1 + 某子节点 action 的访问次数 N (s, a) )
        #mcts_c_puct 是一个决定探索水平的常数；
        #这种搜索控制策略最初倾向于具有高先验概率和低访问次数的行为，
        #但是渐近地倾向于具有高行动价值的行为
        
        return argmax_tiebreaking(self.Q + U)
        #计算过后，我就知道当前局面下，哪个 action 的 Q+U 值最大

    def update(self, action, reward):
        ###更新，回溯
        self.W[action] += reward #reward+1
        self.N[action] += 1      #访问次数+1
        self.Q[action] = self.W[action] / self.N[action] #reward/num 平均奖励

    def probs(self, temperature=1):
        if temperature == 0:
            #如果温度=0，那么先验概率均初始化为0，
            #且根据每个子节点的访问次数，随机抽取一个action，设置概率=1
            p = np.zeros(NUM_ACTIONS)
            p[argmax_tiebreaking(self.N)] = 1
            return p
        else:
            Nt = self.N ** (1.0 / temperature)
            #**表示次方
            return Nt / np.sum(Nt)
            #计算搜索概率推荐的移动向量 π = αθ(s)，
            #它与每次落子动作的访问计数的指数成正比，
            # πa ∝ N (s, a) 1/τ，其中 τ 是温度参数.

class SA(BaseAgent):
    def __init__(self,tree,root,agent_id=0):
        super().__init__()#初始化
        #super() 函数是用于调用父类 (超类) 的一个方法。
        self.agent_id = agent_id
        self.env = self.make_env()
        self.tree=tree
        self.root=root
        

    def make_env(self):#创建智能体对象
        agents = []
        for agent_id in range(NUM_AGENTS):
            if agent_id == self.agent_id:
                agents.append(self)
            else:
                agents.append(SimpleAgent())
        # print(agents)
        return pommerman.make('PommeFFACompetition-v0', agents)

    def search(self):
    
        self.env.training_agent = self.agent_id
        num_iters=args.mcts_iters
        temperature=args.temperature
        # remember current game state
        self.env._init_game_state = self.root
        root = str(self.root)
        # print(root)
        
        for i in range(num_iters):
            # restore game state to root node
            #将游戏状态恢复到根节点
            obs = self.env.reset()
            # serialize game state 序列化游戏状态
            # state = str(self.env.get_json_info())
            state=root
            trace = []
            done = False
            while not done:
                if state in self.tree:
                    node = self.tree[state]
                    # choose actions based on Q + U
                    #action函数，计算Q+U，返回获利最大的操作的标记
                    action = node.action()
                    trace.append((node, action))
                else:
                    # use unfiform distribution for probs
                    probs = np.ones(NUM_ACTIONS) / NUM_ACTIONS
                    #ones () 返回一个全 1 的 n 维数组
                    
                    # use current rewards 目前的奖励for values
                    rewards = self.env._get_rewards()
                    reward = rewards[self.agent_id]

                    # add new node to the tree
                    self.tree[state] = MCTSNode(probs)

                    # stop at leaf node 在叶子节点停止
                    break

                # ensure we are not called recursively递归地
                assert self.env.training_agent == self.agent_id

                # make other agents act
                actions = self.env.act(obs)
                # add my action to list of actions
                actions.insert(self.agent_id, action)
                #print(type(actions))
                # step environment forward
                obs, rewards, done, info = self.env.step(actions)
                reward = rewards[self.agent_id]

                # fetch next state
                state = str(self.env.get_json_info())

            # update tree nodes with rollout results
            for node, action in reversed(trace):
                #reverse（sequence） - > 反转迭代器的序列值
                #返回反向迭代器
                node.update(action, reward)
                reward *= args.discount #0.99

        # reset env back where we were
        self.env.set_json_info()
        self.env._init_game_state = None
        # return action probabilities
        #root = str(self.root)
        values=self.tree[root].probs(temperature)
        # print(values)
        # print("yes")
        # print(self.tree)
        return values, self.tree

    def act(self, obs, action_space):
        # TODO
        assert False




class MCTSAgent(BaseAgent):
    def __init__(self, agent_id=0):
        super().__init__()#初始化
        #super() 函数是用于调用父类 (超类) 的一个方法。
        self.agent_id = agent_id
        self.env = self.make_env()
        self.reset_tree()
        temperature=args.temperature

    def make_env(self):#创建智能体对象
        agents = []
        for agent_id in range(NUM_AGENTS):
            if agent_id == self.agent_id:
                agents.append(self)
            else:
                agents.append(SimpleAgent())
        # print(agents)
        return pommerman.make('PommeFFACompetition-v0', agents)
        #'PommeFFACompetition-v0', 'PommeFFACompetitionFast-v0', 
        #'PommeFFAFast-v0', 'PommeFFA-v1', 'OneVsOne-v0',
        #'PommeRadioCompetition-v2', 'PommeRadio-v2', 
        #'PommeTeamCompetition-v0', 'PommeTeamCompetitionFast-v0', 
        #'PommeTeamCompetition-v1', 'PommeTeam-v0', 'PommeTeamFast-v0'
        #return pommerman.make('OneVsOne-v0',agents)

    def reset_tree(self):#清空树
        self.tree = {}
        #字典形式，对应存储不同state下 MCTSNode类

    def rollout(self):
        # reset search tree in the beginning of each rollout首次展示
        self.reset_tree()

        # guarantees that we are not called recursively保证我们不会被递归调用
        # and episode ends when this agent dies 并且当这名智能体死亡的时候，这一次结束
        self.env.training_agent = self.agent_id
        obs = self.env.reset()
        yes=False
        length = 0
        done = False
        while not done:
            if args.render:
                self.env.render()
                
            root = self.env.get_json_info()
            # do Monte-Carlo tree search
            # start=time.time()
            L=SA(self.tree,root,self.agent_id)
            pi,self.tree = L.search()
        
            # sample action from probabilities
            action = np.random.choice(NUM_ACTIONS, p=pi)
   
            #numpy.random.choice(a, size=None, replace=True, p=None)
            #从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
            #replace:True表示可以取相同数字，False表示不可以取相同数字
            #数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同
        
            # ensure we are not called recursively
            assert self.env.training_agent == self.agent_id
            # make other agents act
            actions = self.env.act(obs)

            # add my action to list of actions
            actions.insert(self.agent_id, action)

            obs, rewards, done, info = self.env.step(actions)
            assert self == self.env._agents[self.agent_id]
            length += 1
        reward = rewards[self.agent_id]
        
        if rewards==[-1,-1,-1,-1]:
            yes=0
        elif reward==1:
            yes=1
        else:
            yes=-1
        return length, reward, rewards,yes

    def act(self, obs, action_space):
        # TODO
        assert False


# def runner(id, num_episodes, fifo, _args):
def runner(id, num_episodes, _args):
    # make args accessible to MCTSAgent
    global args
    args = _args
    # make sure agents play at all positions
    agent_id = id % NUM_AGENTS
    agent = MCTSAgent(agent_id=agent_id)
    
    start_time = time.time()
    length, reward, rewards,yes = agent.rollout()
    elapsed = time.time() - start_time

    return (length, reward, rewards, agent_id, elapsed,yes)

# def profile_runner(id, num_episodes, fifo, _args):
#     import cProfile
#     #cProfile 是一种确定性分析器，只测量 CPU 时间，并不关心内存消耗和其他与内存相关联的信息
#     command = """runner(id, num_episodes, fifo, _args)"""
#     #主要用来测试runner这个函数的性能的
#     cProfile.runctx(command, globals(), locals(), filename=_args.profile)
#     #globals() 函数会以字典类型返回当前位置的全部全局变量
#     #locals() 函数会以字典类型返回当前位置的全部局部变量。

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--profile')
   
    parser.add_argument('--render', action="store_true", default=False)


    # runner params
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--num_runners', type=int, default=4)

    # MCTS params
    parser.add_argument('--mcts_iters', type=int, default=400)
    parser.add_argument('--mcts_c_puct', type=float, default=1.0)
    
    # RL params
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--temperature', type=float, default=0)
    #ArgumentParser 通过 parse_args() 方法解析参数
    args = parser.parse_args()

    # assert args.num_episodes % args.num_runners == 0, "The number of episodes should be divisible by number of runners"

    # # use spawn method for starting subprocesses
    # ctx = multiprocessing.get_context('spawn')
    # #multiprocessing.get_context(method=None)，返回多进程里的环境变量。
    # #使用 multiprocessing 模块提供的 get_context () 函数来设置进程启动的方法，
    # # 调用该函数时可传入 "spawn"、"fork"、"forkserver" 作为参数，用来指定进程启动的方式。
    # #需要注意的一点是，前面在创建进程是，使用的 multiprocessing.Process () 这种形式，而在使用 get_context () 函数设置启动进程方式时，
    # # 需用该函数的返回值，代替 multiprocessing 模块调用 Process ()。
    

    all_time=[]
    all_yes=0
    all_yes1=0
    dataname="mcts_staic_test"+str(args.mcts_iters)+".csv"
    for i in range(args.num_episodes):
        length, reward, rewards, agent_id, elapsed,yes =runner(i, args.num_episodes, args)
        all_time.append(elapsed)
        if yes==1:
            all_yes1=all_yes1+1
            all_yes=all_yes+1
            yes_asign="win"
        elif yes==0:
            all_yes=all_yes+1
            yes_asign="Tie"
        else:
            yes_asign="Fail"
        exp=["Episode:", i+1,"Agent_id",agent_id, "Reward:", reward,"reward:", rewards,"length:",length,"this_time:",elapsed,"victory:",yes_asign,"victory_rate:(in_tie)",all_yes/(i+1),"victory_rate:(no_tie)",all_yes1/(i+1),"All_time",np.sum(all_time),"Avg_time",np.mean(all_time),"Avg_step",elapsed/length]
        with open(dataname,"a",newline="")as out :
            csv_write = csv.writer(out,dialect = 'excel')
            csv_write.writerow(exp)


