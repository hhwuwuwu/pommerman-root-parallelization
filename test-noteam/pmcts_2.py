#optparse 模块主要用来为脚本传递命令参数功能
import argparse
import collections
import csv
import math
import optparse
import queue
import random
#import sets
import time


import numpy as np
import pommerman
from pommerman import constants
from pommerman.agents import BaseAgent, SimpleAgent


NUM_AGENTS = 4
NUM_ACTIONS = len(constants.Action) #6
#Python len () 方法返回对象（字符、列表、元组等）长度或项目个数。
NUM_CHANNELS = 18
# PARALLEL_COUNT = 4
try:
    import java.lang
    PARALLEL_COUNT = java.lang.Runtime.getRuntime().availableProcessors()

except ImportError:
    import multiprocessing
    # PARALLEL_COUNT = multiprocessing.cpu_count()
    PARALLEL_COUNT =2
    #print(PARALLEL_COUNT)自己的电脑，CPU 8核

# def merge_dict(x,y): #关于生成树合并，后续可以考虑
#     result = {}
#     probs=np.ones(NUM_ACTIONS)/NUM_ACTIONS
#     for k,v in x.items():
#         if k in y.keys():
#             result[k]=y[k].add(v)
#         else:
#             result[k]=v
#     return result

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
        # U = args.mcts_c_puct * self.P * np.sqrt(np.sum(self.N)) / (1 + self.N)
        U = self.P * np.sqrt(np.sum(self.N)) / (1 + self.N)
        ##U (s, a) = c_puct × 概率 P (s, a) × np.sqrt (父节点访问次数 N) / ( 1 + 某子节点 action 的访问次数 N (s, a) )
        #mcts_c_puct 是一个决定探索水平的常数；
        #这种搜索控制策略最初倾向于具有高先验概率和低访问次数的行为，
        #但是渐近地倾向于具有高行动价值的行为
        
        return argmax_tiebreaking(self.Q + U)
        #计算过后，我就知道当前局面下，哪个 action 的 Q+U 值最大

    def update(self, action, reward):
        ###更新，回溯
        self.W[action] += reward
        self.N[action] += 1
        self.Q[action] = self.W[action] / self.N[action]

    def probs(self, temperature=1):
        if temperature == 0:
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
    # def add(self,other):
    #     probs=np.ones(NUM_ACTIONS)/NUM_ACTIONS
    #     New=MCTSNode(probs)
    #     New.P=[(self.P[i]+other.P[i])/2 for i in range(0,NUM_ACTIONS)]
    #     New.W=[(self.W[i]+other.W[i])   for i in range(0,NUM_ACTIONS)]
    #     New.N=[(self.N[i]+other.N[i])   for i in range(0,NUM_ACTIONS)]
    #     for  i in range(0,NUM_ACTIONS):
    #         if New.N[i]!=0:
    #             New.Q[i]=New.W[i]/New.N[i]
    #     return New
        
def SW(root_state,tree,info,_args,agent_id=0):
    workers=SearchWorker(root_state,tree,_args,agent_id)
    info.put(workers.run())

class SearchWorker (BaseAgent):
    def __init__(self,root_state,tree,args,agent_id=0):
        super().__init__()#初始化
        self.__root_state = root_state
        # global args
        self.args=args
        self.__iter_max = (self.args.mcts_iters//PARALLEL_COUNT)
        #print(self.__iter_max )
        self.agent_id = agent_id
        self.env=self.make_env()
        self.tree=tree


    # def run(self):
    #     print("puct-2")
    #     # values= self.run1()
    #     # values = np.ones(NUM_ACTIONS) / NUM_ACTIONS
    #     # self.__queue.put(values)

    def run(self):
        self.env.training_agent = self.agent_id     
        # remember3 current game state
        temperature=self.args.temperature
        self.env._init_game_state = self.__root_state
        root = str(self.__root_state)
        for i in range(self.__iter_max):
            # restore game state to self.__root_state node
            #将游戏状态恢复到根节点
            obs = self.env.reset()
            # serialize game state 序列化游戏状态
            # state = str(self.env.get_json_info())
            state= root
            trace = []
            done = False
            while not done:
                # print("uct-1")
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

                    # add new node to the self.tree
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

            # update self.tree nodes with rollout results
            for node, action in reversed(trace):
                #reverse（sequence） - > 反转迭代器的序列值
                #返回反向迭代器
                node.update(action, reward)
                # reward *= args.discount #0.99
                reward *= 0.99
        # reset env back where we were
        self.env.set_json_info()
        self.env._init_game_state = None
        # return action probabilities
        # return (self.tree[root].probs(temperature))
        value=self.tree[root].probs(temperature)
        # res={}
        # res[value]=self.tree
        return value,self.tree
    
    def make_env(self):#创建智能体对象
        agents = []
        for agent_id in range(NUM_AGENTS):
            if agent_id == self.agent_id:
                agents.append(self)
            else:
                agents.append(SimpleAgent())
        return pommerman.make('PommeFFACompetition-v0', agents)

    def act(self, obs, action_space):
        # TODO
        assert False

class PMCTS(BaseAgent):
    def __init__(self, agent_id=0):
        super().__init__()#初始化
        #super() 函数是用于调用父类 (超类) 的一个方法。
        self.agent_id = agent_id
        #self.enemy_id=(agent_id+1)%NUM_AGENTS
        self.env = self.make_env()
        self.reset_tree()
       
    def reset_tree(self):#清空树
        self.tree = {}
        #字典形式，对应存储不同state下 MCTSNode类
    
    def make_env(self):#创建智能体对象
        agents = []
        for agent_id in range(NUM_AGENTS):
            if agent_id == self.agent_id:
                agents.append(self)
            else:
                agents.append(SimpleAgent())
        return pommerman.make('PommeFFACompetition-v0', agents)
    #     #'PommeFFACompetition-v0', 'PommeFFACompetitionFast-v0', 
    #     #'PommeFFAFast-v0', 'PommeFFA-v1', 'OneVsOne-v0',
    #     #'PommeRadioCompetition-v2', 'PommeRadio-v2', 
    #     #'PommeTeamCompetition-v0', 'PommeTeamCompetitionFast-v0', 
    #     #'PommeTeamCompetition-v1', 'PommeTeam-v0', 'PommeTeamFast-v0'
    #     #return pommerman.make('OneVsOne-v0',agents)
    
    def rollout(self):
     # reset search self.tree in the beginning of each rollout首次展示
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
            # print("rollout-1")
            # sample action from probabilities
            self.values = np.zeros(NUM_ACTIONS)
         
            pi = self.puct(root)
            # # sample action from probabilities
     
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
            # step environment
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

    def puct(self,root):
        # start=time.time()
        workers = []
        q=multiprocessing.get_context('spawn')
        info=q.Queue()
        
        for i in range(PARALLEL_COUNT):
            w = q.Process(target=SW,args=(root,self.tree,info,args,self.agent_id));
            w.start()

        values = np.zeros(NUM_ACTIONS)
        for i in range(PARALLEL_COUNT):
            result,tree = info.get()
            values=[values[i]+result[i] for i in range(0,len(result))]

        values=values/sum(values)

        return (values)


def runner(id, num_episodes, _args):

    # make args accessible to MCTSAgent
    global args
    args = _args
    # make sure agents play at all positions
    agent_id = id % NUM_AGENTS
    agent = PMCTS(agent_id=agent_id)
    start_time = time.time()
    length, reward, rewards,yes = agent.rollout()
    elapsed = time.time() - start_time
    return (length, reward, rewards, agent_id, elapsed,yes)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #使用 argparse 的第一步是创建一个 ArgumentParser 对象。
    #ArgumentParser 对象包含将命令行解析成 Python 数据类型所需的全部信息。
    
    parser.add_argument('--profile')
    # ArgumentParser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    # name or flags - 一个命名或者一个选项字符串的列表，例如 foo 或 -f, --foo。
    # action - 当参数在命令行中出现时使用的动作基本类型。
    # nargs - 命令行参数应当消耗的数目。
    # const - 被一些 action 和 nargs 选择所需求的常数。
    # default - 当参数未在命令行中出现时使用的值。
    # type - 命令行参数应当被转换成的类型。
    # choices - 可用的参数的容器。
    # required - 此命令行选项是否可省略 （仅选项可用）。
    # help - 一个此选项作用的简单描述。
    # metavar - 在使用方法消息中使用的参数值示例。
    # dest - 被添加到 parse_args() 所返回对象上的属性名
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

    #multiprocessing.get_context(method=None)，返回多进程里的环境变量。
    #使用 multiprocessing 模块提供的 get_context () 函数来设置进程启动的方法，
    # 调用该函数时可传入 "spawn"、"fork"、"forkserver" 作为参数，用来指定进程启动的方式。
    #需要注意的一点是，前面在创建进程是，使用的 multiprocessing.Process () 这种形式，而在使用 get_context () 函数设置启动进程方式时，
    # 需用该函数的返回值，代替 multiprocessing 模块调用 Process ()。
    all_time=[]
    all_yes=0#表示平局
    all_yes1=0#表示胜利
    dataname="mcts_parallel_test"+str(args.mcts_iters)+"_"+str(PARALLEL_COUNT)+".csv"
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
 