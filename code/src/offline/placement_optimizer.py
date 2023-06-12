import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear, ReLU
from torch_geometric.nn import Sequential, GCNConv
from torch_geometric.nn import global_mean_pool
from torch.distributions import Bernoulli
from torch.autograd import Variable
from itertools import count
import numpy as np
from execution_utils import *
import random
import copy
import math


class Candidate:

    def __init__(self, runtime, cost, graph):
        self.runtime = runtime
        self.cost = cost
        self.placement = []
        for n in graph.nodes:
            self.placement.append(n.placement)

    def __str__ (self):
        return '(rt=' + str(self.runtime) + ' ,cost=' + self.cost + ')'


class PolicyNet(torch.nn.Module):
    # TODO architecture

    def __init__(self):
        super().__init__()
        num_node_features = 8
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 24)
        # self.linear1 = Linear(128,256)
        # self.linear2 = Linear(256,64)
        self.linear3 = Linear(24, 1) # prob of placing in cloud

    def forward(self, data, batch):
        # TODO: Should we append the current runtime to the featuized graph?
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        # x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        # x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x))

        x = global_mean_pool(x, batch)
        x = F.sigmoid(self.linear3(x))
        return x


def comp_lcm(a,b):
  return (a * b) // math.gcd(a,b)


class PlacementOptimizer:

    def __init__(self, workload, realtime,
                learning_rate = 0.01,
                intermediate_rewards=True,
                reward_delta=5,
                batch_size=5,
                runtime_closeness_thresh=0.0,
                epsilon=0.6,
                epsilon_decl=0.998,
                gamma=0.99,
                init_best_cand=False,
                debug=False,
                num_frames=150):
        """
            learning_rate (float): learning rate
            intermediate_rewards (Bool): Reward after each node prediction
            reward_delta (float): how much is too long runtime penalized
            batch_size (int): predict how many graphs at a time
            runtime_closeness_thresh (float): how much too long runtime is okay
            epsilon (float): initial random exploration rate
            epsilon_decl: decline in random exploration rate
            gamma (float): running average factor for discounting rewards
            init_best_cand (bool): input to policy net is best placement
            debug (bool): Print additional stuff
            num_frames (int): Task graph over num_frames frames
        """
        self.workload = workload
        self.realtime = realtime/workload.norm_const

        self.learning_rate = learning_rate
        self.intermediate_rewards = intermediate_rewards
        self.reward_delta = reward_delta
        self.batch_size = batch_size
        self.runtime_closeness_thresh = runtime_closeness_thresh
        self.epsilon = epsilon
        self.epsilon_decl = epsilon_decl
        self.gamma = gamma
        self.intermediate_rewards = intermediate_rewards
        self.init_best_cand = init_best_cand
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = PolicyNet().to(device)
        self.policy_net.train()
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=learning_rate)
        self.num_frames = num_frames
        self.debug = debug

        # init some global counter (for accross place calls)
        self.counter = 0

    # delta is, how much is cost weighted vs runtime
    def get_reward(self, runtime, cost):
        reward = 0
        if runtime > self.realtime:
            reward -= self.reward_delta*(runtime-self.realtime)
        reward -= cost
        return reward


    def eval_rollout(self, candidates, runtime, cost, graph, thresh=0.5, random=False, knob=-1):
        # if abs(runtime - self.realtime) > thresh*self.realtime:
        #     return candidates

        on_pareto = True
        i = 0
        while i < len(candidates):
            c = candidates[i]
            if c.runtime <= runtime and c.cost <= cost:
                on_pareto = False
                break
            elif (c.runtime > runtime and c.cost >= cost) or (c.runtime >= runtime and c.cost > cost):
                candidates =  candidates[:i] + candidates[i+1:]
            else:
                i += 1

        if on_pareto:
            print("on pareto (counter: {}), (rand: {}),(knob: {}), (runtime, cost: {},{})".format(self.counter, random, knob, runtime, cost))
            candidates.append(Candidate(runtime, cost, graph))

        # best_cost = candidates[0].cost
        best_cand = candidates[0]
        for c in candidates:
            if c.runtime <= self.realtime and c.cost < best_cand.cost:
                best_cand = c

        return candidates, best_cand


    def init_graph(self, graph, node_features, placement):
        for i, p in enumerate(placement):
            node_features[i][5] = 0 # visited
            node_features[i][7] = p # placement
            if graph is not None:
                graph.nodes[i].placement = p
        return node_features


    def predict_interm(self, graphs, max_iters):
        """
            Predict with intermediate rewards after each node (=task) placement
            - graphs [TaskGraph]: graphs to be placed
            - max_iters (int): Number of iterations
        """
        # make edge tensor for each graph
        graph_edges = []
        for graph in graphs:
            from_node = []
            to_node = []

            for n in graph.nodes:
                for i in n.dependencies:
                    from_node.append(n.id)
                    to_node.append(i)

            edge_index = torch.tensor([from_node, to_node], dtype=torch.long)
            graph_edges.append(edge_index)

        # init batch tensor: global_mean_pool needs a vec that assigns each
        # input to a pecific example.
        pool_batch = []
        for graph in graphs:
            pool_batch.append(torch.zeros(graph.num_nodes, dtype=torch.long))

        # initial graph node features: Everything on prem
        graph_nodes = []
        for graph in graphs:
            node_features = [[] for n in graph.nodes]
            for n in graph.nodes:
                # node features (8)
                node_features[n.id].append(n.runtime_single)
                node_features[n.id].append(n.runtime_all)
                node_features[n.id].append(n.runtime_cloud)
                node_features[n.id].append(n.input_size)
                node_features[n.id].append(n.output_size)
                node_features[n.id].append(0) # visited
                node_features[n.id].append(0) # current
                node_features[n.id].append(0) # 0 on prem, 1 on cloud
            x = torch.tensor(node_features, dtype=torch.float)
            graph_nodes.append(x)

        # init reward, action, state pools and candidates (remember good placements)
        candidates = [[] for g in graphs]
        init_candidate = []
        reward_pool = []
        reward_discount = []
        state_pool = []
        action_pool = []

        # init with all on prem
        for i, graph in enumerate(graphs):
            runtime, cost = graph.simulate_run()
            candidates[i], cand = self.eval_rollout(candidates[i], runtime, cost, graph)
            init_candidate.append(cand)
            reward_discount.append(self.get_reward(runtime, cost))

        # have a random graph order for training
        lcm = comp_lcm(len(graphs), self.batch_size)
        graph_order = [i%len(graphs) for i in range(lcm)]
        batch_start = 0

        # run num_episode RL iterations
        for e in range(max_iters*len(graphs)):

            # do backward step
            if e % self.batch_size == 0 and e > 0:
                # normalize reward
                reward_mean = np.mean(reward_pool)
                reward_std = np.std(reward_pool)
                assert reward_std != 0
                for r in range(len(reward_pool)):
                    reward_pool[r] = (reward_pool[r] - reward_mean) / reward_std
                if self.debug:
                    print("reward pool", reward_pool)

                # backward
                reward_idx = 0
                self.optimizer.zero_grad()
                for e_back in range(self.batch_size):
                    i = graph_order[batch_start+e_back]
                    graph = graphs[i]

                    # input graph
                    self.init_graph(None, graph_nodes[i], state_pool[e_back])

                    # traverse nodes
                    for node_id in graph.topo_order():
                        graph_nodes[i][node_id][6] = 1 # current
                        data = Data(x=graph_nodes[i], edge_index=graph_edges[i])
                        probs = self.policy_net(data, pool_batch[i])[0]
                        if self.debug:
                            print("probs", i, node_id, probs)
                        m = Bernoulli(probs)
                        task_placement = Variable(torch.FloatTensor([action_pool[e_back][node_id]]))
                        loss = -m.log_prob(task_placement) * reward_pool[reward_idx]
                        if self.debug:
                            print("loss", loss)

                        loss.backward()

                        # update graphs
                        graph_nodes[i][node_id][5] = 1 # visited
                        graph_nodes[i][node_id][6] = 0 # current
                        graph_nodes[i][node_id][7] = task_placement[0] # placement

                        reward_idx += 1

                self.optimizer.step()

                # reset pools
                assert reward_idx == len(reward_pool)
                state_pool = []
                action_pool = []
                reward_pool = []

                # update graphs to use
                batch_start = (batch_start + self.batch_size) % lcm
                if e % lcm == 0:
                    random.shuffle(graph_order)

            assert len(reward_discount) == len(graphs) ==  len(init_candidate)
            assert len(state_pool) == len(action_pool) == e % self.batch_size

            # Roll out / forward
            i = graph_order[batch_start+(e%self.batch_size)]
            graph = graphs[i]

            # init to best placement so far, init reward discount
            self.init_graph(graph, graph_nodes[i], init_candidate[i].placement)
            state_pool.append(copy.deepcopy(init_candidate[i].placement))
            runtime, cost = graph.simulate_run()
            reward_discount[i] = self.get_reward(runtime, cost)

            # predict for each node if on prem (0) or cloud (1)
            random_rollout = random.uniform(0, 1) < self.epsilon
            if random_rollout:
                # sample action random
                min_cloudratio = 0.021
                max_cloudratio = 0.2
                ratio_sample = random.uniform(0.021, 0.2)
                cloud_tasks = int(ratio_sample*graph.num_nodes)
                cloud_idx = [random.randrange(0, graph.num_nodes, 1) for i in range(cloud_tasks)]

            for node_id in graph.topo_order():
                if random_rollout:
                    task_placement = int(node_id in cloud_idx)

                else:
                    # sample from policy net prediction
                    graph_nodes[i][node_id][6] = 1 # current
                    data = Data(x=graph_nodes[i], edge_index=graph_edges[i])
                    probs = self.policy_net(data, pool_batch[i])[0]
                    m = Bernoulli(probs)
                    task_placement = m.sample()
                    task_placement = task_placement.data.numpy().astype(int)[0]

                # get reward TODO: Just subtract by previous reward ...
                graph.nodes[node_id].placement = task_placement
                runtime, cost = graph.simulate_run()
                candidates[i], best_cand = self.eval_rollout(candidates[i], runtime, cost, graph, random=random_rollout, knob=i)
                reward = self.get_reward(runtime, cost)
                reward_pool.append(reward - reward_discount[i])
                reward_discount[i] = reward

                if not random_rollout:
                    # update graphs
                    graph_nodes[i][node_id][5] = 1 # visited
                    graph_nodes[i][node_id][6] = 0 # current
                    graph_nodes[i][node_id][7] = task_placement # placement

            if self.debug:
                print("{},{},{},{},{},{}, --- {}".format(i, random_rollout, reward, reward_pool[-1], runtime, cost, self.realtime))
            action_pool.append([n.placement for n in graph.nodes])

            if self.init_best_cand:
                init_candidate[i] = best_cand

            # reward_pool = next_reward_pool
            self.epsilon *= self.epsilon_decl
            self.counter += 1

            # save and flush print
            if e % 50 == 0 and e > 0:
                torch.save(self.policy_net.state_dict(), "model-knob2-{}-ep06.ckp".format(self.counter))
            print("", end="", flush=True)

        return candidates


    def place(self, knobs, videos, search_placement=True, max_episodes=50):
        """
            Get task graph & acc, search for cheap, realtime cloud placement
            - Knobs [[int/float]]: list of knob settigs to be placed
            - videos [string]: Path to videos to be places
            - search_placement (Bool): Should only accuracy be returned or search
                for placement
            - max_episodes (int): How many episodes should search run for
        """
        # check input
        assert len(knobs) == len(videos)

        # run workload to get graph and accuracy for each knob setting
        graphs = []
        accuracies = []
        for k, v in zip(knobs, videos):
            self.workload.set_knob(k)
            self.workload.process(v)
            g = self.workload.get_taskgraph()
            g.normalize(self.workload.norm_const)
            graphs.append(g)

        if not search_placement:
            return accuracies

        # TODO: This should be done in init
        # self.policy_net.load_state_dict(torch.load("model-knob2-1950.ckp"))
        # self.policy_net.load_state_dict(torch.load("model-fin1-900.ckp"))

        # get candidate placements for each knob: [[Candidate, ...], ...]
        if self.intermediate_rewards:
            candidate_lists = self.predict_interm(graphs, max_episodes)
        else:
            raise NotImplementedError
            # candidate_lists = self.predict(graphs, max_episodes)

        # get cheapest placement for each knob
        if self.debug:
            print("candidates:")
            for candidates in candidate_lists:
                print("=============")
                for c in candidates:
                    print("{},{}".format(c.runtime, c.cost))

        # TODO Save policy net weights
        # torch.save(self.policy_net.state_dict(), "model{}-{}-nofeat-pool.ckp".format(self.learning_rate, self.reward_delta))

        # get cheapest cost among placement candidates
        costs = []
        for candidates in candidate_lists:
            cheapest = np.infty
            best = None
            for c in candidates:
                if c.runtime <= self.runtime and c.cost < cheapest:
                    cheapest = c.cost
                    best = c

            # append to costs
            assert best is not None
            costs.append(cheapest)

        return accuracy, costs, candidate_lists


if __name__ == "__main__":
    po = PlacementOptimizer(realtime=5000)
    print(po.place_debug(knob_duration=5, knob_frequency=120, knob_numsample=900, videos=[]))
