import functools
# import math
# import time
# import random
import heapq
import copy
from collections import defaultdict

# TODO
# cloud collocate


# ==============================================================================
# Placement and Placement Recipe to be used by Execution Manager
# ==============================================================================

class Placement:
    def __init__():
        self.count = 0

    def load(file_path):
        assert False

    def store(file_path):
        assert False

    def get(task_id, knob):
        assert False



# ==============================================================================
# Node: A node = task in the TaskGraph
# ==============================================================================

class Node:
    runtime_single = 0
    runtime_all = 0
    runtime_cloud = 0
    input_size = 0
    output_size = 0

    id = -1
    dependencies = []
    dispatchable = -1
    placement = 0

    def __init__(self, id, rt_single, rt_all, runtime_cloud, input_size,
                 output_size, dep, placement=0):
        self.id = id
        self.runtime_single = rt_single
        self.runtime_all = rt_all
        self.runtime_cloud = runtime_cloud
        self.input_size = input_size
        self.output_size = output_size
        self.dependencies = dep
        self.placement = placement
        if len(dep) == 0:
            self.dispatchable = 0
        else:
            self.dispatchable = -1


    def update_deps(self, id, cur_time):
        for i, dep_id in enumerate(self.dependencies):
            if dep_id == id:
                self.dependencies = self.dependencies[:i] + self.dependencies[i+1:]

        if len(self.dependencies) == 0 and self.dispatchable == -1:
            self.dispatchable = max(self.dispatchable, cur_time)

    # printing
    def __str__(self):
        return "ID: {}\nRuntime: {} / {}\nDeps:{}\nDispatchable: {}\nPlacement: {}\n\n".format(self.id, self.runtime_single, self.runtime_all, self.dependencies, self.dispatchable, self.placement)

    def __repr__(self):
        return self.__str__()

# TODO: Do with __lt__ in node
def compare(a, b):
    if len(a.dependencies) == 0 and len(b.dependencies) == 0:
        if a.dispatchable == b.dispatchable:
            return 0
        if a.dispatchable > b.dispatchable:
            return 1
        return -1
    if len(a.dependencies) > 0 and len(b.dependencies) > 0:
        return 0
    if len(a.dependencies) > 0:
        return 1
    if len(b.dependencies) > 0:
        return -1



# ==============================================================================
# TaskGraph: Graph of tasks
# ==============================================================================

class TaskGraph:

    def __init__(self, cloud_roundtrip = 160, bandwidth_Bps=1850000):
        self.nodes = []
        self.topological_order = []
        self.num_nodes = 0
        self.cloud_roundtrip = cloud_roundtrip
        self.bandwidth = bandwidth_Bps/1000 # bytes / ms


    def insert(self, rt_single, rt_all, dep):
        self.nodes.append(Node(self.num_nodes, rt_single, rt_all, rt_single, 0, 0, dep))
        self.num_nodes += 1

    # def normalize(self):
    #     # TODO: If several graphs
    #     max_rt = 1
    #     for n in self.nodes:
    #         max_rt = max(max_rt, n.runtime_single, n.runtime_all)
    #     for n in self.nodes:
    #         n.runtime_single /= max_rt
    #         n.runtime_all /= max_rt
    #         n.runtime_cloud /= max_rt
    #     self.cloud_roundtrip /= max_rt
    #     return max_rt

    def normalize(self, max_rt):
        # TODO: If several graphs
        for n in self.nodes:
            n.runtime_single /= max_rt
            n.runtime_all /= max_rt
            n.runtime_cloud /= max_rt
        self.cloud_roundtrip /= max_rt
        return max_rt


    def simulate_rt(self, physical_cores):
        # TODO: Use heapq
        # init
        core_q = [0]*physical_cores
        cur_time = 0
        graph = copy.deepcopy(self.nodes)
        assert len(graph) == self.num_nodes
        bandwidth_usage = defaultdict(lambda: 0)

        # compute each node's cloud runtime.
        # single_rt+roundtrip if not all inputs on cloud, else single_rt
        for n in graph:
            if len(n.dependencies) == 0:
                n.runtime_cloud += self.cloud_roundtrip
            else:
                for d in n.dependencies:
                    if graph[d].placement == 0:
                        n.runtime_cloud += self.cloud_roundtrip
                        break

        total_runtime = 0

        while len(graph) > 0:
            graph.sort(key=functools.cmp_to_key(compare))
            # print("="*20)
            # for n in graph:
            #     print(n)
            # print("-"*20)

            n = graph[0]
            assert n.dispatchable > -1
            cur_time = n.dispatchable

            if n.placement == 0:
                # multi threading or not
                # TODO: We currently only support using all cores or 1. Support using any subset
                if abs(n.runtime_all-n.runtime_single) < 0.5*n.runtime_single:
                    # print("single")
                    # print("kcf")
                    core = core_q.index(min(core_q))
                    core_q[core] = max(cur_time, core_q[core]) + n.runtime_single
                    finish_time = core_q[core]
                else:
                    # print("all")
                    # print("yolo")
                    # TODO: If some cores available earlier but others aren't
                    # max core = min(time, core_q)
                    sum_others = 0 # all cores - max core
                    # TODO: handle if done before maxcore reached
                    # rt = n.runtime_single * physical_cores - sum_others
                    rt = n.runtime_all

                    finish_time = 0
                    for i in range(physical_cores):
                        core_q[i] =  max(cur_time, core_q[i]) + rt/physical_cores # int(math.ceil(rt/physical_cores))
                        finish_time = max(finish_time, core_q[i])
            else:
                # TODO: if already on cloud, check if additional inputs etc you
                # need to do this better and together with roundtrip
                # time required to upload data
                transfer_time = 0
                if n.runtime_cloud > n.runtime_single + 10:
                    remaining_bytes = n.input_size
                    while remaining_bytes > 0:
                        avail = self.bandwidth - bandwidth_usage[cur_time+transfer_time]
                        trans = min(avail, remaining_bytes)
                        remaining_bytes -= trans
                        bandwidth_usage[cur_time+transfer_time] += trans
                        transfer_time += 1

                finish_time = cur_time + n.runtime_cloud + transfer_time

                # output
                remaining_bytes = n.output_size
                while remaining_bytes > 0:
                    avail = self.bandwidth - bandwidth_usage[finish_time]
                    trans = min(avail, remaining_bytes)
                    remaining_bytes -= trans
                    bandwidth_usage[finish_time] += trans
                    finish_time += 1

            # print("finish", finish_time)

            total_runtime = max(total_runtime, finish_time)

            # update
            graph = graph[1:]
            for g in graph:
                g.update_deps(n.id, finish_time)

        # print(end-start)
        # print(core_q)
        # print(max(core_q))
        # print("return", total_runtime)
        # print("="*100)
        # print()
        # print("="*100)
        return total_runtime


    def simulate_cost(self):
        # TODO: Cloud cost
        cost = 0
        for n in self.nodes:
            if n.placement == 1:
                cost += n.runtime_single
        return cost


    def simulate_run(self):
        runtime = self.simulate_rt(4)
        cost = self.simulate_cost()
        return runtime, cost


    def topo_order(self):
        return [i for i in range(self.num_nodes)]
        # topo sort
        if len(self.topological_order) != len(self.nodes):
            class TopoNode:
                def __init__(self, node_id, deps):
                    self.id = node_id
                    self.n = len(deps)
                def __lt__(self, other):
                    return self.n < other.n
            out_deps = [[] for n in self.nodes]
            q = []
            for n in self.nodes:
                q.append(TopoNode(n.id, n.dependencies))
                for d in n.dependencies:
                    out_deps[d].append(n.id)
            heapq.heapify(q)
            while len(q) > 0:
                n = heapq.heappop(q)
                assert n.n == 0
                self.topological_order.append(n.id)
                for d in q:
                    if d.id in out_deps[n.id]:
                        d.n -= 1
                heapq.heapify(q)

        return self.topological_order


    def detect_to_track(self, knob, num_frames):
        assert num_frames % knob == 0
        assert len(self.nodes) == 0

        for i in range(int(num_frames/knob)):
            # insert yolo
            self.insert(721, 1941, [])

            # insert tracker tasks
            for j in range(knob):
                self.insert(519, 519, [self.num_nodes-1])


if __name__ == "__main__":
    tg = TaskGraph()
    tg.detect_to_track(60, 120)
    print(tg.simulate_run())
