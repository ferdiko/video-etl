import numpy as np
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from knob_plan import KnobPlanner


class SkyscraperSwitcher:

    def __init__(self, args, buffer):

        # read config
        with open(args["config_file"], "r") as f:
            cfg = json.load(f)

        # parse out values from experiment cfg
        self.cloud_budget = cfg["cloud_budget"]
        self.categories = np.load(args["categories"])
        self.plan_ahead = 24 # TODO: currently fixed according to provided NN weights
        self.time_interval = 2 # TODO: currently fixed according to sim file
        self.planning_interval = cfg["planning_interval"]
        self.budget = self.plan_ahead*3600*2 + self.cloud_budget

        # read profiling results
        with open(args["prof_file"], "r") as f:
            prof = json.load(f)[str(cfg["num_cores"])]

        self.cloud_cost = prof["cloud_cost"]
        self.runtimes = prof["runtime"]
        self.corr_config = prof["knob_config"]

        # configuratoins-placements sorted according to preference:
        # for each content category sort descending to quality, ascending to cost
        quality_sort = []
        for i_qual in range(self.categories.shape[0]):
            quality_sort.append(np.argsort(-self.categories[i_qual]))

        config_prio = []
        for qual_sort in quality_sort:
            cluster_prio = []
            for c in qual_sort:
                config_cluster_prio = []
                # get all placements
                for p, (r, co, con) in enumerate(zip(self.runtimes, self.cloud_cost, self.corr_config)):
                    if con == c:
                        config_cluster_prio.append((co, p, r, con))
                cluster_prio += sorted(config_cluster_prio)
            config_prio.append(cluster_prio)

        self.config_prio = config_prio

        # get knob cost for planner
        self.knob_cost = np.zeros(self.categories.shape[1])
        for r, c, b in zip(self.runtimes, self.cloud_cost, self.corr_config):
            if c == 0:
                self.knob_cost[b] = r

        # instantiate planner
        self.planner = KnobPlanner(self.categories,
                                   self.knob_cost,
                                   hours_plan_ahead=self.plan_ahead,
                                   time_interval=self.time_interval,
                                   forecast_weights=args["weights_file"]
                                   )

        # initialize switcher state
        self.cur_knob = 0
        self.category_counter = np.load(f"{args['workload']}/train_histogram.npy") # first prediction based on training data
        self.counter = 0
        self.buffer = buffer


    def reset_counts(self):
        self.category_counter = np.zeros(self.categories.shape[0])
        self.used_configs_counter = np.ones(self.categories.shape)


    def switch(self, cur_score):

        if self.counter % self.planning_interval == 0:
            self.histogram = self.category_counter/np.sum(self.category_counter)
            self.plan, score = self.planner.plan(input=self.histogram, budget=self.budget,) # TODO: delete out histogram
            self.reset_counts()
            self.buffer.computed_plan()

        # get current content category
        dynamics = np.argmin(np.abs(self.categories[:, self.cur_knob] - cur_score))

        # get preferred knob configuration
        used_configs = self.used_configs_counter[dynamics]/np.sum(self.used_configs_counter[dynamics])
        ratio_error = self.plan[dynamics] - used_configs
        knob_place = np.argmax(ratio_error)

        # search start idx of planned knob config in priority list
        idx = 0
        while self.config_prio[dynamics][idx][3] != knob_place:
            idx += 1

        # choose placement (or other config) such that buffer doesn't overflow
        while not self.buffer.fits(self.config_prio[dynamics][idx][2]):
            idx += 1

        # final knob configuration and placement
        self.cur_knob = self.config_prio[dynamics][idx][3]
        placement = self.config_prio[dynamics][idx][1]
        cloud_cost = self.config_prio[dynamics][idx][0]
        runtime = self.config_prio[dynamics][idx][2]

        # update
        self.category_counter[dynamics] += 1
        self.used_configs_counter[dynamics][self.cur_knob] += 1
        self.buffer.update(self.cur_knob, runtime)
        self.counter += 1

        return self.cur_knob, placement, cloud_cost, self.config_prio[dynamics][idx][2]
