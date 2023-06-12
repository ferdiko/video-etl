import numpy as np
from knob_plan import KnobPlanner
import pickle


def knob_switch(debug):

    """
    Knob switcher when given placements costs and file with
    quality at each video segment
    """

    with open("categories.np", "rb") as f:
        categories = pickle.load(f)

    with open("pred.np", "rb") as f:
        histogram = pickle.load(f)

    with open("knob_cost.np", "rb") as f:
        knob_costs = pickle.load(f)

    with open("covid_bw.place", "rb") as f:
        hw_dict = pickle.load(f)

    corr = np.zeros((3,3)) # TODO

    # ablation
    allowed = ["everything", "buffer_only", "cloud_only", "none"]
    allowed = allowed[0]

    plan_dict = {
                  0: (np.array([0]*len(knob_costs)), np.array(knob_costs), np.array([i for i in range(len(knob_costs))]))
                }

    with open("covid_bw.place", "rb") as f:
        hw_dict = pickle.load(f)

    budgets = np.linspace(24*3600*1000, 3628800000.0/2, 10) #np.linspace(5590080, 3628800000.0/2, 10)
    cloud_budgets = budgets
    cores_list = [2, 4, 8, 16, 24, 32, 48]

    plan_ahead = 24 # plan 48 hours ahead
    knob_order = { "10": 0, "20": 1, "30": 2, "50": 3, "75": 4, "200": 5 }

    buffering_allowed = True
    cloud_allowed = False

    if not buffering_allowed and not cloud_allowed:
        cloud_budgets = np.array([24*3600*1000])

    final_scores = [[] for _ in range(len(cores_list))]
    final_costs = [[] for _ in range(len(cores_list))]

    realtime = 5000
    num_secs = 5 # cost is over how many secs

    # get quality sorted. if a knob config doesnt fit into buffer, we can use
    # next worse one that fits
    quality_sort = []
    for i_qual in range(categories.shape[0]):
        quality_sort.append(np.argsort(-categories[i_qual]))

    # get runtimes for planning
    runtimes, knob_cost, config_belong = plan_dict[0]
    runtimes = np.array(runtimes) #- realtime
    knob_cost = np.array(knob_cost) / num_secs
    config_belong = np.array(config_belong)

    for budget in cloud_budgets:

        for i, num_cores in enumerate(cores_list):

            knob_cost = np.zeros(categories.shape[1])
            # cur = 0
            for r, c, b in zip(hw_dict[num_cores][0], hw_dict[num_cores][1], hw_dict[num_cores][2]):
                if c == 0:
                    knob_cost[b] = r

            runtimes = np.zeros(len(knob_cost))
            config_belong = np.array([i for i in range(len(knob_cost))])
            knob_cost = np.array(knob_cost)

            # plan
            kp = KnobPlanner(categories, knob_cost, plan_ahead, time_interval=2, knob_order=knob_order, verbose = True, corr=corr, runtimes=runtimes, config_belong= config_belong)
            plan, score = kp.plan(input=None, budget=budget, use_gt_histo=True, histogram=histogram)

            # get runtimes for hardware
            runtimes, knob_cost, config_belong = hw_dict[num_cores]
            runtimes = (np.array(runtimes) - realtime)
            knob_cost = np.array(knob_cost) / num_secs
            config_belong = np.array(config_belong)

            # get config priorities
            config_prio = []
            # runtime, cost, config = hw_dict[num_cores]
            for qual_sort in quality_sort:
                cluster_prio = []
                for c in qual_sort:
                    config_cluster_prio = []
                    # get all placements
                    for r, co, con in zip(runtimes, knob_cost, config_belong):
                        if con == c:

                            if not buffering_allowed and r > 0:
                                continue
                            if not cloud_allowed and co > 0:
                                continue
                            config_cluster_prio.append((r, co, con))
                    cluster_prio += sorted(config_cluster_prio, reverse=True)
                config_prio.append(cluster_prio)

            # knob switcher --> test on preproc input
            buffer_size = 180*1000
            buffer = 0

            input = "covid_proc_60.csv"
            runtimes, knob_cost, config_belong = plan_dict[0]
            target_place = plan.reshape((categories.shape[0], knob_cost.shape[0]))

            place_counts = np.ones(target_place.shape)

            file = open(input, 'r')
            file.readline()

            score_sum = 0
            cost_sum = 0
            # policy = 0
            use_knob = 2

            if not buffering_allowed and not cloud_allowed:
                none_dict = { 2: 2, 4: 2, 8: 3, 16: 0, 24: 0, 32: 0, 48: 1 }
                use_knob = none_dict[num_cores]

            consec_zeros = 0
            last_skip = 100

            while True:

                # print(buffer)
                line = file.readline()
                if not line:
                    break

                cur_score = int(line.split(",")[1+use_knob])

                score_sum += cur_score

                # nearest cluster (dynamics)
                dynamics = np.argmin(np.abs(categories[:, use_knob] - cur_score))

                if int(line.split(",")[1]) == 0:
                    consec_zeros += 1
                else:
                    consec_zeros = 0

                if not buffering_allowed and not cloud_allowed:
                    continue

                # all placements that overflow the buffer impossible
                buffer_space = buffer_size - buffer
                ratio_error = target_place[dynamics] - (place_counts[dynamics]/np.sum(place_counts[dynamics]))

                knob_place = np.argmax(ratio_error)

                # search start idx of config prio
                idx = 0
                while config_prio[dynamics][idx][2] != knob_place:
                    idx += 1

                # search for next best placement
                while config_prio[dynamics][idx][0] > buffer_space:
                    idx += 1

                # add cost etc.
                buffer = max(0, buffer + config_prio[dynamics][idx][0])
                cost_sum += config_prio[dynamics][idx][1]
                use_knob = config_prio[dynamics][idx][2]
                place_counts[dynamics][use_knob] += 1

            final_scores[i].append(score_sum)
            final_costs[i].append(cost_sum)

    # print
    print("cost,qual,cores")
    for j, budget in enumerate(cloud_budgets):
        for i, num_cores in enumerate(cores_list):
            # cost = num_cores*3600*1000*24+1.5*budget
            print("{},{},{}".format(final_costs[i][j], final_scores[i][j], num_cores))
