# TODO: Switch to PyTorch
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from keras.models import load_model

import numpy as np
from sklearn.cluster import KMeans
import heapq
# from data_helper import *
# from baselines import *
from scipy.optimize import linprog
from collections import defaultdict

import copy

import os
import random
import pickle


# helper functions
def get_key(date_string, secs):
    date = [int(d) for d in date_string[:-4].split('-')]
    units = [3000, 12, 30, 24, 60, 60]
    carry_over = 0
    date[-1] += secs
    for i in reversed(range(6)):
        tmp = date[i]
        date[i] = (date[i] + carry_over) % units[i]
        carry_over = (tmp + carry_over) // units[i]
    return "{}-{}-{}-{}-{}-{}".format(date[0], date[1], date[2], date[3], date[4], date[5])


class KnobPlanner:
    def __init__(self,
                categories,
                knob_cost,
                hours_plan_ahead,
                time_interval,
                forecast_weights,
                input_hours=96,
                linear_programming=True,
                verbose=False):
        """
        KnobPlanner: Forecast workload and assign knobs to categories
        - categories (np([[float]])): K means cluster centers
        - knob_cost ([[float]]): cost of each placement (row) for each knob(col)
            per second
        - hours_plan_ahead (float): number of hours to plan ahead
        - time_interval (float): how many seconds is each data interval long.
            Needed for determining how many data points 1 input/output day
            consists of
        - input_hours (float): hours of history that forecast NN gets as input
        - linear_programming (Bool): Use linear programming or 0-1 knapsack to
          assign knob settings to categories
        """

        self.categories = categories
        self.knob_cost = np.array(knob_cost)
        self.hours_plan_ahead = hours_plan_ahead
        self.num_cluster = categories.shape[0]
        self.num_knobs = categories.shape[1]
        self.knob_place = knob_cost.shape[0]
        self.time_interval = time_interval
        self.verbose = verbose
        self.forecaster = load_model(forecast_weights)
        self.linear_programming = linear_programming
        self.input_hours = input_hours


    def assign_knobs_lin_prog(self, mixture, budget):
        # define linear program
        A = np.zeros((2*self.num_cluster+1, self.knob_place*self.num_cluster), dtype=float)
        b = np.zeros((2*self.num_cluster+1,), dtype=float)
        c = np.zeros((self.knob_place*self.num_cluster,), dtype=float)

        # enforce that distributions add up to 1
        for i in range(self.num_cluster):
            for j in range(i*self.knob_place, (i+1)*self.knob_place):
                A[2*i][j] = 1
                A[2*i+1][j] = -1
            b[2*i] = 1
            b[2*i+1] = -1

        # enforce below budget
        for i, m in enumerate(mixture):
            for j in range(self.knob_place):
                A[2*self.num_cluster][i*self.knob_place+j] \
                    = m*self.knob_cost[j] *self.hours_plan_ahead*3600
        b[2*self.num_cluster] = budget #/(self.hours_plan_ahead*3600)

        # score TODO maybe just put the mutilplying factors only where we estimate
        # the score. could be better for num stab?
        for i, center in enumerate(self.categories):
            for j in range(self.knob_place):
                c[i*self.knob_place+j] = -mixture[i] * center[j] \
                    * self.hours_plan_ahead*3600/self.time_interval


        # # debug
        # for i in range(3*self.num_cluster+1):
        #     for j in range(self.knob_place*self.num_cluster):
        #         if A[i][j] >= 0.0:
        #             print(" {}".format(round(A[i][j], 2)), end=" ")
        #         else:
        #             print(round(A[i][j], 2), end=" ")
        #
        #     print(" ||  {}".format(round(b[i], 2)))
        #     # print()
        # print("c", c)

        # Solve linear program
        res = linprog(c, A_ub=A, b_ub=b, bounds=(0,1)) #, method='lstsq', options = {"presolve":True})

        if res.fun is None or self.verbose:
            print("\nKnob planner (lin solve):", res.message)
            raise Exception("[Knob planner error] Linear program is infeasible, please adjust the configuration (e.g. increase budget or num_cores).")
        elif self.verbose:
            print("\nKnob planner (lin solve):", res.message)
            print('Expected score:', round(-res.fun, ndigits=4))
            sol = list(res.x)
            print("Solution linear program:")
            for i in range(self.num_cluster):
                for j in range(self.knob_place):
                    print(round(sol[i*self.knob_place+j], ndigits=4), end=" ")
                print()

        res_copy = copy.copy(res.x)

        res_copy = res_copy.reshape((self.num_cluster, self.knob_place))
        return res_copy, -res.fun


    def assign_knobs_knap_sack(self, mixture, budget):
        raise NotImplementedError

        # TODO
        # populate heap
        heap = []
        for p, (c_i, c) in zip(mixture, enumerate(self.categories)):
            for i, k in enumerate(self.category_cost):
                # TODO: Is this the right way to calculate in the cost
                cost = p*24*3600*30/k - p*24*3600*30/150
                if c[i] - c[-1] > 0 and cost > 0:
                    heap.append([cost/(c[i] - c[-1]), cost, i, c_i])

        budget -= 24*3600/5

        # elimate all with cost 0
        # TODO: If pred. mixture is 0 you should still include in case that pred is wrong
        heap = [h for h in heap if h[0] > 0]

        heapq.heapify(heap)

        knob_map = [5]*self.num_cluster
        while len(heap) > 0:
            (_, cost, knob, center) = heapq.heappop(heap)

            if cost < budget:
                knob_map[center] = knob
                budget -= cost

                # elimate all cheaper knobs of that center
                heap = [h for h in heap if h[3] != center or self.categories[center][h[2]] > self.categories[center][knob]]

                # adjust costs of same center knobs
                for i, h in enumerate(heap):
                    if h[3] == center:
                        heap[i][1] = h[1] - cost
                        # denom = cluster_centers[centers][h[2]] - cluster_centers[centers][knob]
                        # if denom > 0:
                        heap[i][0] = heap[i][1]/(self.categories[center][h[2]] - self.categories[center][knob])

                heapq.heapify(heap)

        print("remaining budget:", budget)
        print(knob_map)


    def sample_input_output(self, score, start_day, end_day, num_samples,
                            keys_list, kmeans):
        points_per_hour = int(3600/self.time_interval)

        # history input_hours on from rand. start and histogram for output_days
        X = np.empty((num_samples, self.input_hours*points_per_hour*2))
        y = np.empty((num_samples, self.num_cluster))

        for i in range(num_samples):
            day = random.randrange(start_day, end_day-self.hours_plan_ahead/24)
            hour = random.randrange(24)
            start = "2021-11-{:02d}-{:02d}-00-00".format(day, hour)

            # find starting key
            l = 0
            while keys_list[l] < start:
                l += 1

            # get history (input vector)
            inp_v = []
            for j in range(l, l+points_per_hour*self.input_hours):
                sc = -1
                while sc == -1:
                    knob = random.randrange(6)
                    sc = score[keys_list[j]][knob]
                inp_v.append(knob)
                inp_v.append(sc)
            X[i,:] = inp_v

            # get histogram (output vector)
            histo_vecs = []
            for j in range(l+points_per_hour*self.input_hours, l+points_per_hour*(self.input_hours+self.hours_plan_ahead)):
                v = score[keys_list[j]]
                # assert all([x != -1 for x in v])
                if all([x != -1 for x in v]):
                    histo_vecs.append(v)
            labels = kmeans.predict(histo_vecs)
            histo = np.bincount(labels)
            y[i, :histo.shape[0]] = histo

        # for i in range(self.input_hours*points_per_hour*2):
        #     print("{},".format(X[0,i]), end="")

        # Cast to np array and normalize
        # TODO: When normalizing X you need to make sure that knob idx and score more or less same size
        X /= np.linalg.norm(X)
        y /= y.sum(axis=1)[:,None]
        return X, y


    def get_traindata(self, foldername, num_samples, num_test_samples=0,
        test_start=""):
        """
        Read in data from all workload processing files. Get execution histories
            (x) and target histograms (y).
        - foldername (string): path to workload processing files
        - test_start (string): where to start test split (e.g.
            "2021-11-15-00-00-00"). "" means no test split
        """

        # workload processing files must have this structure:
        # file (date-time.mp4),    knob name, second offset, runtime,           score
        # 2021-11-10-09-47-18.mp4, 75,        0,             6.203967332839966, 70

        # get all knob setting names
        # knob_names = set()
        # for file in os.listdir(foldername):
        #     with open(os.path.join(foldername, file), "r") as f:
        #         lines = f.readlines()
        #     for l in lines:
        #         if len(l.split(",")) < 5 or l.split(",")[3] == "NA" or l[:7] == "file_id":
        #             continue
        #         knob_name = l.split(",")[1]
        #         knob_names.add(knob_name)
        # knobs = dict()
        # for i, knob_name in enumerate(knob_names):
        #     knobs[knob_name] = i

        # build dict which maps time -> knob score vector
        knob_order = { "10": 0, "20": 1, "30": 2, "50": 3, "75": 4, "200": 5 }
        score = defaultdict(lambda: [-1]*6)
        keys_list = []
        for file in os.listdir(foldername):
            with open(os.path.join(foldername, file), "r") as f:
                lines = f.readlines()

            for l in lines:
                if len(l.split(",")) < 5 or l.split(",")[3] == "NA" or l[:7] == "file_id":
                    continue

                key = get_key(l.split(",")[0], int(l.split(",")[2]))
                keys_list.append(key)
                knob = knob_order[l.split(",")[1]]
                sc = int(l.split(",")[4])
                score[key][knob] = sc

        # sample data points
        keys_list.sort() # for sampling
        # TODO: Just get k means from previous offline step
        kmeans = KMeans(n_clusters=self.num_cluster).fit(self.categories) # for building histograms
        kmeans.cluster_centers_ = self.categories
        start_day = 1 # minmum 1, day of Nov where data starts
        end_day = 23 # max 29, day of Nov where data ends
        X_train, y_train = self.sample_input_output(score, start_day, end_day,
            num_samples, keys_list, kmeans)
        return X_train, y_train, [], []


    def fit(self, data_folder="", weights_file=""):
        # load weights
        if weights_file != "":
            self.forecast.load_weights(weights_file)

        # train
        if train_data != "":
            # test_start = "2021-11-15-00-00-00"
            X_train, y_train, X_test, y_test = self.get_traindata(data_folder,
                num_samples=100)

            # fit
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath="100-150-80-30-6.ckpt",
                save_weights_only=True,
                monitor='val_mean_absolute_error',
                mode='max',
                save_best_only=True)

            self.forecast.fit(X_train, y_train, epochs=20, batch_size=128,
                            # validation_data=(X_test, y_test),
                            callbacks=[model_checkpoint_callback])


    def forecast(self, input):
        input = tf.convert_to_tensor([input])
        forecast = self.forecaster(input)[0].numpy()
        # forecast /= forcast.sum()
        return forecast


    def plan(self, input, budget, use_gt_histo=False, histogram=None):
        # predict input
        if use_gt_histo:
            with open("gt_histogram.np", "rb") as f:
                histogram = pickle.load(f)

        else:
            histogram = self.forecast(input)

        if self.verbose:
            print(histogram)

        # assign knobs to categories
        if self.linear_programming:
            return self.assign_knobs_lin_prog(histogram, budget)
        else:
            return self.assign_knobs_knap_sack(histogram, budget)
