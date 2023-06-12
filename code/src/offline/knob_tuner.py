import numpy as np
import argparse
import copy
import sys
import math
import time
import itertools
import pickle
from datetime import datetime
import glob

from placement_optimizer import PlacementOptimizer

import sys


sys.path.append("../../workloads/covid/")
from covid_measures import CovidWorkload

class Knob:
    def __init__(self, name, domain):
        """
           name (string): name of the knob
           domain (np.array): sorted possible values of the knob
           hash_len: number of decimal to hash the value
        """
        self.name = name
        self.domain = domain
        self.hash_len = math.ceil(np.log10(len(domain)))

    def get_neighbour_values(self, value):
        # value (int/float): current knob value
        idx = np.searchsorted(self.domain, value)
        assert self.domain[idx] == value
        if idx == 0:
            return [self.domain[idx + 1]]
        elif idx == len(self.domain) - 1:
            return [self.domain[idx - 1]]
        else:
            return [self.domain[idx - 1], self.domain[idx + 1]]

    def sample(self, size=1):
        return np.random.choice(self.domain, size=size)

    def hash(self, value):
        enc = str(self.domain.index(value))
        if len(enc) < self.hash_len:
            enc = '0' * (self.hash_len - len(enc)) + enc
        return enc

    def dehash(self, enc):
        return self.domain[int(enc)]


class MultiKnob():
    def __init__(self, knobs):
        """
            knobs ([Knob]): list of knobs
        """
        self.knob_names = [k.name for k in knobs]
        self.knobs = knobs
        self.num_assignments = len(self.enumerate())

    def get_neighbours(self, assign_hash, neighbour_name=None):
        assignment = self.dehash(assign_hash)
        video_id = assign_hash.split("v")[1]

        neighbours = []
        for i, k in enumerate(self.knobs):
            vals = k.get_neighbour_values(assignment[i])
            for val in vals:
                tmp = copy.deepcopy(assignment)
                tmp[i] = val
                neighbours.append(self.hash(tmp)+f"v{video_id}")

        return neighbours

    def sample(self):
        res = []
        for i, knob in enumerate(self.knobs):
            res.append(knob.sample(1)[0])
        return res

    def hash(self, assignment):
        enc = ""
        for i, val in enumerate(assignment):
            enc += self.knobs[i].hash(val)
        return enc

    def dehash(self, enc):
        enc = enc.split("v")[0]
        assert len(enc) == sum([knob.hash_len for knob in self.knobs])
        start = 0
        res = []
        for knob in self.knobs:
            end = start + knob.hash_len
            res.append(knob.dehash(enc[start:end]))
            start = end
        return res

    def enumerate(self):
        all_domain = [knob.domain for knob in self.knobs]
        return list(itertools.product(*all_domain))


class KnobTuner:

    def __init__(self, workload,
                realtime,
                starting_points_per_vid=100,
                samples_per_starting_point=2,
                max_iter=20,
                num_etas=10,
                min_eta=0.1,
                max_eta=10):
        """
            workload: workload object
            realtime: how many ms should this workload run at (TODO)
            starting_points_per_vid: number of starting points per video for
                hill climbing
            samples_per_starting_point: number of samples among which starting
                point is elected
            max_iter: maximum number of iteration of hill climbing
        """
        # init
        self.workload = workload
        self.po = PlacementOptimizer(realtime=realtime, workload=workload)
        knobs = []
        for d, n in zip(workload.knob_domains, workload.knob_names):
            knobs.append(Knob(n, d))
        self.multi_knob = MultiKnob(knobs)
        self.realtime = realtime

        # hyperparameters
        assert starting_points_per_vid % num_etas == 0
        assert samples_per_starting_point <= num_etas
        self.starting_points_per_vid = starting_points_per_vid
        self.samples_per_starting_point = samples_per_starting_point
        self.max_iter = max_iter
        self.num_etas = num_etas
        self.min_eta = min_eta
        self.max_eta = max_eta


    def set_knobs(self, multi_knob):
        for i, knob in enumerate(multi_knob):
            self.workload.knobs[i] = knob


    def set_knob_and_place(self, multi_knob, placement):
        for i, knob in enumerate(multi_knob):
            self.workload.knobs[i] = knob


    def eta_normalize(data_points):
        assert False


    def store(self, file_name):
        assert False


    def run(self, videos, out_file="covid.knobs", hw_run=True, cache_file_prefix="cache"):
        # run
        assignments, placements, accs, costs = self.get_parento_frontier(videos, hw_run, cache_file_prefix)

        # persist to disk
        res_dict = dict()
        res_dict["assignments"] = assignments
        res_dict["placements"] = placements
        res_dict["accs"] = accs
        res_dict["costs"] = costs
        with open(out_file, "wb") as f:
            pickle.dump(res_dict, f, pickle.HIGHEST_PROTOCOL)

        return assignments, placements, accs, costs


    def get_parento_frontier(self, video_files, hw_run, cache_file_prefix):
        """
            Function call that returns a set of multi-knob assignments that are
                approximately on the Parento frontier
            video_files: a list of video file locations
        """

        # run hill climbing on all video files
        candidates, acc_cache, cost_cache, cand_cache = self.hill_climbing(video_files, cache_file_prefix)

        # eliminate_duplicates
        seen = set()
        new_candidates = []
        for candidate in candidates:
            assign_hash = candidate.split("v")[0]
            if assign_hash not in seen:
                new_candidates.append(assign_hash)
                seen.add(assign_hash)

        # for each cand assign., get acc on all videos: where is acc. missing?
        po_only_acc = []
        for i, assign_hash in enumerate(new_candidates):
            assert assign_hash in cost_cache
            for video_no, video in enumerate(video_files):
                hash = assign_hash+f"v{video_no}"
                if hash not in acc_cache:
                    po_only_acc.append(hash)

        # get missing accuracies
        _ = self.place([], po_only_acc, video_files, cost_cache, acc_cache, cand_cache)

        # fill out matrix for all costs & accs
        accs = np.zeros((len(new_candidates), len(video_files)))
        costs = np.zeros(len(new_candidates))

        # for each candidate, check if on the pareto front. of at least 1 vid
        for c_i, assign_hash in enumerate(new_candidates):
            costs[c_i] = cost_cache[assign_hash]
            for video_no in range(len(video_files)):
                hash = assign_hash+f"v{video_no}"
                accs[c_i, video_no] = acc_cache[hash]

        # keep all candidates that are on the pareto front. of some video
        keep_idx = []
        for c_i, assign_hash in enumerate(new_candidates):
            less_cost = np.where(costs[c_i] > costs)[0]
            for video_no in range(len(video_files)):
                more_acc = np.where(accs[c_i, video_no] < accs[:, video_no])[0]
                inters = np.intersect1d(less_cost, more_acc)

                if inters.shape[0] == 0:
                    # check if some exactly equal. then only take if smallest idx
                    equ_acc = np.where(accs[c_i, video_no] == accs[:, video_no])[0]
                    equ_cost = np.where(costs == costs[c_i])[0]
                    inters_equ = np.intersect1d(equ_acc, equ_cost)
                    if inters_equ.shape[0] == 1 or np.all(c_i <= inters_equ):
                        keep_idx.append(c_i)
                        break

        # get best placement from placement candidates
        final_assign = [new_candidates[i] for i in keep_idx]
        final_placements = []
        placement_candlists = [cand_cache[h] for h in new_candidates]
        if hw_run:
            raise NotImplementedError
            # actually execute to distinguish between candidates
            for knob, place_cands in zip(parento_frontier, pareto_place_cands):
                for p in place_cands:
                    self.set_knobs(knob, p)
                    start = time.time()
                    workload.process(num_frames=200)
                    end = time.time()
        else:
            for p_cands in placement_candlists:
                best_cost = np.infty
                best_place = None
                for p in p_cands:
                    if p.cost < best_cost and p.runtime <= self.realtime:
                        best_cost = p.cost
                        best_place = p

                assert best_place is not None
                final_placements.append(best_place)

        return final_assign, final_placements, accs[keep_idx], costs[keep_idx]


    def place(self, po_place, po_only_acc, video_files, cost_cache, acc_cache, cand_cache):
        # execute all po_place assignments (search for placement and get acc)
        knob_assignments = [self.multi_knob.dehash(s) for s in po_place]
        videos = [video_files[int(s.split("v")[-1])] for s in po_place]
        acc1, cost1, cand1 = self.po.place(knob_assignments, videos, search_placement=True)
        for a, co, ca, h in zip(acc1, cost1, cand1, po_place):
            assign_hash = h.split("v")[0]
            cost_cache[assign_hash] = co
            cand_cache[assign_hash] = ca
            acc_cache[h] = a

        # execute all po_only_acc assignments (only get the accuracy)
        knob_assignments = [self.multi_knob.dehash(s) for s in po_only_acc]
        videos = [video_files[int(s.split("v")[-1])] for s in po_only_acc]
        acc2 = self.po.place(knob_assignments, videos, search_placement=False)
        cost2 = [cost_cache[h.split("v")[0]] for h in po_only_acc]
        for a, h in zip(acc2, po_only_acc):
            acc_cache[h] = a

        # return
        acc = np.array(acc1 + acc2)
        cost = np.array(cost1 + cost2)
        return acc, cost


    def hill_climbing(self, video_files, cache_file_prefix):
        # TODO check cands propagated
        """
            performing greedy exploration of the parento frontier
            video_files [string]: paths to videos
            returns [hashes]: knob assignments
        """

        # init caches -- load newest file from disk if available
        cache_files = glob.glob(cache_file_prefix+"*.pkl")
        if len(cache_files) == 0:
            cost_cache = dict()
            acc_cache = dict()
            cand_cache = dict()
        else:
            newest = max(cache_files)
            with open(newest, 'rb') as f:
                res_dict = pickle.load(f)
            cost_cache = res_dict["cost_cache"]
            acc_cache = res_dict["acc_cache"]
            cand_cache = res_dict["cand_cache"]

        # init assignment arrays (final result)
        assignments = []
        assignment_scores = []

        num_videos = len(video_files)

        # find starting points
        num_samples = self.samples_per_starting_point * self.starting_points_per_vid
        for video_no in range(num_videos):

            # sample starting_points_per_vid * samples_per_starting_point points
            po_place = []
            po_only_acc = []
            while len(po_place) + len(po_only_acc) < num_samples:
                sample = self.multi_knob.sample()
                sample_hash = self.multi_knob.hash(sample)

                # schedule for execution by PO (either with placement or only acc)
                hash = sample_hash+f"v{video_no}"
                if hash not in acc_cache or acc_cache[hash] != -1:
                    acc_cache[sample_hash+f"v{video_no}"] = -1
                    if sample_hash not in cost_cache:
                        cost_cache[sample_hash] = -1
                        po_place.append(sample_hash+f"v{video_no}")
                    else:
                        po_only_acc.append(sample_hash+f"v{video_no}")

            # get cost and accuracy, calibrate etas
            acc, cost = self.place(po_place, po_only_acc, video_files,
                cost_cache, acc_cache, cand_cache)
            samples = po_place + po_only_acc
            assert len(samples) == num_samples
            eta_fac = np.sum(acc)/np.sum(cost)
            etas = np.linspace(eta_fac*self.min_eta, eta_fac*self.max_eta, num=self.num_etas)

            # only keep starting_points_per_vid best points
            for i in range(self.starting_points_per_vid):
                best_score = -np.infty
                best_sample = None
                for j in range(self.samples_per_starting_point):
                    idx = self.samples_per_starting_point * i + j
                    score = self.calculate_score(acc[idx], cost[idx], etas[int(idx*self.num_etas/num_samples)])
                    if score > best_score:
                        best_score = score
                        best_sample = samples[idx]

                assignments.append(best_sample+f"v{video_no}")
                assignment_scores.append(best_score)


        assert len(assignments) == self.starting_points_per_vid * num_videos

        # update assignments
        for update_iter in range(self.max_iter):
            # compute cost and accuracy for all uncached neighbours
            po_place = []
            po_only_acc = []
            for assign in assignments:
                neighbours = self.multi_knob.get_neighbours(assign)
                for n in neighbours:
                    if n not in acc_cache:
                        n_assign_hash = n.split("v")[0]
                        acc_cache[n] = -1
                        if n_assign_hash not in cost_cache:
                            cost_cache[n_assign_hash] = -1
                            po_place.append(n)
                        else:
                            po_only_acc.append(n)

            # compute cost and accuracy
            _ = self.place(po_place, po_only_acc, video_files, cost_cache,
                acc_cache, cand_cache)

            # for each assignment, update to best neighbour
            for a_i, assign in enumerate(assignments):
                neighbours = self.multi_knob.get_neighbours(assign)
                for n in neighbours:
                    # print(a_i, self.num_etas, self.starting_points_per_vid, len(assignments))
                    score = self.calculate_score(acc_cache[n], cost_cache[n.split("v")[0]], etas[int((a_i%self.starting_points_per_vid)*self.num_etas/self.starting_points_per_vid)])
                    if assignment_scores[a_i] < score:
                        assignment_scores[a_i] = score
                        assignments[a_i] = n

        # persist caches
        now = datetime.now()
        filename = cache_file_prefix + now.strftime("-%Y-%m-%d-%H:%M:%S") + ".pkl"
        res_dict = dict()
        res_dict["acc_cache"] = acc_cache
        res_dict["cost_cache"] = cost_cache
        res_dict["cand_cache"] = cand_cache
        with open(filename, "wb") as f:
            pickle.dump(res_dict, f, pickle.HIGHEST_PROTOCOL)

        return assignments, acc_cache, cost_cache, cand_cache


    def calculate_score(self, acc, cost, eta):
        return acc - eta * cost


if __name__ == '__main__':
    workload = CovidWorkload()
    kt = KnobTuner(workload, realtime=5000)
    kt.run(["a.mp4", "b.mp4"], hw_run=False)
