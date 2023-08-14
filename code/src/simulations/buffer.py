import numpy as np
import copy


# currently, the example simulations only support a switcher interval of 2s
SWITCHER_TIME_INTERVAL = 2
PLANNER_RUNTIME = 0.5 # TODO

class SimBuffer:

    def __init__(self, space, prof):
        self.space = space
        self.prof = prof
        self.obj_sizes = []
        self.obj_times = []

        # get knob config sizes
        self.size_dict = {}
        for c, s in zip(prof["knob_config"], prof["size"]):
            self.size_dict[c] = s

        # buffer size in terms of runtime (assumes the buffer works like a queue)
        max_size = np.max(prof["size"])
        self.time_cap = self.space/(max_size/SWITCHER_TIME_INTERVAL)


    def computed_plan(self):
        if len(self.obj_times) > 0:
            self.obj_times[0] += PLANNER_RUNTIME


    def fits(self, runtime):
        return np.sum(self.obj_times) + runtime <= self.time_cap


    def update(self, configs, runtimes):
        """
        How the buffer simulation works:
        TODO
        """
        # add new objects
        if isinstance(configs, int):
            configs = [configs]
        if isinstance(runtimes, float):
            runtimes = [runtimes]
        # if runtimes is None:
        #     runtimes = np.zeros(len(configs))

        for c, r in zip(configs, runtimes):
            self.obj_sizes.append(self.size_dict[c])
            self.obj_times.append(r)

        # reflect processing progress
        processed = SWITCHER_TIME_INTERVAL
        while processed > 0 and len(self.obj_sizes) > 0:
            if processed >= self.obj_times[0]:
                processed -= self.obj_times[0]
                self.obj_times = self.obj_times[1:]
                self.obj_sizes = self.obj_sizes[1:]
            else:
                new_time = self.obj_times[0] - SWITCHER_TIME_INTERVAL
                self.obj_sizes[0] *=  new_time / self.obj_times[0]
                self.obj_times[0] = new_time
                break

        # return available space after update
        space_used = np.sum(self.obj_sizes)
        if space_used > self.space:
            print("\033[91m" + "Buffer overflow (increase buffer size or number of cores)" + "\033[0;0m")

        return self.space - space_used


    def avail_space(self):
        return self.space - np.sum(self.buffer_sizes)
