import json
import boto3
import time
import asyncio
import ray
import sys
sys.path.append("../../workloads/covid/")
from covid_measures import CovidWorkload


@ray.remote
def ray_run(fn, *args, **kwargs):
    return fn(*args, **kwargs)

# TODO: That has also been defined in KnobTuner
class Knob:

    def __init__(self, domain):
        self.val = domain[0]
        self.domain = domain

    def val(self):
        return self.val


class ExecutionManager:


    def __init__(self, workload, realtime, aws_key_id, aws_key):
        # init boto3, nest_asyncio, ray
        self.loop = asyncio.get_event_loop()
        boto3.setup_default_session(region_name='us-east-2')
        self.client = boto3.client('lambda', aws_access_key_id=aws_key_id,
            aws_secret_access_key=aws_key)
        ray.init()

        # init program state
        self.knobs = []
        self.placement = None
        self.frame = 0

        # init execution manager state
        self.is_fit = False

        # init offline components
        self.knob_tuner = KnobTuner(workload, realtime=realtime)

        # init online components



    def add_knob(self, domain):
        assert not self.is_fit
        k = Knob(domain)
        self.knobs.append(k)
        return k


    def run(self, fn, fn_name, *args, on_prem=True, **kwargs):
        """
        run a function fn with arguments *args, **kwargs as a task.
        - fn: function pointer to function to execute
        - fn_name: String of how function is called in cloud. Will be removed in
          future as functions will be called according to their address (fn)
        - on_prem: delete and use self.placement
        - *args: positional arguments to fn
        - **kwargs: keyword arguments to fn
        """

        # assert self.is_fit

        # execute on cloud or on prem
        future = self.loop.create_future()
        self.loop.create_task(self.launch_task(on_prem, future, fn, fn_name,
            *args, **kwargs))
        return future


    def get_placement(self, fn_name):
        self.placement[fn_name+str(self.frame%self.placement.num_frames)]


    def tick(self):
        """
        indicate next frame
        """
        self.frame += 1


    def set_placement(self, placement):
        self.placement = placement

    def finish(self):
        pending = asyncio.all_tasks(self.loop)
        self.loop.run_until_complete(asyncio.gather(*pending))
        self.loop.close()

    def get(self, future):
        res = self.loop.run_until_complete(self.get_asyncio_result(future))
        if type(res) == ray._raylet.ObjectRef:
            res = ray.get(res)
        return res


    # ==========================================================================
    # helper methods
    # ==========================================================================

    def launch_on_cloud(self, future, fn_name, *args, **kwargs):
        # wait for ray futures
        ray_futures = []
        ray_futures_idx = []
        ray_futures_keys = []

        for i, arg in enumerate(args):
            if type(arg) == ray._raylet.ObjectRef:
                ray_futures.append(arg)
                ray_futures_idx.append(i)

        for key in kwargs:
            if type(kwargs[key]) == ray._raylet.ObjectRef:
                ray_futures.append(arg)
                ray_futures_keys.append(i)

        res = ray.get(ray_futures)

        # insert args into kwargs
        fut_idx = 0
        for i in range(len(args)):
            # insert value from awaited futures or args tuple
            if fut_idx < len(ray_futures_idx) and ray_futures_idx[fut_idx] == i:
                kwargs[f"skyscraper_posarg{i}"] = res[fut_idx]
                fut_idx += 1
            else:
                kwargs[f"skyscraper_posarg{i}"] = args[i]

        # insert awaited ray futures from kwargs into kwargs
        for k in ray_futures_keys:
            kwargs[k] = res[fut_idx]
            fut_idx += 1

        # call lambda function on kwargs input
        response = self.client.invoke(
            FunctionName=fn_name,
            InvocationType='RequestResponse',
            Payload=json.dumps(kwargs))

        res = json.loads(response['Payload'].read())["body"]
        future.set_result(res)
        return future


    async def launch_task(self, on_prem, future, fn, fn_name, *args, **kwargs):
        # wait for asyncio (cloud) futures
        args_list = list(args)
        for i, arg in enumerate(args_list):
            if type(arg) == asyncio.Future:
                args_list[i] = await self.get_asyncio_result(arg)
        args = tuple(args_list)
        for arg in kwargs:
            if type(kwargs[arg]) == asyncio.Future:
                kwargs[arg] = await self.get_asyncio_result(kwargs[arg])

        # launch task
        if on_prem:
            self.launch_on_premise(future, fn, *args, **kwargs)
        else:
            self.launch_on_cloud(future, fn_name, *args, **kwargs)


    async def get_asyncio_result(self, future):
        await future
        return future.result()


    def launch_on_premise(self, future, fn, *args, **kwargs):
        # TODO: For experiments, we used Ray actors. Switch here as well.
        res = ray_run.remote(fn, *args, **kwargs)
        future.set_result(res)
        return future
