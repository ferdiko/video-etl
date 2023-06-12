"""
Informal interface to define a workload.
Any workload must override all of the following methods.
"""

sys.path.append("../../src/offline")
from execution_utils import TaskGraph


class Workload:

    """
    Expose the knobs of the workload like this
    """
    knob_names = []
    knob_types = []
    knobs = []
    knob_ranges = []

    def get_taskgraph(self, num_frames: int) -> TaskGraph:
        """
        Return the task graph for num_frames frames given the current knob setting
        """
        raise NotImplementedError

    def process(self, file: str) -> float:
        """
        Process the file at the path specified by file
        """
        raise NotImplementedError
