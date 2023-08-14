# Examples

## Purpose of examples
Note that this repository does not present a usable system but should merely serve as a reference implementation.

The examples should present an easy way to play around with Skyscraper and observe its performance. Towards that goal, the examples make use of a simulator which we found to be reasonably accurate compared to running the real workload. The simulation allows the examples to run fast (seconds instead of days) and cheap (on a laptop instead of expensive hardware + cloud credits). We might add unsimulated examples in the future.

We are planning to add additional workloads, baselines and configurations soon.

## Running the examples

Running the examples requires tensorflow, sklearn, numpy, scipy.

The following works with python 3.10 and using pip:

```
pip install tensorflow
pip install scikit-learn
```

We provide example configurations to start with. For example, the following should run Skyscraper:

```
python main.py -s skyscraper -w COVID -c COVID/skyscraper_cfg_1.json
```

The following should run the static baseline on the same workload:

```
python main.py -s static -w COVID -c COVID/static_cfg_1.json
```

A list of further argument options can be obtained by `python main.py --help`.
