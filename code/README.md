# Skyscraper code base

Skyscraper allows for cheaper video ETL. This repository contains a prototype implementation used in the experiments.


## Structure
The code base has the following structure. Some file and directory names are different from the ones used in the paper. The names used in the paper are commented to the right.
```
  .
  ├── src/
  │  ├── offline/
  |  |  ├── execution_utils.py         # Task DAG class & runtime simulator
  |  |  ├── knob_tuner.py              # Knob searcher
  |  |  └── placement_optimizer.py     # Placement optimizer
  |  └── online/
  |     ├── execution_manager.py       # ExecutionManager, parallelization, calling functions etc.
  |     ├── knob_plan.py               # Knob planner
  |     └── knob_switcher.py           # Knob switcher
  ├── workloads/
  |  ├── covid/                        # COVID workload
  |  |  └── ...
  |  ├── streaming/                    # MOSEI workload
  |  |  └── ...
  |  └── transMOT/                     # MOT workload
  |     └── ...
  └── README.md
```

## Installation
If using conda, please use the ```environment.yml``` to install all required packages.


## Experiments
We can only provide a subset of the data used in the experiments due to Google Drive's storage limit. Please download the data at the below links.

 - [COVID](https://drive.google.com/drive/folders/1unRQFc4Mh5cfVgIzEv3w6Hr0g5L52yMM?usp=sharing)
 - [MOT](https://drive.google.com/drive/folders/1unRQFc4Mh5cfVgIzEv3w6Hr0g5L52yMM?usp=sharing)
 - [MOSEI](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/)

The code for the three applications is in the respective workloads directories.

Currently, the code in the applications is written for an earlier API, where the user overrides a Workload class that has the tunable knobs as class member variables. This has no effects on the results.
