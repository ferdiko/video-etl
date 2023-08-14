import argparse
import json
import sys
sys.path.append("../src")
from simulations.simulate import *
from online.knob_switcher import *


def parse_args(parser):
    """
    Parse args and set defaults according to workload and system
    """
    args = parser.parse_args()
    assert args.system in ["skyscraper", "static"], f"System {args.system} unknown"
    assert args.workload in ["COVID"], f"Workload {args.workload} unknown"

    arg_dict = {}
    arg_dict["system"] = args.system
    arg_dict["workload"] = args.workload
    arg_dict["config_file"] = f"{args.workload}/{args.system}_cfg_1.json" if args.config is None else args.config
    arg_dict["output_file"] = f"{args.system}_{args.workload}.csv" if args.output is None else args.output
    arg_dict["sim_file"] = f"{args.workload}/{args.workload}_sim.csv" if args.sim_file is None else args.sim_file
    arg_dict["prof_file"] = f"{args.workload}/prof.json" if args.prof_file is None else args.prof_file
    arg_dict["hw_cost"] = args.hw_cost
    arg_dict["weights_file"] = f"{args.workload}/forecast_ckpt" if args.weights is None else args.weights
    arg_dict["categories"] = f"{args.workload}/categories_3.npy" if args.categories is None else args.categories

    return arg_dict


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="Simulation for processing workload")
    parser.add_argument("-s", "--system", help="System to use for processing [ skyscraper | chameleon | static ]", required=True)
    parser.add_argument("-w", "--workload", help="Workload [ COVID | TRANSMOT ]", required=True)
    parser.add_argument("-c", "--config", help="Path to config file", default=None)
    parser.add_argument("-o", "--output", help="File to log output to", default=None)
    parser.add_argument("--sim_file", help="Path to sim file (file containing CV results)", default=None)
    parser.add_argument("--prof_file", help="File to profiling results", default=None)
    parser.add_argument("--hw_cost", help="File to cost of different compute instance types", default="hw_cost.json")
    parser.add_argument("--weights", help="Path to predictor weights", default=None)
    parser.add_argument("--categories", help="Path to content categories (KMeans centers)", default=None)
    args = parse_args(parser)

    # read config
    with open(args["config_file"], "r") as f:
        cfg = json.load(f)

    # read profiling results
    with open(args["prof_file"], "r") as f:
        prof = json.load(f)[str(cfg["num_cores"])]

    # Start simulation
    if args["system"] == "skyscraper":
        buffer = SimBuffer(cfg["buffer_size"], prof)
        switcher = SkyscraperSwitcher(args, buffer)
        simulate_skyscraper(switcher, args, cfg, prof)

    # elif args["system"] == "chameleon":
    #     buffer = SimBuffer(cfg["buffer_size"], prof)
    #     switcher = ChameleonSwitcher()
    #     simulate_chameleon(switcher, args, buffer)

    elif args["system"] == "static":
        simulate_static(args, cfg, prof)

    else:
        assert False, f"System {args['system']} unknown"
