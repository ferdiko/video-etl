import json
import sys
sys.path.append("../src")
from simulations.buffer import SimBuffer

# currently, the example simulations only support a switcher interval of 2s
SWITCHER_TIME_INTERVAL = 2

def simulate_skyscraper(switcher, args, cfg, prof):

    print("Processing using Skyscraper")
    print("Chosen configs and placements are logged to", args["output_file"])

    # Get cost of different instance types
    with open(args["hw_cost"], "r") as f:
        hw_cost = json.load(f)

    # open sim file
    with open(args["sim_file"], "r") as f:
        lines = f.readlines()[1:]

    # open logging file
    out = open(args["output_file"], "w+")
    out.write("chunk_id,chosen_config,chosen_placement,cloud_cost,runtime(s),score\n")

    # Simulate online ingestion
    cur_config = 0
    cur_score = float(lines[0].split(",")[cur_config+1])
    score_sum = cur_score
    cost_sum = len(lines)*hw_cost[str(cfg["num_cores"])]
    out.write(f"{0},{cur_config},{0},{0},{cur_score}\n")
    for i, cur in enumerate(lines[1:]):
        # switch
        cur_config, placement, cost, rt = switcher.switch(cur_score)
        cur_score = float(cur.split(",")[cur_config+1])
        score_sum += cur_score
        cost_sum += cost * hw_cost["lambda"]

        # log to out file
        out.write(f"{i+1},{cur_config},{placement},{cost},{rt:.2f},{cur_score}\n")

    out.close()

    # print
    print("Summed quality:", score_sum)
    print("Summed cost:", cost_sum, "USD")


def simulate_static(args, cfg, prof):

    print("Processing using static configuration")
    print("Log written to", args["output_file"])

    # Get cost of different instance types
    with open(args["hw_cost"], "r") as f:
        hw_cost = json.load(f)

    # open sim file
    with open(args["sim_file"], "r") as f:
        lines = f.readlines()[1:]

    # open logging file
    out = open(args["output_file"], "w+")
    out.write("chunk_id,chosen_config,runtime(s),score\n")

    # get runtime of knob configuration
    static_config = cfg["knob_config"]
    runtimes = prof["runtime"]
    for r, k, c in zip(prof["runtime"], prof["knob_config"], prof["cloud_cost"]):
        if k == static_config and c == 0:
            knob_rt = r

    # warn user if hardware we think knob config can't run in realtime
    if knob_rt > SWITCHER_TIME_INTERVAL:
        print("\033[91m" + f"Knob configuration {static_config} seems to run slower than realtime on the provisioned hardware" + "\033[0;0m")

    # Simulate online ingestion
    score_sum = 0
    cost_sum = len(lines)*hw_cost[str(cfg["num_cores"])]
    for i, cur in enumerate(lines):
        score = float(cur.split(",")[static_config+1])
        score_sum += score

        out.write(f"{i},{static_config},{knob_rt:.2f},{score}\n")

    out.close()

    print("Summed quality:", score_sum)
    print("Summed cost:", cost_sum, "USD")
