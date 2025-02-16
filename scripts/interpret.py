import argparse
import os
import json

import numpy as np


def main(args):
    with open(args.metrics_path, "r") as f:
        metrics = json.load(f)
        
    if not args.method:
        for method, values_dict in metrics.items():
            print(f"Method: {method}")
            for key, values in values_dict.items():
                print(f"\t{key}: {round(np.mean(values), 3)} +- {round(np.std(values), 2)}")
    else:
        values_dict = metrics[args.method]
        for key, values in values_dict.items():
            print(f"{key}: {round(np.mean(values), 3)} +- {round(np.std(values), 2)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mt", "--metrics-type", type=str, required=True)
    parser.add_argument("-m", "--method", type=str, required=False, default=None)
    args = parser.parse_args()
    
    if args.metrics_type == "bio":
        args.metrics_path = "data/analysis/metrics_bio.json"
    else:
        args.metrics_path = "data/analysis/metrics_extr.json"
    
    main(args)