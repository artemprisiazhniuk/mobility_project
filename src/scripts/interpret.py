import argparse
import os
import json

import numpy as np


def main(args):
    with open(args.metrics_path, "r") as f:
        metrics = json.load(f)
        
    if not args.method:
        for method, values_dict in metrics.items():
            print(f"{method}".replace("_", "\_"), end=" ")
            
            values_dict = dict(sorted(values_dict.items()))
            
            for key, values in values_dict.items():
                if isinstance(values, dict):
                    print(f"& {key}".replace("_", "\_"), end=" ")
                    for metric, values_ in values.items():
                        values_ = [v for v in values_ if v != 0]
                        print(f"& {round(np.mean(values_), 3)} $\pm$ {round(np.std(values_), 2)}", end=" ")
                    print("\\\\")                       
                else:
                    print(f"\t{key}: {round(np.mean(values), 3)} $\pm$ {round(np.std(values), 2)}")
            print("\hline")
    else:
        values_dict = metrics[args.method]
        for key, values in values_dict.items():
            if isinstance(values, dict):
                print(f"\t{key}:")
                for metric, values_ in values.items():
                    values_ = [v for v in values_ if v != 0]
                    print(f"\t\t{metric}: {round(np.mean(values_), 3)} $\pm$ {round(np.std(values_), 2)}")                       
            else:
                print(f"\t{key}: {round(np.mean(values), 3)} $\pm$ {round(np.std(values), 2)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mt", "--metrics-type", type=str, required=True)
    parser.add_argument("-m", "--method", type=str, required=False, default=None)
    args = parser.parse_args()
    
    if args.metrics_type == "bio":
        args.metrics_path = "data/analysis/metrics_bio.json"
    elif args.metrics_type == "extr":
        args.metrics_path = "data/analysis/metrics_extr.json"
    else:
        args.metrics_path = "data/analysis/metrics_extr_per_class.json"
    
    main(args)