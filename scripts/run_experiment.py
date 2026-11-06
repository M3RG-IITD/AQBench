"""
    python run_experiment.py --model swin --task_config configs/task/pm25.yaml

"""

import os
import argparse
import yaml
from aqbench.runner import run_experiment, run_persistence
from aqbench.utils.seeding import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       type=str, required=True,
                   help="model directory name under aqbench/models/, e.g. swin")
    p.add_argument("--task_config", type=str, default="configs/task/pm25.yaml")
    p.add_argument("--exp_name",    type=str, default=None,
                   help="override exp_name from model config")
    p.add_argument("--config_name", type=str, default="config",
                   help="model config file basename (default: config.yaml)")
    p.add_argument("--time_embed_dim", type=int, default=None,
                   help="override time_embed_d in model_args (default: from config)")
    return p.parse_args()


def main():
    args = parse_args()

    task_cfg = yaml.safe_load(open(args.task_config))

    model_cfg_path = f"aqbench/models/{args.model}/{args.config_name}.yaml"
    if not os.path.exists(model_cfg_path):
        raise FileNotFoundError(f"Model config not found: {model_cfg_path}")
    model_cfg = yaml.safe_load(open(model_cfg_path))

    if args.exp_name is not None:
        model_cfg["exp_name"] = args.exp_name

    model_cfg.setdefault("model_args", {})

    # Override time_embed_d if provided
    if args.time_embed_dim is not None:
        model_cfg["model_args"]["time_embed_d"] = args.time_embed_dim

    set_seed(task_cfg["seed"])
    if args.model == "persistence":
        run_persistence(task_cfg, model_cfg)
    else:
        run_experiment(task_cfg, model_cfg)


if __name__ == "__main__":
    main()