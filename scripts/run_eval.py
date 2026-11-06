import argparse
import os
import yaml
from aqbench.evaluation.evaluate import evaluate_experiment, evaluate_test_episodes


def task_name_from_config(path):
    return os.path.splitext(os.path.basename(path))[0]


def main(args):
    with open(args.task_config) as f:
        task_cfg = yaml.safe_load(f)
    task_name = task_cfg['task_name']
    target_dir=os.path.join("reports/targets/",task_name)

    report_dir      = os.path.join("reports", task_name_from_config(args.task_config))
    target_features = task_cfg["target_features"]

    if args.split == "test_epi":
        evaluate_test_episodes(
            exp_name        = args.exp_name,
            split           = args.split,
            T_in            = task_cfg["Tin"],
            report_dir      = report_dir,
            epi_mask_dir    = args.epi_mask_dir,
            target_features = target_features,
            target_dir=target_dir
        )
    else:
        evaluate_experiment(
            exp_name        = args.exp_name,
            split           = args.split,
            compute_pde     = not args.no_pde,
            raw_dir         = task_cfg["raw_dir"],
            grid_height_path = args.grid_height_path,
            T_in            = task_cfg["Tin"],
            T_out           = task_cfg["Tout"],
            report_dir      = report_dir,
            episode_eval    = args.episode_eval,
            episode_index   = args.episode_index,
            epi_mask_dir    = args.epi_mask_dir,
            target_features = target_features,
            target_dir=target_dir
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name",          required=True)
    parser.add_argument("--task_config",       required=True)
    parser.add_argument("--split",             default="test")
    parser.add_argument("--no_pde",            action="store_true")
    parser.add_argument("--grid_height_path",  default="data/stats/emission_height.npy")
    parser.add_argument("--episode_eval",      action="store_true", default=False)
    parser.add_argument("--episode_index",     default=None)
    parser.add_argument("--epi_mask_dir",      default="data/processed")
    args = parser.parse_args()
    main(args)