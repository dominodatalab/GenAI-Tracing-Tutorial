#!/usr/bin/env python3
"""Register best judge configurations to MLflow.

This script is called by Domino Jobs to register the best judge configurations
found during optimization experiments.
"""

import argparse
import os
import yaml
import mlflow
from domino.agents.logging import DominoRun


def main():
    parser = argparse.ArgumentParser(description="Register best judge configurations")
    parser.add_argument("--vertical", required=True, help="Industry vertical")
    parser.add_argument("--batch-id", required=True, help="Batch ID from experiment")
    args = parser.parse_args()

    vertical = args.vertical
    batch_id = args.batch_id

    # Load best config from judges.yaml
    with open("/mnt/code/configs/judges.yaml") as f:
        config = yaml.safe_load(f)

    optimized = config.get("optimized_judge", {})

    # Set experiment
    username = os.environ.get("DOMINO_USER_NAME", "job")
    experiment_name = f"judge-optimization-{vertical}-{username}"
    mlflow.set_experiment(experiment_name)

    with DominoRun(agent_config_path="/mnt/code/configs/agents.yaml") as run:
        mlflow.set_tag("mlflow.runName", f"FINAL-BEST-JUDGES-{vertical}")
        mlflow.set_tag("batch_id", batch_id)
        mlflow.set_tag("best_parameters", "true")
        mlflow.set_tag("registered_job", "true")
        mlflow.set_tag("vertical", vertical)

        for jt, jt_configs in optimized.items():
            if vertical in jt_configs:
                cfg = jt_configs[vertical]
                mlflow.log_param(f"{jt}_model", cfg.get("model"))
                mlflow.log_param(f"{jt}_temperature", cfg.get("temperature"))
                mlflow.log_param(f"{jt}_prompt_style", cfg.get("prompt_style"))
                mlflow.log_param(f"{jt}_scale", cfg.get("scale"))
                metrics = cfg.get("validated_metrics", {})
                for k, v in metrics.items():
                    mlflow.log_metric(f"{jt}_{k}", v)

        print(f"Registered FINAL-BEST-JUDGES-{vertical} with batch_id: {batch_id}")


if __name__ == "__main__":
    main()
