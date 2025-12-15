#!/usr/bin/env python3
"""
TriageFlow Model Experiment Runner

Runs experiment for a single model type with temperature grid search.
Creates parent run with child runs for each temperature configuration.
Registers the best child run metrics to the parent run.

This script is designed to be called by launch_experiment.py as a Domino Job.

Usage:
    python run_model_experiment.py --model openai
    python run_model_experiment.py --model local --temperatures 0.1,0.2,0.3
    python run_model_experiment.py --model anthropic --max-incidents 5
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, List

# Ensure project root is in path for imports
sys.path.insert(0, "/mnt/code")

import mlflow
import yaml


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run temperature grid search experiment for a single model type."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["local", "openai", "anthropic"],
        help="Model type to test"
    )
    parser.add_argument(
        "--temperatures",
        type=str,
        default=None,
        help="Comma-separated list of temperatures to test (e.g., 0.1,0.2,0.3)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to agent config file (default: configs/agents.yaml)"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Path to test data JSONL file"
    )
    parser.add_argument(
        "--max-incidents",
        type=int,
        default=None,
        help="Maximum incidents to process per temperature"
    )
    parser.add_argument(
        "--vertical",
        type=str,
        default="financial_services",
        help="Industry vertical for sample incidents (if not using test-data)"
    )
    return parser.parse_args()


def get_temperatures(args, config: dict) -> List[float]:
    """Get list of temperatures to test."""
    if args.temperatures:
        return [float(t) for t in args.temperatures.split(",")]

    # Get from config
    experiment = config.get("experiment", {})
    temps = experiment.get("temperatures", {})

    if isinstance(temps, dict):
        # Use default temperatures or first agent's temperatures
        return temps.get("default", [0.1, 0.2, 0.3, 0.4, 0.5])
    elif isinstance(temps, list):
        return temps

    return [0.1, 0.2, 0.3, 0.4, 0.5]


def run_triage_for_temperature(
    model: str,
    temperature: float,
    config_path: str,
    test_data_path: str = None,
    max_incidents: int = None,
    vertical: str = "financial_services"
) -> Dict:
    """
    Run triage pipeline for a specific temperature configuration.
    Returns the aggregated metrics.
    """
    # Build command
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "run_triage.py"),
        "--experiment",
        "--provider", model,
        "--config", config_path,
        "--temp-override", str(temperature),
    ]

    if test_data_path:
        cmd.extend(["--test-data", test_data_path])
    else:
        cmd.extend(["--vertical", vertical])

    if max_incidents:
        cmd.extend(["--max-incidents", str(max_incidents)])

    # Run the command and capture output
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd="/mnt/code"
    )

    if result.returncode != 0:
        print(f"Error running triage: {result.stderr}", file=sys.stderr)
        return {"error": result.stderr, "combined_quality_score": 0.0}

    # Parse JSON output
    try:
        # The output may contain some logs before the JSON
        # Find the JSON part (starts with { and ends with })
        output = result.stdout.strip()
        json_start = output.rfind("{")
        if json_start >= 0:
            output = output[json_start:]
        metrics = json.loads(output)
        return metrics
    except json.JSONDecodeError as e:
        print(f"Error parsing output: {e}", file=sys.stderr)
        print(f"Output was: {result.stdout[:500]}", file=sys.stderr)
        return {"error": str(e), "combined_quality_score": 0.0}


def main():
    """Main entry point."""
    args = parse_args()

    # Determine paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    if args.config:
        config_path = args.config if os.path.isabs(args.config) else os.path.join(project_root, args.config)
    else:
        config_path = os.path.join(project_root, "configs/agents.yaml")

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Get test data path
    test_data_path = None
    if args.test_data:
        test_data_path = args.test_data if os.path.isabs(args.test_data) else os.path.join(project_root, args.test_data)
    else:
        # Check config for test data
        experiment = config.get("experiment", {})
        test_data_config = experiment.get("test_data", {})
        if isinstance(test_data_config, dict):
            source = test_data_config.get("source")
            if source:
                test_data_path = os.path.join(project_root, source)

    # Get temperatures to test
    temperatures = get_temperatures(args, config)

    # Set up experiment naming
    username = os.environ.get("DOMINO_USER_NAME", os.environ.get("USER", "demo_user"))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f"agent-optimization-{username}"
    parent_run_name = f"{args.model}-{timestamp}"

    print("=" * 60)
    print("TRIAGEFLOW MODEL EXPERIMENT")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Temperatures: {temperatures}")
    print(f"Config: {config_path}")
    print(f"Test data: {test_data_path or 'using CSV from vertical'}")
    print(f"Max incidents: {args.max_incidents or 'all'}")
    print(f"Experiment: {experiment_name}")
    print(f"Parent run: {parent_run_name}")
    print()

    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

    child_results = []

    # Create parent run for this model
    with mlflow.start_run(run_name=parent_run_name) as parent_run:
        parent_run_id = parent_run.info.run_id

        # Log parent run parameters
        mlflow.set_tag("model", args.model)
        mlflow.set_tag("experiment_type", "temperature_grid_search")
        mlflow.log_param("model_type", args.model)
        mlflow.log_param("temperatures_tested", str(temperatures))
        mlflow.log_param("num_temperatures", len(temperatures))

        print(f"Parent run ID: {parent_run_id}")
        print()

        # Create child runs for each temperature
        for temp in temperatures:
            print(f"--- Temperature: {temp} ---")

            with mlflow.start_run(nested=True, run_name=f"temp-{temp}") as child_run:
                child_run_id = child_run.info.run_id
                mlflow.log_param("temperature", temp)
                mlflow.log_param("model_type", args.model)

                # Run triage pipeline
                metrics = run_triage_for_temperature(
                    model=args.model,
                    temperature=temp,
                    config_path=config_path,
                    test_data_path=test_data_path,
                    max_incidents=args.max_incidents,
                    vertical=args.vertical
                )

                # Log metrics to child run
                if "error" not in metrics:
                    for name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(name, value)

                    combined_score = metrics.get("combined_quality_score", 0.0)
                    print(f"  Combined quality score: {combined_score:.3f}")
                else:
                    combined_score = 0.0
                    mlflow.log_metric("error", 1.0)
                    print(f"  Error: {metrics['error'][:100]}")

                child_results.append({
                    "run_id": child_run_id,
                    "temperature": temp,
                    "combined_quality_score": combined_score,
                    "metrics": metrics
                })

            print()

        # Find best child run
        best_child = max(child_results, key=lambda x: x["combined_quality_score"])

        print("=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)

        # Print all results
        for result in child_results:
            marker = " <-- BEST" if result["run_id"] == best_child["run_id"] else ""
            print(f"  temp={result['temperature']}: score={result['combined_quality_score']:.3f}{marker}")

        print()
        print(f"Best temperature: {best_child['temperature']}")
        print(f"Best score: {best_child['combined_quality_score']:.3f}")

        # Log best child metrics to parent
        mlflow.log_param("best_temperature", best_child["temperature"])
        mlflow.set_tag("best_child_run_id", best_child["run_id"])

        best_metrics = best_child.get("metrics", {})
        for key, value in best_metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"best_{key}", value)

        # Also log the raw best score for easy comparison
        mlflow.log_metric("best_combined_quality_score", best_child["combined_quality_score"])

        print()
        print(f"Parent run: {parent_run_id}")
        print(f"Best child run: {best_child['run_id']}")

    print()
    print("=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"View results in Domino Experiments: {experiment_name}")


if __name__ == "__main__":
    main()
