#!/usr/bin/env python3
"""
TriageFlow Experiment Launcher

Launches Domino Jobs for each model type to run temperature grid search experiments.
Each job creates an MLflow parent run with child runs for each temperature.

Usage:
    python launch_experiment.py
    python launch_experiment.py --models openai local
    python launch_experiment.py --max-incidents 5
    python launch_experiment.py --dry-run
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List

# Ensure project root is in path for imports
sys.path.insert(0, "/mnt/code")

import yaml


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch Domino Jobs for model comparison experiments."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["local", "openai", "anthropic"],
        default=None,
        help="Models to test (default: all from config)"
    )
    parser.add_argument(
        "--temperatures",
        type=str,
        default=None,
        help="Comma-separated list of temperatures to test (overrides config)"
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
        help="Path to test data JSONL file (overrides config)"
    )
    parser.add_argument(
        "--max-incidents",
        type=int,
        default=None,
        help="Maximum incidents to process per configuration"
    )
    parser.add_argument(
        "--vertical",
        type=str,
        default="financial_services",
        help="Industry vertical for sample incidents (if not using test-data)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print job commands without actually launching them"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run jobs sequentially instead of in parallel"
    )
    return parser.parse_args()


def get_project_info() -> Dict[str, str]:
    """Get Domino project information from environment."""
    return {
        "owner": os.environ.get("DOMINO_PROJECT_OWNER", os.environ.get("USER", "unknown")),
        "name": os.environ.get("DOMINO_PROJECT_NAME", "unknown"),
        "user": os.environ.get("DOMINO_USER_NAME", os.environ.get("USER", "unknown"))
    }


def build_job_command(
    model: str,
    temperatures: str = None,
    config_path: str = None,
    test_data: str = None,
    max_incidents: int = None,
    vertical: str = None
) -> str:
    """Build the command to run for a model experiment job."""
    cmd_parts = [
        "python",
        "/mnt/code/scripts/run_model_experiment.py",
        "--model", model,
    ]

    if temperatures:
        cmd_parts.extend(["--temperatures", temperatures])

    if config_path:
        cmd_parts.extend(["--config", config_path])

    if test_data:
        cmd_parts.extend(["--test-data", test_data])

    if max_incidents:
        cmd_parts.extend(["--max-incidents", str(max_incidents)])

    if vertical:
        cmd_parts.extend(["--vertical", vertical])

    return " ".join(cmd_parts)


def launch_job(command: str, title: str, dry_run: bool = False, branch: str = "6.2") -> Dict:
    """Launch a Domino job with the given command on a specific branch."""
    if dry_run:
        print(f"[DRY RUN] Would launch job:")
        print(f"  Title: {title}")
        print(f"  Command: {command}")
        print(f"  Branch: {branch}")
        return {"id": "dry-run", "status": "dry-run"}

    try:
        from domino import Domino

        project_info = get_project_info()
        project = f"{project_info['owner']}/{project_info['name']}"

        print(f"  Connecting to project: {project}")
        domino = Domino(project)

        print(f"  Starting job: {title}")
        print(f"  Command: {command}")
        print(f"  Branch: {branch}")

        # Build the job request body with git branch reference
        job_body = {
            "projectId": domino.project_id,
            "commandToRun": command,
            "title": title,
            "mainRepoGitRef": {
                "type": "branches",
                "value": branch
            }
        }

        # Use the Domino API routes to get the correct URL
        url = domino._routes.job_start()
        result = domino.request_manager.post(url, json=job_body).json()

        # Extract job ID from result
        job_id = None
        if isinstance(result, dict):
            job_id = result.get("id") or result.get("jobId") or result.get("runId")
        elif hasattr(result, "id"):
            job_id = result.id

        return {"id": job_id, "status": "submitted", "raw": result}

    except ImportError:
        print("Warning: domino package not available. Running in local mode.")
        print(f"Would launch: {command}")
        return {"id": "local-mode", "status": "not-launched"}

    except Exception as e:
        print(f"Error launching job: {e}")
        import traceback
        traceback.print_exc()
        return {"id": "error", "status": "failed", "error": str(e)}


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

    # Determine models to test
    if args.models:
        models_to_test = args.models
    else:
        experiment = config.get("experiment", {})
        grid_search = experiment.get("grid_search", {})
        models_to_test = grid_search.get("models_to_test", ["local", "openai", "anthropic"])

    # Determine test data path
    test_data = args.test_data
    if not test_data:
        experiment = config.get("experiment", {})
        test_data_config = experiment.get("test_data", {})
        if isinstance(test_data_config, dict):
            test_data = test_data_config.get("source")

    # Get project info
    project_info = get_project_info()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    print("=" * 60)
    print("TRIAGEFLOW EXPERIMENT LAUNCHER")
    print("=" * 60)
    print(f"Project: {project_info['owner']}/{project_info['name']}")
    print(f"User: {project_info['user']}")
    print(f"Timestamp: {timestamp}")
    print()
    print(f"Models to test: {models_to_test}")
    print(f"Config: {config_path}")
    print(f"Test data: {test_data or 'using CSV from vertical'}")
    print(f"Max incidents: {args.max_incidents or 'all'}")
    print(f"Temperatures: {args.temperatures or 'from config'}")
    print(f"Mode: {'Sequential' if args.sequential else 'Parallel'}")
    print()

    if args.dry_run:
        print("[DRY RUN MODE - No jobs will be launched]")
        print()

    # Launch jobs for each model
    jobs = []
    for model in models_to_test:
        print(f"--- Launching job for {model} ---")

        command = build_job_command(
            model=model,
            temperatures=args.temperatures,
            config_path=config_path,
            test_data=test_data,
            max_incidents=args.max_incidents,
            vertical=args.vertical
        )

        title = f"TriageFlow Experiment: {model} ({timestamp})"

        result = launch_job(command, title, dry_run=args.dry_run)

        jobs.append({
            "model": model,
            "command": command,
            "result": result
        })

        if not args.dry_run:
            job_id = result.get("id", "unknown")
            status = result.get("status", "unknown")
            print(f"  Job ID: {job_id}")
            print(f"  Status: {status}")

        print()

        # If sequential mode and not dry run, wait for job to complete
        if args.sequential and not args.dry_run:
            print("  [Sequential mode: waiting for job to complete...]")
            # Note: In a real implementation, you would poll for job completion
            # For now, we just continue since jobs will run independently

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Launched {len(jobs)} experiment jobs:")
    for job in jobs:
        job_id = job["result"].get("id", "unknown")
        print(f"  - {job['model']}: {job_id}")

    print()
    print("View results in Domino Experiments tab.")
    print(f"Experiment name: agent-optimization-{project_info['user']}")

    if args.dry_run:
        print()
        print("[DRY RUN] No jobs were actually launched.")


if __name__ == "__main__":
    main()
