#!/usr/bin/env python3
"""
TriageFlow Agent Experiment Launcher

Launches Domino Jobs for multi-parameter agent experiments.
Submits run_agent_experiment.py as a Domino job on the specified branch.

Usage:
    python launch_agent_experiment.py
    python launch_agent_experiment.py --sample-size 100
    python launch_agent_experiment.py --exhaustive --max-incidents 5
    python launch_agent_experiment.py --dry-run
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Dict

# Ensure project root is in path
sys.path.insert(0, "/mnt/code")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch Domino Job for agent experiments."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of random samples from grid (default: 50)"
    )
    parser.add_argument(
        "--exhaustive",
        action="store_true",
        help="Run all combinations (no sampling)"
    )
    parser.add_argument(
        "--max-incidents",
        type=int,
        default=None,
        help="Maximum incidents to process per experiment"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to experiment grid config"
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="6.2",
        help="Git branch to run job on (default: 6.2)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print job command without launching"
    )
    return parser.parse_args()


def get_project_info() -> Dict[str, str]:
    """Get Domino project information from environment."""
    return {
        "owner": os.environ.get("DOMINO_PROJECT_OWNER", os.environ.get("USER", "unknown")),
        "name": os.environ.get("DOMINO_PROJECT_NAME", "unknown"),
        "user": os.environ.get("DOMINO_USER_NAME", os.environ.get("USER", "unknown"))
    }


def build_job_command(args) -> str:
    """Build the command to run for the experiment job."""
    cmd_parts = [
        "python",
        "/mnt/code/scripts/run_agent_experiment.py",
    ]

    if args.exhaustive:
        cmd_parts.append("--exhaustive")
    else:
        cmd_parts.extend(["--sample-size", str(args.sample_size)])

    if args.max_incidents:
        cmd_parts.extend(["--max-incidents", str(args.max_incidents)])

    if args.seed:
        cmd_parts.extend(["--seed", str(args.seed)])

    if args.config:
        cmd_parts.extend(["--config", args.config])

    return " ".join(cmd_parts)


def launch_job(command: str, title: str, branch: str, dry_run: bool = False) -> Dict:
    """Launch a Domino job with the given command."""
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

    project_info = get_project_info()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    print("=" * 60)
    print("TRIAGEFLOW AGENT EXPERIMENT LAUNCHER")
    print("=" * 60)
    print(f"Project: {project_info['owner']}/{project_info['name']}")
    print(f"User: {project_info['user']}")
    print(f"Timestamp: {timestamp}")
    print()

    if args.exhaustive:
        print("Mode: Exhaustive (all combinations)")
    else:
        print(f"Mode: Random sampling ({args.sample_size} samples)")

    print(f"Max incidents: {args.max_incidents or 'from config'}")
    print(f"Random seed: {args.seed}")
    print(f"Branch: {args.branch}")
    print()

    if args.dry_run:
        print("[DRY RUN MODE - No job will be launched]")
        print()

    # Build command
    command = build_job_command(args)
    title = f"TriageFlow Agent Experiment ({timestamp})"

    # Launch job
    print("--- Launching Agent Experiment Job ---")
    result = launch_job(command, title, args.branch, dry_run=args.dry_run)

    if not args.dry_run:
        job_id = result.get("id", "unknown")
        status = result.get("status", "unknown")
        print(f"  Job ID: {job_id}")
        print(f"  Status: {status}")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if args.dry_run:
        print("[DRY RUN] No job was actually launched.")
    else:
        print(f"Job ID: {result.get('id', 'unknown')}")
        print(f"View results in Domino Experiments tab.")
        print(f"Experiment name: agent-optimization-{project_info['user']}")


if __name__ == "__main__":
    main()
