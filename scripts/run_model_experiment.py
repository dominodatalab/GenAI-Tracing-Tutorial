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
import copy
import io
import json
import os
import sys
from datetime import datetime
from typing import Dict, List

# Ensure project root is in path for imports
sys.path.insert(0, "/mnt/code")

import mlflow
import pandas as pd
import yaml

from domino.agents.logging import DominoRun

from src.models import Incident, IncidentSource
from src.agents import classify_incident, assess_impact, match_resources, draft_response
from src.judges import judge_classification, judge_response, judge_triage


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


def load_incidents(test_data_path: str = None, vertical: str = "financial_services", max_incidents: int = None) -> List[Incident]:
    """Load incidents from test data or CSV."""
    project_root = "/mnt/code"

    if test_data_path:
        # Load from JSONL
        incidents = []
        with open(test_data_path) as f:
            for line in f:
                data = json.loads(line)
                incidents.append(Incident(
                    ticket_id=data.get("ticket_id", f"TEST-{len(incidents)+1}"),
                    title=data.get("title", ""),
                    description=data.get("description", ""),
                    source=IncidentSource(data.get("source", "monitoring")),
                    reported_by=data.get("reported_by", "system"),
                    affected_systems=data.get("affected_systems", []),
                    timestamp=data.get("timestamp", datetime.now().isoformat())
                ))
    else:
        # Load from CSV
        csv_path = os.path.join(project_root, f"example-data/{vertical}.csv")
        df = pd.read_csv(csv_path)
        incidents = []
        for _, row in df.iterrows():
            affected = row.get("affected_systems", "")
            if isinstance(affected, str):
                affected = [s.strip() for s in affected.split(",") if s.strip()]
            else:
                affected = []

            incidents.append(Incident(
                ticket_id=str(row.get("ticket_id", f"INC-{len(incidents)+1}")),
                title=str(row.get("title", "")),
                description=str(row.get("description", "")),
                source=IncidentSource(row.get("source", "monitoring")),
                reported_by=str(row.get("reported_by", "system")),
                affected_systems=affected,
                timestamp=str(row.get("timestamp", datetime.now().isoformat()))
            ))

    if max_incidents:
        incidents = incidents[:max_incidents]

    return incidents


def initialize_client(provider: str, config: dict):
    """Initialize LLM client and enable auto-tracing."""
    if provider == "openai":
        from openai import OpenAI
        mlflow.openai.autolog()
        return OpenAI()

    elif provider == "local":
        from local_model.domino_model_client import get_domino_model_client
        mlflow.openai.autolog()  # Local model uses OpenAI-compatible API
        return get_domino_model_client()

    elif provider == "anthropic":
        from anthropic import Anthropic
        mlflow.anthropic.autolog()
        return Anthropic()

    raise ValueError(f"Unknown provider: {provider}")


def apply_temp_override(config: dict, temperature: float) -> dict:
    """Apply temperature override to all agents in config."""
    config = copy.deepcopy(config)
    for agent_name in ["classifier", "impact_assessor", "resource_matcher", "response_drafter"]:
        if agent_name in config.get("agents", {}):
            config["agents"][agent_name]["temperature"] = temperature
    return config


def create_triage_function(client, provider: str, model: str, config: dict):
    """Create the triage pipeline function with tracing."""
    from domino.agents.tracing import add_tracing

    def pipeline_evaluator(inputs, output, **kwargs):
        """Evaluate the triage pipeline output."""
        incident = inputs.get("incident")
        if not incident or not output:
            return {"combined_quality_score": 0.0}

        eval_result = {}
        try:
            # Run classification judge
            classification_eval = judge_classification(
                client, provider, model, incident, output["classification"], config
            )
            eval_result["classification_eval"] = classification_eval

            # Run response judge
            response_evals = judge_response(
                client, provider, model, incident, output["response"], config
            )
            eval_result["response_evals"] = response_evals

            # Run triage judge
            triage_eval = judge_triage(
                client, provider, model, incident, output, config
            )
            eval_result["triage_eval"] = triage_eval

            # Compute combined score
            scores = []
            if classification_eval:
                scores.append(classification_eval.get("score", 3))
            if response_evals:
                scores.extend([e.get("score", 3) for e in response_evals])
            if triage_eval:
                scores.append(triage_eval.get("score", 3))

            eval_result["combined_quality_score"] = sum(scores) / len(scores) if scores else 3.0

        except Exception as e:
            print(f"Evaluation error: {e}", file=sys.stderr)
            eval_result["combined_quality_score"] = 3.0

        return eval_result

    @add_tracing(name="triage_incident", autolog_frameworks=[provider], evaluator=pipeline_evaluator)
    def triage_incident(incident: Incident) -> dict:
        """Run the full triage pipeline for an incident."""
        # Agent 1: Classify
        classification = classify_incident(client, provider, model, incident, config)

        # Agent 2: Assess Impact
        impact = assess_impact(client, provider, model, incident, classification, config)

        # Agent 3: Match Resources
        assignment = match_resources(client, provider, model, incident, classification, impact, config)

        # Agent 4: Draft Response
        response = draft_response(client, provider, model, incident, classification, impact, assignment, config)

        return {
            "classification": classification,
            "impact": impact,
            "assignment": assignment,
            "response": response,
        }

    return triage_incident


def compute_aggregated_metrics(results: List[dict]) -> dict:
    """Compute aggregated metrics from results."""
    if not results:
        return {"combined_quality_score": 0.0}

    metrics = {}

    # Numeric fields to aggregate
    numeric_fields = [
        "classification_confidence",
        "impact_score",
        "resource_match_score",
        "completeness_score",
        "combined_quality_score",
        "classification_judge_score",
        "response_judge_score",
        "triage_judge_score",
    ]

    for field in numeric_fields:
        values = []
        for r in results:
            if field in r and isinstance(r[field], (int, float)):
                values.append(r[field])
            # Also check nested objects
            elif "classification" in r and hasattr(r["classification"], "confidence") and field == "classification_confidence":
                values.append(r["classification"].confidence)
            elif "impact" in r and hasattr(r["impact"], "impact_score") and field == "impact_score":
                values.append(r["impact"].impact_score)
            elif "assignment" in r and hasattr(r["assignment"], "primary_responder") and field == "resource_match_score":
                values.append(r["assignment"].primary_responder.match_score)
            elif "response" in r and hasattr(r["response"], "completeness_score") and field == "completeness_score":
                values.append(r["response"].completeness_score)

        if values:
            metrics[f"{field}_mean"] = sum(values) / len(values)
            metrics[f"{field}_min"] = min(values)
            metrics[f"{field}_max"] = max(values)

    # Add combined quality score at top level for easy access
    if "combined_quality_score_mean" in metrics:
        metrics["combined_quality_score"] = metrics["combined_quality_score_mean"]

    return metrics


def run_temperature_experiment(
    model: str,
    temperature: float,
    config: dict,
    config_path: str,
    incidents: List[Incident],
) -> Dict:
    """
    Run triage pipeline for a specific temperature configuration.
    Returns the aggregated metrics.
    """
    # Apply temperature override
    temp_config = apply_temp_override(config, temperature)

    # Get model name
    model_config = temp_config["models"].get(model, {})
    if isinstance(model_config, str):
        model_name = model_config
    else:
        model_name = model_config.get("name", "gpt-4o-mini")

    # Initialize client
    client = initialize_client(model, temp_config)

    # Create triage function
    triage_incident = create_triage_function(client, model, model_name, temp_config)

    # Aggregated metrics for DominoRun
    aggregated_metrics = [
        ("classification_confidence", "mean"),
        ("impact_score", "median"),
        ("resource_match_score", "mean"),
        ("completeness_score", "mean"),
        ("classification_judge_score", "mean"),
        ("response_judge_score", "mean"),
        ("triage_judge_score", "mean"),
    ]

    # Process incidents within DominoRun for tracing
    results = []
    with DominoRun(agent_config_path=config_path, custom_summary_metrics=aggregated_metrics) as run:
        mlflow.set_tag("mode", "temperature_experiment")
        mlflow.set_tag("model", model)
        mlflow.set_tag("provider", model)
        mlflow.set_tag("temperature", str(temperature))
        mlflow.set_tag("experiment_type", "temperature_grid_search")
        mlflow.set_tag("mlflow.runName", f"{model}-temp-{temperature}")

        for incident in incidents:
            try:
                result = triage_incident(incident)

                # Extract metrics from result objects
                result_with_metrics = {
                    "ticket_id": incident.ticket_id,
                    "classification": result["classification"],
                    "impact": result["impact"],
                    "assignment": result["assignment"],
                    "response": result["response"],
                    "classification_confidence": result["classification"].confidence,
                    "impact_score": result["impact"].impact_score,
                    "resource_match_score": result["assignment"].primary_responder.match_score,
                    "completeness_score": result["response"].completeness_score,
                }
                results.append(result_with_metrics)

            except Exception as e:
                print(f"Error processing {incident.ticket_id}: {e}", file=sys.stderr)
                continue

        # Suppress DominoRun exit messages
        _stdout = sys.stdout
        sys.stdout = io.StringIO()

    sys.stdout = _stdout

    # Compute and return aggregated metrics
    metrics = compute_aggregated_metrics(results)
    metrics["incidents_processed"] = len(results)
    metrics["incidents_failed"] = len(incidents) - len(results)

    return metrics


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

    # Load incidents once (reused for each temperature)
    incidents = load_incidents(test_data_path, args.vertical, args.max_incidents)

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
    print(f"Incidents loaded: {len(incidents)}")
    print(f"Experiment: {experiment_name}")
    print(f"Parent run: {parent_run_name}")
    print()

    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

    child_results = []

    # Run experiment for each temperature (each creates its own DominoRun)
    for temp in temperatures:
        print(f"--- Temperature: {temp} ---")

        # Run triage pipeline with tracing
        metrics = run_temperature_experiment(
            model=args.model,
            temperature=temp,
            config=config,
            config_path=config_path,
            incidents=incidents,
        )

        # Log metrics summary
        combined_score = metrics.get("combined_quality_score", 0.0)
        if combined_score > 0:
            print(f"  Combined quality score: {combined_score:.3f}")
            print(f"  Incidents processed: {metrics.get('incidents_processed', 0)}")
        else:
            print(f"  Error or no results")

        child_results.append({
            "temperature": temp,
            "combined_quality_score": combined_score,
            "metrics": metrics
        })

        print()

    # Find best temperature
    best_result = max(child_results, key=lambda x: x["combined_quality_score"])

    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Print all results
    for result in child_results:
        marker = " <-- BEST" if result["temperature"] == best_result["temperature"] else ""
        print(f"  temp={result['temperature']}: score={result['combined_quality_score']:.3f}{marker}")

    print()
    print(f"Best temperature: {best_result['temperature']}")
    print(f"Best score: {best_result['combined_quality_score']:.3f}")

    print()
    print("=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"View results in Domino Experiments: {experiment_name}")


if __name__ == "__main__":
    main()
