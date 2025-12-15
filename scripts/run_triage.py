#!/usr/bin/env python3
"""
TriageFlow: Incident Triage Demo

Multi-agent incident triage with Domino GenAI tracing.
Runs a 4-agent pipeline to classify, assess, assign, and respond to incidents.

Usage:
    # Standard mode (single provider for all agents)
    python run_triage.py
    python run_triage.py --provider anthropic --vertical healthcare

    # Multi-model mode (different models per agent based on experiment results)
    python run_triage.py --multimodel
    python run_triage.py --multimodel --config configs/agents.yaml

    # Experiment mode (used by run_model_experiment.py)
    python run_triage.py --experiment --provider openai --temp-override 0.3
"""

import argparse
import io
import json
import os
import sys
from datetime import datetime

# Ensure project root is in path for imports
sys.path.insert(0, "/mnt/code")

import mlflow
import pandas as pd
import yaml

from domino.agents.tracing import add_tracing, search_traces
from domino.agents.logging import DominoRun, log_evaluation

from src.models import Incident, IncidentSource
from src.agents import classify_incident, assess_impact, match_resources, draft_response
from src.judges import judge_classification, judge_response, judge_triage


# Valid options
PROVIDERS = ["openai", "anthropic", "local"]
VERTICALS = ["financial_services", "healthcare", "energy", "public_sector"]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the TriageFlow incident triage pipeline with Domino tracing."
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=PROVIDERS,
        default="openai",
        help="LLM provider to use (default: openai)"
    )
    parser.add_argument(
        "--vertical",
        type=str,
        choices=VERTICALS,
        default="financial_services",
        help="Industry vertical for sample incidents (default: financial_services)"
    )
    parser.add_argument(
        "-n", "--num-tickets",
        type=int,
        default=0,
        help="Number of tickets to process (default: all in CSV)"
    )
    parser.add_argument(
        "--multimodel",
        action="store_true",
        help="Use multi-model mode (different model per agent)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config file path (default: configs/agents.yaml)"
    )
    parser.add_argument(
        "--experiment",
        action="store_true",
        help="Run in experiment mode (quiet output, returns metrics JSON)"
    )
    parser.add_argument(
        "--temp-override",
        type=float,
        default=None,
        help="Override temperature for all agents (experiment mode)"
    )
    parser.add_argument(
        "--judge-config",
        type=str,
        default=None,
        help="Judge config file path (default: configs/judges.yaml)"
    )
    parser.add_argument(
        "--max-incidents",
        type=int,
        default=None,
        help="Maximum number of incidents to process (default: all)"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Path to test data file (JSONL format for experiments)"
    )
    return parser.parse_args()


def initialize_client(provider: str, config: dict = None, quiet: bool = False):
    """Initialize LLM client and enable auto-tracing."""
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        mlflow.openai.autolog()
    elif provider == "local":
        from local_model.domino_model_client import get_local_model_client
        endpoint = None
        if config and "models" in config:
            local_config = config["models"].get("local", {})
            endpoint = local_config.get("endpoint")
        client = get_local_model_client(endpoint_url=endpoint)
        mlflow.openai.autolog()  # Local model uses OpenAI-compatible API
    else:
        from anthropic import Anthropic
        client = Anthropic()
        mlflow.anthropic.autolog()

    if not quiet:
        print(f"Auto-tracing enabled for {provider}")
    return client


def row_to_incident(row) -> Incident:
    """Convert a DataFrame row to an Incident object."""
    return Incident(
        ticket_id=row["ticket_id"],
        description=row["description"],
        source=IncidentSource(row["source"]),
        reporter=row["reporter"] if pd.notna(row["reporter"]) else None,
        affected_system=row["affected_system"] if pd.notna(row["affected_system"]) else None,
        initial_severity=int(row["initial_severity"]) if pd.notna(row["initial_severity"]) else None
    )


def jsonl_to_incident(data: dict) -> Incident:
    """Convert JSONL data to an Incident object."""
    # Map source to valid IncidentSource value, default to 'monitoring'
    source_str = data.get("source", "monitoring")
    valid_sources = ["monitoring", "user_report", "automated_scan", "external_notification", "audit"]
    if source_str not in valid_sources:
        source_str = "monitoring"

    return Incident(
        ticket_id=data.get("ticket_id", f"TEST-{hash(data.get('incident_text', ''))%10000}"),
        description=data.get("incident_text", data.get("description", "")),
        source=IncidentSource(source_str),
        reporter=data.get("reporter"),
        affected_system=data.get("affected_system"),
        initial_severity=data.get("initial_severity")
    )


def load_test_incidents(test_data_path: str, max_incidents: int = None) -> list:
    """Load incidents from JSONL test data file."""
    incidents = []
    with open(test_data_path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                incidents.append(jsonl_to_incident(data))
                if max_incidents and len(incidents) >= max_incidents:
                    break
    return incidents


def apply_temp_override(config: dict, temp: float) -> dict:
    """Apply temperature override to all agents in config."""
    config = config.copy()
    if "agents" in config:
        config["agents"] = config["agents"].copy()
        for agent_name in config["agents"]:
            config["agents"][agent_name] = config["agents"][agent_name].copy()
            config["agents"][agent_name]["temperature"] = temp
    return config


def compute_aggregated_metrics(results: list) -> dict:
    """Compute aggregated metrics from results list."""
    if not results:
        return {}

    metrics = {
        "classification_confidence": [],
        "impact_score": [],
        "resource_match_score": [],
        "completeness_score": [],
        "classification_judge_score": [],
        "response_judge_score": [],
        "triage_judge_score": [],
        "combined_quality_score": [],
    }

    for r in results:
        metrics["classification_confidence"].append(r.get("classification_confidence", 0.5))
        metrics["impact_score"].append(r.get("impact_score", 5.0))
        metrics["resource_match_score"].append(r.get("resource_match_score", 0.5))
        metrics["completeness_score"].append(r.get("completeness_score", 0.5))
        metrics["classification_judge_score"].append(r.get("classification_judge_score", 3))
        metrics["response_judge_score"].append(r.get("response_judge_score", 3))
        metrics["triage_judge_score"].append(r.get("triage_judge_score", 3))

        # Compute combined quality score for this result
        combined = (
            r.get("classification_judge_score", 3) +
            r.get("response_judge_score", 3) +
            r.get("triage_judge_score", 3)
        ) / 3
        metrics["combined_quality_score"].append(combined)

    import statistics

    return {
        "classification_confidence": statistics.mean(metrics["classification_confidence"]),
        "impact_score": statistics.median(metrics["impact_score"]),
        "resource_match_score": statistics.mean(metrics["resource_match_score"]),
        "completeness_score": statistics.mean(metrics["completeness_score"]),
        "classification_judge_score": statistics.mean(metrics["classification_judge_score"]),
        "response_judge_score": statistics.mean(metrics["response_judge_score"]),
        "triage_judge_score": statistics.mean(metrics["triage_judge_score"]),
        "combined_quality_score": statistics.mean(metrics["combined_quality_score"]),
    }


def pipeline_evaluator(span) -> dict:
    """Extract pre-computed metrics from pipeline outputs."""
    outputs = span.outputs or {}
    if not hasattr(outputs, "get"):
        return {}

    return {
        "classification_confidence": outputs.get("classification_confidence", 0.5),
        "impact_score": outputs.get("impact_score", 5.0),
        "resource_match_score": outputs.get("resource_match_score", 0.5),
        "completeness_score": outputs.get("completeness_score", 0.5),
        "classification_judge_score": outputs.get("classification_judge_score", 3),
        "response_judge_score": outputs.get("response_judge_score", 3),
        "triage_judge_score": outputs.get("triage_judge_score", 3),
    }


def create_triage_function(client, provider: str, model: str, config: dict):
    """Create the traced triage pipeline function."""

    @add_tracing(name="triage_incident", autolog_frameworks=[provider], evaluator=pipeline_evaluator)
    def triage_incident(incident: Incident):
        """Run the 4-agent triage pipeline with LLM judges."""
        # Agent 1: Classify the incident
        classification = classify_incident(client, provider, model, incident, config)

        # Agent 2: Assess impact
        impact = assess_impact(client, provider, model, incident, classification, config)

        # Agent 3: Match resources
        resources = match_resources(client, provider, model, classification, impact, config)

        # Agent 4: Draft response communications
        response = draft_response(client, provider, model, incident, classification, impact, resources, config)

        # Convert to dicts for judges
        class_dict = classification.model_dump()
        impact_dict = impact.model_dump()
        resources_dict = resources.model_dump()
        response_dict = response.model_dump()
        primary = resources_dict.get("primary_responder", {})

        # Run LLM judges to evaluate output quality
        class_judge = judge_classification(client, provider, incident.description, class_dict)

        comms = response_dict.get("communications", [])
        if comms:
            resp_judge = judge_response(client, provider, incident.description, class_dict.get("urgency", 3), comms[0])
        else:
            resp_judge = {"score": 1}

        triage_judge = judge_triage(client, provider, incident.description, class_dict, impact_dict, resources_dict, response_dict)

        return {
            "classification": classification,
            "impact": impact,
            "resources": resources,
            "response": response,
            # Metrics for evaluator
            "classification_confidence": class_dict.get("confidence", 0.5),
            "impact_score": impact_dict.get("impact_score", 5.0),
            "resource_match_score": primary.get("match_score", 0.5) if isinstance(primary, dict) else 0.5,
            "completeness_score": response_dict.get("completeness_score", 0.5),
            "classification_judge_score": class_judge.get("score", 3),
            "response_judge_score": resp_judge.get("score", 3),
            "triage_judge_score": triage_judge.get("score", 3),
        }

    return triage_incident


def print_results_summary(results: list):
    """Print a summary table of triage results."""
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    summary = pd.DataFrame([{
        "Ticket": r["ticket_id"],
        "Category": r["classification"].category.value,
        "Urgency": r["classification"].urgency,
        "Impact": r["impact"].impact_score,
        "Responder": r["resources"].primary_responder.name,
        "SLA Met": r["resources"].sla_met
    } for r in results])

    print(summary.to_string(index=False))


def print_sample_communication(results: list):
    """Print a sample communication from the first result."""
    if not results:
        return

    print("\n" + "=" * 80)
    print("SAMPLE COMMUNICATION")
    print("=" * 80)

    sample = results[0]
    print(f"Ticket: {sample['ticket_id']}\n")

    for comm in sample["response"].communications:
        print(f"--- {comm.audience.upper()} ---")
        print(f"Subject: {comm.subject}")
        print(f"{comm.body[:300]}...\n")


def add_adhoc_evaluations(run_id: str, results: list):
    """Add ad hoc evaluations to traces after pipeline completes."""
    print("\n" + "-" * 40)
    print("Adding ad hoc evaluations...")

    traces = search_traces(run_id=run_id)

    for i, trace in enumerate(traces.data):
        result = results[i]

        # Compute combined quality score from judge evaluations
        combined_quality = (
            result["classification_judge_score"] +
            result["response_judge_score"] +
            result["triage_judge_score"]
        ) / 3

        # Flag high-urgency incidents that may need manual review
        needs_review = result["classification"].urgency >= 4 and result["impact"].impact_score >= 7

        log_evaluation(trace_id=trace.id, name="combined_quality_score", value=round(combined_quality, 2))
        log_evaluation(trace_id=trace.id, name="needs_manual_review", value=1.0 if needs_review else 0.0)

    print(f"Added evaluations to {len(traces.data)} traces")


def run_multimodel_mode(args, config_path: str, incidents: list):
    """Run triage in multi-model mode with different models per agent."""
    from src.multimodel_agents import MultiModelTriage
    from src.deployable_judges import CompositeJudge, JudgeConfig

    print("\n[MULTI-MODEL MODE]")

    # Load multimodel config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize multi-model triage
    triage = MultiModelTriage(config)

    # Show model assignments
    print("Model assignments:")
    for agent, model in triage.get_model_info().items():
        print(f"  {agent}: {model}")
    print()

    # Initialize judge
    judge_config = JudgeConfig(model="gpt-4o-mini")
    judge = CompositeJudge(judge_config)

    # Set up experiment naming
    username = os.environ.get("DOMINO_USER_NAME", os.environ.get("USER", "demo_user"))
    project_name = os.environ.get("DOMINO_PROJECT_NAME", "default")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f"tracing-multimodel-{project_name}-{username}"
    run_name = f"multimodel-{args.vertical}-{timestamp}"

    print(f"Experiment: {experiment_name}")
    print(f"Run: {run_name}")
    print()

    # Aggregated metrics
    aggregated_metrics = [
        ("classification_confidence", "mean"),
        ("impact_score", "median"),
        ("resource_match_score", "mean"),
        ("completeness_score", "mean"),
        ("combined_quality_score", "mean"),
    ]

    # Enable auto-tracing for all providers
    mlflow.openai.autolog()
    mlflow.anthropic.autolog()

    mlflow.set_experiment(experiment_name)

    results = []
    run_id = None

    with DominoRun(agent_config_path=config_path, custom_summary_metrics=aggregated_metrics) as run:
        mlflow.set_tag("mlflow.runName", run_name)
        mlflow.set_tag("mode", "multimodel")
        run_id = run.info.run_id

        for incident in incidents:
            print(f"Processing {incident.ticket_id}...")

            try:
                # Run multi-model pipeline
                classification, impact, resources, response = triage.run(incident)

                # Run judges
                eval_result = judge.evaluate_all(
                    incident=incident.description,
                    classification=classification.model_dump(),
                    impact=impact.model_dump(),
                    resources=resources.model_dump(),
                    response=response.model_dump()
                )

                primary = resources.primary_responder

                result = {
                    "ticket_id": incident.ticket_id,
                    "classification": classification,
                    "impact": impact,
                    "resources": resources,
                    "response": response,
                    "classification_confidence": classification.confidence,
                    "impact_score": impact.impact_score,
                    "resource_match_score": primary.match_score,
                    "completeness_score": response.completeness_score,
                    "combined_quality_score": eval_result.get("combined_quality_score", 3.0),
                    "classification_judge_score": eval_result.get("classification_eval", {}).get("score", 3),
                    "response_judge_score": eval_result.get("response_evals", [{}])[0].get("score", 3) if eval_result.get("response_evals") else 3,
                    "triage_judge_score": eval_result.get("triage_eval", {}).get("score", 3),
                }

                results.append(result)
                print(f"  -> {classification.category.value} | Urgency: {classification.urgency} | Impact: {impact.impact_score}")

            except Exception as e:
                print(f"  ERROR: {e}")
                continue

        # Suppress DominoRun exit messages
        _stdout = sys.stdout
        sys.stdout = io.StringIO()

    sys.stdout = _stdout
    return results, run_id


def run_experiment_mode(args, config_path: str, config: dict, incidents: list) -> dict:
    """
    Run triage in experiment mode - quiet, returns metrics as JSON.
    Used by run_model_experiment.py for hyperparameter tuning.
    """
    # Apply temperature override if specified
    if args.temp_override is not None:
        config = apply_temp_override(config, args.temp_override)

    # Get model info
    model_config = config["models"].get(args.provider, {})
    if isinstance(model_config, str):
        model = model_config
    else:
        model = model_config.get("name", "gpt-4o-mini")

    # Initialize client
    client = initialize_client(args.provider, config, quiet=True)

    # Create triage function
    triage_incident = create_triage_function(client, args.provider, model, config)

    # Process incidents
    results = []
    for incident in incidents:
        try:
            result = triage_incident(incident)
            results.append({
                "ticket_id": incident.ticket_id,
                **result
            })
        except Exception as e:
            # Log error but continue
            print(f"Error processing {incident.ticket_id}: {e}", file=sys.stderr)
            continue

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

    # Determine max incidents
    max_incidents = args.max_incidents or (args.num_tickets if args.num_tickets > 0 else None)

    # Load incidents - from test data or CSV
    if args.test_data:
        test_data_path = args.test_data if os.path.isabs(args.test_data) else os.path.join(project_root, args.test_data)
        incidents = load_test_incidents(test_data_path, max_incidents)
    else:
        data_path = os.path.join(project_root, f"example-data/{args.vertical}.csv")
        df = pd.read_csv(data_path)
        incidents = [row_to_incident(row) for _, row in df.iterrows()]
        if max_incidents:
            incidents = incidents[:max_incidents]

    # Experiment mode - quiet output, return JSON metrics
    if args.experiment:
        metrics = run_experiment_mode(args, config_path, config, incidents)
        # Output metrics as JSON to stdout
        print(json.dumps(metrics, indent=2))
        return

    # Standard interactive mode
    print("=" * 80)
    print("TriageFlow: Incident Triage Demo")
    print("=" * 80)
    print(f"Provider: {args.provider}")
    print(f"Vertical: {args.vertical}")
    if args.multimodel:
        print("Mode: Multi-Model (different model per agent)")
    print()
    print(f"Config: {config_path}")
    print(f"Processing {len(incidents)} incidents")

    # Run in appropriate mode
    if args.multimodel:
        results, run_id = run_multimodel_mode(args, config_path, incidents)
        print_results_summary(results)
        print_sample_communication(results)
        if run_id:
            add_adhoc_evaluations(run_id, results)
        print("\n" + "=" * 80)
        print("Done! View traces in Domino Experiment Manager.")
        print("=" * 80)
        return

    # Standard single-provider mode
    model_config = config["models"].get(args.provider, {})
    if isinstance(model_config, str):
        model = model_config
    else:
        model = model_config.get("name", "gpt-4o-mini")
    print(f"Model: {model}")

    # Initialize client with auto-tracing
    client = initialize_client(args.provider, config)

    # Set up experiment and run naming
    username = os.environ.get("DOMINO_USER_NAME", os.environ.get("USER", "demo_user"))
    project_name = os.environ.get("DOMINO_PROJECT_NAME", "default")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f"tracing-{project_name}-{username}"
    run_name = f"{args.vertical}-{username}-{timestamp}"

    print(f"\nExperiment: {experiment_name}")
    print(f"Run: {run_name}")
    print()

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

    # Create the traced triage function
    triage_incident = create_triage_function(client, args.provider, model, config)

    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

    # Run the pipeline
    results = []
    run_id = None

    with DominoRun(agent_config_path=config_path, custom_summary_metrics=aggregated_metrics) as run:
        mlflow.set_tag("mlflow.runName", run_name)
        run_id = run.info.run_id

        for incident in incidents:
            print(f"Processing {incident.ticket_id}...")

            result = triage_incident(incident)

            results.append({
                "ticket_id": incident.ticket_id,
                **result
            })
            print(f"  -> {result['classification'].category.value} | Urgency: {result['classification'].urgency} | Impact: {result['impact'].impact_score}")

        # Suppress DominoRun exit messages
        _stdout = sys.stdout
        sys.stdout = io.StringIO()

    sys.stdout = _stdout
    print(f"\nProcessed {len(results)} incidents")

    # Print results
    print_results_summary(results)
    print_sample_communication(results)

    # Add ad hoc evaluations
    add_adhoc_evaluations(run_id, results)

    print("\n" + "=" * 80)
    print("Done! View traces in Domino Experiment Manager.")
    print("=" * 80)


if __name__ == "__main__":
    main()
