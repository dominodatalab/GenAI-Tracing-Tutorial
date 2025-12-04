#!/usr/bin/env python3
"""
TriageFlow: Incident Triage Demo

Multi-agent incident triage with Domino GenAI tracing.
Runs a 4-agent pipeline to classify, assess, assign, and respond to incidents.

Usage:
    python run_triage.py
    python run_triage.py --provider anthropic --vertical healthcare
"""

import argparse
import io
import os
import sys
from datetime import datetime

import mlflow
import pandas as pd
import yaml

from domino.agents.tracing import add_tracing, search_traces
from domino.agents.logging import DominoRun, log_evaluation

from src.models import Incident, IncidentSource
from src.agents import classify_incident, assess_impact, match_resources, draft_response
from src.judges import judge_classification, judge_response, judge_triage


# Valid options
PROVIDERS = ["openai", "anthropic"]
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
    return parser.parse_args()


def initialize_client(provider: str):
    """Initialize LLM client and enable auto-tracing."""
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        mlflow.openai.autolog()
    else:
        from anthropic import Anthropic
        client = Anthropic()
        mlflow.anthropic.autolog()

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


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 80)
    print("TriageFlow: Incident Triage Demo")
    print("=" * 80)
    print(f"Provider: {args.provider}")
    print(f"Vertical: {args.vertical}")
    print()

    # Load configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = config["models"][args.provider]
    print(f"Model: {model}")

    # Initialize client with auto-tracing
    client = initialize_client(args.provider)

    # Load sample incidents
    data_path = os.path.join(script_dir, f"example-data/{args.vertical}.csv")
    df = pd.read_csv(data_path)
    incidents = [row_to_incident(row) for _, row in df.iterrows()]
    print(f"Loaded {len(incidents)} incidents from {args.vertical}")

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
