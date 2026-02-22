#!/usr/bin/env python3
"""
TriageFlow: Incident Triage Demo

Multi-agent incident triage with Domino GenAI tracing.
Runs a 4-agent pipeline to classify, assess, assign, and respond to incidents.

This script demonstrates the three key steps of Domino tracing:
1. Enable MLflow autologging to capture LLM calls
2. Use @add_tracing decorator to create traced functions with evaluators
3. Run within DominoRun context to aggregate metrics across traces

Usage:
    python run_triage.py
    python run_triage.py --provider anthropic --vertical healthcare -n 5
"""

import argparse
import io
import os
import sys
from datetime import datetime

import pandas as pd
import yaml

# =============================================================================
# TRACING SETUP: Import Domino tracing components
# =============================================================================
# - mlflow: Provides autologging for LLM calls (openai, anthropic, etc.)
# - add_tracing: Decorator that creates traced spans with inputs/outputs
# - search_traces: Retrieves traces from a completed run for post-hoc evaluation
# - DominoRun: Context manager that creates an MLflow run and aggregates metrics
# - log_evaluation: Attaches evaluation scores to specific traces
# =============================================================================
import mlflow
from domino.agents.tracing import add_tracing, search_traces
from domino.agents.logging import DominoRun, log_evaluation

from src.models import Incident, IncidentSource
from src.agents import classify_incident, assess_impact, match_resources, draft_response
from src.judges import judge_classification, judge_response, judge_triage


PROVIDERS = ["openai", "anthropic"]
VERTICALS = ["financial_services", "healthcare", "energy", "public_sector"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the TriageFlow incident triage pipeline with Domino tracing."
    )
    parser.add_argument("--provider", choices=PROVIDERS, default="openai")
    parser.add_argument("--vertical", choices=VERTICALS, default="financial_services")
    parser.add_argument("-n", "--num-tickets", type=int, default=0,
                        help="Number of tickets to process (0 = all)")
    return parser.parse_args()


# =============================================================================
# STEP 1: Enable MLflow Autologging
# =============================================================================
# Autologging automatically captures all LLM API calls including:
# - Request parameters (model, temperature, messages)
# - Response content and token usage
# - Latency for each call
# This creates detailed spans under your traced function.
# =============================================================================
def initialize_client(provider: str):
    """Initialize LLM client and enable MLflow autologging."""
    if provider == "openai":
        from openai import OpenAI
        mlflow.openai.autolog()  # <-- Captures all OpenAI API calls
        return OpenAI()
    else:
        from anthropic import Anthropic
        mlflow.anthropic.autolog()  # <-- Captures all Anthropic API calls
        return Anthropic()


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


# =============================================================================
# STEP 2A: Define an Evaluator Function
# =============================================================================
# The evaluator extracts metrics from the traced function's output.
# It receives a span object with .outputs containing the function's return value.
# Return a dict of metric_name -> numeric_value pairs.
# These metrics are automatically attached to each trace.
# =============================================================================
def pipeline_evaluator(span) -> dict:
    """
    Extract metrics from pipeline outputs for automatic evaluation.

    Args:
        span: MLflow span object with .outputs containing function return value

    Returns:
        Dict of metric names to numeric values
    """
    outputs = span.outputs or {}
    if not hasattr(outputs, "get"):
        return {}

    return {
        # Agent output metrics
        "classification_confidence": outputs.get("classification_confidence", 0.5),
        "impact_score": outputs.get("impact_score", 5.0),
        "resource_match_score": outputs.get("resource_match_score", 0.5),
        "completeness_score": outputs.get("completeness_score", 0.5),
        # LLM judge scores
        "classification_judge_score": outputs.get("classification_judge_score", 3),
        "response_judge_score": outputs.get("response_judge_score", 3),
        "triage_judge_score": outputs.get("triage_judge_score", 3),
    }


# =============================================================================
# STEP 2B: Create a Traced Function with @add_tracing
# =============================================================================
# The @add_tracing decorator:
# - Creates a parent span that captures function inputs and outputs
# - Nests all LLM calls (from autolog) under this parent span
# - Runs the evaluator on completion to extract metrics
#
# Parameters:
# - name: Name shown in the trace explorer
# - autolog_frameworks: List of frameworks to capture (e.g., ["openai"])
# - evaluator: Function that extracts metrics from the output
# =============================================================================
def create_triage_function(client, provider: str, model: str, config: dict):
    """Create a traced triage pipeline function."""

    @add_tracing(
        name="triage_incident",           # Span name in trace explorer
        autolog_frameworks=[provider],     # Capture LLM calls for this provider
        evaluator=pipeline_evaluator       # Extract metrics from output
    )
    def triage_incident(incident: Incident):
        """
        Run the 4-agent triage pipeline.

        All LLM calls within this function are automatically captured
        as child spans under the 'triage_incident' parent span.
        """
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

        # Run LLM judges (also captured by autolog)
        class_judge = judge_classification(client, provider, model, incident.description, class_dict)

        resp_judges = judge_response(client, provider, model, incident.description, response_dict)
        resp_score = sum(r.get("score", 3) for r in resp_judges) / len(resp_judges) if resp_judges else 3

        triage_output = {
            "classification": class_dict,
            "impact": impact_dict,
            "assignment": resources_dict,
            "response": response_dict
        }
        triage_judge = judge_triage(client, provider, model, incident.description, triage_output)

        # Return results with metrics for the evaluator
        return {
            "classification": classification,
            "impact": impact,
            "resources": resources,
            "response": response,
            # These values are extracted by pipeline_evaluator
            "classification_confidence": class_dict.get("confidence", 0.5),
            "impact_score": impact_dict.get("impact_score", 5.0),
            "resource_match_score": primary.get("match_score", 0.5) if isinstance(primary, dict) else 0.5,
            "completeness_score": response_dict.get("completeness_score", 0.5),
            "classification_judge_score": class_judge.get("score", 3),
            "response_judge_score": resp_score,
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


# =============================================================================
# STEP 4: Post-hoc Evaluation with search_traces and log_evaluation
# =============================================================================
# After the run completes, you can add additional evaluations:
# - search_traces(run_id=...) retrieves all traces from a run
# - log_evaluation(trace_id=..., name=..., value=...) attaches a metric
#
# This is useful for:
# - Adding human feedback scores
# - Computing derived metrics
# - Flagging traces for review
# =============================================================================
def add_posthoc_evaluations(run_id: str, results: list):
    """Add post-hoc evaluations to traces after pipeline completes."""
    print("\n" + "-" * 40)
    print("Adding post-hoc evaluations...")

    # Retrieve all traces from this run
    traces = search_traces(run_id=run_id)

    for i, trace in enumerate(traces.data):
        result = results[i]

        # Compute combined quality score
        combined_quality = (
            result["classification_judge_score"] +
            result["response_judge_score"] +
            result["triage_judge_score"]
        ) / 3

        # Flag high-urgency incidents for manual review
        needs_review = (
            result["classification"].urgency >= 4 and
            result["impact"].impact_score >= 7
        )

        # Attach evaluations to the trace
        log_evaluation(
            trace_id=trace.id,
            name="combined_quality_score",
            value=round(combined_quality, 2)
        )
        log_evaluation(
            trace_id=trace.id,
            name="needs_manual_review",
            value=1.0 if needs_review else 0.0
        )

    print(f"Added evaluations to {len(traces.data)} traces")


def main():
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

    # STEP 1: Initialize client with autologging
    client = initialize_client(args.provider)
    print(f"MLflow autologging enabled for {args.provider}")

    # Load sample incidents
    data_path = os.path.join(script_dir, f"example-data/{args.vertical}.csv")
    df = pd.read_csv(data_path)
    incidents = [row_to_incident(row) for _, row in df.iterrows()]
    if args.num_tickets > 0:
        incidents = incidents[:args.num_tickets]
    print(f"Processing {len(incidents)} incidents from {args.vertical}")

    # Set up experiment naming
    username = os.environ.get("DOMINO_USER_NAME", os.environ.get("USER", "demo_user"))
    project_name = os.environ.get("DOMINO_PROJECT_NAME", "default")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f"tracing-{project_name}-{username}"
    run_name = f"{args.vertical}-{username}-{timestamp}"

    print(f"\nExperiment: {experiment_name}")
    print(f"Run: {run_name}")
    print()

    # =========================================================================
    # STEP 3: Run within DominoRun Context
    # =========================================================================
    # DominoRun creates an MLflow run that:
    # - Stores all traces from @add_tracing functions
    # - Aggregates metrics across traces (mean, median, etc.)
    # - Links configuration for reproducibility
    #
    # Parameters:
    # - agent_config_path: Path to config file (stored with the run)
    # - custom_summary_metrics: List of (metric_name, aggregation) tuples
    # =========================================================================
    aggregated_metrics = [
        ("classification_confidence", "mean"),
        ("impact_score", "median"),
        ("resource_match_score", "mean"),
        ("completeness_score", "mean"),
        ("classification_judge_score", "mean"),
        ("response_judge_score", "mean"),
        ("triage_judge_score", "mean"),
    ]

    # STEP 2: Create the traced function
    triage_incident = create_triage_function(client, args.provider, model, config)

    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

    # Run pipeline within DominoRun context
    results = []
    run_id = None

    with DominoRun(
        agent_config_path=config_path,
        custom_summary_metrics=aggregated_metrics
    ) as run:
        mlflow.set_tag("mlflow.runName", run_name)
        run_id = run.info.run_id

        for incident in incidents:
            print(f"Processing {incident.ticket_id}...")

            # Each call creates a trace under this run
            result = triage_incident(incident)

            results.append({"ticket_id": incident.ticket_id, **result})
            print(f"  -> {result['classification'].category.value} | "
                  f"Urgency: {result['classification'].urgency} | "
                  f"Impact: {result['impact'].impact_score}")

        # Suppress DominoRun exit messages
        _stdout = sys.stdout
        sys.stdout = io.StringIO()

    sys.stdout = _stdout
    print(f"\nProcessed {len(results)} incidents")

    # Print results
    print_results_summary(results)
    print_sample_communication(results)

    # STEP 4: Add post-hoc evaluations
    add_posthoc_evaluations(run_id, results)

    print("\n" + "=" * 80)
    print("Done! View traces in Domino Experiment Manager.")
    print("=" * 80)


if __name__ == "__main__":
    main()
