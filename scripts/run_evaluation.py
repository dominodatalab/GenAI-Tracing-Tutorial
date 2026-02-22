#!/usr/bin/env python3
"""
TriageFlow Production Evaluation Script

This script supports two modes:
1. Batch Processing: Run triage on multiple tickets with full tracing
2. Post-hoc Evaluation: Add evaluations to existing traces from previous runs

Usage:
    # Batch processing with tracing
    python run_evaluation.py batch --provider openai --vertical financial_services -n 10

    # Post-hoc evaluation on existing run
    python run_evaluation.py evaluate --run-id <mlflow_run_id>

    # List recent runs
    python run_evaluation.py list-runs --experiment <experiment_name>

See: https://docs.dominodatalab.com/en/cloud/user_guide/fc1922/set-up-and-run-genai-traces/
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, "/mnt/code")

# Import tracing modules
try:
    from domino.agents.tracing import add_tracing, init_tracing, search_traces
    from domino.agents.logging import DominoRun, log_evaluation
    import mlflow
    TRACING_AVAILABLE = True
    init_tracing()
except ImportError as e:
    print(f"Warning: Domino tracing SDK not available: {e}")
    print("Some features will be disabled.")
    TRACING_AVAILABLE = False
    mlflow = None

from src.models import Incident, IncidentSource
from src.agents import classify_incident, assess_impact, match_resources, draft_response
from src.judges import judge_classification, judge_response, judge_triage

# Constants
PROVIDERS = ["openai", "anthropic"]
VERTICALS = ["financial_services", "healthcare", "energy", "public_sector"]
CONFIG_PATH = "/mnt/code/config.yaml"


def load_config(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def initialize_client(provider: str):
    """Initialize LLM client with autologging."""
    if provider == "openai":
        from openai import OpenAI
        if mlflow:
            mlflow.openai.autolog()
        return OpenAI()
    else:
        from anthropic import Anthropic
        if mlflow:
            mlflow.anthropic.autolog()
        return Anthropic()


def row_to_incident(row: pd.Series) -> Incident:
    """Convert a DataFrame row to an Incident object."""
    return Incident(
        ticket_id=row["ticket_id"],
        description=row["description"],
        source=IncidentSource(row["source"]),
        reporter=row["reporter"] if pd.notna(row.get("reporter")) else None,
        affected_system=row["affected_system"] if pd.notna(row.get("affected_system")) else None,
        initial_severity=int(row["initial_severity"]) if pd.notna(row.get("initial_severity")) else None
    )


def pipeline_evaluator(span) -> Dict[str, Any]:
    """Extract metrics from the triage pipeline span for automatic evaluation."""
    try:
        output = span.outputs or {}
        judge_scores = output.get("judge_scores", {})

        classification = output.get("classification")
        if hasattr(classification, "model_dump"):
            classification = classification.model_dump()
        elif classification is None:
            classification = {}

        impact = output.get("impact")
        if hasattr(impact, "model_dump"):
            impact = impact.model_dump()
        elif impact is None:
            impact = {}

        assignment = output.get("assignment")
        if hasattr(assignment, "model_dump"):
            assignment = assignment.model_dump()
        elif assignment is None:
            assignment = {}

        response = output.get("response")
        if hasattr(response, "model_dump"):
            response = response.model_dump()
        elif response is None:
            response = {}

        primary = assignment.get("primary_responder", {})
        if hasattr(primary, "model_dump"):
            primary = primary.model_dump()

        category = classification.get("category", "unknown")
        if hasattr(category, "value"):
            category = category.value

        return {
            "classification_judge_score": float(judge_scores.get("classification_score", 0.0)),
            "response_judge_score": float(judge_scores.get("response_score", 0.0)),
            "triage_judge_score": float(judge_scores.get("triage_score", 0.0)),
            "combined_quality_score": float(judge_scores.get("combined_score", 0.0)),
            "classification_confidence": float(classification.get("confidence", 0.0)),
            "urgency": int(classification.get("urgency", 0)),
            "impact_score": float(impact.get("impact_score", 0.0)),
            "affected_users_estimate": int(impact.get("affected_users_estimate", 0)),
            "completeness_score": float(response.get("completeness_score", 0.0)),
            "resource_match_score": float(primary.get("match_score", 0.0) if isinstance(primary, dict) else 0.0),
            "pipeline_success": 1 if output and not output.get("error") else 0,
            "sla_met": 1 if assignment.get("sla_met") else 0,
            "category": str(category),
            "blast_radius": str(impact.get("blast_radius", "unknown")),
            "responder": str(primary.get("name", "unknown") if isinstance(primary, dict) else "unknown"),
        }
    except Exception as e:
        print(f"Evaluation error: {e}")
        return {"pipeline_success": 0, "combined_quality_score": 0.0}


def create_traced_triage_function(client, provider: str, model: str, config: Dict[str, Any]):
    """Create a traced triage function for batch processing."""
    tracing_framework = "openai" if provider == "openai" else provider

    @add_tracing(name="triage_incident", autolog_frameworks=[tracing_framework], evaluator=pipeline_evaluator)
    def triage_incident(incident: Incident) -> Dict[str, Any]:
        """Run the full triage pipeline with judges."""
        classification = classify_incident(client, provider, model, incident, config)
        impact = assess_impact(client, provider, model, incident, classification, config)
        assignment = match_resources(client, provider, model, classification, impact, config)
        response = draft_response(client, provider, model, incident, classification, impact, assignment, config)

        triage_output = {
            "classification": classification,
            "impact": impact,
            "assignment": assignment,
            "response": response,
        }

        # Run judges
        classification_eval = judge_classification(
            client, provider, model, incident.description, classification.model_dump()
        )
        response_evals = judge_response(
            client, provider, model, incident.description, response.model_dump()
        )
        triage_eval = judge_triage(
            client, provider, model, incident.description, {
                "classification": classification.model_dump(),
                "impact": impact.model_dump(),
                "assignment": assignment.model_dump(),
                "response": response.model_dump()
            }
        )

        classification_score = classification_eval.get("score", 3) if classification_eval else 3
        triage_score = triage_eval.get("score", 3) if triage_eval else 3
        response_scores = [e.get("score", 3) for e in response_evals] if response_evals else [3]
        response_score = sum(response_scores) / len(response_scores)
        combined_score = (classification_score + response_score + triage_score) / 3

        triage_output["judge_scores"] = {
            "classification_score": classification_score,
            "response_score": response_score,
            "triage_score": triage_score,
            "combined_score": combined_score,
        }

        # Log evaluations to current span
        if mlflow and log_evaluation:
            span = mlflow.get_current_active_span()
            if span:
                class_dict = classification.model_dump()
                impact_dict = impact.model_dump()
                assignment_dict = assignment.model_dump()
                response_dict = response.model_dump()
                primary = assignment_dict.get("primary_responder", {})

                log_evaluation(trace_id=span.request_id, name="classification_judge_score", value=float(classification_score))
                log_evaluation(trace_id=span.request_id, name="response_judge_score", value=float(response_score))
                log_evaluation(trace_id=span.request_id, name="triage_judge_score", value=float(triage_score))
                log_evaluation(trace_id=span.request_id, name="combined_quality_score", value=round(combined_score, 2))
                log_evaluation(trace_id=span.request_id, name="classification_confidence", value=float(class_dict.get("confidence", 0.5)))
                log_evaluation(trace_id=span.request_id, name="urgency", value=float(class_dict.get("urgency", 0)))
                log_evaluation(trace_id=span.request_id, name="impact_score", value=float(impact_dict.get("impact_score", 0.0)))
                log_evaluation(trace_id=span.request_id, name="completeness_score", value=float(response_dict.get("completeness_score", 0.0)))
                log_evaluation(trace_id=span.request_id, name="resource_match_score", value=float(primary.get("match_score", 0.0) if isinstance(primary, dict) else 0.0))
                log_evaluation(trace_id=span.request_id, name="pipeline_success", value=1.0)
                log_evaluation(trace_id=span.request_id, name="sla_met", value=1.0 if assignment_dict.get("sla_met") else 0.0)

                needs_review = class_dict.get("urgency", 0) >= 4 and impact_dict.get("impact_score", 0) >= 7
                log_evaluation(trace_id=span.request_id, name="needs_manual_review", value=1.0 if needs_review else 0.0)

        return triage_output

    return triage_incident


def run_batch_processing(args):
    """Run batch triage processing with full tracing."""
    if not TRACING_AVAILABLE:
        print("ERROR: Domino tracing SDK is required for batch processing.")
        sys.exit(1)

    print("=" * 80)
    print("TriageFlow Batch Evaluation")
    print("=" * 80)
    print(f"Provider: {args.provider}")
    print(f"Vertical: {args.vertical}")
    print(f"Config: {args.config}")
    print()

    config = load_config(args.config)
    model = config["models"][args.provider]
    print(f"Model: {model}")

    # Initialize client
    client = initialize_client(args.provider)

    # Load incidents
    data_path = f"/mnt/code/example-data/{args.vertical}.csv"
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)
    incidents = [row_to_incident(row) for _, row in df.iterrows()]
    if args.num_tickets > 0:
        incidents = incidents[:args.num_tickets]
    print(f"Processing {len(incidents)} incidents")

    # Set up experiment naming
    username = os.environ.get("DOMINO_USER_NAME", os.environ.get("USER", "demo_user"))
    project_name = os.environ.get("DOMINO_PROJECT_NAME", "triageflow")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = args.experiment or f"tracing-{project_name}-{username}"
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
        ("combined_quality_score", "mean"),
    ]

    # Create traced function
    triage_incident = create_traced_triage_function(client, args.provider, model, config)

    # Set experiment
    mlflow.set_experiment(experiment_name)

    # Run with DominoRun context
    results = []
    run_id = None

    with DominoRun(agent_config_path=args.config, custom_summary_metrics=aggregated_metrics) as run:
        mlflow.set_tag("mlflow.runName", run_name)
        mlflow.set_tag("vertical", args.vertical)
        mlflow.set_tag("provider", args.provider)
        mlflow.set_tag("model", model)
        mlflow.set_tag("launched_from", "cli")
        run_id = run.info.run_id

        for i, incident in enumerate(incidents, 1):
            print(f"[{i}/{len(incidents)}] Processing {incident.ticket_id}...")

            try:
                result = triage_incident(incident)
                results.append({
                    "ticket_id": incident.ticket_id,
                    "success": True,
                    **result
                })
                cat = result["classification"].category.value
                urg = result["classification"].urgency
                imp = result["impact"].impact_score
                print(f"    -> {cat} | Urgency: {urg} | Impact: {imp:.1f}")
            except Exception as e:
                print(f"    -> ERROR: {e}")
                results.append({
                    "ticket_id": incident.ticket_id,
                    "success": False,
                    "error": str(e)
                })

    print(f"\nProcessed {len(results)} incidents")
    print(f"Run ID: {run_id}")

    # Add post-hoc evaluations
    if run_id:
        add_posthoc_evaluations(run_id, results)

    # Print summary
    print_results_summary(results)

    print("\n" + "=" * 80)
    print("Done. View traces in Domino Experiment Manager.")
    print(f"Run ID: {run_id}")
    print("=" * 80)

    return run_id


def add_posthoc_evaluations(run_id: str, results: List[Dict[str, Any]]):
    """Add post-hoc evaluations to traces after pipeline completes."""
    print("\n" + "-" * 40)
    print("Adding post-hoc evaluations...")

    traces = search_traces(run_id=run_id)

    successful_results = [r for r in results if r.get("success", False)]

    for i, trace in enumerate(traces.data):
        if i >= len(successful_results):
            break

        result = successful_results[i]

        # Compute derived metrics
        judge_scores = result.get("judge_scores", {})
        combined_quality = judge_scores.get("combined_score", 3.0)

        classification = result.get("classification")
        if hasattr(classification, "model_dump"):
            classification = classification.model_dump()
        else:
            classification = classification or {}

        impact = result.get("impact")
        if hasattr(impact, "model_dump"):
            impact = impact.model_dump()
        else:
            impact = impact or {}

        urgency = classification.get("urgency", 0)
        impact_score = impact.get("impact_score", 0)

        # Flag high-urgency incidents needing manual review
        needs_review = urgency >= 4 and impact_score >= 7

        # Quality tier classification
        if combined_quality >= 4.0:
            quality_tier = "excellent"
        elif combined_quality >= 3.0:
            quality_tier = "good"
        elif combined_quality >= 2.0:
            quality_tier = "fair"
        else:
            quality_tier = "poor"

        # Log evaluations
        log_evaluation(trace_id=trace.id, name="combined_quality_score", value=round(combined_quality, 2))
        log_evaluation(trace_id=trace.id, name="needs_manual_review", value=1.0 if needs_review else 0.0)
        log_evaluation(trace_id=trace.id, name="quality_tier", value=quality_tier)

    print(f"Added evaluations to {len(traces.data)} traces")


def run_posthoc_evaluation(args):
    """Add post-hoc evaluations to an existing run."""
    if not TRACING_AVAILABLE:
        print("ERROR: Domino tracing SDK is required for post-hoc evaluation.")
        sys.exit(1)

    print("=" * 80)
    print("TriageFlow Post-Hoc Evaluation")
    print("=" * 80)
    print(f"Run ID: {args.run_id}")
    print()

    # Search for traces
    print("Searching for traces...")
    traces = search_traces(run_id=args.run_id)

    if not traces.data:
        print("No traces found for this run.")
        sys.exit(1)

    print(f"Found {len(traces.data)} traces")
    print()

    # Process each trace
    for i, trace in enumerate(traces.data, 1):
        print(f"[{i}/{len(traces.data)}] Processing trace {trace.id[:8]}...")

        # Extract existing evaluations if available
        existing_evals = {}
        if hasattr(trace, "evaluations"):
            for eval_item in trace.evaluations:
                existing_evals[eval_item.name] = eval_item.value

        # Compute new evaluations based on existing metrics
        classification_score = existing_evals.get("classification_judge_score", 3.0)
        response_score = existing_evals.get("response_judge_score", 3.0)
        triage_score = existing_evals.get("triage_judge_score", 3.0)
        urgency = existing_evals.get("urgency", 0)
        impact_score = existing_evals.get("impact_score", 0)

        # Recompute combined quality
        combined_quality = (classification_score + response_score + triage_score) / 3

        # Flag for manual review
        needs_review = urgency >= 4 and impact_score >= 7

        # Quality tier
        if combined_quality >= 4.0:
            quality_tier = "excellent"
        elif combined_quality >= 3.0:
            quality_tier = "good"
        elif combined_quality >= 2.0:
            quality_tier = "fair"
        else:
            quality_tier = "poor"

        # SLA compliance analysis
        sla_met = existing_evals.get("sla_met", 0)
        sla_status = "compliant" if sla_met else "non_compliant"

        # Log new evaluations
        log_evaluation(trace_id=trace.id, name="combined_quality_score_v2", value=round(combined_quality, 2))
        log_evaluation(trace_id=trace.id, name="needs_manual_review", value=1.0 if needs_review else 0.0)
        log_evaluation(trace_id=trace.id, name="quality_tier", value=quality_tier)
        log_evaluation(trace_id=trace.id, name="sla_status", value=sla_status)

        # Custom evaluations from args
        if args.add_label:
            for label in args.add_label:
                name, value = label.split("=", 1)
                try:
                    numeric_value = float(value)
                    log_evaluation(trace_id=trace.id, name=name, value=numeric_value)
                except ValueError:
                    log_evaluation(trace_id=trace.id, name=name, value=value)

        print(f"    -> Quality: {quality_tier} | Review needed: {needs_review}")

    print("\n" + "=" * 80)
    print(f"Post-hoc evaluation complete. Updated {len(traces.data)} traces.")
    print("=" * 80)


def list_runs(args):
    """List recent runs from an experiment."""
    if not TRACING_AVAILABLE or not mlflow:
        print("ERROR: MLflow is required to list runs.")
        sys.exit(1)

    experiment_name = args.experiment
    if not experiment_name:
        username = os.environ.get("DOMINO_USER_NAME", os.environ.get("USER", "demo_user"))
        project_name = os.environ.get("DOMINO_PROJECT_NAME", "triageflow")
        experiment_name = f"tracing-{project_name}-{username}"

    print("=" * 80)
    print(f"Recent Runs: {experiment_name}")
    print("=" * 80)
    print()

    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"Experiment not found: {experiment_name}")
            sys.exit(1)

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=args.limit,
            order_by=["start_time DESC"]
        )

        if runs.empty:
            print("No runs found.")
            return

        print(f"{'Run ID':<36} {'Run Name':<30} {'Status':<10} {'Start Time'}")
        print("-" * 100)

        for _, run in runs.iterrows():
            run_id = run.get("run_id", "N/A")
            run_name = run.get("tags.mlflow.runName", "N/A")
            status = run.get("status", "N/A")
            start_time = run.get("start_time", "N/A")
            if hasattr(start_time, "strftime"):
                start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"{run_id:<36} {run_name[:30]:<30} {status:<10} {start_time}")

    except Exception as e:
        print(f"Error listing runs: {e}")
        sys.exit(1)


def analyze_traces(args):
    """Analyze traces from a run and generate a report."""
    if not TRACING_AVAILABLE:
        print("ERROR: Domino tracing SDK is required.")
        sys.exit(1)

    print("=" * 80)
    print("TriageFlow Trace Analysis")
    print("=" * 80)
    print(f"Run ID: {args.run_id}")
    print()

    traces = search_traces(run_id=args.run_id)

    if not traces.data:
        print("No traces found.")
        return

    print(f"Found {len(traces.data)} traces")
    print()

    # Collect metrics
    metrics = {
        "total_traces": len(traces.data),
        "quality_scores": [],
        "urgency_levels": [],
        "impact_scores": [],
        "sla_compliance": [],
        "categories": {},
    }

    for trace in traces.data:
        if hasattr(trace, "evaluations"):
            evals = {e.name: e.value for e in trace.evaluations}

            if "combined_quality_score" in evals:
                metrics["quality_scores"].append(evals["combined_quality_score"])
            if "urgency" in evals:
                metrics["urgency_levels"].append(int(evals["urgency"]))
            if "impact_score" in evals:
                metrics["impact_scores"].append(evals["impact_score"])
            if "sla_met" in evals:
                metrics["sla_compliance"].append(evals["sla_met"])
            if "category" in evals:
                cat = evals["category"]
                metrics["categories"][cat] = metrics["categories"].get(cat, 0) + 1

    # Print analysis
    print("Quality Scores:")
    if metrics["quality_scores"]:
        avg_quality = sum(metrics["quality_scores"]) / len(metrics["quality_scores"])
        min_quality = min(metrics["quality_scores"])
        max_quality = max(metrics["quality_scores"])
        print(f"  Average: {avg_quality:.2f}")
        print(f"  Min: {min_quality:.2f}, Max: {max_quality:.2f}")
    else:
        print("  No quality scores found")

    print("\nUrgency Distribution:")
    if metrics["urgency_levels"]:
        for level in range(1, 6):
            count = metrics["urgency_levels"].count(level)
            pct = count / len(metrics["urgency_levels"]) * 100
            bar = "#" * int(pct / 2)
            print(f"  Level {level}: {count:3d} ({pct:5.1f}%) {bar}")
    else:
        print("  No urgency data found")

    print("\nSLA Compliance:")
    if metrics["sla_compliance"]:
        compliant = sum(1 for s in metrics["sla_compliance"] if s)
        total = len(metrics["sla_compliance"])
        pct = compliant / total * 100
        print(f"  Compliant: {compliant}/{total} ({pct:.1f}%)")
    else:
        print("  No SLA data found")

    print("\nCategories:")
    if metrics["categories"]:
        for cat, count in sorted(metrics["categories"].items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}")
    else:
        print("  No category data found")

    # Export to JSON if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics exported to: {args.output}")


def print_results_summary(results: List[Dict[str, Any]]):
    """Print a summary table of results."""
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    successful = [r for r in results if r.get("success", False)]
    failed = [r for r in results if not r.get("success", False)]

    print(f"Total: {len(results)} | Success: {len(successful)} | Failed: {len(failed)}")
    print()

    if successful:
        summary_data = []
        for r in successful:
            classification = r.get("classification")
            if hasattr(classification, "model_dump"):
                classification = classification.model_dump()

            impact = r.get("impact")
            if hasattr(impact, "model_dump"):
                impact = impact.model_dump()

            resources = r.get("assignment")
            if hasattr(resources, "model_dump"):
                resources = resources.model_dump()

            primary = resources.get("primary_responder", {}) if resources else {}
            if hasattr(primary, "model_dump"):
                primary = primary.model_dump()

            category = classification.get("category", "unknown") if classification else "unknown"
            if hasattr(category, "value"):
                category = category.value

            summary_data.append({
                "Ticket": r["ticket_id"],
                "Category": category,
                "Urgency": classification.get("urgency", 0) if classification else 0,
                "Impact": impact.get("impact_score", 0) if impact else 0,
                "Responder": primary.get("name", "N/A") if isinstance(primary, dict) else "N/A",
                "SLA": "Yes" if resources and resources.get("sla_met") else "No"
            })

        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="TriageFlow Production Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run batch processing
  python run_evaluation.py batch --provider openai --vertical financial_services -n 5

  # Add evaluations to existing run
  python run_evaluation.py evaluate --run-id abc123

  # List recent runs
  python run_evaluation.py list-runs

  # Analyze traces from a run
  python run_evaluation.py analyze --run-id abc123 --output report.json
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Batch processing command
    batch_parser = subparsers.add_parser("batch", help="Run batch triage processing with tracing")
    batch_parser.add_argument("--provider", choices=PROVIDERS, default="openai", help="LLM provider")
    batch_parser.add_argument("--vertical", choices=VERTICALS, default="financial_services", help="Industry vertical")
    batch_parser.add_argument("-n", "--num-tickets", type=int, default=0, help="Number of tickets (0 = all)")
    batch_parser.add_argument("--config", default=CONFIG_PATH, help="Path to config file")
    batch_parser.add_argument("--experiment", help="MLflow experiment name")

    # Post-hoc evaluation command
    eval_parser = subparsers.add_parser("evaluate", help="Add post-hoc evaluations to existing traces")
    eval_parser.add_argument("--run-id", required=True, help="MLflow run ID")
    eval_parser.add_argument("--add-label", action="append", help="Add custom label (name=value)")

    # List runs command
    list_parser = subparsers.add_parser("list-runs", help="List recent runs")
    list_parser.add_argument("--experiment", help="Experiment name")
    list_parser.add_argument("--limit", type=int, default=20, help="Maximum number of runs to show")

    # Analyze traces command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze traces from a run")
    analyze_parser.add_argument("--run-id", required=True, help="MLflow run ID")
    analyze_parser.add_argument("--output", help="Output file for JSON report")

    args = parser.parse_args()

    if args.command == "batch":
        run_batch_processing(args)
    elif args.command == "evaluate":
        run_posthoc_evaluation(args)
    elif args.command == "list-runs":
        list_runs(args)
    elif args.command == "analyze":
        analyze_traces(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
