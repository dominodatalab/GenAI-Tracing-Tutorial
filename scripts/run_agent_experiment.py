#!/usr/bin/env python3
"""
TriageFlow Agent Experiment Runner

Runs multi-parameter grid search experiments across all agents.
Each parameter combination runs as its own DominoRun with full tracing.

Usage:
    python run_agent_experiment.py
    python run_agent_experiment.py --sample-size 50
    python run_agent_experiment.py --exhaustive --max-incidents 5
"""

import argparse
import copy
import io
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

# Ensure project root is in path
sys.path.insert(0, "/mnt/code")

import mlflow
import yaml

from domino.agents.logging import DominoRun
from domino.agents.tracing import add_tracing

from src.models import Incident, IncidentSource
from src.agents import classify_incident, assess_impact, match_resources, draft_response
from src.judges import judge_classification, judge_response, judge_triage
from src.experiment_utils import (
    load_yaml_config,
    generate_parameter_grid,
    sample_parameter_grid,
    apply_agent_overrides,
    log_experiment_params,
    create_experiment_signature,
    rank_results,
    summarize_best_params,
    build_prompt_with_variants,
    format_few_shot_examples,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run multi-parameter agent experiments with grid search."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to experiment grid config (default: configs/experiment_grid.yaml)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of random samples from grid (overrides config)"
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
        default=None,
        help="Random seed for sampling"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without executing"
    )
    return parser.parse_args()


def load_incidents(test_data_path: str, max_incidents: int = None) -> List[Incident]:
    """Load incidents from JSONL test data file."""
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
        mlflow.openai.autolog()
        return get_domino_model_client()
    elif provider == "anthropic":
        from anthropic import Anthropic
        mlflow.anthropic.autolog()
        return Anthropic()
    raise ValueError(f"Unknown provider: {provider}")


def get_model_info(model_key: str, models_config: dict) -> Tuple[str, str]:
    """Get provider and model name from model key."""
    if model_key in models_config:
        model_info = models_config[model_key]
        if isinstance(model_info, dict):
            provider = model_info.get("provider", model_key)
            model_name = model_info.get("name", "gpt-4o-mini")
        else:
            provider = model_key
            model_name = model_info
    else:
        provider = model_key
        model_name = "gpt-4o-mini"
    return provider, model_name


def create_triage_function(client, provider: str, model: str, config: dict):
    """Create the triage pipeline function with tracing and evaluation."""

    def pipeline_evaluator(inputs, output, **kwargs):
        """Evaluate the triage pipeline output."""
        incident = inputs.get("incident")
        if not incident or not output:
            return {"combined_quality_score": 0.0}

        eval_result = {}
        try:
            classification_eval = judge_classification(
                client, provider, model, incident, output["classification"], config
            )
            eval_result["classification_eval"] = classification_eval

            response_evals = judge_response(
                client, provider, model, incident, output["response"], config
            )
            eval_result["response_evals"] = response_evals

            triage_eval = judge_triage(
                client, provider, model, incident, output, config
            )
            eval_result["triage_eval"] = triage_eval

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
        classification = classify_incident(client, provider, model, incident, config)
        impact = assess_impact(client, provider, model, incident, classification, config)
        assignment = match_resources(client, provider, model, incident, classification, impact, config)
        response = draft_response(client, provider, model, incident, classification, impact, assignment, config)

        return {
            "classification": classification,
            "impact": impact,
            "assignment": assignment,
            "response": response,
        }

    return triage_incident


def build_agent_prompts(
    base_config: dict,
    param_set: dict,
    prompt_variants: dict,
    few_shot_examples: dict,
) -> dict:
    """
    Build prompts for each agent based on parameter settings.

    Handles system_prompt_variant, few_shot_count, chain_of_thought, etc.
    """
    config = copy.deepcopy(base_config)

    for agent_name in ["classifier", "impact_assessor", "resource_matcher", "response_drafter"]:
        agent_params = param_set.get(agent_name, {})

        # Get base prompt from config
        base_prompt = config["agents"][agent_name].get("prompt", "")

        # Get variant name (if specified)
        variant_name = agent_params.get("system_prompt_variant", "default")

        # Get few-shot count
        few_shot_count = agent_params.get("few_shot_count", 0)

        # Get chain of thought setting
        chain_of_thought = agent_params.get("chain_of_thought", False)

        # Get agent-specific prompt variants and examples
        agent_variants = prompt_variants.get(agent_name, {})
        agent_examples = few_shot_examples.get(agent_name, [])

        # Build the prompt
        final_prompt = build_prompt_with_variants(
            base_prompt=base_prompt,
            prompt_variants=agent_variants,
            variant_name=variant_name,
            few_shot_examples=agent_examples,
            few_shot_count=few_shot_count,
            chain_of_thought=chain_of_thought,
            length_guidance=agent_params.get("length_guidance"),
        )

        # Update config with constructed prompt
        config["agents"][agent_name]["prompt"] = final_prompt

    return config


def run_single_experiment(
    param_set: dict,
    base_config: dict,
    models_config: dict,
    agents_config_path: str,
    incidents: List[Incident],
    prompt_variants: dict,
    few_shot_examples: dict,
) -> dict:
    """
    Run a single experiment with one parameter combination.

    Returns dict with parameters and metrics.
    """
    # Build prompts with variants and few-shot examples
    exp_config = build_agent_prompts(base_config, param_set, prompt_variants, few_shot_examples)

    # Apply other parameter overrides (temperature, etc.)
    exp_config = apply_agent_overrides(exp_config, param_set)

    # Determine which model/provider to use (use classifier's model as primary)
    classifier_model_key = param_set.get("classifier", {}).get("model", "openai")
    provider, model_name = get_model_info(classifier_model_key, models_config)

    # Initialize client
    client = initialize_client(provider, exp_config)

    # Create triage function
    triage_incident = create_triage_function(client, provider, model_name, exp_config)

    # Create experiment signature for run naming
    exp_signature = create_experiment_signature(param_set)

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

    results = []

    with DominoRun(agent_config_path=agents_config_path, custom_summary_metrics=aggregated_metrics) as run:
        # Set run name
        mlflow.set_tag("mlflow.runName", f"agent-exp-{exp_signature[:50]}")

        # Log experiment mode and type
        mlflow.set_tag("mode", "agent_experiment")
        mlflow.set_tag("experiment_type", "multi_parameter_grid_search")

        # Log all parameters including model types for each agent
        log_experiment_params(param_set)

        # Log model type tags for easy filtering
        for agent_name, params in param_set.items():
            model_key = params.get("model", "openai")
            mlflow.set_tag(f"{agent_name}_model", model_key)

        # Process each incident
        for incident in incidents:
            try:
                result = triage_incident(incident)

                result_metrics = {
                    "ticket_id": incident.ticket_id,
                    "classification_confidence": result["classification"].confidence,
                    "impact_score": result["impact"].impact_score,
                    "resource_match_score": result["assignment"].primary_responder.match_score,
                    "completeness_score": result["response"].completeness_score,
                }
                results.append(result_metrics)

            except Exception as e:
                print(f"Error processing {incident.ticket_id}: {e}", file=sys.stderr)
                continue

        # Suppress DominoRun exit messages
        _stdout = sys.stdout
        sys.stdout = io.StringIO()

    sys.stdout = _stdout

    # Compute aggregated metrics
    metrics = {}
    if results:
        for key in ["classification_confidence", "impact_score", "resource_match_score", "completeness_score"]:
            values = [r[key] for r in results if key in r]
            if values:
                metrics[f"{key}_mean"] = sum(values) / len(values)

        metrics["incidents_processed"] = len(results)
        metrics["combined_quality_score"] = sum(
            metrics.get(f"{k}_mean", 0) for k in
            ["classification_confidence", "impact_score", "resource_match_score", "completeness_score"]
        ) / 4

    return {
        "parameters": param_set,
        "signature": exp_signature,
        "metrics": metrics,
    }


def main():
    """Main entry point."""
    args = parse_args()

    # Paths
    project_root = "/mnt/code"

    if args.config:
        grid_config_path = args.config if os.path.isabs(args.config) else os.path.join(project_root, args.config)
    else:
        grid_config_path = os.path.join(project_root, "configs/experiment_grid.yaml")

    agents_config_path = os.path.join(project_root, "configs/agents.yaml")
    prompt_variants_path = os.path.join(project_root, "configs/prompt_variants.yaml")
    few_shot_path = os.path.join(project_root, "configs/few_shot_examples.yaml")

    # Load configurations
    grid_config = load_yaml_config(grid_config_path)
    agents_config = load_yaml_config(agents_config_path)
    prompt_variants = load_yaml_config(prompt_variants_path)
    few_shot_examples = load_yaml_config(few_shot_path)

    exp_config = grid_config.get("agent_experiment", {})
    models_config = grid_config.get("models", agents_config.get("models", {}))

    # Get test data settings
    test_data_config = exp_config.get("test_data", {})
    test_data_path = test_data_config.get("source", "example-data/test_incidents.jsonl")
    if not os.path.isabs(test_data_path):
        test_data_path = os.path.join(project_root, test_data_path)

    max_incidents = args.max_incidents or test_data_config.get("max_incidents", 10)

    # Sampling settings
    sampling_config = exp_config.get("sampling", {})
    sample_size = args.sample_size or sampling_config.get("sample_size", 50)
    seed = args.seed or sampling_config.get("seed", 42)

    if args.exhaustive:
        sample_size = None  # Run all

    # Generate parameter grid
    param_grid = generate_parameter_grid(exp_config)
    total_combinations = len(param_grid)

    # Sample if needed
    if sample_size and sample_size < total_combinations:
        param_grid = sample_parameter_grid(param_grid, sample_size, seed)

    # Load incidents
    incidents = load_incidents(test_data_path, max_incidents)

    # Setup experiment naming
    username = os.environ.get("DOMINO_USER_NAME", os.environ.get("USER", "demo_user"))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f"agent-optimization-{username}"

    print("=" * 70)
    print("TRIAGEFLOW AGENT EXPERIMENT")
    print("=" * 70)
    print(f"Experiment: {experiment_name}")
    print(f"Total parameter combinations: {total_combinations}")
    print(f"Running: {len(param_grid)} combinations")
    print(f"Incidents per run: {len(incidents)}")
    print(f"Timestamp: {timestamp}")
    print()

    if args.dry_run:
        print("[DRY RUN] Would run the following parameter combinations:")
        for i, params in enumerate(param_grid[:10], 1):
            sig = create_experiment_signature(params)
            print(f"  {i}. {sig[:60]}...")
        if len(param_grid) > 10:
            print(f"  ... and {len(param_grid) - 10} more")
        return

    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

    # Run experiments
    all_results = []

    for i, param_set in enumerate(param_grid, 1):
        sig = create_experiment_signature(param_set)
        print(f"[{i}/{len(param_grid)}] Running: {sig[:50]}...")

        try:
            result = run_single_experiment(
                param_set=param_set,
                base_config=agents_config,
                models_config=models_config,
                agents_config_path=agents_config_path,
                incidents=incidents,
                prompt_variants=prompt_variants,
                few_shot_examples=few_shot_examples,
            )
            all_results.append(result)

            score = result["metrics"].get("combined_quality_score", 0)
            processed = result["metrics"].get("incidents_processed", 0)
            print(f"  Score: {score:.3f} | Processed: {processed}/{len(incidents)}")

        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            all_results.append({
                "parameters": param_set,
                "signature": sig,
                "metrics": {"combined_quality_score": 0, "error": str(e)},
            })

        print()

    # Rank results
    ranked_results = rank_results(all_results)

    # Print summary
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print("Top 10 configurations:")
    print("-" * 70)

    for result in ranked_results[:10]:
        rank = result["rank"]
        score = result["metrics"].get("combined_quality_score", 0)
        sig = result["signature"]
        print(f"  #{rank}: {score:.3f} - {sig[:50]}")

    print()
    print("Best configuration:")
    print("-" * 70)
    best_params = summarize_best_params(ranked_results)
    for agent, params in best_params.items():
        print(f"  {agent}:")
        for k, v in params.items():
            print(f"    {k}: {v}")

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Results logged to experiment: {experiment_name}")
    print(f"Total runs: {len(all_results)}")


if __name__ == "__main__":
    main()
