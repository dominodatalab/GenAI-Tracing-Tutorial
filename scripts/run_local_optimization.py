#!/usr/bin/env python3
"""
Local Model Optimization Script

Establishes baseline performance with OpenAI, then optimizes the local Qwen model
through grid search. Produces a comparison report with recommendations.

Approach:
1. Run baseline with known-good OpenAI configuration (single run per agent)
2. Run parameter grid search for local model only
3. Compare best local config against baseline
4. Save best configuration and optionally submit as Domino Job

Experiment naming: agent-optimization-{vertical}-{username}

Usage:
    python run_local_optimization.py --vertical healthcare -n 5
    python run_local_optimization.py --vertical healthcare --baseline-only
    python run_local_optimization.py --vertical healthcare --local-only
    python run_local_optimization.py --vertical healthcare --submit-job
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, "/mnt/code")

import mlflow
import yaml

from domino.agents.logging import DominoRun
from domino.agents.tracing import add_tracing

from src.models import Incident, IncidentSource
from src.agents import classify_incident, assess_impact, match_resources, draft_response


# =============================================================================
# CONFIGURATION
# =============================================================================

VERTICALS = ["financial_services", "healthcare", "energy", "public_sector"]
GROUND_TRUTH_PATH = "/mnt/code/example-data/ground_truth_judgments.yaml"
AGENTS_CONFIG_PATH = "/mnt/code/configs/agents.yaml"

# Baseline configuration: Known-good settings for OpenAI
# Temperatures tuned per agent type based on task characteristics
BASELINE_CONFIG = {
    "classifier": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.2,  # Low for consistency
    },
    "impact_assessor": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.3,  # Medium for reasoning exploration
    },
    "resource_matcher": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.1,  # Low for tool call precision
    },
    "response_drafter": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.6,  # Higher for natural language
    },
}

# Local model optimization grid
# Searches around baseline temperatures to find optimal local settings
LOCAL_GRID = {
    "classifier": {
        "temperature": [0.1, 0.2, 0.3],
    },
    "impact_assessor": {
        "temperature": [0.2, 0.3, 0.4],
    },
    "resource_matcher": {
        "temperature": [0.0, 0.1, 0.2],
    },
    "response_drafter": {
        "temperature": [0.5, 0.6, 0.7],
    },
}


# =============================================================================
# UTILITIES
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Optimize local model against OpenAI baseline.")
    parser.add_argument("--vertical", type=str, choices=VERTICALS, required=True)
    parser.add_argument("-n", "--max-tickets", type=int, default=5, help="Max tickets to process")
    parser.add_argument("--baseline-only", action="store_true", help="Run baseline only")
    parser.add_argument("--local-only", action="store_true", help="Run local optimization only (assumes baseline exists)")
    parser.add_argument("--dry-run", action="store_true", help="Show configuration without running")
    parser.add_argument("--submit-job", action="store_true", help="Submit best config as Domino Job")
    return parser.parse_args()


def load_ground_truth(vertical: str) -> List[dict]:
    """Load ground truth data for a vertical."""
    with open(GROUND_TRUTH_PATH) as f:
        data = yaml.safe_load(f)
    return data.get(vertical, [])


def load_agents_config() -> dict:
    """Load the agents configuration."""
    with open(AGENTS_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def ground_truth_to_incident(gt: dict) -> Incident:
    """Convert ground truth entry to Incident."""
    return Incident(
        ticket_id=gt["ticket_id"],
        description=gt["description"],
        source=IncidentSource.USER_REPORT,
    )


def get_client(provider: str):
    """Get LLM client for provider."""
    if provider == "openai":
        from openai import OpenAI
        return OpenAI()
    elif provider == "local":
        from local_model.domino_model_client import get_local_model_client
        return get_local_model_client()
    elif provider == "anthropic":
        from anthropic import Anthropic
        return Anthropic()
    raise ValueError(f"Unknown provider: {provider}")


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_classification(result: dict, ground_truth: dict) -> dict:
    """Evaluate classifier output against ground truth."""
    gt = ground_truth.get("evaluation_ground_truth", {})
    expected = ground_truth.get("expected_classification", {})

    category_match = (
        result.get("category", "").lower() == expected.get("category", "").lower()
    )
    urgency_match = (
        result.get("urgency") == expected.get("urgency") or
        abs(result.get("urgency", 0) - expected.get("urgency", 0)) <= 1  # Allow 1 level off
    )

    return {
        "category_correct": category_match,
        "urgency_correct": urgency_match,
        "score": 1.0 if (category_match and urgency_match) else (0.5 if category_match else 0.0),
    }


def evaluate_impact(result: dict, ground_truth: dict) -> dict:
    """Evaluate impact assessment output."""
    gt = ground_truth.get("evaluation_ground_truth", {})
    # Binary: is the assessment reasonable?
    quality = gt.get("impact_assessment_quality", 2)
    return {
        "quality_target": quality,
        "score": 1.0 if quality >= 2 else 0.0,
    }


def evaluate_resources(result: dict, ground_truth: dict) -> dict:
    """Evaluate resource assignment output."""
    gt = ground_truth.get("evaluation_ground_truth", {})
    correct = gt.get("resource_assignment_correct", False)
    return {
        "assignment_correct": correct,
        "score": 1.0 if correct else 0.0,
    }


def evaluate_response(result: dict, ground_truth: dict) -> dict:
    """Evaluate response draft output."""
    gt = ground_truth.get("evaluation_ground_truth", {})
    appropriateness = gt.get("response_appropriateness", 2)
    return {
        "appropriateness_target": appropriateness,
        "score": 1.0 if appropriateness >= 2 else 0.0,
    }


# =============================================================================
# PIPELINE EXECUTION
# =============================================================================

def run_agent_pipeline(
    incident: Incident,
    ground_truth: dict,
    agents_config: dict,
    agent_overrides: Dict[str, dict],
) -> dict:
    """Run the full agent pipeline with specified configuration overrides."""
    results = {}

    # Get clients based on provider overrides
    classifier_cfg = agent_overrides.get("classifier", {})
    provider = classifier_cfg.get("provider", "openai")
    model = classifier_cfg.get("model", "gpt-4o-mini")
    temperature = classifier_cfg.get("temperature", 0.2)

    client = get_client(provider)

    import time

    # Classifier
    start = time.time()
    try:
        classification = classify_incident(client, provider, model, incident, agents_config)
        results["classifier"] = {
            "success": True,
            "category": classification.category.value if hasattr(classification.category, 'value') else str(classification.category),
            "urgency": classification.urgency,
            "latency_ms": (time.time() - start) * 1000,
        }
        results["classifier"]["eval"] = evaluate_classification(results["classifier"], ground_truth)
    except Exception as e:
        results["classifier"] = {"success": False, "error": str(e), "latency_ms": (time.time() - start) * 1000}

    # Impact Assessor
    impact_cfg = agent_overrides.get("impact_assessor", {})
    start = time.time()
    try:
        impact = assess_impact(client, provider, model, incident, classification, agents_config)
        results["impact_assessor"] = {
            "success": True,
            "impact_score": impact.impact_score,
            "blast_radius": impact.blast_radius,
            "latency_ms": (time.time() - start) * 1000,
        }
        results["impact_assessor"]["eval"] = evaluate_impact(results["impact_assessor"], ground_truth)
    except Exception as e:
        results["impact_assessor"] = {"success": False, "error": str(e), "latency_ms": (time.time() - start) * 1000}

    # Resource Matcher
    resource_cfg = agent_overrides.get("resource_matcher", {})
    start = time.time()
    try:
        resources = match_resources(client, provider, model, classification, impact, agents_config)
        results["resource_matcher"] = {
            "success": True,
            "primary_responder": resources.primary_responder.name if resources.primary_responder else "unknown",
            "sla_hours": resources.sla_target_hours,
            "latency_ms": (time.time() - start) * 1000,
        }
        results["resource_matcher"]["eval"] = evaluate_resources(results["resource_matcher"], ground_truth)
    except Exception as e:
        results["resource_matcher"] = {"success": False, "error": str(e), "latency_ms": (time.time() - start) * 1000}

    # Response Drafter
    response_cfg = agent_overrides.get("response_drafter", {})
    start = time.time()
    try:
        response = draft_response(client, provider, model, incident, classification, impact, resources, agents_config)
        results["response_drafter"] = {
            "success": True,
            "num_communications": len(response.communications),
            "latency_ms": (time.time() - start) * 1000,
        }
        results["response_drafter"]["eval"] = evaluate_response(results["response_drafter"], ground_truth)
    except Exception as e:
        results["response_drafter"] = {"success": False, "error": str(e), "latency_ms": (time.time() - start) * 1000}

    return results


def aggregate_results(all_results: List[dict]) -> dict:
    """Aggregate results across multiple incidents."""
    agents = ["classifier", "impact_assessor", "resource_matcher", "response_drafter"]
    summary = {}

    for agent in agents:
        scores = []
        latencies = []
        successes = 0

        for result in all_results:
            agent_result = result.get(agent, {})
            if agent_result.get("success"):
                successes += 1
                latencies.append(agent_result.get("latency_ms", 0))
                if "eval" in agent_result:
                    scores.append(agent_result["eval"].get("score", 0))

        summary[agent] = {
            "success_rate": successes / len(all_results) if all_results else 0,
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "n": len(all_results),
        }

    return summary


# =============================================================================
# MAIN EXPERIMENT FUNCTIONS
# =============================================================================

def run_baseline(
    incidents: List[Incident],
    ground_truth_data: List[dict],
    agents_config: dict,
    experiment_name: str,
) -> dict:
    """Run baseline evaluation with OpenAI."""
    print("\n" + "=" * 70)
    print("BASELINE EVALUATION (OpenAI gpt-4o-mini)")
    print("=" * 70)

    all_results = []

    mlflow.set_experiment(experiment_name)

    with DominoRun(agent_config_path=AGENTS_CONFIG_PATH) as run:
        mlflow.set_tag("mlflow.runName", "BASELINE-openai-gpt-4o-mini")
        mlflow.set_tag("run_type", "baseline")
        mlflow.set_tag("provider", "openai")
        mlflow.set_tag("model", "gpt-4o-mini")

        # Log baseline config
        for agent, cfg in BASELINE_CONFIG.items():
            mlflow.log_param(f"{agent}_temperature", cfg["temperature"])

        for i, (incident, gt) in enumerate(zip(incidents, ground_truth_data)):
            print(f"  [{i+1}/{len(incidents)}] {incident.ticket_id}...", end=" ")
            result = run_agent_pipeline(incident, gt, agents_config, BASELINE_CONFIG)
            all_results.append(result)

            # Quick status
            scores = [r.get("eval", {}).get("score", 0) for r in result.values() if r.get("success")]
            avg = sum(scores) / len(scores) if scores else 0
            print(f"avg_score={avg:.2f}")

        summary = aggregate_results(all_results)

        # Log metrics
        for agent, metrics in summary.items():
            mlflow.log_metric(f"{agent}_score", metrics["avg_score"])
            mlflow.log_metric(f"{agent}_latency_ms", metrics["avg_latency_ms"])
            mlflow.log_metric(f"{agent}_success_rate", metrics["success_rate"])

    print("\nBaseline Results:")
    for agent, metrics in summary.items():
        print(f"  {agent}: score={metrics['avg_score']:.1%}, latency={metrics['avg_latency_ms']:.0f}ms")

    return summary


def run_local_optimization(
    incidents: List[Incident],
    ground_truth_data: List[dict],
    agents_config: dict,
    experiment_name: str,
) -> Tuple[dict, dict]:
    """Run grid search optimization for local model."""
    print("\n" + "=" * 70)
    print("LOCAL MODEL OPTIMIZATION (Qwen 2.5 3B)")
    print("=" * 70)

    best_configs = {}
    best_results = {}

    mlflow.set_experiment(experiment_name)

    for agent, grid in LOCAL_GRID.items():
        print(f"\n--- Optimizing {agent} ---")
        agent_best_score = -1
        agent_best_config = None
        agent_best_metrics = None

        temperatures = grid.get("temperature", [0.3])

        for temp in temperatures:
            config_name = f"local-t{temp}"
            print(f"  Testing {config_name}...", end=" ")

            # Build config for this run
            run_config = {
                agent: {
                    "provider": "local",
                    "model": "qwen2.5-3b-instruct",
                    "temperature": temp,
                }
            }
            # Use baseline for other agents
            for other_agent in BASELINE_CONFIG:
                if other_agent != agent:
                    run_config[other_agent] = BASELINE_CONFIG[other_agent]

            with DominoRun(agent_config_path=AGENTS_CONFIG_PATH) as run:
                mlflow.set_tag("mlflow.runName", f"{agent}-{config_name}")
                mlflow.set_tag("run_type", "local_optimization")
                mlflow.set_tag("agent", agent)
                mlflow.set_tag("provider", "local")
                mlflow.log_param("temperature", temp)

                all_results = []
                for incident, gt in zip(incidents, ground_truth_data):
                    result = run_agent_pipeline(incident, gt, agents_config, run_config)
                    all_results.append(result)

                summary = aggregate_results(all_results)
                agent_metrics = summary[agent]

                mlflow.log_metric("score", agent_metrics["avg_score"])
                mlflow.log_metric("latency_ms", agent_metrics["avg_latency_ms"])
                mlflow.log_metric("success_rate", agent_metrics["success_rate"])

                print(f"score={agent_metrics['avg_score']:.1%}, latency={agent_metrics['avg_latency_ms']:.0f}ms")

                if agent_metrics["avg_score"] > agent_best_score:
                    agent_best_score = agent_metrics["avg_score"]
                    agent_best_config = {"temperature": temp}
                    agent_best_metrics = agent_metrics

        best_configs[agent] = agent_best_config
        best_results[agent] = agent_best_metrics
        print(f"  Best: temperature={agent_best_config['temperature']}, score={agent_best_score:.1%}")

    return best_configs, best_results


def generate_comparison_report(
    baseline: dict,
    local_best_configs: dict,
    local_best_results: dict,
    vertical: str,
) -> str:
    """Generate comparison report with recommendations."""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("OPTIMIZATION RESULTS")
    lines.append("=" * 70)
    lines.append(f"Vertical: {vertical}")
    lines.append("")

    # Comparison table
    lines.append(f"{'Agent':<20} {'Baseline':<15} {'Local Best':<15} {'Delta':<10} {'Speedup':<10}")
    lines.append("-" * 70)

    recommendations = {"local": [], "api": []}

    for agent in ["classifier", "impact_assessor", "resource_matcher", "response_drafter"]:
        base = baseline.get(agent, {})
        local = local_best_results.get(agent, {})

        base_score = base.get("avg_score", 0)
        local_score = local.get("avg_score", 0)
        delta = local_score - base_score

        base_latency = base.get("avg_latency_ms", 1)
        local_latency = local.get("avg_latency_ms", 1)
        speedup = base_latency / local_latency if local_latency > 0 else 0

        delta_str = f"{delta:+.1%}"
        speedup_str = f"{speedup:.1f}x"

        lines.append(f"{agent:<20} {base_score:.1%}{'':>8} {local_score:.1%}{'':>8} {delta_str:<10} {speedup_str:<10}")

        # Recommendation logic
        # Accept local if: within 10% of baseline AND at least 2x faster
        if delta >= -0.10 and speedup >= 2.0:
            recommendations["local"].append(agent)
        else:
            recommendations["api"].append(agent)

    lines.append("")
    lines.append("-" * 70)
    lines.append("RECOMMENDATIONS")
    lines.append("-" * 70)

    if recommendations["local"]:
        lines.append(f"Use LOCAL model for: {', '.join(recommendations['local'])}")
        lines.append("  (Acceptable quality with significant latency improvement)")

    if recommendations["api"]:
        lines.append(f"Use API model for:   {', '.join(recommendations['api'])}")
        lines.append("  (Quality gap too large or insufficient speedup)")

    lines.append("")
    lines.append("Best Local Configurations:")
    for agent, cfg in local_best_configs.items():
        lines.append(f"  {agent}: temperature={cfg.get('temperature', 'N/A')}")

    return "\n".join(lines)


def save_best_config(
    baseline: dict,
    local_configs: dict,
    local_results: dict,
    vertical: str,
    batch_id: str,
) -> str:
    """Save the best configuration to agents.yaml."""
    agents_path = "/mnt/code/configs/agents.yaml"

    with open(agents_path) as f:
        config = yaml.safe_load(f)

    if "optimized_agents" not in config:
        config["optimized_agents"] = {}

    if vertical not in config["optimized_agents"]:
        config["optimized_agents"][vertical] = {}

    for agent in ["classifier", "impact_assessor", "resource_matcher", "response_drafter"]:
        base = baseline.get(agent, {})
        local = local_results.get(agent, {})
        local_cfg = local_configs.get(agent, {})

        base_score = base.get("avg_score", 0)
        local_score = local.get("avg_score", 0)
        delta = local_score - base_score

        base_latency = base.get("avg_latency_ms", 1)
        local_latency = local.get("avg_latency_ms", 1)
        speedup = base_latency / local_latency if local_latency > 0 else 0

        # Recommendation: use local if within 10% and 2x faster
        use_local = delta >= -0.10 and speedup >= 2.0

        config["optimized_agents"][vertical][agent] = {
            "recommended_provider": "local" if use_local else "openai",
            "local_config": {
                "model": "qwen2.5-3b-instruct",
                "temperature": local_cfg.get("temperature", 0.3),
            },
            "baseline_config": BASELINE_CONFIG[agent],
            "metrics": {
                "baseline_score": round(base_score, 3),
                "local_score": round(local_score, 3),
                "delta": round(delta, 3),
                "speedup": round(speedup, 2),
            },
            "batch_id": batch_id,
            "optimized_at": datetime.now().isoformat(),
        }

    with open(agents_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Saved optimized config to {agents_path}")
    return agents_path


def submit_optimization_job(
    vertical: str,
    batch_id: str,
    local_configs: dict,
) -> Optional[str]:
    """Submit a Domino Job to register the best agent configurations."""
    try:
        from domino import Domino

        project_owner = os.environ.get("DOMINO_PROJECT_OWNER", "")
        project_name = os.environ.get("DOMINO_PROJECT_NAME", "")

        if not project_owner or not project_name:
            print("WARNING: DOMINO_PROJECT_OWNER or DOMINO_PROJECT_NAME not set. Skipping job submission.")
            return None

        project = f"{project_owner}/{project_name}"
        domino = Domino(project)

        job_title = f"BEST-AGENTS-{vertical}-{batch_id}"

        # Build config summary for the job
        config_summary = json.dumps(local_configs)

        command = f'''python -c "
import mlflow
import yaml
import os
from datetime import datetime
from domino.agents.logging import DominoRun

vertical = '{vertical}'
batch_id = '{batch_id}'

# Load optimized config from agents.yaml
with open('/mnt/code/configs/agents.yaml') as f:
    config = yaml.safe_load(f)

optimized = config.get('optimized_agents', {{}}).get(vertical, {{}})

# Set experiment
experiment_name = 'agent-optimization-' + vertical + '-' + os.environ.get('DOMINO_USER_NAME', 'job')
mlflow.set_experiment(experiment_name)

with DominoRun(agent_config_path='/mnt/code/configs/agents.yaml') as run:
    mlflow.set_tag('mlflow.runName', 'FINAL-BEST-AGENTS-{vertical}')
    mlflow.set_tag('batch_id', batch_id)
    mlflow.set_tag('best_parameters', 'true')
    mlflow.set_tag('registered_job', 'true')
    mlflow.set_tag('vertical', vertical)

    for agent, agent_cfg in optimized.items():
        mlflow.log_param(f'{{agent}}_provider', agent_cfg.get('recommended_provider'))
        local_cfg = agent_cfg.get('local_config', {{}})
        mlflow.log_param(f'{{agent}}_temperature', local_cfg.get('temperature'))
        metrics = agent_cfg.get('metrics', {{}})
        for k, v in metrics.items():
            mlflow.log_metric(f'{{agent}}_{{k}}', v)

    print(f'Registered FINAL-BEST-AGENTS-{vertical} with batch_id: {{batch_id}}')
"'''

        result = domino.job_start(command=command, title=job_title)

        job_id = None
        if isinstance(result, dict):
            job_id = result.get("id") or result.get("jobId") or result.get("runId")

        if job_id:
            domino_host = os.environ.get("DOMINO_USER_HOST", os.environ.get("DOMINO_API_HOST", ""))
            for suffix in ["/v4/api", "/api/api", "/api", "/v4"]:
                if domino_host.endswith(suffix):
                    domino_host = domino_host[:-len(suffix)]
                    break
            domino_host = domino_host.rstrip("/")

            job_url = f"{domino_host}/jobs/{project_owner}/{project_name}/{job_id}" if domino_host else ""
            print(f"Submitted Domino Job: {job_title}")
            print(f"Job ID: {job_id}")
            if job_url:
                print(f"Job URL: {job_url}")
            return job_id

        print(f"Job submitted but no ID returned: {result}")
        return None

    except ImportError:
        print("WARNING: python-domino not installed. Skipping job submission.")
        return None
    except Exception as e:
        print(f"WARNING: Failed to submit Domino Job: {e}")
        return None


def create_final_mlflow_run(
    baseline: dict,
    local_configs: dict,
    local_results: dict,
    vertical: str,
    batch_id: str,
    experiment_name: str,
):
    """Create a combined MLflow run with all best configurations."""
    print("\n" + "=" * 70)
    print("CREATING FINAL BEST-AGENTS RUN")
    print("=" * 70)

    mlflow.set_experiment(experiment_name)

    with DominoRun(agent_config_path=AGENTS_CONFIG_PATH) as run:
        mlflow.set_tag("mlflow.runName", f"BEST-AGENTS-{vertical}")
        mlflow.set_tag("batch_id", batch_id)
        mlflow.set_tag("best_parameters", "true")
        mlflow.set_tag("vertical", vertical)

        for agent in ["classifier", "impact_assessor", "resource_matcher", "response_drafter"]:
            base = baseline.get(agent, {})
            local = local_results.get(agent, {})
            local_cfg = local_configs.get(agent, {})

            base_score = base.get("avg_score", 0)
            local_score = local.get("avg_score", 0)
            delta = local_score - base_score
            base_latency = base.get("avg_latency_ms", 1)
            local_latency = local.get("avg_latency_ms", 1)
            speedup = base_latency / local_latency if local_latency > 0 else 0

            use_local = delta >= -0.10 and speedup >= 2.0

            mlflow.log_param(f"{agent}_recommended", "local" if use_local else "openai")
            mlflow.log_param(f"{agent}_temperature", local_cfg.get("temperature", 0.3))
            mlflow.log_metric(f"{agent}_baseline_score", base_score)
            mlflow.log_metric(f"{agent}_local_score", local_score)
            mlflow.log_metric(f"{agent}_delta", delta)
            mlflow.log_metric(f"{agent}_speedup", speedup)

        print(f"Created run: BEST-AGENTS-{vertical}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    # Load data
    ground_truth_data = load_ground_truth(args.vertical)
    if not ground_truth_data:
        print(f"ERROR: No ground truth found for vertical '{args.vertical}'")
        sys.exit(1)

    ground_truth_data = ground_truth_data[:args.max_tickets]
    incidents = [ground_truth_to_incident(gt) for gt in ground_truth_data]
    agents_config = load_agents_config()

    # Experiment setup
    username = os.environ.get("DOMINO_USER_NAME", os.environ.get("USER", "user"))
    experiment_name = f"agent-optimization-{args.vertical}-{username}"

    print("=" * 70)
    print("LOCAL MODEL OPTIMIZATION EXPERIMENT")
    print("=" * 70)
    print(f"Vertical: {args.vertical}")
    print(f"Incidents: {len(incidents)}")
    print(f"Experiment: {experiment_name}")

    if args.dry_run:
        print("\n[DRY RUN] Would run:")
        print(f"  Baseline: {BASELINE_CONFIG}")
        print(f"  Local grid: {LOCAL_GRID}")
        return

    baseline_results = None
    local_configs = None
    local_results = None

    # Run baseline
    if not args.local_only:
        baseline_results = run_baseline(incidents, ground_truth_data, agents_config, experiment_name)

    # Run local optimization
    if not args.baseline_only:
        local_configs, local_results = run_local_optimization(
            incidents, ground_truth_data, agents_config, experiment_name
        )

    # Generate batch ID
    batch_id = f"{username}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Generate report if we have both
    if baseline_results and local_results:
        report = generate_comparison_report(
            baseline_results, local_configs, local_results, args.vertical
        )
        print(report)

        # Save report
        report_path = f"/mnt/code/results/local_optimization_{args.vertical}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")

        # Save best configuration to agents.yaml
        save_best_config(baseline_results, local_configs, local_results, args.vertical, batch_id)

        # Create final MLflow run with all best configs
        create_final_mlflow_run(
            baseline_results, local_configs, local_results,
            args.vertical, batch_id, experiment_name
        )

        # Submit Domino Job if requested
        if args.submit_job:
            print("\n" + "=" * 70)
            print("SUBMITTING DOMINO JOB")
            print("=" * 70)
            submit_optimization_job(args.vertical, batch_id, local_configs)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Batch ID: {batch_id}")
    print(f"Vertical: {args.vertical}")
    print(f"Experiment: {experiment_name}")
    if local_configs:
        print("\nBest local configurations:")
        for agent, cfg in local_configs.items():
            print(f"  {agent}: temperature={cfg.get('temperature', 'N/A')}")


if __name__ == "__main__":
    main()
