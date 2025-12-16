#!/usr/bin/env python3
"""
TriageFlow Judge Experiment Runner

Optimizes judge configurations via grid search over models, temperatures,
prompt styles, and scales. Uses ground truth labels from YAML for validation.

Experiment name: judge-optimization-{username}
Tag: model type only (e.g., gpt-4o-mini)

Usage:
    python run_judge_experiment.py --vertical financial_services --dry-run
    python run_judge_experiment.py --vertical healthcare --model gpt-4o-mini
    python run_judge_experiment.py --vertical energy --runs-per-config 2
"""

import argparse
import itertools
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, "/mnt/code")

import mlflow
import yaml

from domino.agents.tracing import add_tracing
from domino.agents.logging import DominoRun

from src.models import Incident, IncidentSource
from src.agents import classify_incident


# =============================================================================
# CONSTANTS
# =============================================================================

VERTICALS = ["financial_services", "healthcare", "energy", "public_sector"]
GROUND_TRUTH_PATH = "/mnt/code/example-data/ground_truth_judgments.yaml"


# =============================================================================
# JUDGE PROMPT TEMPLATES
# =============================================================================

JUDGE_PROMPTS = {
    ("direct", "binary"): """Evaluate this classification. Is it correct?

Incident: {incident}
Classification: {category} (Urgency: {urgency})
Reasoning: {reasoning}

Return JSON: {{"score": 0 or 1, "rationale": "brief explanation"}}""",

    ("direct", "three_point"): """Evaluate this classification quality (1-3).

Incident: {incident}
Classification: {category} (Urgency: {urgency})
Reasoning: {reasoning}

Score: 1=poor, 2=acceptable, 3=good
Return JSON: {{"score": 1-3, "rationale": "brief explanation"}}""",

    ("cot", "binary"): """Think step by step about this classification.

Incident: {incident}
Classification: {category} (Urgency: {urgency})
Reasoning: {reasoning}

Consider: Is the category appropriate? Is urgency justified? Is reasoning sound?
First explain your analysis, then decide: pass (1) or fail (0).
Return JSON: {{"score": 0 or 1, "analysis": "step-by-step", "rationale": "summary"}}""",

    ("cot", "three_point"): """Think step by step about this classification.

Incident: {incident}
Classification: {category} (Urgency: {urgency})
Reasoning: {reasoning}

Consider: Is the category appropriate? Is urgency justified? Is reasoning sound?
First explain your analysis, then score 1-3 (1=poor, 2=acceptable, 3=good).
Return JSON: {{"score": 1-3, "analysis": "step-by-step", "rationale": "summary"}}""",

    ("rubric", "binary"): """Evaluate using this rubric:
- PASS (1): Category matches incident, urgency justified, reasoning references specifics
- FAIL (0): Category mismatch, urgency unjustified, or vague reasoning

Incident: {incident}
Classification: {category} (Urgency: {urgency})
Reasoning: {reasoning}

Return JSON: {{"score": 0 or 1, "rubric_check": "pass/fail criteria", "rationale": "summary"}}""",

    ("rubric", "three_point"): """Evaluate using this rubric:
- GOOD (3): Perfect category, well-justified urgency, specific reasoning
- ACCEPTABLE (2): Correct category, reasonable urgency, adequate reasoning
- POOR (1): Wrong category, off urgency, or weak reasoning

Incident: {incident}
Classification: {category} (Urgency: {urgency})
Reasoning: {reasoning}

Return JSON: {{"score": 1-3, "rubric_check": "criteria assessment", "rationale": "summary"}}""",
}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Run judge optimization experiment.")
    parser.add_argument(
        "--vertical",
        type=str,
        choices=VERTICALS,
        required=True,
        help="Industry vertical to evaluate"
    )
    parser.add_argument("--model", type=str, help="Single model to test (default: all)")
    parser.add_argument("--config", type=str, help="Config path (default: configs/judge_experiment_grid.yaml)")
    parser.add_argument("--runs-per-config", type=int, help="Runs per config for consistency")
    parser.add_argument("--dry-run", action="store_true", help="Show configs without running")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_ground_truth(vertical: str) -> List[dict]:
    """Load ground truth data for a specific vertical."""
    with open(GROUND_TRUTH_PATH) as f:
        data = yaml.safe_load(f)
    return data.get(vertical, [])


def ground_truth_to_incident(gt: dict) -> Incident:
    """Convert ground truth entry to Incident object."""
    return Incident(
        ticket_id=gt["ticket_id"],
        description=gt["description"],
        source=IncidentSource("monitoring"),
        reporter=None,
        affected_system=None,
        initial_severity=None
    )


def get_client(provider: str):
    """Initialize LLM client with autolog enabled."""
    if provider == "openai":
        from openai import OpenAI
        mlflow.openai.autolog()
        return OpenAI()
    elif provider == "anthropic":
        from anthropic import Anthropic
        mlflow.anthropic.autolog()
        return Anthropic()
    raise ValueError(f"Unknown provider: {provider}")


def call_judge(client, provider: str, model: str, prompt: str, temperature: float) -> Tuple[dict, float]:
    """Call judge LLM and return (parsed_result, latency_ms)."""
    start = time.time()
    try:
        if provider == "openai":
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=500
            )
            content = resp.choices[0].message.content
        else:
            resp = client.messages.create(
                model=model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            content = resp.content[0].text

        latency = (time.time() - start) * 1000
        match = re.search(r'\{[\s\S]*\}', content)
        if match:
            result = json.loads(match.group(0))
            result["_valid"] = True
            return result, latency
    except Exception as e:
        latency = (time.time() - start) * 1000
        return {"_valid": False, "_error": str(e)}, latency

    return {"_valid": False}, latency


def judge_evaluator(span) -> dict:
    """Extract metrics from judge evaluation outputs."""
    outputs = span.outputs or {}
    if not hasattr(outputs, "get"):
        return {}
    return {
        "json_valid": 1.0 if outputs.get("json_valid", False) else 0.0,
        "latency_ms": outputs.get("latency_ms", 0),
        "score": outputs.get("score"),
        "human_match": outputs.get("human_match"),
    }


def compute_human_agreement(judge_score: int, ground_truth: dict, scale: str) -> Optional[float]:
    """
    Compare judge score to ground truth.
    For binary: check if judge agrees with category_correct AND urgency_correct.
    For three_point: compare to reasoning_quality (1-3 scale).
    """
    if judge_score is None:
        return None

    gt = ground_truth.get("evaluation_ground_truth", {})

    if scale == "binary":
        # Binary: 1 = pass (correct), 0 = fail (incorrect)
        # Ground truth has category_correct and urgency_correct as booleans
        human_pass = gt.get("category_correct", False) and gt.get("urgency_correct", False)
        judge_pass = judge_score == 1
        return 1.0 if human_pass == judge_pass else 0.0
    else:
        # Three-point: compare to reasoning_quality (1-3)
        human_score = gt.get("reasoning_quality", 2)
        # Exact match = 1.0, off by 1 = 0.5, off by 2 = 0.0
        diff = abs(judge_score - human_score)
        if diff == 0:
            return 1.0
        elif diff == 1:
            return 0.5
        else:
            return 0.0


def run_judge_config(
    model_key: str,
    temperature: float,
    prompt_style: str,
    scale: str,
    ground_truth_data: List[dict],
    agent_outputs: List[dict],
    models_config: dict,
    agents_config: dict,
    runs_per_config: int,
) -> dict:
    """Run a single judge configuration and return metrics."""

    # Get provider and model name
    model_info = models_config.get(model_key, {})
    provider = model_info.get("provider", "openai") if isinstance(model_info, dict) else "openai"
    model_name = model_info.get("name", model_key) if isinstance(model_info, dict) else model_key

    client = get_client(provider)
    prompt_template = JUDGE_PROMPTS.get((prompt_style, scale), JUDGE_PROMPTS[("direct", "three_point")])

    all_scores = []
    all_latencies = []
    json_successes = 0
    human_agreements = []
    total_calls = 0

    # Create traced evaluation function
    @add_tracing(name="judge_evaluation", autolog_frameworks=[provider], evaluator=judge_evaluator)
    def evaluate_classification(gt: dict, agent_output: dict) -> dict:
        prompt = prompt_template.format(
            incident=gt["description"],
            category=agent_output.get("category", "unknown"),
            urgency=agent_output.get("urgency", 3),
            reasoning=agent_output.get("reasoning", ""),
        )
        result, latency = call_judge(client, provider, model_name, prompt, temperature)

        output = {
            "json_valid": result.get("_valid", False),
            "latency_ms": latency,
            "score": result.get("score"),
        }

        # Compute human agreement
        if output["score"] is not None:
            agreement = compute_human_agreement(output["score"], gt, scale)
            output["human_match"] = agreement

        return output

    # Run evaluations
    for gt, agent_output in zip(ground_truth_data, agent_outputs):
        for _ in range(runs_per_config):
            result = evaluate_classification(gt, agent_output)
            total_calls += 1
            all_latencies.append(result["latency_ms"])

            if result["json_valid"]:
                json_successes += 1
                if result["score"] is not None:
                    all_scores.append(float(result["score"]))

            if result.get("human_match") is not None:
                human_agreements.append(result["human_match"])

    # Compute metrics
    metrics = {
        "total_calls": total_calls,
        "json_parse_rate": json_successes / total_calls if total_calls else 0,
        "avg_latency_ms": sum(all_latencies) / len(all_latencies) if all_latencies else 0,
    }

    # Consistency metrics
    if all_scores:
        mean = sum(all_scores) / len(all_scores)
        variance = sum((s - mean) ** 2 for s in all_scores) / len(all_scores)
        metrics["consistency_std"] = variance ** 0.5
        metrics["score_mean"] = mean

    # Human agreement
    if human_agreements:
        metrics["human_agreement"] = sum(human_agreements) / len(human_agreements)

    return metrics


def save_best_config(best_result: dict, vertical: str, batch_id: str):
    """Save the best judge configuration to judges.yaml."""
    judges_path = "/mnt/code/configs/judges.yaml"

    with open(judges_path) as f:
        judges_config = yaml.safe_load(f)

    # Extract parameters from config name (model-tX.X-style-scale)
    # Format: gpt-4o-mini-t0.0-direct-binary
    config_name = best_result["config"]
    metrics = best_result["metrics"]

    # Parse config name - find temperature marker to split model from params
    t_index = config_name.find("-t0.") if "-t0." in config_name else config_name.find("-t0")
    if t_index == -1:
        t_index = config_name.find("-t")  # fallback

    model = config_name[:t_index]
    remainder = config_name[t_index+1:]  # e.g., "t0.0-direct-binary"

    parts = remainder.split("-")
    temp = float(parts[0].replace("t", ""))
    prompt_style = parts[1]
    scale = parts[2]

    # Add or update optimized_judge section
    if "optimized_judge" not in judges_config:
        judges_config["optimized_judge"] = {}

    judges_config["optimized_judge"][vertical] = {
        "model": model,
        "temperature": temp,
        "prompt_style": prompt_style,
        "scale": scale,
        "validated_metrics": {
            "human_agreement": round(metrics["human_agreement"], 3),
            "consistency_std": round(metrics["consistency_std"], 3),
            "json_parse_rate": round(metrics["json_parse_rate"], 3),
            "avg_latency_ms": round(metrics["avg_latency_ms"], 1),
        },
        "batch_id": batch_id,
        "validated_at": datetime.now().isoformat(),
    }

    with open(judges_path, "w") as f:
        yaml.dump(judges_config, f, default_flow_style=False, sort_keys=False)

    print(f"Saved best config to {judges_path}")


def run_best_config_validation(
    best_result: dict,
    ground_truth_data: List[dict],
    agent_outputs: List[dict],
    models_config: dict,
    agents_config: dict,
    vertical: str,
    batch_id: str,
    experiment_name: str,
):
    """Run a final validation with the best config and tag as best_parameters."""
    config_name = best_result["config"]

    # Parse config name - find temperature marker to split model from params
    t_index = config_name.find("-t0.") if "-t0." in config_name else config_name.find("-t0")
    if t_index == -1:
        t_index = config_name.find("-t")

    model = config_name[:t_index]
    remainder = config_name[t_index+1:]
    parts = remainder.split("-")
    temp = float(parts[0].replace("t", ""))
    prompt_style = parts[1]
    scale = parts[2]

    print("\n" + "=" * 70)
    print("REGISTERING BEST CONFIGURATION")
    print("=" * 70)
    print(f"Running validation with: {config_name}")

    agents_path = "/mnt/code/configs/agents.yaml"

    with DominoRun(agent_config_path=agents_path) as run:
        mlflow.set_tag("mlflow.runName", f"BEST-{vertical}-{config_name}")
        mlflow.set_tag("model", model)
        mlflow.set_tag("batch_id", batch_id)
        mlflow.set_tag("best_parameters", "true")  # Key tag for identifying best config
        mlflow.set_tag("vertical", vertical)

        mlflow.log_param("judge_model", model)
        mlflow.log_param("temperature", temp)
        mlflow.log_param("prompt_style", prompt_style)
        mlflow.log_param("scale", scale)
        mlflow.log_param("vertical", vertical)
        mlflow.log_param("is_best", True)

        # Run the config one more time for validation
        metrics = run_judge_config(
            model_key=model,
            temperature=temp,
            prompt_style=prompt_style,
            scale=scale,
            ground_truth_data=ground_truth_data,
            agent_outputs=agent_outputs,
            models_config=models_config,
            agents_config=agents_config,
            runs_per_config=3,
        )

        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)

        print(f"  Validation complete: agreement={metrics.get('human_agreement', 0):.1%}")

    return metrics


def main():
    args = parse_args()
    project_root = "/mnt/code"

    # Load configs
    config_path = args.config or f"{project_root}/configs/judge_experiment_grid.yaml"
    agents_path = f"{project_root}/configs/agents.yaml"

    config = load_config(config_path)
    agents_config = load_config(agents_path)

    exp_config = config.get("judge_experiment", {})
    models_config = config.get("models", {})
    grid = exp_config.get("grid", {})

    # Parameters
    models = [args.model] if args.model else grid.get("model", ["gpt-4o-mini"])
    temperatures = grid.get("temperature", [0.0, 0.1, 0.3])
    prompt_styles = grid.get("prompt_style", ["direct", "cot", "rubric"])
    scales = grid.get("scale", ["binary", "three_point"])
    runs_per_config = args.runs_per_config or exp_config.get("consistency", {}).get("runs_per_config", 3)

    # Load ground truth for vertical
    ground_truth_data = load_ground_truth(args.vertical)
    if not ground_truth_data:
        print(f"ERROR: No ground truth found for vertical '{args.vertical}'")
        sys.exit(1)

    # Convert to incidents
    incidents = [ground_truth_to_incident(gt) for gt in ground_truth_data]

    # Build parameter grid
    param_grid = list(itertools.product(models, temperatures, prompt_styles, scales))

    # Experiment setup with batch ID
    username = os.environ.get("DOMINO_USER_NAME", os.environ.get("USER", "demo_user"))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_id = f"{username}-{timestamp}"
    experiment_name = f"judge-optimization-{username}"

    print("=" * 70)
    print("JUDGE OPTIMIZATION EXPERIMENT")
    print("=" * 70)
    print(f"Experiment: {experiment_name}")
    print(f"Batch ID: {batch_id}")
    print(f"Vertical: {args.vertical}")
    print(f"Incidents: {len(ground_truth_data)} (with ground truth)")
    print(f"Configurations: {len(param_grid)}")
    print(f"Runs/config: {runs_per_config}")
    print()

    if args.dry_run:
        print("[DRY RUN] Configurations:")
        for i, (m, t, p, s) in enumerate(param_grid[:10], 1):
            print(f"  {i}. {m}, temp={t}, {p}, {s}")
        if len(param_grid) > 10:
            print(f"  ... and {len(param_grid) - 10} more")
        return

    # Generate agent outputs using baseline model
    print("Generating agent outputs...")
    baseline_client = get_client("openai")
    agent_outputs = []
    for incident in incidents:
        try:
            classification = classify_incident(baseline_client, "openai", "gpt-4o-mini", incident, agents_config)
            agent_outputs.append({
                "category": classification.category.value if hasattr(classification.category, 'value') else str(classification.category),
                "urgency": classification.urgency,
                "reasoning": classification.reasoning,
            })
        except Exception as e:
            agent_outputs.append({"category": "unknown", "urgency": 3, "reasoning": f"Error: {e}"})
    print(f"Generated {len(agent_outputs)} outputs\n")

    # Run experiments
    mlflow.set_experiment(experiment_name)
    results = []

    for i, (model, temp, style, scale) in enumerate(param_grid, 1):
        config_name = f"{model}-t{temp}-{style}-{scale}"
        print(f"[{i}/{len(param_grid)}] {config_name}")

        with DominoRun(agent_config_path=agents_path) as run:
            mlflow.set_tag("mlflow.runName", config_name)
            mlflow.set_tag("model", model)  # Only tag = model type
            mlflow.set_tag("batch_id", batch_id)

            mlflow.log_param("judge_model", model)
            mlflow.log_param("temperature", temp)
            mlflow.log_param("prompt_style", style)
            mlflow.log_param("scale", scale)
            mlflow.log_param("vertical", args.vertical)

            try:
                metrics = run_judge_config(
                    model_key=model,
                    temperature=temp,
                    prompt_style=style,
                    scale=scale,
                    ground_truth_data=ground_truth_data,
                    agent_outputs=agent_outputs,
                    models_config=models_config,
                    agents_config=agents_config,
                    runs_per_config=runs_per_config,
                )

                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)

                results.append({"config": config_name, "model": model, "metrics": metrics})

                agreement = metrics.get("human_agreement", "N/A")
                std = metrics.get("consistency_std", "N/A")
                print(f"  agreement={agreement:.1%}" if isinstance(agreement, float) else f"  agreement={agreement}", end="")
                print(f", std={std:.3f}" if isinstance(std, float) else f", std={std}")

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({"config": config_name, "model": model, "metrics": {"error": str(e)}})

    # Summary with multi-objective filtering
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Thresholds for filtering
    MIN_JSON_PARSE_RATE = 0.95
    MAX_CONSISTENCY_STD = 0.35  # Slightly above 0.3 to handle floating point

    # Filter valid results
    all_valid = [r for r in results if isinstance(r["metrics"].get("human_agreement"), float)]

    # Apply thresholds
    qualified = [
        r for r in all_valid
        if r["metrics"].get("json_parse_rate", 0) >= MIN_JSON_PARSE_RATE
        and r["metrics"].get("consistency_std", 1.0) <= MAX_CONSISTENCY_STD
    ]

    print(f"Total configurations: {len(results)}")
    print(f"With valid metrics: {len(all_valid)}")
    print(f"Meeting thresholds (json>={MIN_JSON_PARSE_RATE:.0%}, std<={MAX_CONSISTENCY_STD}): {len(qualified)}")
    print()

    if qualified:
        print("TOP QUALIFIED CONFIGURATIONS (by human agreement):")
        print("-" * 70)
        ranked = sorted(qualified, key=lambda x: -x["metrics"]["human_agreement"])
        for i, r in enumerate(ranked[:10], 1):
            m = r["metrics"]
            print(f"  {i}. {r['config']}")
            print(f"     agreement={m['human_agreement']:.1%}, std={m['consistency_std']:.3f}, json={m['json_parse_rate']:.0%}")

        # Log best config
        best = ranked[0]
        print()
        print("=" * 70)
        print("BEST CONFIGURATION")
        print("=" * 70)
        print(f"  {best['config']}")
        print(f"  Human agreement: {best['metrics']['human_agreement']:.1%}")
        print(f"  Consistency std: {best['metrics']['consistency_std']:.3f}")
        print(f"  JSON parse rate: {best['metrics']['json_parse_rate']:.0%}")
        print(f"  Avg latency: {best['metrics']['avg_latency_ms']:.0f}ms")

        # Save best config to judges.yaml
        save_best_config(best, args.vertical, batch_id)

        # Run final validation with best_parameters tag
        run_best_config_validation(
            best_result=best,
            ground_truth_data=ground_truth_data,
            agent_outputs=agent_outputs,
            models_config=models_config,
            agents_config=agents_config,
            vertical=args.vertical,
            batch_id=batch_id,
            experiment_name=experiment_name,
        )
    else:
        print("WARNING: No configurations met the quality thresholds.")
        print("Consider relaxing thresholds or reviewing judge prompts.")
        print()
        print("Top by human agreement (unfiltered):")
        for r in sorted(all_valid, key=lambda x: -x["metrics"]["human_agreement"])[:5]:
            m = r["metrics"]
            print(f"  {r['config']}: agreement={m['human_agreement']:.1%}, std={m.get('consistency_std', 0):.3f}, json={m.get('json_parse_rate', 0):.0%}")

    print(f"\nExperiment complete: {experiment_name}")
    print(f"Batch ID: {batch_id}")


if __name__ == "__main__":
    main()
