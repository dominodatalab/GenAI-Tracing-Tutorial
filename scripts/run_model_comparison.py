#!/usr/bin/env python3
"""
TriageFlow Model Comparison Experiment

Compares frontier models (OpenAI, Anthropic) against local Qwen model
with few-shot learning from frontier outputs.

Each model gets its own DominoRun with full tracing and metrics logging.

Workflow:
1. Run frontier models on sample incidents to generate high-quality outputs
2. Use those outputs as few-shot examples for the local Qwen model
3. Run all three models on test incidents (each as separate DominoRun)
4. Evaluate with LLM judges and compare results

Usage:
    python run_model_comparison.py
    python run_model_comparison.py --few-shot-count 3 --test-incidents 10
    python run_model_comparison.py --skip-frontier  # Use cached examples
"""

import argparse
import copy
import io
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

sys.path.insert(0, "/mnt/code")

import mlflow
import pandas as pd
import requests
import yaml

from domino.agents.tracing import add_tracing
from domino.agents.logging import DominoRun

from src.models import (
    Incident, IncidentSource, Classification, ImpactAssessment,
    ResourceAssignment, ResponsePlan
)
from src.agents import (
    classify_incident, assess_impact, match_resources, draft_response,
    call_with_tools, parse_json_response, get_all_tools
)
from src.judges import judge_classification, judge_response, judge_triage


# =============================================================================
# CONSTANTS
# =============================================================================

CONFIG_PATH = "/mnt/code/configs/agents.yaml"
LOCAL_ENDPOINT = "https://genai-llm.domino-eval.com/endpoints/bf209962-1bd0-4524-87c8-2d0ac662a022/v1"

# Model-specific config files
MODEL_CONFIGS = {
    "openai": "/mnt/code/configs/model_openai.yaml",
    "anthropic": "/mnt/code/configs/model_anthropic.yaml",
    "local": "/mnt/code/configs/model_local.yaml",
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FewShotExample:
    """A single few-shot example for one agent."""
    agent: str
    input_text: str
    output_json: dict
    source_model: str
    ticket_id: str


@dataclass
class FewShotBank:
    """Collection of few-shot examples by agent."""
    classifier: List[FewShotExample] = field(default_factory=list)
    impact_assessor: List[FewShotExample] = field(default_factory=list)
    resource_matcher: List[FewShotExample] = field(default_factory=list)
    response_drafter: List[FewShotExample] = field(default_factory=list)

    def add(self, example: FewShotExample):
        getattr(self, example.agent).append(example)

    def get_examples(self, agent: str, count: int) -> List[FewShotExample]:
        examples = getattr(self, agent, [])
        return examples[:count]

    def to_dict(self) -> dict:
        return {
            "classifier": [asdict(e) for e in self.classifier],
            "impact_assessor": [asdict(e) for e in self.impact_assessor],
            "resource_matcher": [asdict(e) for e in self.resource_matcher],
            "response_drafter": [asdict(e) for e in self.response_drafter],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FewShotBank":
        bank = cls()
        for agent in ["classifier", "impact_assessor", "resource_matcher", "response_drafter"]:
            for ex_data in data.get(agent, []):
                setattr(bank, agent, getattr(bank, agent) + [FewShotExample(**ex_data)])
        return bank

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FewShotBank":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_config(config_path: str = CONFIG_PATH) -> dict:
    """Load configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_incidents(path: str, limit: int = None) -> List[Incident]:
    """Load incidents from CSV or JSONL."""
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        incidents = []
        for _, row in df.iterrows():
            incidents.append(Incident(
                ticket_id=row["ticket_id"],
                description=row["description"],
                source=IncidentSource(row.get("source", "monitoring")),
                reporter=row.get("reporter") if pd.notna(row.get("reporter")) else None,
                affected_system=row.get("affected_system") if pd.notna(row.get("affected_system")) else None,
                initial_severity=int(row.get("initial_severity")) if pd.notna(row.get("initial_severity")) else None
            ))
    else:
        incidents = []
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                incidents.append(Incident(
                    ticket_id=data.get("ticket_id", f"INC-{len(incidents)+1}"),
                    description=data["description"],
                    source=IncidentSource(data.get("source", "monitoring")),
                    reporter=data.get("reporter"),
                    affected_system=data.get("affected_system"),
                    initial_severity=data.get("initial_severity")
                ))

    if limit:
        incidents = incidents[:limit]
    return incidents


def get_client(provider: str, config: dict = None):
    """Initialize LLM client."""
    if provider == "openai":
        from openai import OpenAI
        return OpenAI()
    elif provider == "anthropic":
        from anthropic import Anthropic
        return Anthropic()
    elif provider == "local":
        from openai import OpenAI
        # Get access token from Domino's local token service
        api_key = requests.get("http://localhost:8899/access-token").text
        return OpenAI(
            base_url=LOCAL_ENDPOINT,
            api_key=api_key
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


def format_few_shot_for_prompt(examples: List[FewShotExample]) -> str:
    """Format few-shot examples for injection into prompt."""
    if not examples:
        return ""

    parts = ["EXAMPLES FROM PREVIOUS SUCCESSFUL TRIAGES:\n"]
    for i, ex in enumerate(examples, 1):
        parts.append(f"Example {i}:")
        parts.append(f"Input: {ex.input_text[:500]}...")
        parts.append(f"Output: {json.dumps(ex.output_json, indent=2)}")
        parts.append("")

    parts.append("NOW PROCESS THE FOLLOWING:\n")
    return "\n".join(parts)


def pipeline_evaluator(span) -> dict:
    """Extract metrics from pipeline outputs for DominoRun."""
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


# =============================================================================
# PHASE 1: GENERATE FEW-SHOT EXAMPLES FROM FRONTIER MODELS
# =============================================================================

def generate_few_shot_examples(
    incidents: List[Incident],
    config: dict,
    providers: List[str] = ["openai", "anthropic"]
) -> FewShotBank:
    """Run frontier models on sample incidents to generate few-shot examples."""
    bank = FewShotBank()

    for provider in providers:
        print(f"\n{'='*60}")
        print(f"Generating examples from {provider.upper()}")
        print(f"{'='*60}")

        client = get_client(provider, config)
        model_config = config["models"][provider]
        model = model_config["name"] if isinstance(model_config, dict) else model_config

        for incident in incidents:
            print(f"\n  Processing {incident.ticket_id}...")

            try:
                classification = classify_incident(client, provider, model, incident, config)
                bank.add(FewShotExample(
                    agent="classifier",
                    input_text=incident.description,
                    output_json=classification.model_dump(),
                    source_model=f"{provider}/{model}",
                    ticket_id=incident.ticket_id
                ))
                print(f"    Classifier: {classification.category.value} (urgency={classification.urgency})")

                impact = assess_impact(client, provider, model, incident, classification, config)
                bank.add(FewShotExample(
                    agent="impact_assessor",
                    input_text=f"Incident: {incident.description}\nClassification: {classification.model_dump_json()}",
                    output_json=impact.model_dump(),
                    source_model=f"{provider}/{model}",
                    ticket_id=incident.ticket_id
                ))
                print(f"    Impact: score={impact.impact_score}, radius={impact.blast_radius}")

                resources = match_resources(client, provider, model, classification, impact, config)
                bank.add(FewShotExample(
                    agent="resource_matcher",
                    input_text=f"Classification: {classification.model_dump_json()}\nImpact: {impact.model_dump_json()}",
                    output_json=resources.model_dump(),
                    source_model=f"{provider}/{model}",
                    ticket_id=incident.ticket_id
                ))
                print(f"    Resources: {resources.primary_responder.name}")

                response = draft_response(client, provider, model, incident, classification, impact, resources, config)
                bank.add(FewShotExample(
                    agent="response_drafter",
                    input_text=f"Incident: {incident.description}",
                    output_json=response.model_dump(),
                    source_model=f"{provider}/{model}",
                    ticket_id=incident.ticket_id
                ))
                comm_type = "external" if response.communications else ("slack" if response.slack_messages else "none")
                print(f"    Response: {comm_type}")

            except Exception as e:
                print(f"    ERROR: {e}")
                continue

    print(f"\n  Generated {len(bank.classifier)} classifier examples")
    return bank


# =============================================================================
# PHASE 2: CREATE TRACED PIPELINE FUNCTIONS
# =============================================================================

def create_frontier_pipeline(client, provider: str, model: str, config: dict, judge_client):
    """Create a traced pipeline function for frontier models."""

    @add_tracing(name="triage_incident", autolog_frameworks=[provider], evaluator=pipeline_evaluator)
    def triage_incident(incident: Incident) -> dict:
        """Run the 4-agent triage pipeline with judges."""
        # Run agents
        classification = classify_incident(client, provider, model, incident, config)
        impact = assess_impact(client, provider, model, incident, classification, config)
        resources = match_resources(client, provider, model, classification, impact, config)
        response = draft_response(client, provider, model, incident, classification, impact, resources, config)

        # Convert to dicts
        class_dict = classification.model_dump()
        impact_dict = impact.model_dump()
        resources_dict = resources.model_dump()
        response_dict = response.model_dump()

        # Run judges
        class_judge = judge_classification(judge_client, "openai", incident.description, class_dict)

        resp_judge = {"score": 3}
        if response.communications:
            resp_judge = judge_response(
                judge_client, "openai", incident.description,
                classification.urgency, response.communications[0].model_dump()
            )

        triage_judge = judge_triage(
            judge_client, "openai", incident.description,
            class_dict, impact_dict, resources_dict, response_dict
        )

        return {
            "classification": classification,
            "impact": impact,
            "resources": resources,
            "response": response,
            # Metrics for evaluator
            "classification_confidence": class_dict.get("confidence", 0.5),
            "impact_score": impact_dict.get("impact_score", 5.0),
            "resource_match_score": resources_dict.get("primary_responder", {}).get("match_score", 0.5),
            "completeness_score": response_dict.get("completeness_score", 0.5),
            "classification_judge_score": class_judge.get("score", 3),
            "response_judge_score": resp_judge.get("score", 3),
            "triage_judge_score": triage_judge.get("score", 3),
        }

    return triage_incident


def create_local_pipeline(client, config: dict, few_shot_bank: FewShotBank, few_shot_count: int, judge_client):
    """Create a traced pipeline function for local model with few-shot."""

    local_config = config.get("local_prompts", config["agents"])
    model = ""  # Local endpoint expects empty string

    @add_tracing(name="triage_incident", autolog_frameworks=["openai"], evaluator=pipeline_evaluator)
    def triage_incident(incident: Incident) -> dict:
        """Run the 4-agent triage pipeline with few-shot examples."""

        # Classifier with few-shot
        classifier_examples = few_shot_bank.get_examples("classifier", few_shot_count)
        few_shot_text = format_few_shot_for_prompt(classifier_examples)
        prompt = local_config["classifier"]["prompt"].format(
            few_shot_examples=few_shot_text,
            incident=incident.model_dump_json()
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=local_config["classifier"]["temperature"],
            max_tokens=local_config["classifier"]["max_tokens"]
        )
        classification = Classification(**parse_json_response(response.choices[0].message.content))

        # Impact Assessor with few-shot
        impact_examples = few_shot_bank.get_examples("impact_assessor", few_shot_count)
        few_shot_text = format_few_shot_for_prompt(impact_examples)
        prompt = local_config["impact_assessor"]["prompt"].format(
            few_shot_examples=few_shot_text,
            incident=incident.model_dump_json(),
            classification=classification.model_dump_json()
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=local_config["impact_assessor"]["temperature"],
            max_tokens=local_config["impact_assessor"]["max_tokens"]
        )
        impact = ImpactAssessment(**parse_json_response(response.choices[0].message.content))

        # Resource Matcher with few-shot
        resource_examples = few_shot_bank.get_examples("resource_matcher", few_shot_count)
        few_shot_text = format_few_shot_for_prompt(resource_examples)
        prompt = local_config["resource_matcher"]["prompt"].format(
            few_shot_examples=few_shot_text,
            classification=classification.model_dump_json(),
            impact=impact.model_dump_json()
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=local_config["resource_matcher"]["temperature"],
            max_tokens=local_config["resource_matcher"]["max_tokens"]
        )
        resources = ResourceAssignment(**parse_json_response(response.choices[0].message.content))

        # Response Drafter with few-shot
        response_examples = few_shot_bank.get_examples("response_drafter", few_shot_count)
        few_shot_text = format_few_shot_for_prompt(response_examples)
        prompt = local_config["response_drafter"]["prompt"].format(
            few_shot_examples=few_shot_text,
            incident=incident.model_dump_json(),
            classification=classification.model_dump_json(),
            impact=impact.model_dump_json(),
            resources=resources.model_dump_json()
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=local_config["response_drafter"]["temperature"],
            max_tokens=local_config["response_drafter"]["max_tokens"]
        )
        response_plan = ResponsePlan(**parse_json_response(response.choices[0].message.content))

        # Convert to dicts
        class_dict = classification.model_dump()
        impact_dict = impact.model_dump()
        resources_dict = resources.model_dump()
        response_dict = response_plan.model_dump()

        # Run judges
        class_judge = judge_classification(judge_client, "openai", incident.description, class_dict)

        resp_judge = {"score": 3}
        if response_plan.communications:
            resp_judge = judge_response(
                judge_client, "openai", incident.description,
                classification.urgency, response_plan.communications[0].model_dump()
            )

        triage_judge = judge_triage(
            judge_client, "openai", incident.description,
            class_dict, impact_dict, resources_dict, response_dict
        )

        return {
            "classification": classification,
            "impact": impact,
            "resources": resources,
            "response": response_plan,
            # Metrics
            "classification_confidence": class_dict.get("confidence", 0.5),
            "impact_score": impact_dict.get("impact_score", 5.0),
            "resource_match_score": resources_dict.get("primary_responder", {}).get("match_score", 0.5),
            "completeness_score": response_dict.get("completeness_score", 0.5),
            "classification_judge_score": class_judge.get("score", 3),
            "response_judge_score": resp_judge.get("score", 3),
            "triage_judge_score": triage_judge.get("score", 3),
        }

    return triage_incident


# =============================================================================
# PHASE 3: RUN COMPARISON WITH DOMINO LOGGING
# =============================================================================

def load_model_config(provider: str) -> dict:
    """Load model-specific configuration."""
    config_path = MODEL_CONFIGS.get(provider)
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def run_model_experiment(
    provider: str,
    model_name: str,
    incidents: List[Incident],
    config: dict,
    few_shot_bank: Optional[FewShotBank] = None,
    few_shot_count: int = 2
) -> List[dict]:
    """
    Run a single model experiment with full Domino tracing.

    Creates its own DominoRun with parameters and metrics logged.
    """
    results = []

    # Load model-specific config
    model_config = load_model_config(provider)
    model_config_path = MODEL_CONFIGS.get(provider, CONFIG_PATH)

    # Get clients
    client = get_client(provider, config)
    judge_client = get_client("openai", config)

    # Create pipeline function
    if provider == "local":
        triage_fn = create_local_pipeline(client, config, few_shot_bank, few_shot_count, judge_client)
    else:
        triage_fn = create_frontier_pipeline(client, provider, model_name, config, judge_client)

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

    # Get username for experiment name
    username = os.environ.get("DOMINO_USER_NAME", os.environ.get("USER", "unknown"))

    # Set experiment name
    experiment_name = f"agent-workflow-{username}"
    mlflow.set_experiment(experiment_name)

    # Run with DominoRun for proper Domino integration
    with DominoRun(agent_config_path=model_config_path, custom_summary_metrics=aggregated_metrics) as run:
        # Set run name and tags
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = model_config.get("name", f"{provider}-{model_name}")
        mlflow.set_tag("mlflow.runName", f"{run_name}-{timestamp}")
        mlflow.set_tag("mode", "model_comparison")
        mlflow.set_tag("provider", provider)
        mlflow.set_tag("model", model_name)
        mlflow.set_tag("config_name", model_config.get("name", "unknown"))

        # Log parameters from model config
        mlflow.log_param("name", model_config.get("name", model_name))
        mlflow.log_param("provider", provider)
        mlflow.log_param("model", model_config.get("model", model_name))
        mlflow.log_param("prompt_type", model_config.get("prompt_type", "standard"))
        mlflow.log_param("temperature", model_config.get("temperature", 0.3))
        mlflow.log_param("num_incidents", len(incidents))

        if provider == "local":
            mlflow.log_param("few_shot_count", model_config.get("few_shot_count", few_shot_count))
            mlflow.set_tag("prompt_type", "strict_few_shot")
        else:
            mlflow.set_tag("prompt_type", "standard")

        # Process each incident
        for incident in incidents:
            print(f"    Processing {incident.ticket_id}...")
            start = time.time()

            try:
                result = triage_fn(incident)
                duration = time.time() - start

                results.append({
                    "ticket_id": incident.ticket_id,
                    "category": result["classification"].category.value,
                    "urgency": result["classification"].urgency,
                    "impact_score": result["impact"].impact_score,
                    "blast_radius": result["impact"].blast_radius,
                    "classification_judge": result["classification_judge_score"],
                    "response_judge": result["response_judge_score"],
                    "triage_judge": result["triage_judge_score"],
                    "duration_sec": round(duration, 2),
                    "error": None
                })

                print(f"      Category: {result['classification'].category.value}, "
                      f"Urgency: {result['classification'].urgency}, "
                      f"Judges: C={result['classification_judge_score']}, "
                      f"R={result['response_judge_score']}, T={result['triage_judge_score']}")

            except Exception as e:
                duration = time.time() - start
                results.append({
                    "ticket_id": incident.ticket_id,
                    "category": None,
                    "urgency": None,
                    "impact_score": None,
                    "blast_radius": None,
                    "classification_judge": None,
                    "response_judge": None,
                    "triage_judge": None,
                    "duration_sec": round(duration, 2),
                    "error": str(e)
                })
                print(f"      ERROR: {e}")

        # Log aggregate metrics
        successful = [r for r in results if r["error"] is None]
        if successful:
            mlflow.log_metric("avg_classification_judge", sum(r["classification_judge"] for r in successful) / len(successful))
            mlflow.log_metric("avg_triage_judge", sum(r["triage_judge"] for r in successful) / len(successful))
            mlflow.log_metric("avg_duration_sec", sum(r["duration_sec"] for r in successful) / len(successful))
            mlflow.log_metric("success_rate", len(successful) / len(results))

        # Suppress DominoRun exit messages
        _stdout = sys.stdout
        sys.stdout = io.StringIO()

    sys.stdout = _stdout
    return results


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Run model comparison experiment")
    parser.add_argument("--few-shot-incidents", type=int, default=3,
                        help="Number of incidents for generating few-shot examples")
    parser.add_argument("--few-shot-count", type=int, default=2,
                        help="Number of few-shot examples per agent for local model")
    parser.add_argument("--test-incidents", type=int, default=5,
                        help="Number of incidents to test on")
    parser.add_argument("--data-path", type=str, default="/mnt/code/example-data/financial_services.csv",
                        help="Path to incident data")
    parser.add_argument("--skip-frontier", action="store_true",
                        help="Skip frontier generation, use cached examples")
    parser.add_argument("--cache-path", type=str, default="/tmp/few_shot_cache.json",
                        help="Path to cache few-shot examples")
    parser.add_argument("--models", type=str, default="openai,anthropic,local",
                        help="Comma-separated list of models to test")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()

    print("=" * 70)
    print("TRIAGEFLOW MODEL COMPARISON EXPERIMENT")
    print("=" * 70)
    print(f"Few-shot generation incidents: {args.few_shot_incidents}")
    print(f"Few-shot examples per agent: {args.few_shot_count}")
    print(f"Test incidents: {args.test_incidents}")
    print(f"Models to test: {args.models}")
    print(f"Data path: {args.data_path}")
    print()

    # Load all incidents
    all_incidents = load_incidents(args.data_path)
    print(f"Loaded {len(all_incidents)} total incidents")

    if args.dry_run:
        print("\n[DRY RUN] Would process:")
        print(f"  - {args.few_shot_incidents} incidents for few-shot generation")
        print(f"  - {args.test_incidents} incidents for testing")
        print(f"  - Models: {args.models}")
        return

    # Split into few-shot generation set and test set
    few_shot_incidents = all_incidents[:args.few_shot_incidents]
    test_incidents = all_incidents[args.few_shot_incidents:args.few_shot_incidents + args.test_incidents]

    print(f"Few-shot incidents: {[i.ticket_id for i in few_shot_incidents]}")
    print(f"Test incidents: {[i.ticket_id for i in test_incidents]}")

    # Phase 1: Generate or load few-shot examples
    if args.skip_frontier and os.path.exists(args.cache_path):
        print(f"\nLoading cached few-shot examples from {args.cache_path}")
        few_shot_bank = FewShotBank.load(args.cache_path)
    else:
        print("\n" + "=" * 70)
        print("PHASE 1: GENERATING FEW-SHOT EXAMPLES FROM FRONTIER MODELS")
        print("=" * 70)
        few_shot_bank = generate_few_shot_examples(few_shot_incidents, config)
        few_shot_bank.save(args.cache_path)
        print(f"\nCached few-shot examples to {args.cache_path}")

    # Phase 2: Run comparison - each model gets its own DominoRun
    print("\n" + "=" * 70)
    print("PHASE 2: RUNNING MODEL EXPERIMENTS (each model = separate DominoRun)")
    print("=" * 70)

    models_to_test = args.models.split(",")
    all_results = {}

    for model_key in models_to_test:
        model_key = model_key.strip()
        model_config = config["models"].get(model_key, {})
        model_name = model_config.get("name", model_key) if isinstance(model_config, dict) else model_config

        print(f"\n{'='*60}")
        print(f"Running {model_key.upper()} ({model_name})")
        print(f"{'='*60}")

        results = run_model_experiment(
            provider=model_key,
            model_name=model_name,
            incidents=test_incidents,
            config=config,
            few_shot_bank=few_shot_bank if model_key == "local" else None,
            few_shot_count=args.few_shot_count
        )

        all_results[model_key] = results

    # Phase 3: Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Create summary DataFrame
    rows = []
    for model_key, results in all_results.items():
        for r in results:
            rows.append({
                "model": model_key,
                **r
            })

    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))

    # Aggregate by model
    print("\n" + "-" * 40)
    print("AVERAGE SCORES BY MODEL")
    print("-" * 40)

    for model_key, results in all_results.items():
        successful = [r for r in results if r["error"] is None]
        if successful:
            avg_class = sum(r["classification_judge"] for r in successful) / len(successful)
            avg_triage = sum(r["triage_judge"] for r in successful) / len(successful)
            avg_duration = sum(r["duration_sec"] for r in successful) / len(successful)
            success_rate = len(successful) / len(results)
            print(f"{model_key:12} | Class: {avg_class:.2f} | Triage: {avg_triage:.2f} | "
                  f"Duration: {avg_duration:.1f}s | Success: {success_rate:.0%}")

    # Save results
    results_path = f"/tmp/model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE - View runs in Domino Experiment Manager")
    print("=" * 70)


if __name__ == "__main__":
    main()
