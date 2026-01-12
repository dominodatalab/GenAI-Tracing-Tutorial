#!/usr/bin/env python3
"""
TriageFlow Model Comparison Experiment

Compares frontier models (OpenAI, Anthropic) against local Qwen model
with few-shot learning from frontier outputs.

Workflow:
1. Run frontier models on sample incidents to generate high-quality outputs
2. Use those outputs as few-shot examples for the local Qwen model
3. Run all three models on test incidents
4. Evaluate with LLM judges and compare results

Usage:
    python run_model_comparison.py
    python run_model_comparison.py --few-shot-count 3 --test-incidents 10
    python run_model_comparison.py --skip-frontier  # Use cached examples
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

sys.path.insert(0, "/mnt/code")

import mlflow
import pandas as pd
import yaml

from src.models import Incident, IncidentSource, Classification, ImpactAssessment, ResourceAssignment, ResponsePlan
from src.agents import (
    classify_incident, assess_impact, match_resources, draft_response,
    call_with_tools, parse_json_response, get_all_tools
)
from src.judges import judge_classification, judge_response, judge_triage


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


@dataclass
class ModelResult:
    """Results from running one model on one incident."""
    model_name: str
    provider: str
    ticket_id: str
    classification: Optional[dict] = None
    impact: Optional[dict] = None
    resources: Optional[dict] = None
    response: Optional[dict] = None
    judge_scores: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    duration_seconds: float = 0.0


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_config(config_path: str = "/mnt/code/configs/agents.yaml") -> dict:
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


def get_client(provider: str):
    """Initialize LLM client."""
    if provider == "openai":
        from openai import OpenAI
        return OpenAI()
    elif provider == "anthropic":
        from anthropic import Anthropic
        return Anthropic()
    elif provider == "local":
        from openai import OpenAI
        import requests
        config = load_config()
        endpoint = config["models"]["local"]["endpoint"]
        # Get access token from Domino's local token service
        api_key = requests.get("http://localhost:8899/access-token").text
        return OpenAI(
            base_url=endpoint,
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


# =============================================================================
# PHASE 1: GENERATE FEW-SHOT EXAMPLES FROM FRONTIER MODELS
# =============================================================================

def generate_few_shot_examples(
    incidents: List[Incident],
    config: dict,
    providers: List[str] = ["openai", "anthropic"]
) -> FewShotBank:
    """
    Run frontier models on sample incidents to generate few-shot examples.

    Args:
        incidents: Sample incidents to process
        config: Agent configuration
        providers: Which frontier providers to use

    Returns:
        FewShotBank with examples from all agents
    """
    bank = FewShotBank()

    for provider in providers:
        print(f"\n{'='*60}")
        print(f"Generating examples from {provider.upper()}")
        print(f"{'='*60}")

        client = get_client(provider)
        model_config = config["models"][provider]
        model = model_config["name"] if isinstance(model_config, dict) else model_config

        for incident in incidents:
            print(f"\n  Processing {incident.ticket_id}...")

            try:
                # Classifier
                classification = classify_incident(client, provider, model, incident, config)
                bank.add(FewShotExample(
                    agent="classifier",
                    input_text=incident.description,
                    output_json=classification.model_dump(),
                    source_model=f"{provider}/{model}",
                    ticket_id=incident.ticket_id
                ))
                print(f"    Classifier: {classification.category.value} (urgency={classification.urgency})")

                # Impact Assessor
                impact = assess_impact(client, provider, model, incident, classification, config)
                bank.add(FewShotExample(
                    agent="impact_assessor",
                    input_text=f"Incident: {incident.description}\nClassification: {classification.model_dump_json()}",
                    output_json=impact.model_dump(),
                    source_model=f"{provider}/{model}",
                    ticket_id=incident.ticket_id
                ))
                print(f"    Impact: score={impact.impact_score}, radius={impact.blast_radius}")

                # Resource Matcher
                resources = match_resources(client, provider, model, classification, impact, config)
                bank.add(FewShotExample(
                    agent="resource_matcher",
                    input_text=f"Classification: {classification.model_dump_json()}\nImpact: {impact.model_dump_json()}",
                    output_json=resources.model_dump(),
                    source_model=f"{provider}/{model}",
                    ticket_id=incident.ticket_id
                ))
                print(f"    Resources: {resources.primary_responder.name}")

                # Response Drafter
                response = draft_response(client, provider, model, incident, classification, impact, resources, config)
                bank.add(FewShotExample(
                    agent="response_drafter",
                    input_text=f"Incident: {incident.description}\nClassification: {classification.model_dump_json()}\nImpact: {impact.model_dump_json()}",
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
    print(f"  Generated {len(bank.impact_assessor)} impact assessor examples")
    print(f"  Generated {len(bank.resource_matcher)} resource matcher examples")
    print(f"  Generated {len(bank.response_drafter)} response drafter examples")

    return bank


# =============================================================================
# PHASE 2: RUN LOCAL MODEL WITH FEW-SHOT
# =============================================================================

def run_local_with_few_shot(
    client,
    incident: Incident,
    config: dict,
    few_shot_bank: FewShotBank,
    few_shot_count: int = 2
) -> Dict[str, Any]:
    """
    Run the local Qwen model using strict prompts with few-shot examples.

    Args:
        client: OpenAI-compatible client pointing to local endpoint
        incident: Incident to process
        config: Full configuration
        few_shot_bank: Bank of few-shot examples
        few_shot_count: Number of examples to include per agent

    Returns:
        Dict with classification, impact, resources, response
    """
    local_config = config.get("local_prompts", config["agents"])
    # Local Qwen endpoint expects empty string for model
    model = ""
    results = {}

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
    results["classification"] = classification

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
    results["impact"] = impact

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
    results["resources"] = resources

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
    results["response"] = response_plan

    return results


# =============================================================================
# PHASE 3: FULL COMPARISON
# =============================================================================

def run_comparison(
    test_incidents: List[Incident],
    config: dict,
    few_shot_bank: FewShotBank,
    few_shot_count: int = 2
) -> List[ModelResult]:
    """
    Run all three models on test incidents and collect results.

    Args:
        test_incidents: Incidents to test on
        config: Full configuration
        few_shot_bank: Few-shot examples for local model
        few_shot_count: Number of examples for local model

    Returns:
        List of ModelResult for each model/incident combination
    """
    results = []
    models_to_test = [
        ("openai", "openai"),
        ("anthropic", "anthropic"),
        ("local", "local")  # Will use few-shot
    ]

    for incident in test_incidents:
        print(f"\n{'='*60}")
        print(f"Testing: {incident.ticket_id}")
        print(f"{'='*60}")

        for provider, model_key in models_to_test:
            import time
            start = time.time()

            model_config = config["models"][model_key]
            model_name = model_config["name"] if isinstance(model_config, dict) else model_config

            print(f"\n  {provider.upper()} ({model_name})...")

            result = ModelResult(
                model_name=model_name,
                provider=provider,
                ticket_id=incident.ticket_id
            )

            try:
                client = get_client(provider)

                if provider == "local":
                    # Use few-shot approach for local model
                    outputs = run_local_with_few_shot(
                        client, incident, config, few_shot_bank, few_shot_count
                    )
                    classification = outputs["classification"]
                    impact = outputs["impact"]
                    resources = outputs["resources"]
                    response = outputs["response"]
                else:
                    # Standard approach for frontier models
                    classification = classify_incident(client, provider, model_name, incident, config)
                    impact = assess_impact(client, provider, model_name, incident, classification, config)
                    resources = match_resources(client, provider, model_name, classification, impact, config)
                    response = draft_response(client, provider, model_name, incident, classification, impact, resources, config)

                result.classification = classification.model_dump()
                result.impact = impact.model_dump()
                result.resources = resources.model_dump()
                result.response = response.model_dump()

                # Run judges (use frontier model for judging)
                judge_client = get_client("openai")

                class_judge = judge_classification(
                    judge_client, "openai", incident.description, result.classification
                )
                result.judge_scores["classification"] = class_judge.get("score", 0)

                if response.communications:
                    resp_judge = judge_response(
                        judge_client, "openai", incident.description,
                        classification.urgency, response.communications[0].model_dump()
                    )
                    result.judge_scores["response"] = resp_judge.get("score", 0)

                triage_judge = judge_triage(
                    judge_client, "openai", incident.description,
                    result.classification, result.impact,
                    result.resources, result.response
                )
                result.judge_scores["triage"] = triage_judge.get("score", 0)

                print(f"    Category: {classification.category.value}")
                print(f"    Urgency: {classification.urgency}")
                print(f"    Impact: {impact.impact_score}")
                print(f"    Judge scores: {result.judge_scores}")

            except Exception as e:
                result.error = str(e)
                print(f"    ERROR: {e}")

            result.duration_seconds = time.time() - start
            results.append(result)

    return results


def summarize_results(results: List[ModelResult]) -> pd.DataFrame:
    """Create summary DataFrame from results."""
    rows = []
    for r in results:
        row = {
            "ticket_id": r.ticket_id,
            "model": r.model_name,
            "provider": r.provider,
            "category": r.classification.get("category") if r.classification else None,
            "urgency": r.classification.get("urgency") if r.classification else None,
            "impact_score": r.impact.get("impact_score") if r.impact else None,
            "blast_radius": r.impact.get("blast_radius") if r.impact else None,
            "judge_classification": r.judge_scores.get("classification"),
            "judge_response": r.judge_scores.get("response"),
            "judge_triage": r.judge_scores.get("triage"),
            "duration_sec": round(r.duration_seconds, 2),
            "error": r.error
        }
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Run model comparison experiment")
    parser.add_argument("--few-shot-incidents", type=int, default=3,
                        help="Number of incidents to use for generating few-shot examples")
    parser.add_argument("--few-shot-count", type=int, default=2,
                        help="Number of few-shot examples to include in local model prompts")
    parser.add_argument("--test-incidents", type=int, default=5,
                        help="Number of incidents to test on")
    parser.add_argument("--data-path", type=str, default="/mnt/code/example-data/financial_services.csv",
                        help="Path to incident data")
    parser.add_argument("--skip-frontier", action="store_true",
                        help="Skip frontier generation, use cached examples")
    parser.add_argument("--cache-path", type=str, default="/tmp/few_shot_cache.json",
                        help="Path to cache few-shot examples")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()

    print("="*70)
    print("TRIAGEFLOW MODEL COMPARISON EXPERIMENT")
    print("="*70)
    print(f"Few-shot generation incidents: {args.few_shot_incidents}")
    print(f"Few-shot examples per agent: {args.few_shot_count}")
    print(f"Test incidents: {args.test_incidents}")
    print(f"Data path: {args.data_path}")
    print()

    # Load all incidents
    all_incidents = load_incidents(args.data_path)
    print(f"Loaded {len(all_incidents)} total incidents")

    if args.dry_run:
        print("\n[DRY RUN] Would process:")
        print(f"  - {args.few_shot_incidents} incidents for few-shot generation")
        print(f"  - {args.test_incidents} incidents for testing")
        return

    # Split into few-shot generation set and test set
    few_shot_incidents = all_incidents[:args.few_shot_incidents]
    test_incidents = all_incidents[args.few_shot_incidents:args.few_shot_incidents + args.test_incidents]

    print(f"Few-shot incidents: {[i.ticket_id for i in few_shot_incidents]}")
    print(f"Test incidents: {[i.ticket_id for i in test_incidents]}")

    # Phase 1: Generate few-shot examples
    if args.skip_frontier and os.path.exists(args.cache_path):
        print(f"\nLoading cached few-shot examples from {args.cache_path}")
        few_shot_bank = FewShotBank.load(args.cache_path)
    else:
        print("\n" + "="*70)
        print("PHASE 1: GENERATING FEW-SHOT EXAMPLES FROM FRONTIER MODELS")
        print("="*70)
        few_shot_bank = generate_few_shot_examples(few_shot_incidents, config)

        # Cache the examples
        os.makedirs(os.path.dirname(args.cache_path), exist_ok=True)
        few_shot_bank.save(args.cache_path)
        print(f"\nCached few-shot examples to {args.cache_path}")

    # Phase 2 & 3: Run comparison
    print("\n" + "="*70)
    print("PHASE 2 & 3: RUNNING MODEL COMPARISON")
    print("="*70)

    # Set up MLflow
    experiment_name = f"model-comparison-{os.environ.get('DOMINO_PROJECT_NAME', 'local')}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"comparison-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
        mlflow.log_param("few_shot_incidents", args.few_shot_incidents)
        mlflow.log_param("few_shot_count", args.few_shot_count)
        mlflow.log_param("test_incidents", args.test_incidents)
        mlflow.log_param("data_path", args.data_path)

        results = run_comparison(test_incidents, config, few_shot_bank, args.few_shot_count)

        # Summarize
        summary_df = summarize_results(results)

        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        print(summary_df.to_string(index=False))

        # Aggregate by model
        print("\n" + "-"*40)
        print("AVERAGE SCORES BY MODEL")
        print("-"*40)
        model_summary = summary_df.groupby("provider").agg({
            "judge_classification": "mean",
            "judge_response": "mean",
            "judge_triage": "mean",
            "duration_sec": "mean"
        }).round(2)
        print(model_summary)

        # Log to MLflow
        for provider in ["openai", "anthropic", "local"]:
            provider_results = summary_df[summary_df["provider"] == provider]
            if len(provider_results) > 0:
                mlflow.log_metric(f"{provider}_avg_classification", provider_results["judge_classification"].mean())
                mlflow.log_metric(f"{provider}_avg_triage", provider_results["judge_triage"].mean())
                mlflow.log_metric(f"{provider}_avg_duration", provider_results["duration_sec"].mean())

        # Save results
        results_path = f"/tmp/comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        summary_df.to_csv(results_path, index=False)
        mlflow.log_artifact(results_path)
        print(f"\nResults saved to {results_path}")

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
