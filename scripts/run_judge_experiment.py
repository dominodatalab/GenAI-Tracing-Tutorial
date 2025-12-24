#!/usr/bin/env python3
"""
TriageFlow Judge Experiment Runner

Optimizes judge configurations via grid search over models, temperatures,
prompt styles, and scales. Uses ground truth labels from YAML for validation.

Supports all 4 judges:
  - classifier: Evaluates incident classification
  - impact_assessor: Evaluates impact assessment quality
  - resource_matcher: Evaluates resource assignment quality
  - response_drafter: Evaluates communication quality

Experiment name: judge-optimization-{username}
Tag: model type only (e.g., gpt-4o-mini)

Usage:
    python run_judge_experiment.py --vertical financial_services --dry-run
    python run_judge_experiment.py --vertical healthcare --model gpt-4o-mini
    python run_judge_experiment.py --vertical energy --judge classifier
    python run_judge_experiment.py --vertical financial_services --judge all
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
from src.agents import (
    classify_incident,
    assess_impact,
    match_resources,
    draft_response,
)


# =============================================================================
# CONSTANTS
# =============================================================================

VERTICALS = ["financial_services", "healthcare", "energy", "public_sector"]
JUDGE_TYPES = ["classifier", "impact_assessor", "resource_matcher", "response_drafter", "all"]
GROUND_TRUTH_PATH = "/mnt/code/example-data/ground_truth_judgments.yaml"


# =============================================================================
# JUDGE PROMPT TEMPLATES (by judge type)
# =============================================================================

# Classifier judge prompts
CLASSIFIER_PROMPTS = {
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

# Impact Assessor judge prompts
IMPACT_ASSESSOR_PROMPTS = {
    ("direct", "binary"): """Evaluate this impact assessment. Is it reasonable?

Incident: {incident}
Impact Score: {impact_score}/10
Affected Users: {affected_users}
Blast Radius: {blast_radius}
Reasoning: {reasoning}

Return JSON: {{"score": 0 or 1, "rationale": "brief explanation"}}""",

    ("direct", "three_point"): """Evaluate this impact assessment quality (1-3).

Incident: {incident}
Impact Score: {impact_score}/10
Affected Users: {affected_users}
Blast Radius: {blast_radius}
Reasoning: {reasoning}

Score: 1=poor, 2=acceptable, 3=good
Return JSON: {{"score": 1-3, "rationale": "brief explanation"}}""",

    ("cot", "binary"): """Think step by step about this impact assessment.

Incident: {incident}
Impact Score: {impact_score}/10
Affected Users: {affected_users}
Blast Radius: {blast_radius}
Reasoning: {reasoning}

Consider: Is the impact score proportional? Is the blast radius reasonable? Is the user estimate logical?
First explain your analysis, then decide: pass (1) or fail (0).
Return JSON: {{"score": 0 or 1, "analysis": "step-by-step", "rationale": "summary"}}""",

    ("cot", "three_point"): """Think step by step about this impact assessment.

Incident: {incident}
Impact Score: {impact_score}/10
Affected Users: {affected_users}
Blast Radius: {blast_radius}
Reasoning: {reasoning}

Consider: Is the impact score proportional? Is the blast radius reasonable? Is the user estimate logical?
First explain your analysis, then score 1-3 (1=poor, 2=acceptable, 3=good).
Return JSON: {{"score": 1-3, "analysis": "step-by-step", "rationale": "summary"}}""",

    ("rubric", "binary"): """Evaluate using this rubric:
- PASS (1): Impact score justified by incident, blast radius reasonable, estimates logical
- FAIL (0): Impact score over/underestimated, unreasonable blast radius, poor estimates

Incident: {incident}
Impact Score: {impact_score}/10
Affected Users: {affected_users}
Blast Radius: {blast_radius}
Reasoning: {reasoning}

Return JSON: {{"score": 0 or 1, "rubric_check": "pass/fail criteria", "rationale": "summary"}}""",

    ("rubric", "three_point"): """Evaluate using this rubric:
- GOOD (3): Well-calibrated impact score, accurate blast radius, detailed reasoning
- ACCEPTABLE (2): Reasonable impact score, adequate blast radius, some reasoning
- POOR (1): Miscalibrated impact, wrong blast radius, weak reasoning

Incident: {incident}
Impact Score: {impact_score}/10
Affected Users: {affected_users}
Blast Radius: {blast_radius}
Reasoning: {reasoning}

Return JSON: {{"score": 1-3, "rubric_check": "criteria assessment", "rationale": "summary"}}""",
}

# Resource Matcher judge prompts
RESOURCE_MATCHER_PROMPTS = {
    ("direct", "binary"): """Evaluate this resource assignment. Is it appropriate?

Incident Category: {category}
Urgency: {urgency}
Primary Responder: {responder_name} (Skills: {responder_skills})
Match Score: {match_score}
SLA Target: {sla_hours} hours
SLA Met: {sla_met}

Return JSON: {{"score": 0 or 1, "rationale": "brief explanation"}}""",

    ("direct", "three_point"): """Evaluate this resource assignment quality (1-3).

Incident Category: {category}
Urgency: {urgency}
Primary Responder: {responder_name} (Skills: {responder_skills})
Match Score: {match_score}
SLA Target: {sla_hours} hours
SLA Met: {sla_met}

Score: 1=poor, 2=acceptable, 3=good
Return JSON: {{"score": 1-3, "rationale": "brief explanation"}}""",

    ("cot", "binary"): """Think step by step about this resource assignment.

Incident Category: {category}
Urgency: {urgency}
Primary Responder: {responder_name} (Skills: {responder_skills})
Match Score: {match_score}
SLA Target: {sla_hours} hours
SLA Met: {sla_met}

Consider: Do the responder's skills match the incident type? Is the SLA appropriate for the urgency?
First explain your analysis, then decide: pass (1) or fail (0).
Return JSON: {{"score": 0 or 1, "analysis": "step-by-step", "rationale": "summary"}}""",

    ("cot", "three_point"): """Think step by step about this resource assignment.

Incident Category: {category}
Urgency: {urgency}
Primary Responder: {responder_name} (Skills: {responder_skills})
Match Score: {match_score}
SLA Target: {sla_hours} hours
SLA Met: {sla_met}

Consider: Do the responder's skills match the incident type? Is the SLA appropriate for the urgency?
First explain your analysis, then score 1-3 (1=poor, 2=acceptable, 3=good).
Return JSON: {{"score": 1-3, "analysis": "step-by-step", "rationale": "summary"}}""",

    ("rubric", "binary"): """Evaluate using this rubric:
- PASS (1): Responder skills match incident, SLA appropriate for urgency, logical assignment
- FAIL (0): Skill mismatch, SLA too loose for urgency, poor assignment

Incident Category: {category}
Urgency: {urgency}
Primary Responder: {responder_name} (Skills: {responder_skills})
Match Score: {match_score}
SLA Target: {sla_hours} hours
SLA Met: {sla_met}

Return JSON: {{"score": 0 or 1, "rubric_check": "pass/fail criteria", "rationale": "summary"}}""",

    ("rubric", "three_point"): """Evaluate using this rubric:
- GOOD (3): Perfect skill match, SLA well-calibrated, excellent assignment
- ACCEPTABLE (2): Adequate skills, reasonable SLA, acceptable assignment
- POOR (1): Skill mismatch, SLA issues, poor assignment

Incident Category: {category}
Urgency: {urgency}
Primary Responder: {responder_name} (Skills: {responder_skills})
Match Score: {match_score}
SLA Target: {sla_hours} hours
SLA Met: {sla_met}

Return JSON: {{"score": 1-3, "rubric_check": "criteria assessment", "rationale": "summary"}}""",
}

# Response Drafter judge prompts
RESPONSE_DRAFTER_PROMPTS = {
    ("direct", "binary"): """Evaluate this incident communication. Is it appropriate?

Incident: {incident}
Urgency: {urgency}
Audience: {audience}
Subject: {subject}
Body: {body}

Return JSON: {{"score": 0 or 1, "rationale": "brief explanation"}}""",

    ("direct", "three_point"): """Evaluate this incident communication quality (1-3).

Incident: {incident}
Urgency: {urgency}
Audience: {audience}
Subject: {subject}
Body: {body}

Score: 1=poor, 2=acceptable, 3=good
Return JSON: {{"score": 1-3, "rationale": "brief explanation"}}""",

    ("cot", "binary"): """Think step by step about this incident communication.

Incident: {incident}
Urgency: {urgency}
Audience: {audience}
Subject: {subject}
Body: {body}

Consider: Is the tone appropriate for the audience and urgency? Is information clear? Are next steps actionable?
First explain your analysis, then decide: pass (1) or fail (0).
Return JSON: {{"score": 0 or 1, "analysis": "step-by-step", "rationale": "summary"}}""",

    ("cot", "three_point"): """Think step by step about this incident communication.

Incident: {incident}
Urgency: {urgency}
Audience: {audience}
Subject: {subject}
Body: {body}

Consider: Is the tone appropriate for the audience and urgency? Is information clear? Are next steps actionable?
First explain your analysis, then score 1-3 (1=poor, 2=acceptable, 3=good).
Return JSON: {{"score": 1-3, "analysis": "step-by-step", "rationale": "summary"}}""",

    ("rubric", "binary"): """Evaluate using this rubric:
- PASS (1): Tone matches urgency/audience, clear information, actionable next steps
- FAIL (0): Wrong tone, unclear information, no actionable steps

Incident: {incident}
Urgency: {urgency}
Audience: {audience}
Subject: {subject}
Body: {body}

Return JSON: {{"score": 0 or 1, "rubric_check": "pass/fail criteria", "rationale": "summary"}}""",

    ("rubric", "three_point"): """Evaluate using this rubric:
- GOOD (3): Perfect tone, comprehensive info, clear actionable steps
- ACCEPTABLE (2): Appropriate tone, adequate info, some actions
- POOR (1): Wrong tone, missing info, unclear actions

Incident: {incident}
Urgency: {urgency}
Audience: {audience}
Subject: {subject}
Body: {body}

Return JSON: {{"score": 1-3, "rubric_check": "criteria assessment", "rationale": "summary"}}""",
}

# Map judge type to prompts
JUDGE_PROMPTS = {
    "classifier": CLASSIFIER_PROMPTS,
    "impact_assessor": IMPACT_ASSESSOR_PROMPTS,
    "resource_matcher": RESOURCE_MATCHER_PROMPTS,
    "response_drafter": RESPONSE_DRAFTER_PROMPTS,
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
    parser.add_argument(
        "--judge",
        type=str,
        choices=JUDGE_TYPES,
        default="classifier",
        help="Judge type to optimize (default: classifier, use 'all' for all judges)"
    )
    parser.add_argument("--model", type=str, help="Single model to test (default: all)")
    parser.add_argument("--config", type=str, help="Config path (default: configs/judge_experiment_grid.yaml)")
    parser.add_argument("--runs-per-config", type=int, help="Runs per config for consistency")
    parser.add_argument("--max-tickets", "-n", type=int, help="Maximum number of tickets to process")
    parser.add_argument("--dry-run", action="store_true", help="Show configs without running")
    parser.add_argument("--submit-job", action="store_true", help="Submit best config as a Domino Job for final registration")
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


def compute_human_agreement(judge_score: int, ground_truth: dict, scale: str, judge_type: str = "classifier") -> Optional[float]:
    """
    Compare judge score to ground truth based on judge type.

    Judge-specific ground truth fields:
    - classifier: category_correct, urgency_correct, reasoning_quality
    - impact_assessor: impact_assessment_quality
    - resource_matcher: resource_assignment_correct, resource_assignment_quality
    - response_drafter: response_appropriateness
    """
    if judge_score is None:
        return None

    gt = ground_truth.get("evaluation_ground_truth", {})

    if scale == "binary":
        # Binary: 1 = pass (correct), 0 = fail (incorrect)
        if judge_type == "classifier":
            human_pass = gt.get("category_correct", False) and gt.get("urgency_correct", False)
        elif judge_type == "impact_assessor":
            # Impact is "correct" if quality >= 2 (acceptable or good)
            human_pass = gt.get("impact_assessment_quality", 2) >= 2
        elif judge_type == "resource_matcher":
            human_pass = gt.get("resource_assignment_correct", False)
        elif judge_type == "response_drafter":
            # Response is "correct" if appropriateness >= 2
            human_pass = gt.get("response_appropriateness", 2) >= 2
        else:
            human_pass = False

        judge_pass = judge_score == 1
        return 1.0 if human_pass == judge_pass else 0.0
    else:
        # Three-point: compare to quality score (1-3)
        if judge_type == "classifier":
            human_score = gt.get("reasoning_quality", 2)
        elif judge_type == "impact_assessor":
            human_score = gt.get("impact_assessment_quality", 2)
        elif judge_type == "resource_matcher":
            human_score = gt.get("resource_assignment_quality", 2)
        elif judge_type == "response_drafter":
            human_score = gt.get("response_appropriateness", 2)
        else:
            human_score = 2

        # Exact match = 1.0, off by 1 = 0.5, off by 2 = 0.0
        diff = abs(judge_score - human_score)
        if diff == 0:
            return 1.0
        elif diff == 1:
            return 0.5
        else:
            return 0.0


def format_judge_prompt(
    judge_type: str,
    prompt_style: str,
    scale: str,
    gt: dict,
    agent_output: dict,
) -> str:
    """Format the judge prompt based on judge type and agent output."""
    prompts = JUDGE_PROMPTS.get(judge_type, CLASSIFIER_PROMPTS)
    template = prompts.get((prompt_style, scale), prompts[("direct", "three_point")])

    if judge_type == "classifier":
        return template.format(
            incident=gt["description"],
            category=agent_output.get("category", "unknown"),
            urgency=agent_output.get("urgency", 3),
            reasoning=agent_output.get("reasoning", ""),
        )
    elif judge_type == "impact_assessor":
        return template.format(
            incident=gt["description"],
            impact_score=agent_output.get("impact_score", 5),
            affected_users=agent_output.get("affected_users_estimate", "unknown"),
            blast_radius=agent_output.get("blast_radius", "unknown"),
            reasoning=agent_output.get("reasoning", ""),
        )
    elif judge_type == "resource_matcher":
        primary = agent_output.get("primary_responder", {})
        return template.format(
            category=agent_output.get("category", "unknown"),
            urgency=agent_output.get("urgency", 3),
            responder_name=primary.get("name", "unknown"),
            responder_skills=", ".join(primary.get("skills", [])),
            match_score=primary.get("match_score", 0),
            sla_hours=agent_output.get("sla_target_hours", 24),
            sla_met=agent_output.get("sla_met", False),
        )
    elif judge_type == "response_drafter":
        # Get first communication if available
        comms = agent_output.get("communications", [{}])
        first_comm = comms[0] if comms else {}
        return template.format(
            incident=gt["description"],
            urgency=agent_output.get("urgency", 3),
            audience=first_comm.get("audience", "unknown"),
            subject=first_comm.get("subject", ""),
            body=first_comm.get("body", "")[:500],
        )
    else:
        raise ValueError(f"Unknown judge type: {judge_type}")


def run_judge_config(
    judge_type: str,
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

    all_scores = []
    all_latencies = []
    json_successes = 0
    human_agreements = []
    total_calls = 0

    # Create traced evaluation function
    @add_tracing(name=f"{judge_type}_judge_evaluation", autolog_frameworks=[provider], evaluator=judge_evaluator)
    def evaluate_agent_output(gt: dict, agent_output: dict) -> dict:
        prompt = format_judge_prompt(judge_type, prompt_style, scale, gt, agent_output)
        result, latency = call_judge(client, provider, model_name, prompt, temperature)

        output = {
            "json_valid": result.get("_valid", False),
            "latency_ms": latency,
            "score": result.get("score"),
        }

        # Compute human agreement using judge-specific ground truth
        if output["score"] is not None:
            agreement = compute_human_agreement(output["score"], gt, scale, judge_type)
            output["human_match"] = agreement

        return output

    # Run evaluations
    for gt, agent_output in zip(ground_truth_data, agent_outputs):
        for _ in range(runs_per_config):
            result = evaluate_agent_output(gt, agent_output)
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


def save_best_config(best_result: dict, vertical: str, judge_type: str, batch_id: str):
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

    # Add or update optimized_judge section (nested by judge_type and vertical)
    if "optimized_judge" not in judges_config:
        judges_config["optimized_judge"] = {}

    if judge_type not in judges_config["optimized_judge"]:
        judges_config["optimized_judge"][judge_type] = {}

    judges_config["optimized_judge"][judge_type][vertical] = {
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


def submit_best_config_job(
    best_configs: Dict[str, dict],
    vertical: str,
    batch_id: str,
) -> Optional[str]:
    """Submit a Domino Job to register the best judge configurations.

    Returns the job ID if successful, None otherwise.
    """
    try:
        from domino import Domino

        # Get project info
        project_owner = os.environ.get("DOMINO_PROJECT_OWNER", "")
        project_name = os.environ.get("DOMINO_PROJECT_NAME", "")

        if not project_owner or not project_name:
            print("WARNING: DOMINO_PROJECT_OWNER or DOMINO_PROJECT_NAME not set. Skipping job submission.")
            return None

        project = f"{project_owner}/{project_name}"
        domino = Domino(project)

        # Build command to run a simple registration script
        # The job will load the best configs from judges.yaml and create a final MLflow run
        job_title = f"BEST-JUDGES-{vertical}-{batch_id}"

        # Call the dedicated registration script with arguments
        command = f"python scripts/register_best_judges.py --vertical {vertical} --batch-id {batch_id}"

        result = domino.job_start(command=command, title=job_title)

        job_id = None
        if isinstance(result, dict):
            job_id = result.get("id") or result.get("jobId") or result.get("runId")

        if job_id:
            # Build job URL
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
        else:
            print(f"Job submitted but no ID returned: {result}")
            return None

    except ImportError:
        print("WARNING: python-domino not installed. Skipping job submission.")
        return None
    except Exception as e:
        print(f"WARNING: Failed to submit Domino Job: {e}")
        return None


def run_best_config_validation(
    best_result: dict,
    ground_truth_data: List[dict],
    agent_outputs: List[dict],
    models_config: dict,
    agents_config: dict,
    vertical: str,
    judge_type: str,
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
        mlflow.set_tag("mlflow.runName", f"BEST-{judge_type}-{vertical}-{config_name}")
        mlflow.set_tag("model", model)
        mlflow.set_tag("batch_id", batch_id)
        mlflow.set_tag("best_parameters", "true")  # Key tag for identifying best config
        mlflow.set_tag("vertical", vertical)
        mlflow.set_tag("judge_type", judge_type)

        mlflow.log_param("judge_model", model)
        mlflow.log_param("temperature", temp)
        mlflow.log_param("prompt_style", prompt_style)
        mlflow.log_param("scale", scale)
        mlflow.log_param("vertical", vertical)
        mlflow.log_param("judge_type", judge_type)
        mlflow.log_param("is_best", True)

        # Run the config one more time for validation
        metrics = run_judge_config(
            judge_type=judge_type,
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


def generate_agent_outputs(
    judge_type: str,
    incidents: List[Incident],
    ground_truth_data: List[dict],
    client,
    agents_config: dict,
) -> List[dict]:
    """Generate agent outputs for the specified judge type."""
    outputs = []

    for incident, gt in zip(incidents, ground_truth_data):
        try:
            if judge_type == "classifier":
                classification = classify_incident(client, "openai", "gpt-4o-mini", incident, agents_config)
                outputs.append({
                    "category": classification.category.value if hasattr(classification.category, 'value') else str(classification.category),
                    "urgency": classification.urgency,
                    "reasoning": classification.reasoning,
                })

            elif judge_type == "impact_assessor":
                # First classify, then assess impact
                classification = classify_incident(client, "openai", "gpt-4o-mini", incident, agents_config)
                impact = assess_impact(client, "openai", "gpt-4o-mini", incident, classification, agents_config)
                outputs.append({
                    "impact_score": impact.impact_score,
                    "affected_users_estimate": impact.affected_users_estimate,
                    "blast_radius": impact.blast_radius,
                    "reasoning": impact.reasoning,
                })

            elif judge_type == "resource_matcher":
                # Classify, assess impact, then match resources
                classification = classify_incident(client, "openai", "gpt-4o-mini", incident, agents_config)
                impact = assess_impact(client, "openai", "gpt-4o-mini", incident, classification, agents_config)
                resources = match_resources(client, "openai", "gpt-4o-mini", classification, impact, agents_config)
                primary = resources.primary_responder
                outputs.append({
                    "category": classification.category.value if hasattr(classification.category, 'value') else str(classification.category),
                    "urgency": classification.urgency,
                    "primary_responder": {
                        "name": primary.name,
                        "skills": primary.skills,
                        "match_score": primary.match_score,
                    },
                    "sla_target_hours": resources.sla_target_hours,
                    "sla_met": resources.sla_met,
                })

            elif judge_type == "response_drafter":
                # Full pipeline: classify, assess, match, draft
                classification = classify_incident(client, "openai", "gpt-4o-mini", incident, agents_config)
                impact = assess_impact(client, "openai", "gpt-4o-mini", incident, classification, agents_config)
                resources = match_resources(client, "openai", "gpt-4o-mini", classification, impact, agents_config)
                response = draft_response(client, "openai", "gpt-4o-mini", incident, classification, impact, resources, agents_config)
                comms = [{"audience": c.audience, "subject": c.subject, "body": c.body} for c in response.communications]
                outputs.append({
                    "urgency": classification.urgency,
                    "communications": comms,
                })

        except Exception as e:
            # Default fallback for each type
            if judge_type == "classifier":
                outputs.append({"category": "unknown", "urgency": 3, "reasoning": f"Error: {e}"})
            elif judge_type == "impact_assessor":
                outputs.append({"impact_score": 5, "affected_users_estimate": "unknown", "blast_radius": "unknown", "reasoning": f"Error: {e}"})
            elif judge_type == "resource_matcher":
                outputs.append({"category": "unknown", "urgency": 3, "primary_responder": {"name": "unknown", "skills": [], "match_score": 0}, "sla_target_hours": 24, "sla_met": False})
            elif judge_type == "response_drafter":
                outputs.append({"urgency": 3, "communications": [{"audience": "unknown", "subject": "", "body": f"Error: {e}"}]})

    return outputs


def run_judge_experiment_for_type(
    judge_type: str,
    args,
    ground_truth_data: List[dict],
    incidents: List[Incident],
    param_grid: List[tuple],
    models_config: dict,
    agents_config: dict,
    agents_path: str,
    batch_id: str,
    experiment_name: str,
    runs_per_config: int,
):
    """Run the judge experiment for a specific judge type."""
    print()
    print("=" * 70)
    print(f"JUDGE: {judge_type.upper()}")
    print("=" * 70)

    # Generate agent outputs for this judge type
    print(f"Generating {judge_type} agent outputs...")
    baseline_client = get_client("openai")
    agent_outputs = generate_agent_outputs(
        judge_type=judge_type,
        incidents=incidents,
        ground_truth_data=ground_truth_data,
        client=baseline_client,
        agents_config=agents_config,
    )
    print(f"Generated {len(agent_outputs)} outputs\n")

    # Run experiments
    mlflow.set_experiment(experiment_name)
    results = []

    for i, (model, temp, style, scale) in enumerate(param_grid, 1):
        config_name = f"{model}-t{temp}-{style}-{scale}"
        print(f"[{i}/{len(param_grid)}] {config_name}")

        with DominoRun(agent_config_path=agents_path) as run:
            mlflow.set_tag("mlflow.runName", f"{judge_type}-{config_name}")
            mlflow.set_tag("model", model)
            mlflow.set_tag("batch_id", batch_id)
            mlflow.set_tag("judge_type", judge_type)

            mlflow.log_param("judge_model", model)
            mlflow.log_param("temperature", temp)
            mlflow.log_param("prompt_style", style)
            mlflow.log_param("scale", scale)
            mlflow.log_param("vertical", args.vertical)
            mlflow.log_param("judge_type", judge_type)

            try:
                metrics = run_judge_config(
                    judge_type=judge_type,
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
    print("\n" + "-" * 70)
    print(f"RESULTS SUMMARY ({judge_type})")
    print("-" * 70)

    # Thresholds for filtering
    MIN_JSON_PARSE_RATE = 0.95
    MAX_CONSISTENCY_STD = 0.35

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
        ranked = sorted(qualified, key=lambda x: -x["metrics"]["human_agreement"])
        for i, r in enumerate(ranked[:5], 1):
            m = r["metrics"]
            print(f"  {i}. {r['config']}")
            print(f"     agreement={m['human_agreement']:.1%}, std={m['consistency_std']:.3f}, json={m['json_parse_rate']:.0%}")

        # Best config
        best = ranked[0]
        print()
        print(f"BEST {judge_type.upper()} CONFIGURATION: {best['config']}")
        print(f"  Human agreement: {best['metrics']['human_agreement']:.1%}")

        # Save best config
        save_best_config(best, args.vertical, judge_type, batch_id)

        # Run final validation
        run_best_config_validation(
            best_result=best,
            ground_truth_data=ground_truth_data,
            agent_outputs=agent_outputs,
            models_config=models_config,
            agents_config=agents_config,
            vertical=args.vertical,
            judge_type=judge_type,
            batch_id=batch_id,
            experiment_name=experiment_name,
        )
        return best
    else:
        print("WARNING: No configurations met the quality thresholds.")
        if all_valid:
            print("Top by human agreement (unfiltered):")
            for r in sorted(all_valid, key=lambda x: -x["metrics"]["human_agreement"])[:3]:
                m = r["metrics"]
                print(f"  {r['config']}: agreement={m['human_agreement']:.1%}")
        return None


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

    # Determine which judges to run
    if args.judge == "all":
        judge_types = ["classifier", "impact_assessor", "resource_matcher", "response_drafter"]
    else:
        judge_types = [args.judge]

    # Load ground truth for vertical
    ground_truth_data = load_ground_truth(args.vertical)
    if not ground_truth_data:
        print(f"ERROR: No ground truth found for vertical '{args.vertical}'")
        sys.exit(1)

    # Limit tickets if specified
    if args.max_tickets:
        ground_truth_data = ground_truth_data[:args.max_tickets]

    # Convert to incidents
    incidents = [ground_truth_to_incident(gt) for gt in ground_truth_data]

    # Build parameter grid
    param_grid = list(itertools.product(models, temperatures, prompt_styles, scales))

    # Experiment setup with batch ID
    username = os.environ.get("DOMINO_USER_NAME", os.environ.get("USER", "demo_user"))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_id = f"{username}-{timestamp}"
    experiment_name = f"judge-optimization-{args.vertical}-{username}"

    print("=" * 70)
    print("JUDGE OPTIMIZATION EXPERIMENT")
    print("=" * 70)
    print(f"Experiment: {experiment_name}")
    print(f"Batch ID: {batch_id}")
    print(f"Vertical: {args.vertical}")
    print(f"Judge types: {', '.join(judge_types)}")
    print(f"Incidents: {len(ground_truth_data)} (with ground truth)")
    print(f"Configurations per judge: {len(param_grid)}")
    print(f"Runs/config: {runs_per_config}")

    if args.dry_run:
        print("\n[DRY RUN] Configurations:")
        for i, (m, t, p, s) in enumerate(param_grid[:10], 1):
            print(f"  {i}. {m}, temp={t}, {p}, {s}")
        if len(param_grid) > 10:
            print(f"  ... and {len(param_grid) - 10} more")
        print(f"\nWould run for judges: {', '.join(judge_types)}")
        return

    # Run experiment for each judge type
    best_configs = {}
    for judge_type in judge_types:
        best = run_judge_experiment_for_type(
            judge_type=judge_type,
            args=args,
            ground_truth_data=ground_truth_data,
            incidents=incidents,
            param_grid=param_grid,
            models_config=models_config,
            agents_config=agents_config,
            agents_path=agents_path,
            batch_id=batch_id,
            experiment_name=experiment_name,
            runs_per_config=runs_per_config,
        )
        if best:
            best_configs[judge_type] = best

    # Create a combined final run with all best configurations
    if best_configs and len(best_configs) > 1:
        print("\n" + "=" * 70)
        print("CREATING COMBINED BEST-ALL-JUDGES RUN")
        print("=" * 70)

        mlflow.set_experiment(experiment_name)

        with DominoRun(agent_config_path=agents_path) as run:
            mlflow.set_tag("mlflow.runName", f"BEST-ALL-JUDGES-{args.vertical}")
            mlflow.set_tag("batch_id", batch_id)
            mlflow.set_tag("best_parameters", "true")
            mlflow.set_tag("all_judges", "true")
            mlflow.set_tag("vertical", args.vertical)

            # Log best config for each judge
            for jt, best in best_configs.items():
                config_name = best["config"]
                metrics = best["metrics"]

                # Parse config
                t_index = config_name.find("-t0.") if "-t0." in config_name else config_name.find("-t0")
                if t_index == -1:
                    t_index = config_name.find("-t")
                model = config_name[:t_index]
                remainder = config_name[t_index+1:]
                parts = remainder.split("-")
                temp = float(parts[0].replace("t", ""))
                prompt_style = parts[1]
                scale = parts[2]

                # Log params and metrics with judge prefix
                mlflow.log_param(f"{jt}_model", model)
                mlflow.log_param(f"{jt}_temperature", temp)
                mlflow.log_param(f"{jt}_prompt_style", prompt_style)
                mlflow.log_param(f"{jt}_scale", scale)
                mlflow.log_metric(f"{jt}_agreement", metrics.get("human_agreement", 0))
                mlflow.log_metric(f"{jt}_std", metrics.get("consistency_std", 0))
                mlflow.log_metric(f"{jt}_json_rate", metrics.get("json_parse_rate", 0))

            # Log the full config as an artifact
            best_config_summary = {
                "batch_id": batch_id,
                "vertical": args.vertical,
                "timestamp": datetime.now().isoformat(),
                "judges": {}
            }
            for jt, best in best_configs.items():
                config_name = best["config"]
                t_index = config_name.find("-t0.") if "-t0." in config_name else config_name.find("-t0")
                if t_index == -1:
                    t_index = config_name.find("-t")
                model = config_name[:t_index]
                remainder = config_name[t_index+1:]
                parts = remainder.split("-")
                temp = float(parts[0].replace("t", ""))
                prompt_style = parts[1]
                scale = parts[2]

                best_config_summary["judges"][jt] = {
                    "model": model,
                    "temperature": temp,
                    "prompt_style": prompt_style,
                    "scale": scale,
                    "metrics": {
                        "human_agreement": round(best["metrics"].get("human_agreement", 0), 3),
                        "consistency_std": round(best["metrics"].get("consistency_std", 0), 3),
                        "json_parse_rate": round(best["metrics"].get("json_parse_rate", 0), 3),
                    }
                }

            # Save to temp file and log as artifact
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(best_config_summary, f, default_flow_style=False)
                temp_path = f.name
            mlflow.log_artifact(temp_path, "best_configs")
            os.unlink(temp_path)

            print(f"Created combined run: BEST-ALL-JUDGES-{args.vertical}")
            print("Best configurations logged:")
            for jt, best in best_configs.items():
                print(f"  {jt}: {best['config']} (agreement={best['metrics']['human_agreement']:.1%})")

    # Submit Domino Job if requested
    if args.submit_job and best_configs:
        print("\n" + "=" * 70)
        print("SUBMITTING DOMINO JOB FOR BEST CONFIG REGISTRATION")
        print("=" * 70)
        submit_best_config_job(
            best_configs=best_configs,
            vertical=args.vertical,
            batch_id=batch_id,
        )

    # Final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Batch ID: {batch_id}")
    print(f"Vertical: {args.vertical}")
    print()
    if best_configs:
        print("Best configurations found:")
        for jt, best in best_configs.items():
            print(f"  {jt}: {best['config']} (agreement={best['metrics']['human_agreement']:.1%})")
    else:
        print("WARNING: No configurations met the quality thresholds for any judge.")
        print("Consider relaxing thresholds or reviewing judge prompts.")

    print(f"\nExperiment complete: {experiment_name}")


if __name__ == "__main__":
    main()
