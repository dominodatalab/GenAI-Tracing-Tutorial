"""
Experiment Utilities for Agent Experimentation

Provides utilities for:
- Parameter grid generation
- Configuration overrides
- Few-shot example formatting
- Prompt building with variants
- MLflow parameter logging
"""

import copy
import itertools
import random
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import yaml


def load_yaml_config(path: str) -> dict:
    """Load a YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def generate_parameter_grid(grid_config: dict, agents: List[str] = None) -> List[dict]:
    """
    Generate all parameter combinations from a grid configuration.

    Args:
        grid_config: Dict with agent names as keys and param dicts as values.
                    Each param can be a single value or list of values.
        agents: Optional list of agents to include. If None, uses all.

    Returns:
        List of parameter combination dicts, each with structure:
        {
            "classifier": {"temperature": 0.1, "few_shot_count": 2, ...},
            "impact_assessor": {"temperature": 0.3, ...},
            ...
        }
    """
    if agents is None:
        agents = ["classifier", "impact_assessor", "resource_matcher", "response_drafter"]

    # Build parameter lists for each agent
    agent_param_lists = {}
    for agent in agents:
        if agent not in grid_config:
            continue

        agent_config = grid_config[agent]
        param_names = []
        param_values = []

        for param_name, values in agent_config.items():
            param_names.append(param_name)
            # Ensure values is a list
            if not isinstance(values, list):
                values = [values]
            param_values.append(values)

        # Generate all combinations for this agent
        agent_combinations = []
        for combo in itertools.product(*param_values):
            agent_combinations.append(dict(zip(param_names, combo)))

        agent_param_lists[agent] = agent_combinations

    # Generate cross-product of all agents
    all_combinations = []
    agent_names = list(agent_param_lists.keys())

    if not agent_names:
        return []

    agent_combos = [agent_param_lists[name] for name in agent_names]

    for combo in itertools.product(*agent_combos):
        param_set = {}
        for i, agent_name in enumerate(agent_names):
            param_set[agent_name] = combo[i]
        all_combinations.append(param_set)

    return all_combinations


def sample_parameter_grid(
    grid: List[dict],
    sample_size: int,
    seed: int = None
) -> List[dict]:
    """
    Randomly sample from a parameter grid.

    Args:
        grid: Full parameter grid from generate_parameter_grid
        sample_size: Number of samples to take
        seed: Random seed for reproducibility

    Returns:
        Sampled subset of parameter combinations
    """
    if seed is not None:
        random.seed(seed)

    if sample_size >= len(grid):
        return grid

    return random.sample(grid, sample_size)


def apply_agent_overrides(config: dict, overrides: dict) -> dict:
    """
    Apply parameter overrides to agent configurations.

    Args:
        config: Base configuration dict with "agents" key
        overrides: Dict of agent name -> param overrides

    Returns:
        New config with overrides applied (deep copy)
    """
    config = copy.deepcopy(config)

    for agent_name, params in overrides.items():
        if agent_name in config.get("agents", {}):
            for param_name, value in params.items():
                config["agents"][agent_name][param_name] = value

    return config


def format_few_shot_examples(
    examples: List[dict],
    count: int,
    include_reasoning: bool = True
) -> str:
    """
    Format few-shot examples for prompt injection.

    Args:
        examples: List of example dicts with 'input' and 'output' keys
        count: Number of examples to include (0 for none)
        include_reasoning: Whether to include reasoning/notes if available

    Returns:
        Formatted string of examples, or empty string if count=0
    """
    if count == 0 or not examples:
        return ""

    selected = examples[:count]
    formatted_parts = ["Here are some examples:\n"]

    for i, ex in enumerate(selected, 1):
        formatted_parts.append(f"Example {i}:")
        formatted_parts.append(f"Input: {ex['input']}")
        formatted_parts.append(f"Output: {ex['output']}")
        if include_reasoning and 'note' in ex:
            formatted_parts.append(f"Note: {ex['note']}")
        formatted_parts.append("")

    return "\n".join(formatted_parts)


def build_prompt_with_variants(
    base_prompt: str,
    prompt_variants: dict,
    variant_name: str,
    few_shot_examples: List[dict] = None,
    few_shot_count: int = 0,
    chain_of_thought: bool = False,
    length_guidance: str = None
) -> str:
    """
    Construct final prompt with variant, few-shot examples, and modifiers.

    Args:
        base_prompt: Original prompt template (may be ignored if variant exists)
        prompt_variants: Dict of variant_name -> prompt template
        variant_name: Which variant to use (or "default" for base_prompt)
        few_shot_examples: List of examples to potentially include
        few_shot_count: Number of examples to include
        chain_of_thought: Whether to add chain-of-thought prefix
        length_guidance: One of "concise", "standard", "detailed" or None

    Returns:
        Constructed prompt string
    """
    # Select base prompt
    if variant_name and variant_name != "default" and variant_name in prompt_variants:
        prompt = prompt_variants[variant_name]
    else:
        prompt = base_prompt

    parts = []

    # Add chain of thought prefix if requested
    if chain_of_thought and "chain_of_thought_prefix" in prompt_variants:
        parts.append(prompt_variants["chain_of_thought_prefix"])

    # Add few-shot examples
    if few_shot_count > 0 and few_shot_examples:
        parts.append(format_few_shot_examples(few_shot_examples, few_shot_count))

    # Add main prompt
    parts.append(prompt)

    # Add length guidance suffix
    if length_guidance and "length_guidance" in prompt_variants:
        guidance = prompt_variants["length_guidance"].get(length_guidance, "")
        if guidance:
            parts.append(guidance)

    return "\n".join(parts)


def log_experiment_params(param_set: dict, prefix: str = ""):
    """
    Log all experiment parameters to MLflow as params and tags.

    Args:
        param_set: Dict of agent_name -> param dict
        prefix: Optional prefix for parameter names
    """
    for agent_name, params in param_set.items():
        for param_name, value in params.items():
            full_name = f"{prefix}{agent_name}.{param_name}" if prefix else f"{agent_name}.{param_name}"

            # Log as MLflow param (for filtering)
            try:
                mlflow.log_param(full_name, value)
            except Exception:
                pass  # Params may already be logged

            # Also log as tag (always works, good for filtering)
            mlflow.set_tag(full_name, str(value))


def get_model_for_agent(param_set: dict, agent_name: str, models_config: dict) -> Tuple[str, str]:
    """
    Get the provider and model name for an agent from the parameter set.

    Args:
        param_set: Current parameter set
        agent_name: Name of the agent
        models_config: Model configuration dict

    Returns:
        Tuple of (provider, model_name)
    """
    agent_params = param_set.get(agent_name, {})
    model_key = agent_params.get("model", "openai")

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


def create_experiment_signature(param_set: dict) -> str:
    """
    Create a unique signature string for a parameter combination.

    Useful for naming runs and tracking results.

    Args:
        param_set: Parameter combination dict

    Returns:
        String signature like "clf_t0.2_fs2_ia_t0.4_cot1_..."
    """
    parts = []

    agent_abbrevs = {
        "classifier": "clf",
        "impact_assessor": "ia",
        "resource_matcher": "rm",
        "response_drafter": "rd"
    }

    param_abbrevs = {
        "temperature": "t",
        "few_shot_count": "fs",
        "system_prompt_variant": "pv",
        "chain_of_thought": "cot",
        "model": "m",
        "tone": "tone",
        "length_guidance": "len"
    }

    for agent_name, params in param_set.items():
        agent_abbrev = agent_abbrevs.get(agent_name, agent_name[:3])
        param_parts = []

        for param_name, value in sorted(params.items()):
            param_abbrev = param_abbrevs.get(param_name, param_name[:3])

            # Format value
            if isinstance(value, bool):
                value_str = "1" if value else "0"
            elif isinstance(value, float):
                value_str = f"{value:.1f}".replace(".", "")
            else:
                value_str = str(value)[:3]

            param_parts.append(f"{param_abbrev}{value_str}")

        parts.append(f"{agent_abbrev}_{'_'.join(param_parts)}")

    return "__".join(parts)


def rank_results(results: List[dict], metric_key: str = "combined_quality_score") -> List[dict]:
    """
    Rank experiment results by a metric.

    Args:
        results: List of result dicts with metrics
        metric_key: Key to sort by (higher is better)

    Returns:
        Results sorted by metric with 'rank' field added
    """
    sorted_results = sorted(
        results,
        key=lambda x: x.get("metrics", {}).get(metric_key, 0),
        reverse=True
    )

    for i, result in enumerate(sorted_results, 1):
        result["rank"] = i

    return sorted_results


def summarize_best_params(results: List[dict]) -> dict:
    """
    Extract best parameters from ranked results.

    Args:
        results: Ranked results from rank_results

    Returns:
        Dict with best parameters per agent
    """
    if not results:
        return {}

    best_result = results[0]
    return best_result.get("parameters", {})
