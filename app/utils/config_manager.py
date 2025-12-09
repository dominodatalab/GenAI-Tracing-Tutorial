"""
Configuration management utilities for TriageFlow App.

Handles loading base configuration and saving timestamped configs to the Domino dataset.
"""

import os
import json
import fcntl
from datetime import datetime
from typing import Dict, List, Optional, Any

import yaml


# Constants
BASE_CONFIG_PATH = "/mnt/code/config.yaml"
EXAMPLE_DATA_PATH = "/mnt/code/example-data"
VERTICALS = ["financial_services", "healthcare", "energy", "public_sector"]
PROVIDERS = ["openai", "anthropic"]


def get_project_name() -> str:
    """Get the current Domino project name."""
    return os.environ.get("DOMINO_PROJECT_NAME", "default")


def get_dataset_path() -> str:
    """Get the path to the Domino project dataset."""
    project_name = get_project_name()
    return f"/mnt/data/{project_name}"


def get_configs_dir() -> str:
    """Get the configs directory in the dataset, creating if needed."""
    path = os.path.join(get_dataset_path(), "configs")
    os.makedirs(path, exist_ok=True)
    return path


def get_job_history_dir() -> str:
    """Get the job history directory in the dataset, creating if needed."""
    path = os.path.join(get_dataset_path(), "job_history")
    os.makedirs(path, exist_ok=True)
    return path


def load_base_config() -> Dict[str, Any]:
    """Load the base configuration from the project."""
    with open(BASE_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any]) -> str:
    """
    Save configuration to dataset with timestamp.

    Returns:
        The full path to the saved config file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"config_{timestamp}.yaml"
    filepath = os.path.join(get_configs_dir(), filename)

    with open(filepath, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return filepath


def load_config(filepath: str) -> Dict[str, Any]:
    """Load a configuration file from the given path."""
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def list_saved_configs() -> List[Dict[str, Any]]:
    """
    List all saved configuration files in the dataset.

    Returns:
        List of dicts with 'filename', 'filepath', and 'timestamp' keys.
    """
    configs_dir = get_configs_dir()
    configs = []

    if os.path.exists(configs_dir):
        for filename in os.listdir(configs_dir):
            if filename.startswith("config_") and filename.endswith(".yaml"):
                filepath = os.path.join(configs_dir, filename)
                # Extract timestamp from filename: config_YYYYMMDD_HHMMSS.yaml
                try:
                    timestamp_str = filename[7:-5]  # Remove "config_" and ".yaml"
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    configs.append({
                        "filename": filename,
                        "filepath": filepath,
                        "timestamp": timestamp
                    })
                except ValueError:
                    continue

    # Sort by timestamp descending (newest first)
    configs.sort(key=lambda x: x["timestamp"], reverse=True)
    return configs


def load_sample_tickets(vertical: str) -> List[Dict[str, Any]]:
    """
    Load sample tickets for a given vertical.

    Args:
        vertical: One of 'financial_services', 'healthcare', 'energy', 'public_sector'

    Returns:
        List of ticket dictionaries.
    """
    import pandas as pd

    filepath = os.path.join(EXAMPLE_DATA_PATH, f"{vertical}.csv")
    if not os.path.exists(filepath):
        return []

    df = pd.read_csv(filepath)
    return df.to_dict("records")


def get_job_history_path() -> str:
    """Get the path to the job history JSON file."""
    return os.path.join(get_job_history_dir(), "history.json")


def load_job_history() -> List[Dict[str, Any]]:
    """
    Load job history from the dataset.

    Returns:
        List of job history entries.
    """
    history_path = get_job_history_path()

    if not os.path.exists(history_path):
        return []

    try:
        with open(history_path, "r") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                return json.load(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except (json.JSONDecodeError, IOError):
        return []


def save_job_to_history(job_info: Dict[str, Any]) -> None:
    """
    Append a job entry to the history file with file locking for concurrency.

    Args:
        job_info: Dictionary containing job details.
    """
    history_path = get_job_history_path()

    # Ensure directory exists
    os.makedirs(os.path.dirname(history_path), exist_ok=True)

    # Use file locking for concurrent access safety
    with open(history_path, "a+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            content = f.read()

            if content:
                try:
                    history = json.loads(content)
                except json.JSONDecodeError:
                    history = []
            else:
                history = []

            history.insert(0, job_info)  # Add newest first

            # Keep only last 100 entries
            history = history[:100]

            f.seek(0)
            f.truncate()
            json.dump(history, f, indent=2, default=str)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def get_default_agent_config(agent_name: str) -> Dict[str, Any]:
    """Get default configuration for a specific agent."""
    config = load_base_config()
    return config.get("agents", {}).get(agent_name, {})


def get_default_tools_config(agent_name: str) -> List[Dict[str, Any]]:
    """Get default tools configuration for a specific agent."""
    config = load_base_config()
    return config.get("tools", {}).get(agent_name, [])


def merge_config(base_config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge updates into base config.

    Args:
        base_config: The base configuration dictionary.
        updates: Updates to apply.

    Returns:
        Merged configuration.
    """
    import copy
    result = copy.deepcopy(base_config)

    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def config_to_yaml_string(config: Dict[str, Any]) -> str:
    """Convert a config dictionary to a YAML string for display."""
    return yaml.dump(config, default_flow_style=False, sort_keys=False)
