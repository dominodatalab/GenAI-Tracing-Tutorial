"""
Domino API client utilities for TriageFlow App.

Handles job submission and status tracking via the python-domino SDK.
"""

import os
from datetime import datetime
from typing import Dict, Any, Optional

from domino import Domino


def get_project_info() -> Dict[str, str]:
    """Get current Domino project information from environment."""
    return {
        "owner": os.environ.get("DOMINO_PROJECT_OWNER", ""),
        "name": os.environ.get("DOMINO_PROJECT_NAME", ""),
        "user": os.environ.get("DOMINO_USER_NAME", os.environ.get("USER", "unknown")),
    }


def get_domino_client() -> Domino:
    """
    Initialize and return a Domino client for the current project.

    The client uses environment variables for authentication:
    - DOMINO_API_HOST: API server URL
    - DOMINO_USER_API_KEY or DOMINO_TOKEN_FILE: Authentication credentials
    """
    project_info = get_project_info()
    project = f"{project_info['owner']}/{project_info['name']}"
    return Domino(project)


def start_triage_job(
    config_path: str,
    provider: str,
    vertical: str,
    num_tickets: int,
    title: Optional[str] = None
) -> Dict[str, Any]:
    """
    Start a TriageFlow triage job.

    Args:
        config_path: Full path to the config YAML file in the dataset.
        provider: LLM provider ('openai' or 'anthropic').
        vertical: Industry vertical for sample incidents.
        num_tickets: Number of tickets to process (0 for all).
        title: Optional job title. Auto-generated if not provided.

    Returns:
        Dictionary containing job submission result with 'id' and other metadata.
    """
    if not title:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        title = f"TriageFlow-{vertical}-{timestamp}"

    command = (
        f"python /mnt/code/run_triage_app.py "
        f"--config-path {config_path} "
        f"--provider {provider} "
        f"--vertical {vertical} "
        f"-n {num_tickets}"
    )

    domino = get_domino_client()
    result = domino.job_start(command=command, title=title)

    return result


def get_job_status(job_id: str) -> Dict[str, Any]:
    """
    Get the status of a job.

    Args:
        job_id: The Domino job ID.

    Returns:
        Dictionary containing job status information.
    """
    domino = get_domino_client()
    # The python-domino SDK provides job status via runs_list or similar
    # For now, we return a basic structure
    try:
        # Note: The exact method depends on python-domino version
        # This is a placeholder that should work with most versions
        return {"job_id": job_id, "status": "submitted"}
    except Exception as e:
        return {"job_id": job_id, "status": "unknown", "error": str(e)}


def get_domino_host() -> str:
    """
    Get the Domino host URL from environment or session.

    Checks multiple environment variables and cleans up API suffixes.
    """
    # Try various environment variables that might contain the host
    domino_host = (
        os.environ.get("DOMINO_USER_HOST", "") or
        os.environ.get("DOMINO_API_HOST", "") or
        ""
    )

    # Clean up common API path suffixes
    for suffix in ["/v4/api", "/api/api", "/api", "/v4"]:
        if domino_host.endswith(suffix):
            domino_host = domino_host[:-len(suffix)]
            break

    return domino_host.rstrip("/")


def build_job_url(job_id: str, domino_host: str = None) -> str:
    """
    Build the URL to view a job in the Domino UI.

    Args:
        job_id: The Domino job ID.
        domino_host: Optional override for Domino host URL.
                     If not provided, attempts to detect from environment.

    Returns:
        URL string to the job in Domino, or empty string if host unknown.
    """
    if not job_id:
        return ""

    project_info = get_project_info()

    if not domino_host:
        domino_host = get_domino_host()

    if not domino_host:
        # Cannot construct URL without host
        return ""

    # Clean the host URL
    domino_host = domino_host.rstrip("/")

    # URL pattern: {host}/jobs/{owner}/{project}/{job_id}
    return f"{domino_host}/jobs/{project_info['owner']}/{project_info['name']}/{job_id}"


def create_job_history_entry(
    job_result: Dict[str, Any],
    config_path: str,
    provider: str,
    vertical: str,
    num_tickets: int,
    domino_host: str = None
) -> Dict[str, Any]:
    """
    Create a job history entry from job submission result.

    Args:
        job_result: Result from job_start().
        config_path: Path to the config file used.
        provider: LLM provider.
        vertical: Industry vertical.
        num_tickets: Number of tickets.
        domino_host: Optional Domino host URL for building job links.

    Returns:
        Dictionary suitable for saving to job history.
    """
    project_info = get_project_info()

    # Extract job ID from result
    # The structure varies by python-domino version
    job_id = None
    if isinstance(job_result, dict):
        job_id = job_result.get("id") or job_result.get("jobId") or job_result.get("runId")

    return {
        "job_id": job_id,
        "timestamp": datetime.now().isoformat(),
        "config_file": os.path.basename(config_path),
        "config_path": config_path,
        "provider": provider,
        "vertical": vertical,
        "num_tickets": num_tickets,
        "status": "submitted",
        "user": project_info["user"],
        "project_owner": project_info["owner"],
        "project_name": project_info["name"],
        "job_url": build_job_url(job_id, domino_host) if job_id else None
    }
