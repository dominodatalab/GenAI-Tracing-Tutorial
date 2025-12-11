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
        f"python /mnt/code/app/run_triage_app.py "
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


def get_job_logs(job_id: str, log_type: str = "stdout") -> Optional[str]:
    """
    Retrieve logs for a completed job.

    Args:
        job_id: The Domino job ID.
        log_type: Type of log to retrieve ('stdout' or 'stderr').

    Returns:
        Log content as a string, or None if unavailable.
    """
    try:
        domino = get_domino_client()
        # Use the runs_stdout method from python-domino SDK
        if log_type == "stdout":
            logs = domino.runs_stdout(job_id)
        else:
            logs = domino.runs_stderr(job_id)
        return logs
    except Exception as e:
        # Return None if logs aren't available yet or job doesn't exist
        return None


def parse_triage_results(stdout: str) -> Dict[str, Any]:
    """
    Parse the stdout from a triage job to extract structured results.

    Args:
        stdout: The raw stdout content from the job.

    Returns:
        Dictionary containing parsed results with keys:
        - header: Job header information
        - summary_table: List of dicts with ticket results
        - sample_communication: Sample communication details
        - status: Overall job status
    """
    result = {
        "header": {},
        "summary_table": [],
        "sample_communication": None,
        "evaluations_added": 0,
        "status": "unknown",
        "raw": stdout
    }

    if not stdout:
        return result

    lines = stdout.strip().split("\n")

    # Parse header section
    for line in lines:
        if line.startswith("Config Path:"):
            result["header"]["config_path"] = line.split(":", 1)[1].strip()
        elif line.startswith("Provider:"):
            result["header"]["provider"] = line.split(":", 1)[1].strip()
        elif line.startswith("Vertical:"):
            result["header"]["vertical"] = line.split(":", 1)[1].strip()
        elif line.startswith("Model:"):
            result["header"]["model"] = line.split(":", 1)[1].strip()
        elif line.startswith("Processing") and "incidents from" in line:
            parts = line.split()
            try:
                result["header"]["num_incidents"] = int(parts[1])
            except (ValueError, IndexError):
                pass
        elif line.startswith("Experiment:"):
            result["header"]["experiment"] = line.split(":", 1)[1].strip()
        elif line.startswith("Run:"):
            result["header"]["run"] = line.split(":", 1)[1].strip()

    # Parse results summary table
    in_summary = False
    summary_lines = []
    for i, line in enumerate(lines):
        if "RESULTS SUMMARY" in line:
            in_summary = True
            continue
        if in_summary:
            if line.startswith("="):
                if summary_lines:  # End of summary
                    break
                continue
            if line.strip():
                summary_lines.append(line)

    # Parse summary table (header + data rows)
    if len(summary_lines) >= 2:
        # First line is header
        headers = summary_lines[0].split()
        for data_line in summary_lines[1:]:
            # Handle the data - columns are: Ticket, Category, Urgency, Impact, Responder, SLA Met
            parts = data_line.split()
            if len(parts) >= 6:
                # Responder name might have spaces, SLA Met is last two words
                sla_met = parts[-1]  # True/False
                responder_parts = []
                # Work backwards to find where responder name starts
                idx = len(parts) - 2  # Skip SLA Met value
                while idx >= 4:  # After Ticket, Category, Urgency, Impact
                    responder_parts.insert(0, parts[idx])
                    idx -= 1

                result["summary_table"].append({
                    "ticket": parts[0],
                    "category": parts[1],
                    "urgency": int(parts[2]) if parts[2].isdigit() else parts[2],
                    "impact": float(parts[3]) if parts[3].replace(".", "").isdigit() else parts[3],
                    "responder": " ".join(responder_parts) if responder_parts else parts[4],
                    "sla_met": sla_met == "True"
                })

    # Parse sample communication
    in_communication = False
    comm_section = []
    for line in lines:
        if "SAMPLE COMMUNICATION" in line:
            in_communication = True
            continue
        if in_communication:
            if line.startswith("=") and len(comm_section) > 0:
                break
            if line.startswith("="):
                continue
            comm_section.append(line)

    if comm_section:
        result["sample_communication"] = {
            "raw": "\n".join(comm_section),
            "audiences": []
        }
        # Parse individual audience communications
        current_audience = None
        current_content = []
        for line in comm_section:
            if line.startswith("---") and line.endswith("---"):
                if current_audience and current_content:
                    result["sample_communication"]["audiences"].append({
                        "audience": current_audience,
                        "content": "\n".join(current_content)
                    })
                current_audience = line.strip("- ").strip()
                current_content = []
            elif current_audience:
                current_content.append(line)
        # Add last audience
        if current_audience and current_content:
            result["sample_communication"]["audiences"].append({
                "audience": current_audience,
                "content": "\n".join(current_content)
            })

    # Parse evaluations
    for line in lines:
        if "Added evaluations to" in line:
            parts = line.split()
            try:
                result["evaluations_added"] = int(parts[3])
            except (ValueError, IndexError):
                pass

    # Determine status
    if "Done!" in stdout:
        result["status"] = "completed"
    elif "ERROR" in stdout or "Traceback" in stdout:
        result["status"] = "failed"
    elif result["summary_table"]:
        result["status"] = "completed"
    else:
        result["status"] = "running"

    return result


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
