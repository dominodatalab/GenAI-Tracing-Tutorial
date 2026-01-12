"""
Guardrails client for input validation and output sanitization.

Integrates with the external Guardrails service to:
- Validate ticket descriptions before processing (prompt injection, toxicity)
- Sanitize final responses (PII redaction)
"""

import os
import json
import requests
import mlflow
from mlflow.entities import SpanType
from dataclasses import dataclass, field
from typing import Optional


GUARDRAILS_URL = "https://genai-llm.domino-eval.com/apps/guardrails"


@dataclass
class GuardrailsResult:
    """Result from guardrails check."""
    passed: bool
    blocked_reason: Optional[str] = None
    sanitized_text: Optional[str] = None
    checks: dict = field(default_factory=dict)
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "blocked_reason": self.blocked_reason,
            "sanitized_text": self.sanitized_text,
            "checks": self.checks,
            "latency_ms": self.latency_ms
        }


def _get_headers() -> dict:
    """Get authentication headers for guardrails API."""
    api_key = os.environ.get("DOMINO_USER_API_KEY")
    if not api_key:
        raise ValueError("DOMINO_USER_API_KEY environment variable not set")
    return {
        "X-Domino-Api-Key": api_key,
        "Content-Type": "application/json"
    }


@mlflow.trace(span_type=SpanType.TOOL, name="guardrails_check_input")
def check_input(text: str) -> GuardrailsResult:
    """
    Validate ticket description before processing.

    Checks for:
    - Prompt injection attempts
    - Toxic/harmful content
    - PII presence (flagged but not blocked)

    Returns:
        GuardrailsResult with passed=False if content should be blocked
    """
    try:
        response = requests.post(
            f"{GUARDRAILS_URL}/check_input",
            headers=_get_headers(),
            json={"text": text},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        return GuardrailsResult(
            passed=data.get("passed", False),
            blocked_reason=data.get("blocked_reason"),
            sanitized_text=data.get("sanitized_text"),
            checks=data.get("checks", {}),
            latency_ms=data.get("latency_ms", 0.0)
        )
    except requests.exceptions.RequestException as e:
        # If guardrails service is unavailable, log and continue
        # (fail-open behavior - can be changed to fail-closed if needed)
        mlflow.log_param("guardrails_input_error", str(e))
        return GuardrailsResult(
            passed=True,
            blocked_reason=None,
            checks={"error": str(e), "service_available": False},
            latency_ms=0.0
        )


@mlflow.trace(span_type=SpanType.TOOL, name="guardrails_check_output")
def check_output(text: str) -> GuardrailsResult:
    """
    Sanitize final response before returning to user.

    Performs:
    - PII redaction (names, emails, phone numbers, SSNs, etc.)
    - Toxicity checking

    Returns:
        GuardrailsResult with sanitized_text containing PII-redacted content
    """
    try:
        response = requests.post(
            f"{GUARDRAILS_URL}/check_output",
            headers=_get_headers(),
            json={"text": text},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        return GuardrailsResult(
            passed=data.get("passed", True),
            blocked_reason=data.get("blocked_reason"),
            sanitized_text=data.get("sanitized_text"),
            checks=data.get("checks", {}),
            latency_ms=data.get("latency_ms", 0.0)
        )
    except requests.exceptions.RequestException as e:
        # If guardrails service is unavailable, return original text
        mlflow.log_param("guardrails_output_error", str(e))
        return GuardrailsResult(
            passed=True,
            blocked_reason=None,
            sanitized_text=text,  # Return original if service unavailable
            checks={"error": str(e), "service_available": False},
            latency_ms=0.0
        )


@mlflow.trace(span_type=SpanType.TOOL, name="guardrails_sanitize")
def sanitize_for_logging(text: str) -> str:
    """
    Redact PII for external logging without full output check.

    Use this for logging incident data to external systems.

    Returns:
        Sanitized text with PII redacted
    """
    try:
        response = requests.post(
            f"{GUARDRAILS_URL}/sanitize",
            headers=_get_headers(),
            json={"text": text},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data.get("sanitized_text", text)
    except requests.exceptions.RequestException:
        return text  # Return original if service unavailable


def health_check() -> bool:
    """Check if guardrails service is available."""
    try:
        api_key = os.environ.get("DOMINO_USER_API_KEY")
        if not api_key:
            return False
        headers = {
            "X-Domino-Api-Key": api_key,
            "Content-Type": "application/json"
        }
        response = requests.get(
            f"{GUARDRAILS_URL}/health",
            headers=headers,
            timeout=5
        )
        return response.status_code == 200
    except Exception:
        return False


class GuardrailsWrapper:
    """
    Context manager for wrapping pipeline execution with guardrails.

    Usage:
        with GuardrailsWrapper(incident_description) as gw:
            if not gw.input_passed:
                return gw.blocked_response()
            # ... run pipeline ...
            result = gw.sanitize_output(result_json)
    """

    def __init__(self, input_text: str, fail_closed: bool = False):
        """
        Initialize guardrails wrapper.

        Args:
            input_text: The incident description to validate
            fail_closed: If True, block processing when service unavailable
        """
        self.input_text = input_text
        self.fail_closed = fail_closed
        self.input_result: Optional[GuardrailsResult] = None
        self.output_result: Optional[GuardrailsResult] = None

    def __enter__(self):
        self.input_result = check_input(self.input_text)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @property
    def input_passed(self) -> bool:
        """Whether input validation passed."""
        if self.input_result is None:
            return True
        if self.fail_closed and not self.input_result.checks.get("service_available", True):
            return False
        return self.input_result.passed

    @property
    def blocked_reason(self) -> Optional[str]:
        """Reason input was blocked, if any."""
        if self.input_result is None:
            return None
        return self.input_result.blocked_reason

    def blocked_response(self) -> dict:
        """Generate a blocked response for rejected input."""
        return {
            "status": "blocked",
            "reason": self.blocked_reason or "Input blocked by guardrails",
            "guardrails": self.input_result.to_dict() if self.input_result else {}
        }

    def sanitize_output(self, output_text: str) -> tuple[str, GuardrailsResult]:
        """
        Sanitize output text, returning sanitized version.

        Returns:
            Tuple of (sanitized_text, GuardrailsResult)
        """
        self.output_result = check_output(output_text)
        sanitized = self.output_result.sanitized_text or output_text
        return sanitized, self.output_result

    def get_metrics(self) -> dict:
        """Get guardrails metrics for logging."""
        metrics = {}
        if self.input_result:
            metrics["guardrails_input_passed"] = self.input_result.passed
            metrics["guardrails_input_latency_ms"] = self.input_result.latency_ms
            if self.input_result.checks.get("prompt_injection"):
                metrics["guardrails_injection_detected"] = self.input_result.checks["prompt_injection"].get("is_injection", False)
            if self.input_result.checks.get("pii"):
                metrics["guardrails_input_pii_count"] = self.input_result.checks["pii"].get("count", 0)

        if self.output_result:
            metrics["guardrails_output_passed"] = self.output_result.passed
            metrics["guardrails_output_latency_ms"] = self.output_result.latency_ms
            if self.output_result.checks.get("pii"):
                metrics["guardrails_output_pii_count"] = self.output_result.checks["pii"].get("count", 0)

        return metrics
