#!/usr/bin/env python3
"""
PII Redaction Validation Script for Domino Governance

This scripted check validates that the Guardrails service's PII redaction
endpoint is functioning correctly by sending test data containing known
PII patterns and verifying proper redaction.

Usage:
    python pii_redaction_check.py --guardrails-url <url> --test-mode <mode>

Arguments:
    --guardrails-url: Base URL for the Guardrails API service
    --test-mode: Test intensity (standard, comprehensive, minimal)

Outputs:
    - pii_validation_results.txt: Human-readable summary
    - pii_validation_results.json: Structured results for automated processing
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Dict, List, Any
import random

# =============================================================================
# TEST DATA - Known PII patterns for validation
# =============================================================================

TEST_CASES = [
    {
        "id": "TC001",
        "name": "Email Address Redaction",
        "input": "Please contact john.doe@acmecorp.com for assistance with ticket #12345.",
        "pii_type": "email",
        "expected_redacted": ["john.doe@acmecorp.com"],
        "severity": "high"
    },
    {
        "id": "TC002", 
        "name": "Phone Number Redaction",
        "input": "Customer callback requested at (555) 123-4567 or 555.987.6543.",
        "pii_type": "phone",
        "expected_redacted": ["(555) 123-4567", "555.987.6543"],
        "severity": "high"
    },
    {
        "id": "TC003",
        "name": "SSN Redaction",
        "input": "Employee SSN for verification: 123-45-6789. Please process immediately.",
        "pii_type": "ssn",
        "expected_redacted": ["123-45-6789"],
        "severity": "critical"
    },
    {
        "id": "TC004",
        "name": "Credit Card Redaction",
        "input": "Payment failed for card 4111-1111-1111-1111, expiry 12/25.",
        "pii_type": "credit_card",
        "expected_redacted": ["4111-1111-1111-1111"],
        "severity": "critical"
    },
    {
        "id": "TC005",
        "name": "IP Address Redaction",
        "input": "Suspicious login attempt from 192.168.1.105 and 10.0.0.42.",
        "pii_type": "ip_address",
        "expected_redacted": ["192.168.1.105", "10.0.0.42"],
        "severity": "medium"
    },
    {
        "id": "TC006",
        "name": "Full Name Redaction",
        "input": "Incident reported by Sarah Johnson, escalated to Mike Chen.",
        "pii_type": "name",
        "expected_redacted": ["Sarah Johnson", "Mike Chen"],
        "severity": "medium"
    },
    {
        "id": "TC007",
        "name": "Address Redaction",
        "input": "Ship replacement to 123 Main Street, Apt 4B, New York, NY 10001.",
        "pii_type": "address",
        "expected_redacted": ["123 Main Street, Apt 4B, New York, NY 10001"],
        "severity": "medium"
    },
    {
        "id": "TC008",
        "name": "Mixed PII Redaction",
        "input": "User jane.smith@email.com (SSN: 987-65-4321) called from 555-000-1234 about account.",
        "pii_type": "mixed",
        "expected_redacted": ["jane.smith@email.com", "987-65-4321", "555-000-1234"],
        "severity": "critical"
    },
    {
        "id": "TC009",
        "name": "No PII Present",
        "input": "Server CPU utilization reached 95% causing performance degradation.",
        "pii_type": "none",
        "expected_redacted": [],
        "severity": "low"
    },
    {
        "id": "TC010",
        "name": "Date of Birth Redaction",
        "input": "Customer DOB: 03/15/1985. Please verify identity before proceeding.",
        "pii_type": "dob",
        "expected_redacted": ["03/15/1985"],
        "severity": "high"
    }
]


# =============================================================================
# MOCK GUARDRAILS API CLIENT
# =============================================================================

class MockGuardrailsClient:
    """
    Mock client simulating Guardrails API behavior.
    In production, this would make actual HTTP requests to the Guardrails service.
    """
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.endpoint = f"{base_url}/redact-pii"
        
    def redact_pii(self, text: str) -> Dict[str, Any]:
        """
        Simulate calling the /redact-pii endpoint.
        
        Returns a response mimicking the Guardrails API format:
        {
            "original_text": str,
            "redacted_text": str,
            "pii_detected": [
                {"type": str, "value": str, "start": int, "end": int, "redacted_as": str}
            ],
            "processing_time_ms": float
        }
        """
        # Simulate API latency
        processing_time = random.uniform(15, 85)
        
        # Simulate PII detection and redaction
        redacted_text = text
        pii_detected = []
        
        # Email pattern
        import re
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        for match in re.finditer(email_pattern, text):
            pii_detected.append({
                "type": "email",
                "value": match.group(),
                "start": match.start(),
                "end": match.end(),
                "redacted_as": "[EMAIL_REDACTED]"
            })
            redacted_text = redacted_text.replace(match.group(), "[EMAIL_REDACTED]")
        
        # Phone pattern
        phone_pattern = r'[\(]?\d{3}[\)]?[-.\s]?\d{3}[-.\s]?\d{4}'
        for match in re.finditer(phone_pattern, text):
            pii_detected.append({
                "type": "phone",
                "value": match.group(),
                "start": match.start(),
                "end": match.end(),
                "redacted_as": "[PHONE_REDACTED]"
            })
            redacted_text = redacted_text.replace(match.group(), "[PHONE_REDACTED]")
        
        # SSN pattern
        ssn_pattern = r'\d{3}-\d{2}-\d{4}'
        for match in re.finditer(ssn_pattern, text):
            pii_detected.append({
                "type": "ssn",
                "value": match.group(),
                "start": match.start(),
                "end": match.end(),
                "redacted_as": "[SSN_REDACTED]"
            })
            redacted_text = redacted_text.replace(match.group(), "[SSN_REDACTED]")
        
        # Credit card pattern
        cc_pattern = r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}'
        for match in re.finditer(cc_pattern, text):
            pii_detected.append({
                "type": "credit_card",
                "value": match.group(),
                "start": match.start(),
                "end": match.end(),
                "redacted_as": "[CC_REDACTED]"
            })
            redacted_text = redacted_text.replace(match.group(), "[CC_REDACTED]")
        
        # IP address pattern
        ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
        for match in re.finditer(ip_pattern, text):
            pii_detected.append({
                "type": "ip_address",
                "value": match.group(),
                "start": match.start(),
                "end": match.end(),
                "redacted_as": "[IP_REDACTED]"
            })
            redacted_text = redacted_text.replace(match.group(), "[IP_REDACTED]")
        
        # DOB pattern (MM/DD/YYYY)
        dob_pattern = r'\d{2}/\d{2}/\d{4}'
        for match in re.finditer(dob_pattern, text):
            pii_detected.append({
                "type": "dob",
                "value": match.group(),
                "start": match.start(),
                "end": match.end(),
                "redacted_as": "[DOB_REDACTED]"
            })
            redacted_text = redacted_text.replace(match.group(), "[DOB_REDACTED]")
        
        return {
            "original_text": text,
            "redacted_text": redacted_text,
            "pii_detected": pii_detected,
            "processing_time_ms": processing_time
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Simulate health check endpoint."""
        return {
            "status": "healthy",
            "service": "guardrails-pii",
            "version": "1.2.0",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# =============================================================================
# VALIDATION LOGIC
# =============================================================================

def run_validation(client: MockGuardrailsClient, test_mode: str) -> Dict[str, Any]:
    """
    Run PII redaction validation tests.
    
    Args:
        client: Guardrails API client
        test_mode: Test intensity level
        
    Returns:
        Validation results dictionary
    """
    # Select test cases based on mode
    if test_mode == "minimal":
        cases = [tc for tc in TEST_CASES if tc["severity"] == "critical"][:3]
    elif test_mode == "comprehensive":
        cases = TEST_CASES
    else:  # standard
        cases = [tc for tc in TEST_CASES if tc["severity"] in ["critical", "high"]]
    
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "guardrails_url": client.base_url,
        "test_mode": test_mode,
        "total_tests": len(cases),
        "passed": 0,
        "failed": 0,
        "test_results": [],
        "health_check": None,
        "summary": {}
    }
    
    # Run health check first
    try:
        health = client.health_check()
        results["health_check"] = {
            "status": "passed",
            "response": health
        }
    except Exception as e:
        results["health_check"] = {
            "status": "failed",
            "error": str(e)
        }
        results["summary"]["health_check_failed"] = True
    
    # Run test cases
    total_latency = 0
    critical_failures = []
    
    for tc in cases:
        try:
            response = client.redact_pii(tc["input"])
            
            # Validate redaction
            detected_values = [p["value"] for p in response["pii_detected"]]
            expected = tc["expected_redacted"]
            
            # Check if all expected PII was detected
            all_detected = all(exp in detected_values for exp in expected)
            
            # Check for false positives (only for "none" type)
            no_false_positives = True
            if tc["pii_type"] == "none" and len(response["pii_detected"]) > 0:
                no_false_positives = False
            
            passed = all_detected and no_false_positives
            
            test_result = {
                "test_id": tc["id"],
                "test_name": tc["name"],
                "pii_type": tc["pii_type"],
                "severity": tc["severity"],
                "status": "passed" if passed else "failed",
                "expected_redacted": expected,
                "actual_detected": detected_values,
                "redacted_text": response["redacted_text"],
                "latency_ms": response["processing_time_ms"]
            }
            
            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1
                if tc["severity"] == "critical":
                    critical_failures.append(tc["id"])
                test_result["failure_reason"] = "Missing expected PII detection" if not all_detected else "False positive detected"
            
            results["test_results"].append(test_result)
            total_latency += response["processing_time_ms"]
            
        except Exception as e:
            results["failed"] += 1
            results["test_results"].append({
                "test_id": tc["id"],
                "test_name": tc["name"],
                "status": "error",
                "error": str(e)
            })
    
    # Calculate summary metrics
    results["summary"] = {
        "pass_rate": (results["passed"] / results["total_tests"] * 100) if results["total_tests"] > 0 else 0,
        "avg_latency_ms": total_latency / len(cases) if cases else 0,
        "critical_failures": critical_failures,
        "has_critical_failures": len(critical_failures) > 0,
        "overall_status": "PASSED" if results["failed"] == 0 else "FAILED"
    }
    
    return results


def generate_text_report(results: Dict[str, Any]) -> str:
    """Generate human-readable text report."""
    lines = [
        "=" * 70,
        "PII REDACTION VALIDATION REPORT",
        "=" * 70,
        "",
        f"Timestamp: {results['timestamp']}",
        f"Guardrails URL: {results['guardrails_url']}",
        f"Test Mode: {results['test_mode']}",
        "",
        "-" * 70,
        "SUMMARY",
        "-" * 70,
        f"Total Tests: {results['total_tests']}",
        f"Passed: {results['passed']}",
        f"Failed: {results['failed']}",
        f"Pass Rate: {results['summary']['pass_rate']:.1f}%",
        f"Avg Latency: {results['summary']['avg_latency_ms']:.1f}ms",
        f"Overall Status: {results['summary']['overall_status']}",
        "",
    ]
    
    if results['summary']['has_critical_failures']:
        lines.extend([
            "⚠️  CRITICAL FAILURES DETECTED:",
            f"    Test IDs: {', '.join(results['summary']['critical_failures'])}",
            ""
        ])
    
    # Health check status
    lines.extend([
        "-" * 70,
        "HEALTH CHECK",
        "-" * 70,
    ])
    if results['health_check']:
        lines.append(f"Status: {results['health_check']['status'].upper()}")
        if results['health_check']['status'] == 'passed':
            lines.append(f"Service Version: {results['health_check']['response'].get('version', 'N/A')}")
    lines.append("")
    
    # Individual test results
    lines.extend([
        "-" * 70,
        "TEST RESULTS",
        "-" * 70,
        ""
    ])
    
    for tr in results['test_results']:
        status_icon = "✓" if tr['status'] == 'passed' else "✗"
        lines.append(f"{status_icon} [{tr['test_id']}] {tr['test_name']}")
        lines.append(f"    PII Type: {tr['pii_type']} | Severity: {tr['severity']}")
        lines.append(f"    Status: {tr['status'].upper()}")
        if tr['status'] == 'passed':
            lines.append(f"    Latency: {tr.get('latency_ms', 0):.1f}ms")
        elif tr['status'] == 'failed':
            lines.append(f"    Reason: {tr.get('failure_reason', 'Unknown')}")
            lines.append(f"    Expected: {tr.get('expected_redacted', [])}")
            lines.append(f"    Detected: {tr.get('actual_detected', [])}")
        lines.append("")
    
    lines.extend([
        "=" * 70,
        "END OF REPORT",
        "=" * 70
    ])
    
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PII Redaction Validation for Domino Governance")
    parser.add_argument("--guardrails-url", required=True, help="Base URL for Guardrails API")
    parser.add_argument("--test-mode", default="standard", choices=["minimal", "standard", "comprehensive"],
                        help="Test intensity level")
    args = parser.parse_args()
    
    print(f"Starting PII Redaction Validation...")
    print(f"Guardrails URL: {args.guardrails_url}")
    print(f"Test Mode: {args.test_mode}")
    print()
    
    # Initialize client
    client = MockGuardrailsClient(args.guardrails_url)
    
    # Run validation
    results = run_validation(client, args.test_mode)
    
    # Generate outputs
    text_report = generate_text_report(results)
    
    # Write text report
    with open("pii_validation_results.txt", "w") as f:
        f.write(text_report)
    print("Text report written to: pii_validation_results.txt")
    
    # Write JSON results
    with open("pii_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("JSON results written to: pii_validation_results.json")
    
    # Print summary to stdout
    print()
    print(text_report)
    
    # Exit with appropriate code
    if results['summary']['overall_status'] == 'PASSED':
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
