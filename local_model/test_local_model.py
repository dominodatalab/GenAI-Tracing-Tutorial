#!/usr/bin/env python3
"""
Test script for Domino-hosted local model.

Validates:
1. Connection to the model endpoint
2. Basic completion
3. JSON output capability
4. Tool calling (if supported)

Usage:
    python local_model/test_local_model.py
    python local_model/test_local_model.py --verbose
"""

import argparse
import json
import sys
import time
from typing import Tuple

from domino_model_client import (
    get_local_model_client,
    DominoLocalModelClient,
    test_connection
)


def test_basic_completion(client: DominoLocalModelClient, verbose: bool = False) -> Tuple[bool, str]:
    """Test basic completion capability."""
    try:
        start = time.time()
        response = client.complete(
            prompt="What is 2 + 2? Answer with just the number.",
            temperature=0.1,
            max_tokens=10
        )
        latency = (time.time() - start) * 1000

        if verbose:
            print(f"  Response: {response}")
            print(f"  Latency: {latency:.0f}ms")

        # Check if response contains "4"
        if "4" in response:
            return True, f"OK ({latency:.0f}ms)"
        else:
            return False, f"Unexpected response: {response}"

    except Exception as e:
        return False, str(e)


def test_json_output(client: DominoLocalModelClient, verbose: bool = False) -> Tuple[bool, str]:
    """Test JSON output capability (critical for agents)."""
    try:
        start = time.time()
        response = client.complete(
            prompt="""Classify this incident and return ONLY valid JSON (no markdown, no explanation):

Incident: Production database is down. All customer-facing services affected.

Required JSON format:
{"category": "<string>", "urgency": <1-5>, "reasoning": "<string>"}""",
            temperature=0.2,
            max_tokens=200
        )
        latency = (time.time() - start) * 1000

        if verbose:
            print(f"  Response: {response}")
            print(f"  Latency: {latency:.0f}ms")

        # Try to parse as JSON
        # Handle potential markdown code blocks
        json_str = response.strip()
        if json_str.startswith("```"):
            lines = json_str.split("\n")
            json_str = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        parsed = json.loads(json_str)

        # Validate required fields
        required = ["category", "urgency", "reasoning"]
        missing = [f for f in required if f not in parsed]

        if missing:
            return False, f"Missing fields: {missing}"

        if not isinstance(parsed["urgency"], int) or not 1 <= parsed["urgency"] <= 5:
            return False, f"Invalid urgency: {parsed['urgency']}"

        return True, f"Valid JSON ({latency:.0f}ms)"

    except json.JSONDecodeError as e:
        return False, f"JSON parse error: {e}"
    except Exception as e:
        return False, str(e)


def test_tool_calling(client: DominoLocalModelClient, verbose: bool = False) -> Tuple[bool, str]:
    """Test tool/function calling capability."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    try:
        start = time.time()
        result = client.complete_with_tools(
            prompt="What's the weather in San Francisco?",
            tools=tools,
            temperature=0.1,
            max_tokens=200
        )
        latency = (time.time() - start) * 1000

        if verbose:
            print(f"  Content: {result['content']}")
            print(f"  Tool calls: {result['tool_calls']}")
            print(f"  Latency: {latency:.0f}ms")

        if result["tool_calls"]:
            tc = result["tool_calls"][0]
            if tc["name"] == "get_weather":
                return True, f"Tool call detected ({latency:.0f}ms)"
            else:
                return False, f"Wrong tool called: {tc['name']}"
        else:
            # Model might respond directly instead of using tools
            return False, "No tool call made (model may not support function calling)"

    except Exception as e:
        return False, str(e)


def test_incident_classification(client: DominoLocalModelClient, verbose: bool = False) -> Tuple[bool, str]:
    """Test full incident classification like the TriageFlow classifier."""
    try:
        start = time.time()
        response = client.complete(
            prompt="""You are an incident classification specialist. Analyze this incident and return JSON.

Incident: Multiple users reporting they cannot access the CRM system. Error message shows 'Authentication failed - LDAP server unreachable'. Started approximately 30 minutes ago. Affecting the entire sales team (~50 users).

Return JSON with:
- category: one of [security, operational, performance, data_integrity, compliance, infrastructure, user_access]
- subcategory: specific subcategory
- urgency: 1-5 (5=critical)
- confidence: 0.0-1.0
- reasoning: brief explanation
- affected_domain: department/area affected

Return ONLY valid JSON, no markdown or explanation.""",
            temperature=0.3,
            max_tokens=500
        )
        latency = (time.time() - start) * 1000

        if verbose:
            print(f"  Response: {response}")
            print(f"  Latency: {latency:.0f}ms")

        # Parse and validate
        json_str = response.strip()
        if json_str.startswith("```"):
            lines = json_str.split("\n")
            json_str = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        parsed = json.loads(json_str)

        valid_categories = ["security", "operational", "performance", "data_integrity",
                          "compliance", "infrastructure", "user_access"]

        if parsed.get("category") not in valid_categories:
            return False, f"Invalid category: {parsed.get('category')}"

        # For this incident, infrastructure or user_access would be correct
        if parsed.get("category") in ["infrastructure", "user_access"]:
            return True, f"Correct classification ({latency:.0f}ms)"
        else:
            return True, f"Valid but unexpected category: {parsed.get('category')} ({latency:.0f}ms)"

    except json.JSONDecodeError as e:
        return False, f"JSON parse error: {e}"
    except Exception as e:
        return False, str(e)


def run_all_tests(verbose: bool = False):
    """Run all tests and report results."""
    print("=" * 60)
    print("DOMINO LOCAL MODEL TEST SUITE")
    print("=" * 60)
    print()

    # Test connection first
    print("1. Connection Test")
    if test_connection():
        print("   [PASS] Connected to model endpoint")
    else:
        print("   [FAIL] Could not connect to model endpoint")
        print("\n   Check:")
        print("   - DOMINO_MODEL_ENDPOINT environment variable")
        print("   - Token endpoint at localhost:8899")
        print("   - Network connectivity to the model endpoint")
        sys.exit(1)

    # Initialize client
    client = DominoLocalModelClient()
    results = []

    # Run tests
    tests = [
        ("2. Basic Completion", test_basic_completion),
        ("3. JSON Output", test_json_output),
        ("4. Tool Calling", test_tool_calling),
        ("5. Incident Classification", test_incident_classification),
    ]

    for name, test_fn in tests:
        print(f"\n{name}")
        passed, message = test_fn(client, verbose)
        status = "[PASS]" if passed else "[FAIL]"
        print(f"   {status} {message}")
        results.append((name, passed))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, p in results if p)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\nAll tests passed! Model is ready for TriageFlow experiments.")
    elif passed >= total - 1:
        print("\nMost tests passed. Tool calling may not be fully supported by this model.")
        print("The model can still be used for agent experiments with prompt-based tool descriptions.")
    else:
        print("\nSome tests failed. Review the output above for details.")

    return passed == total


def main():
    parser = argparse.ArgumentParser(description="Test Domino-hosted local model")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed output for each test")
    args = parser.parse_args()

    success = run_all_tests(verbose=args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
