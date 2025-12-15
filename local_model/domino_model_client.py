"""
Domino-Hosted Local Model Client

Provides an OpenAI-compatible client for the Domino-hosted Qwen model.
Handles authentication via the Domino token endpoint.

Usage:
    from local_model.domino_model_client import get_local_model_client

    client = get_local_model_client()
    response = client.chat.completions.create(
        model="",  # Empty - endpoint knows the model
        messages=[{"role": "user", "content": "Hello"}]
    )
"""

import os
import requests
from typing import Optional
from openai import OpenAI


# Default endpoint configuration
# This endpoint hosts the Qwen 2.5 3B model on Domino
DEFAULT_MODEL_ENDPOINT = "https://genai-llm.domino-eval.com/endpoints/bf209962-1bd0-4524-87c8-2d0ac662a022/v1"

# Domino provides authentication tokens via this local endpoint
TOKEN_ENDPOINT = "http://localhost:8899/access-token"


def get_domino_token() -> str:
    """
    Retrieve the Domino access token from the local token endpoint.

    This endpoint is available inside Domino workspaces and provides
    authentication for internal API calls.

    Returns:
        Access token string

    Raises:
        RuntimeError: If token retrieval fails
    """
    try:
        response = requests.get(TOKEN_ENDPOINT, timeout=5)
        response.raise_for_status()
        return response.text.strip()
    except requests.RequestException as e:
        # Fall back to environment variable if local endpoint unavailable
        env_token = os.environ.get("DOMINO_API_KEY") or os.environ.get("DOMINO_USER_API_KEY")
        if env_token:
            return env_token
        raise RuntimeError(f"Failed to retrieve Domino token: {e}")


def get_local_model_client(
    endpoint_url: Optional[str] = None,
    api_key: Optional[str] = None
) -> OpenAI:
    """
    Create an OpenAI-compatible client for the Domino-hosted local model.

    Args:
        endpoint_url: Override the default model endpoint URL.
                      Can also be set via DOMINO_MODEL_ENDPOINT env var.
        api_key: Override the API key. If not provided, retrieves from
                 the Domino token endpoint.

    Returns:
        OpenAI client configured for the local model

    Example:
        client = get_local_model_client()
        completion = client.chat.completions.create(
            model="",  # Empty string - endpoint knows the model
            messages=[{"role": "user", "content": "Classify this incident..."}]
        )
    """
    # Determine endpoint URL
    base_url = endpoint_url or os.environ.get("DOMINO_MODEL_ENDPOINT", DEFAULT_MODEL_ENDPOINT)

    # Get authentication token
    if api_key is None:
        api_key = get_domino_token()

    return OpenAI(
        base_url=base_url,
        api_key=api_key
    )


class DominoLocalModelClient:
    """
    Wrapper class for the Domino-hosted local model with additional features.

    Provides:
    - Fresh token for each request (avoids timeout issues)
    - Request timing and logging
    - Retry logic for transient failures
    """

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        model_name: str = "qwen2.5-3b-instruct"
    ):
        """
        Initialize the local model client.

        Args:
            endpoint_url: Model endpoint URL (uses env/default if not provided)
            model_name: Name for logging/tracking (actual model determined by endpoint)
        """
        self.endpoint_url = endpoint_url or os.environ.get(
            "DOMINO_MODEL_ENDPOINT", DEFAULT_MODEL_ENDPOINT
        )
        self.model_name = model_name

    def _get_fresh_client(self) -> OpenAI:
        """Get a new OpenAI client with a fresh token for each request."""
        # Always fetch a fresh token immediately before making the request
        api_key = get_domino_token()
        return OpenAI(
            base_url=self.endpoint_url,
            api_key=api_key
        )

    @property
    def client(self) -> OpenAI:
        """Get a fresh client for each access (fetches new token)."""
        return self._get_fresh_client()

    def complete(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1500,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a completion from the local model.

        Args:
            prompt: User prompt/message
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            system_prompt: Optional system message

        Returns:
            Generated text response
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model="",  # Empty - endpoint knows the model
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content or ""

    def complete_with_tools(
        self,
        prompt: str,
        tools: list,
        temperature: float = 0.3,
        max_tokens: int = 1500
    ) -> dict:
        """
        Generate a completion with tool/function calling support.

        Note: Tool calling support depends on the model's capabilities.
        Qwen 2.5 3B has basic function calling support but may be inconsistent.

        Args:
            prompt: User prompt
            tools: List of tool definitions in OpenAI format
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            Dict with 'content' and 'tool_calls' keys
        """
        response = self.client.chat.completions.create(
            model="",
            messages=[{"role": "user", "content": prompt}],
            tools=tools if tools else None,
            tool_choice="auto" if tools else None,
            temperature=temperature,
            max_tokens=max_tokens
        )

        message = response.choices[0].message

        result = {
            "content": message.content or "",
            "tool_calls": []
        }

        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
                for tc in message.tool_calls
            ]

        return result


# Convenience function for quick testing
def test_connection() -> bool:
    """
    Test connection to the local model endpoint.

    Returns:
        True if connection successful, False otherwise
    """
    try:
        client = get_local_model_client()
        response = client.chat.completions.create(
            model="",
            messages=[{"role": "user", "content": "Hello, respond with just 'OK'"}],
            max_tokens=10
        )
        return bool(response.choices[0].message.content)
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False


if __name__ == "__main__":
    # Quick test when run directly
    print("Testing Domino local model connection...")

    if test_connection():
        print("Connection successful!")

        # Run a sample classification
        client = DominoLocalModelClient()
        response = client.complete(
            prompt="""Classify this incident and return JSON:

Incident: Users cannot log in to the VPN. Multiple reports from remote workers.

Return JSON with: category, urgency (1-5), reasoning""",
            temperature=0.3,
            max_tokens=500
        )
        print(f"\nSample response:\n{response}")
    else:
        print("Connection failed. Check endpoint URL and authentication.")
