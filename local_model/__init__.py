"""
Local Model Integration Module

Provides clients for Domino-hosted local models (e.g., Qwen 2.5 3B).
"""

from .domino_model_client import (
    get_local_model_client,
    get_domino_token,
    DominoLocalModelClient,
    test_connection,
    DEFAULT_MODEL_ENDPOINT,
    TOKEN_ENDPOINT
)

__all__ = [
    "get_local_model_client",
    "get_domino_token",
    "DominoLocalModelClient",
    "test_connection",
    "DEFAULT_MODEL_ENDPOINT",
    "TOKEN_ENDPOINT"
]
