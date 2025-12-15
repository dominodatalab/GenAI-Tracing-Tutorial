"""
Multi-Model Agent Implementations

Provides agent functions that support per-agent model configuration.
Each agent can use a different LLM provider (local Qwen, OpenAI, Anthropic).

Usage:
    from src.multimodel_agents import MultiModelTriage

    triage = MultiModelTriage.from_config("config_multimodel.yaml")
    result = triage.run(incident)
"""

import json
import os
import re
import sys
from typing import Dict, List, Optional, Any, Tuple

import mlflow
from mlflow.entities import SpanType

# Ensure project root is in path
sys.path.insert(0, "/mnt/code")

from .models import (
    Incident, Classification, ImpactAssessment, ResourceAssignment,
    ResponsePlan, Resource, Communication
)
from .tools import (
    lookup_category_definitions, lookup_historical_incidents, calculate_impact_score,
    check_resource_availability, get_sla_requirements,
    get_communication_templates, get_stakeholder_list
)


# Tool function registry
TOOL_FUNCTIONS = {
    "lookup_category_definitions": lookup_category_definitions,
    "lookup_historical_incidents": lookup_historical_incidents,
    "calculate_impact_score": calculate_impact_score,
    "check_resource_availability": check_resource_availability,
    "get_sla_requirements": get_sla_requirements,
    "get_communication_templates": get_communication_templates,
    "get_stakeholder_list": get_stakeholder_list,
}


class ModelClient:
    """
    Unified model client that supports multiple providers.
    Handles the differences between OpenAI, Anthropic, and local model APIs.
    """

    def __init__(self, model_config: Dict, model_type: str = "api"):
        """
        Initialize a model client.

        Args:
            model_config: Configuration dict with 'name', 'type', and optionally 'endpoint'
            model_type: Type of model - 'api' for cloud models, 'local' for Domino-hosted
        """
        self.model_name = model_config.get("name", "gpt-4o-mini")
        self.model_type = model_config.get("type", model_type)
        self.endpoint = model_config.get("endpoint")
        self._client = None
        self._client_type = None
        self._model_for_api = None

    def _init_client(self):
        """Initialize the client type and model name (but not the client for local models)."""
        if self._client_type is not None:
            return

        if self.model_type == "local":
            # For local models, don't cache the client - we'll get a fresh one each time
            self._client_type = "openai"
            self._model_for_api = ""  # Local models use empty string

        elif "claude" in self.model_name.lower() or "anthropic" in self.model_type.lower():
            from anthropic import Anthropic
            self._client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            self._client_type = "anthropic"
            self._model_for_api = self.model_name

        else:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            self._client_type = "openai"
            self._model_for_api = self.model_name

    def _get_client(self):
        """Get the client, fetching a fresh token for local models."""
        self._init_client()
        if self.model_type == "local":
            # Always get a fresh token immediately before making the request
            from local_model.domino_model_client import get_local_model_client
            return get_local_model_client(endpoint_url=self.endpoint)
        return self._client

    @property
    def client(self):
        """Get the underlying client, initializing if needed."""
        return self._get_client()

    @property
    def client_type(self) -> str:
        """Get the client type (openai or anthropic)."""
        self._init_client()
        return self._client_type

    def complete_with_tools(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.3,
        max_tokens: int = 1500
    ) -> str:
        """
        Generate a completion with tool support.

        Args:
            messages: List of message dicts
            tools: Tool definitions in OpenAI format
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Final text response after tool execution
        """
        self._init_client()

        if self._client_type == "anthropic":
            return self._anthropic_tool_loop(messages, tools, temperature, max_tokens)
        else:
            return self._openai_tool_loop(messages, tools, temperature, max_tokens)

    def _openai_tool_loop(self, messages, tools, temperature, max_tokens) -> str:
        """Handle OpenAI-compatible tool calling loop."""
        openai_tools = self._convert_tools_to_openai(tools) if tools else None

        while True:
            kwargs = {
                "model": self._model_for_api,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if openai_tools:
                kwargs["tools"] = openai_tools

            response = self.client.chat.completions.create(**kwargs)
            msg = response.choices[0].message

            if not msg.tool_calls:
                return msg.content or ""

            messages.append(msg)
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                result = TOOL_FUNCTIONS[fn_name](**fn_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })

    def _anthropic_tool_loop(self, messages, tools, temperature, max_tokens) -> str:
        """Handle Anthropic tool calling loop."""
        anthropic_tools = self._convert_tools_to_anthropic(tools) if tools else None
        system_msg = "You are an assistant that uses tools and returns structured JSON. Always return valid JSON in your final response."

        while True:
            kwargs = {
                "model": self._model_for_api,
                "max_tokens": max_tokens,
                "messages": messages,
                "system": system_msg
            }
            if anthropic_tools:
                kwargs["tools"] = anthropic_tools

            response = self.client.messages.create(**kwargs)

            text_parts = []
            tool_uses = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_uses.append(block)

            if not tool_uses:
                result = "\n".join(text_parts).strip()
                if not result and response.stop_reason == "end_turn":
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": "Please provide your analysis as valid JSON."})
                    continue
                return result

            tool_results = []
            for tool_use in tool_uses:
                fn_name = tool_use.name
                fn_args = tool_use.input
                result = TOOL_FUNCTIONS[fn_name](**fn_args)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": json.dumps(result)
                })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

    def _convert_tools_to_openai(self, tools: List[Dict]) -> List[Dict]:
        """Convert tool configs to OpenAI format."""
        return [{
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"]
            }
        } for t in tools]

    def _convert_tools_to_anthropic(self, tools: List[Dict]) -> List[Dict]:
        """Convert tool configs to Anthropic format."""
        return [{
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"]
        } for t in tools]


class MultiModelTriage:
    """
    Multi-model triage pipeline where each agent can use a different model.

    Usage:
        triage = MultiModelTriage.from_config("config_multimodel.yaml")
        result = triage.run(incident)
    """

    def __init__(self, config: Dict):
        """
        Initialize with configuration dict.

        Args:
            config: Full configuration including models, agent_models, tools, agents
        """
        self.config = config

        # Initialize model clients for each agent
        self.agent_clients: Dict[str, ModelClient] = {}
        self._init_agent_clients()

    @classmethod
    def from_config(cls, config_path: str) -> "MultiModelTriage":
        """Create from a YAML config file."""
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls(config)

    def _init_agent_clients(self):
        """Initialize a model client for each agent based on config."""
        models = self.config.get("models", {})
        agent_models = self.config.get("agent_models", {})

        for agent_name in ["classifier", "impact_assessor", "resource_matcher", "response_drafter"]:
            model_key = agent_models.get(agent_name, "openai")  # Default to OpenAI
            model_config = models.get(model_key, {"name": "gpt-4o-mini", "type": "api"})
            self.agent_clients[agent_name] = ModelClient(model_config)

    def _parse_json_response(self, response: str) -> Dict:
        """Extract JSON from LLM response."""
        if not response:
            raise ValueError("Empty response from LLM")

        response = response.strip()

        # Try to find JSON in code blocks
        code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response)
        if code_block_match:
            response = code_block_match.group(1).strip()

        # Try to find JSON object directly
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            response = json_match.group(0)

        return json.loads(response)

    @mlflow.trace(span_type=SpanType.AGENT)
    def classify_incident(self, incident: Incident) -> Classification:
        """Classify the incident using the configured model."""
        client = self.agent_clients["classifier"]
        agent_config = self.config["agents"]["classifier"]

        prompt = agent_config["prompt"].format(incident=incident.model_dump_json())
        messages = [{"role": "user", "content": prompt}]
        tools = self.config["tools"]["classifier"]

        response = client.complete_with_tools(
            messages=messages,
            tools=tools,
            temperature=agent_config["temperature"],
            max_tokens=agent_config["max_tokens"]
        )

        return Classification(**self._parse_json_response(response))

    @mlflow.trace(span_type=SpanType.AGENT)
    def assess_impact(
        self,
        incident: Incident,
        classification: Classification
    ) -> ImpactAssessment:
        """Assess impact using the configured model."""
        client = self.agent_clients["impact_assessor"]
        agent_config = self.config["agents"]["impact_assessor"]

        prompt = agent_config["prompt"].format(
            classification=classification.model_dump_json(),
            incident=incident.model_dump_json()
        )
        messages = [{"role": "user", "content": prompt}]
        tools = self.config["tools"]["impact_assessor"]

        response = client.complete_with_tools(
            messages=messages,
            tools=tools,
            temperature=agent_config["temperature"],
            max_tokens=agent_config["max_tokens"]
        )

        return ImpactAssessment(**self._parse_json_response(response))

    @mlflow.trace(span_type=SpanType.AGENT)
    def match_resources(
        self,
        classification: Classification,
        impact: ImpactAssessment
    ) -> ResourceAssignment:
        """Match resources using the configured model."""
        client = self.agent_clients["resource_matcher"]
        agent_config = self.config["agents"]["resource_matcher"]

        prompt = agent_config["prompt"].format(
            classification=classification.model_dump_json(),
            impact=impact.model_dump_json()
        )
        messages = [{"role": "user", "content": prompt}]
        tools = self.config["tools"]["resource_matcher"]

        response = client.complete_with_tools(
            messages=messages,
            tools=tools,
            temperature=agent_config["temperature"],
            max_tokens=agent_config["max_tokens"]
        )

        data = self._parse_json_response(response)
        data["primary_responder"] = Resource(**data["primary_responder"])
        data["backup_responders"] = [Resource(**r) for r in data.get("backup_responders", [])]

        return ResourceAssignment(**data)

    @mlflow.trace(span_type=SpanType.AGENT)
    def draft_response(
        self,
        incident: Incident,
        classification: Classification,
        impact: ImpactAssessment,
        resources: ResourceAssignment
    ) -> ResponsePlan:
        """Draft response using the configured model."""
        client = self.agent_clients["response_drafter"]
        agent_config = self.config["agents"]["response_drafter"]

        prompt = agent_config["prompt"].format(
            incident=incident.model_dump_json(),
            classification=classification.model_dump_json(),
            impact=impact.model_dump_json(),
            resources=resources.model_dump_json()
        )
        messages = [{"role": "user", "content": prompt}]
        tools = self.config["tools"]["response_drafter"]

        response = client.complete_with_tools(
            messages=messages,
            tools=tools,
            temperature=agent_config["temperature"],
            max_tokens=agent_config["max_tokens"]
        )

        data = self._parse_json_response(response)
        data["communications"] = [Communication(**c) for c in data.get("communications", [])]

        return ResponsePlan(**data)

    def run(self, incident: Incident) -> Tuple[Classification, ImpactAssessment, ResourceAssignment, ResponsePlan]:
        """
        Run the full triage pipeline.

        Args:
            incident: The incident to triage

        Returns:
            Tuple of (Classification, ImpactAssessment, ResourceAssignment, ResponsePlan)
        """
        classification = self.classify_incident(incident)
        impact = self.assess_impact(incident, classification)
        resources = self.match_resources(classification, impact)
        response = self.draft_response(incident, classification, impact, resources)

        return classification, impact, resources, response

    def get_model_info(self) -> Dict[str, str]:
        """Return information about which model each agent is using."""
        return {
            agent: client.model_name
            for agent, client in self.agent_clients.items()
        }
