import json
import re
import mlflow
from mlflow.entities import SpanType
from .models import (
    Incident, Classification, ImpactAssessment, ResourceAssignment,
    ResponsePlan, Resource, Communication
)
from .tools import (
    lookup_category_definitions, lookup_historical_incidents, calculate_impact_score,
    check_resource_availability, get_sla_requirements,
    get_communication_templates, get_stakeholder_list
)

TOOL_FUNCTIONS = {
    "lookup_category_definitions": lookup_category_definitions,
    "lookup_historical_incidents": lookup_historical_incidents,
    "calculate_impact_score": calculate_impact_score,
    "check_resource_availability": check_resource_availability,
    "get_sla_requirements": get_sla_requirements,
    "get_communication_templates": get_communication_templates,
    "get_stakeholder_list": get_stakeholder_list,
}


def config_to_openai_tools(tool_configs: list) -> list:
    """Convert config tool schemas to OpenAI format."""
    return [{
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["parameters"]
        }
    } for t in tool_configs]


def config_to_anthropic_tools(tool_configs: list) -> list:
    """Convert config tool schemas to Anthropic format."""
    return [{
        "name": t["name"],
        "description": t["description"],
        "input_schema": t["parameters"]
    } for t in tool_configs]


def call_with_tools(client, provider: str, model: str, messages: list,
                    tool_configs: list, temperature: float, max_tokens: int) -> str:
    """Call LLM with tools, handling tool calls until final response."""
    if provider == "openai":
        tools = config_to_openai_tools(tool_configs) if tool_configs else None
        return _openai_tool_loop(client, model, messages, tools, temperature, max_tokens)
    else:
        tools = config_to_anthropic_tools(tool_configs) if tool_configs else None
        return _anthropic_tool_loop(client, model, messages, tools, temperature, max_tokens)


def _openai_tool_loop(client, model, messages, tools, temperature, max_tokens):
    """Handle OpenAI tool calling loop."""
    while True:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens
        )
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


def _anthropic_tool_loop(client, model, messages, tools, temperature, max_tokens):
    """Handle Anthropic tool calling loop."""
    system_msg = "You are an assistant that uses tools and returns structured JSON. Always return valid JSON in your final response."

    while True:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            tools=tools,
            messages=messages,
            system=system_msg
        )

        # Collect text and tool uses from response
        text_parts = []
        tool_uses = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_uses.append(block)

        # If no tool calls, return the text
        if not tool_uses:
            result = "\n".join(text_parts).strip()
            # If we got an empty response after tools, prompt for JSON output
            if not result and response.stop_reason == "end_turn":
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": "Please provide your analysis as valid JSON."})
                continue
            return result

        # Execute tool calls
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

        # Add assistant response and tool results to conversation
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})


def parse_json_response(response: str) -> dict:
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
def classify_incident(client, provider: str, model: str,
                      incident: Incident, config: dict) -> Classification:
    """Classify the incident using tools from config."""
    prompt = config["agents"]["classifier"]["prompt"].format(incident=incident.model_dump_json())
    messages = [{"role": "user", "content": prompt}]
    tools = config["tools"]["classifier"]

    response = call_with_tools(client, provider, model, messages, tools,
                               config["agents"]["classifier"]["temperature"],
                               config["agents"]["classifier"]["max_tokens"])
    return Classification(**parse_json_response(response))


@mlflow.trace(span_type=SpanType.AGENT)
def assess_impact(client, provider: str, model: str, incident: Incident,
                  classification: Classification, config: dict) -> ImpactAssessment:
    """Assess impact using tools from config."""
    prompt = config["agents"]["impact_assessor"]["prompt"].format(
        classification=classification.model_dump_json(),
        incident=incident.model_dump_json()
    )
    messages = [{"role": "user", "content": prompt}]
    tools = config["tools"]["impact_assessor"]

    response = call_with_tools(client, provider, model, messages, tools,
                               config["agents"]["impact_assessor"]["temperature"],
                               config["agents"]["impact_assessor"]["max_tokens"])
    return ImpactAssessment(**parse_json_response(response))


@mlflow.trace(span_type=SpanType.AGENT)
def match_resources(client, provider: str, model: str,
                    classification: Classification,
                    impact: ImpactAssessment, config: dict) -> ResourceAssignment:
    """Match resources using tools from config."""
    prompt = config["agents"]["resource_matcher"]["prompt"].format(
        classification=classification.model_dump_json(),
        impact=impact.model_dump_json()
    )
    messages = [{"role": "user", "content": prompt}]
    tools = config["tools"]["resource_matcher"]

    response = call_with_tools(client, provider, model, messages, tools,
                               config["agents"]["resource_matcher"]["temperature"],
                               config["agents"]["resource_matcher"]["max_tokens"])
    data = parse_json_response(response)
    data["primary_responder"] = Resource(**data["primary_responder"])
    data["backup_responders"] = [Resource(**r) for r in data.get("backup_responders", [])]
    return ResourceAssignment(**data)


@mlflow.trace(span_type=SpanType.AGENT)
def draft_response(client, provider: str, model: str, incident: Incident,
                   classification: Classification, impact: ImpactAssessment,
                   resources: ResourceAssignment, config: dict) -> ResponsePlan:
    """Draft response using tools from config."""
    prompt = config["agents"]["response_drafter"]["prompt"].format(
        incident=incident.model_dump_json(),
        classification=classification.model_dump_json(),
        impact=impact.model_dump_json(),
        resources=resources.model_dump_json()
    )
    messages = [{"role": "user", "content": prompt}]
    tools = config["tools"]["response_drafter"]

    response = call_with_tools(client, provider, model, messages, tools,
                               config["agents"]["response_drafter"]["temperature"],
                               config["agents"]["response_drafter"]["max_tokens"])
    data = parse_json_response(response)
    data["communications"] = [Communication(**c) for c in data.get("communications", [])]
    return ResponsePlan(**data)
