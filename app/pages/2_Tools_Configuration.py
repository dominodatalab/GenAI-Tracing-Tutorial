"""
Tools Configuration Page

View and configure the tools available to each agent.
"""

import streamlit as st
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_manager import load_base_config, config_to_yaml_string

st.set_page_config(
    page_title="Tools Configuration - TriageFlow",
    layout="wide"
)


# Tool documentation for reference
TOOL_DOCS = {
    "lookup_category_definitions": {
        "description": "Returns the official category definitions with subcategories and keywords.",
        "returns": "Dictionary with 7 categories: security, operational, performance, data_integrity, compliance, infrastructure, user_access",
        "example_output": "{'security': {'subcategories': ['breach', 'unauthorized_access', ...], 'keywords': [...]}}"
    },
    "lookup_historical_incidents": {
        "description": "Searches for similar historical incidents to inform impact assessment.",
        "returns": "List of similar past incidents with ticket_id, impact_score, resolution_hours, etc.",
        "example_output": "[{'ticket_id': 'INC-001', 'category': 'security', 'impact_score': 8.5, ...}]"
    },
    "calculate_impact_score": {
        "description": "Calculates a normalized impact score (0-10) based on multiple factors.",
        "returns": "Float between 0 and 10",
        "example_output": "7.5"
    },
    "check_resource_availability": {
        "description": "Checks available resources matching required skills with availability and load information.",
        "returns": "List of available resources sorted by match score",
        "example_output": "[{'resource_id': 'R001', 'name': 'Alice Chen', 'match_score': 0.85, ...}]"
    },
    "get_sla_requirements": {
        "description": "Gets SLA requirements based on urgency level and incident category.",
        "returns": "Dictionary with response_hours, resolution_hours, escalation_threshold_hours",
        "example_output": "{'response_hours': 1, 'resolution_hours': 4, 'escalation_threshold_hours': 2}"
    },
    "get_communication_templates": {
        "description": "Gets communication templates for different audiences based on category and urgency.",
        "returns": "Dictionary with templates for technical_team, management, affected_users",
        "example_output": "{'technical_team': {'tone': 'direct', ...}, 'management': {...}}"
    },
    "get_stakeholder_list": {
        "description": "Determines which stakeholders need notification based on impact and blast radius.",
        "returns": "List of stakeholders with notification methods and timing",
        "example_output": "[{'stakeholder': 'technical_team', 'notification': 'immediate'}, ...]"
    }
}


def init_page_state():
    """Initialize page-specific session state."""
    if "config" not in st.session_state:
        st.session_state.config = load_base_config()


def render_tool_card(tool: dict, agent_name: str, tool_index: int):
    """Render a single tool configuration card."""
    tool_name = tool.get("name", "Unknown Tool")
    tool_desc = tool.get("description", "No description available")
    tool_params = tool.get("parameters", {})

    with st.container(border=True):
        col1, col2 = st.columns([4, 1])

        with col1:
            st.markdown(f"**{tool_name}**")

        # Tool description (editable)
        new_desc = st.text_input(
            "Description",
            value=tool_desc,
            key=f"{agent_name}_{tool_index}_desc",
            label_visibility="collapsed"
        )

        # Update in session state
        st.session_state.config["tools"][agent_name][tool_index]["description"] = new_desc

        # Parameters display
        if tool_params.get("properties"):
            st.markdown("**Parameters:**")
            params_display = []
            for param_name, param_info in tool_params.get("properties", {}).items():
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "")
                required = param_name in tool_params.get("required", [])
                req_marker = " (required)" if required else ""
                params_display.append(f"- `{param_name}` ({param_type}){req_marker}: {param_desc}")

            st.markdown("\n".join(params_display))
        else:
            st.caption("No parameters required")

        # Documentation reference
        if tool_name in TOOL_DOCS:
            with st.expander("Documentation", expanded=False):
                doc = TOOL_DOCS[tool_name]
                st.markdown(f"**Returns:** {doc['returns']}")
                st.code(doc['example_output'], language="python")


def render_agent_tools(agent_name: str, display_name: str):
    """Render tools for a specific agent."""
    config = st.session_state.config
    tools = config.get("tools", {}).get(agent_name, [])

    st.markdown(f"### {display_name}")

    if not tools:
        st.info(f"No tools configured for {display_name}")
        return

    for i, tool in enumerate(tools):
        render_tool_card(tool, agent_name, i)


def main():
    """Main page content."""
    init_page_state()

    st.title("Tools Configuration")
    st.markdown("View and configure the tools available to each agent in the triage pipeline.")

    st.divider()

    # Overview
    st.markdown("""
    Tools are functions that agents can call during their reasoning process to retrieve
    information or perform calculations. Each agent has access to specific tools relevant
    to its responsibilities.
    """)

    # Tool count summary
    config = st.session_state.config
    tools_config = config.get("tools", {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        classifier_count = len(tools_config.get("classifier", []))
        st.metric("Classifier Tools", classifier_count)
    with col2:
        impact_count = len(tools_config.get("impact_assessor", []))
        st.metric("Impact Assessor Tools", impact_count)
    with col3:
        resource_count = len(tools_config.get("resource_matcher", []))
        st.metric("Resource Matcher Tools", resource_count)
    with col4:
        response_count = len(tools_config.get("response_drafter", []))
        st.metric("Response Drafter Tools", response_count)

    st.divider()

    # Tabs for each agent's tools
    tab1, tab2, tab3, tab4 = st.tabs([
        "Classifier",
        "Impact Assessor",
        "Resource Matcher",
        "Response Drafter"
    ])

    with tab1:
        render_agent_tools("classifier", "Classifier Agent Tools")
        st.caption("The classifier uses category definitions to properly categorize incidents.")

    with tab2:
        render_agent_tools("impact_assessor", "Impact Assessor Agent Tools")
        st.caption("The impact assessor uses historical data and scoring algorithms to evaluate incident impact.")

    with tab3:
        render_agent_tools("resource_matcher", "Resource Matcher Agent Tools")
        st.caption("The resource matcher checks availability and SLA requirements to assign responders.")

    with tab4:
        render_agent_tools("response_drafter", "Response Drafter Agent Tools")
        st.caption("The response drafter uses templates and stakeholder information to create communications.")

    st.divider()

    # Configuration Preview
    st.markdown("## Configuration Preview")

    with st.expander("View Tools Configuration (YAML)", expanded=False):
        preview_config = {"tools": st.session_state.config.get("tools", {})}
        st.code(config_to_yaml_string(preview_config), language="yaml")

    # Reset button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Reset to Defaults", type="secondary"):
            base_config = load_base_config()
            st.session_state.config["tools"] = base_config.get("tools", {})
            st.success("Tools configuration reset to defaults.")
            st.rerun()


if __name__ == "__main__":
    main()
