"""
Agent Configuration Page

Configure agent prompts, models, temperature, and token limits.
"""

import streamlit as st
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_manager import load_base_config, config_to_yaml_string

st.set_page_config(
    page_title="Agent Configuration - TriageFlow",
    layout="wide"
)


def init_page_state():
    """Initialize page-specific session state."""
    if "config" not in st.session_state:
        st.session_state.config = load_base_config()

    if "provider" not in st.session_state:
        st.session_state.provider = "openai"


def render_agent_config(agent_name: str, display_name: str, description: str):
    """Render configuration controls for a single agent."""
    config = st.session_state.config
    agent_config = config.get("agents", {}).get(agent_name, {})

    with st.expander(f"{display_name}", expanded=False):
        st.caption(description)

        col1, col2 = st.columns(2)

        with col1:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=float(agent_config.get("temperature", 0.5)),
                step=0.1,
                key=f"{agent_name}_temp",
                help="Lower values produce more deterministic outputs, higher values increase creativity"
            )

        with col2:
            max_tokens = st.number_input(
                "Max Tokens",
                min_value=100,
                max_value=4000,
                value=int(agent_config.get("max_tokens", 1000)),
                step=100,
                key=f"{agent_name}_tokens",
                help="Maximum number of tokens in the response"
            )

        prompt = st.text_area(
            "System Prompt",
            value=agent_config.get("prompt", ""),
            height=300,
            key=f"{agent_name}_prompt",
            help="The system prompt sent to the LLM for this agent"
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Reset to Default", key=f"{agent_name}_reset"):
                base_config = load_base_config()
                base_agent = base_config.get("agents", {}).get(agent_name, {})
                st.session_state.config["agents"][agent_name] = base_agent.copy()
                st.rerun()

        # Update config in session state
        if "agents" not in st.session_state.config:
            st.session_state.config["agents"] = {}
        if agent_name not in st.session_state.config["agents"]:
            st.session_state.config["agents"][agent_name] = {}

        st.session_state.config["agents"][agent_name]["temperature"] = temperature
        st.session_state.config["agents"][agent_name]["max_tokens"] = max_tokens
        st.session_state.config["agents"][agent_name]["prompt"] = prompt


def main():
    """Main page content."""
    init_page_state()

    st.title("Agent Configuration")
    st.markdown("Configure the four agents in the TriageFlow pipeline.")

    st.divider()

    # Provider and Model Selection
    st.markdown("## Provider and Model")

    col1, col2 = st.columns(2)

    with col1:
        provider = st.radio(
            "LLM Provider",
            options=["openai", "anthropic"],
            index=0 if st.session_state.provider == "openai" else 1,
            horizontal=True,
            help="Select the LLM provider for all agents"
        )
        st.session_state.provider = provider

    with col2:
        current_model = st.session_state.config.get("models", {}).get(provider, "")
        default_models = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-sonnet-4-20250514"
        }

        model = st.text_input(
            f"{provider.title()} Model",
            value=current_model or default_models[provider],
            help="Model identifier to use for this provider"
        )

        # Update model in config
        if "models" not in st.session_state.config:
            st.session_state.config["models"] = {}
        st.session_state.config["models"][provider] = model

    st.divider()

    # Agent Configurations
    st.markdown("## Agent Settings")

    st.markdown("""
    Each agent has specific responsibilities in the triage pipeline:

    - **Classifier**: Categorizes incidents and assigns urgency levels
    - **Impact Assessor**: Evaluates blast radius and affected users/systems
    - **Resource Matcher**: Assigns responders based on skills and availability
    - **Response Drafter**: Creates communications for different stakeholders
    """)

    # Classifier Agent
    render_agent_config(
        "classifier",
        "Classifier Agent",
        "Categorizes incidents into predefined categories and assigns urgency levels (1-5)."
    )

    # Impact Assessor Agent
    render_agent_config(
        "impact_assessor",
        "Impact Assessor Agent",
        "Evaluates the impact of incidents including affected users, systems, and blast radius."
    )

    # Resource Matcher Agent
    render_agent_config(
        "resource_matcher",
        "Resource Matcher Agent",
        "Matches incidents to available resources based on required skills and SLA requirements."
    )

    # Response Drafter Agent
    render_agent_config(
        "response_drafter",
        "Response Drafter Agent",
        "Drafts communications for technical teams, management, and affected users."
    )

    st.divider()

    # Configuration Preview
    st.markdown("## Configuration Preview")

    with st.expander("View Current Configuration (YAML)", expanded=False):
        # Show only agents and models sections
        preview_config = {
            "models": st.session_state.config.get("models", {}),
            "agents": st.session_state.config.get("agents", {})
        }
        st.code(config_to_yaml_string(preview_config), language="yaml")

    # Actions
    col1, col2, col3 = st.columns([1, 1, 3])

    with col1:
        if st.button("Reset All to Defaults", type="secondary"):
            st.session_state.config = load_base_config()
            st.success("Configuration reset to defaults.")
            st.rerun()

    with col2:
        if st.button("Apply Changes", type="primary"):
            st.success("Changes applied to current session.")


if __name__ == "__main__":
    main()
