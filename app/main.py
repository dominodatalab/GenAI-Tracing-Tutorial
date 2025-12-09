"""
TriageFlow Configuration App

A Streamlit application for configuring and launching the TriageFlow
multi-agent incident triage system.
"""

import streamlit as st

# Page configuration must be first Streamlit command
st.set_page_config(
    page_title="TriageFlow Configuration",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, professional styling
st.markdown("""
<style>
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Headers */
    h1 {
        color: #1a1a2e;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    h2 {
        color: #2d2d44;
        font-weight: 500;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 0.5rem;
    }

    h3 {
        color: #3d3d5c;
        font-weight: 500;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        font-size: 0.9rem;
    }

    /* Card-like containers */
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        margin-bottom: 1rem;
    }

    /* Button styling */
    .stButton > button {
        border-radius: 4px;
        font-weight: 500;
    }

    /* Primary buttons */
    .stButton > button[kind="primary"] {
        background-color: #1a73e8;
        border-color: #1a73e8;
    }

    /* Text inputs */
    .stTextInput > div > div > input {
        border-radius: 4px;
    }

    /* Select boxes */
    .stSelectbox > div > div {
        border-radius: 4px;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: #1a1a2e;
    }

    /* Info boxes */
    .stAlert {
        border-radius: 4px;
    }

    /* Tables */
    .stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 4px;
    }

    /* Code blocks */
    .stCodeBlock {
        border-radius: 4px;
    }

    /* Dividers */
    hr {
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        border-color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state with default values."""
    from utils.config_manager import load_base_config

    if "config" not in st.session_state:
        st.session_state.config = load_base_config()

    if "provider" not in st.session_state:
        st.session_state.provider = "openai"

    if "selected_vertical" not in st.session_state:
        st.session_state.selected_vertical = "financial_services"

    if "num_tickets" not in st.session_state:
        st.session_state.num_tickets = 0

    if "last_job_result" not in st.session_state:
        st.session_state.last_job_result = None

    # Judge configuration (not in base config.yaml, so we store separately)
    if "judge_config" not in st.session_state:
        st.session_state.judge_config = {
            "classification_judge": {
                "temperature": 0.1,
                "max_tokens": 200,
                "prompt": """Evaluate this incident classification. Score 1-5 (5=excellent).

Incident: {incident}

Classification:
- Category: {category}
- Subcategory: {subcategory}
- Urgency: {urgency}
- Reasoning: {reasoning}

Evaluate:
1. Is the category appropriate for this incident?
2. Is the urgency level justified by the description?
3. Is the reasoning sound?

Return JSON: {{"score": <1-5>, "rationale": "<brief explanation>"}}"""
            },
            "response_judge": {
                "temperature": 0.1,
                "max_tokens": 200,
                "prompt": """Evaluate this incident communication. Score 1-5 (5=excellent).

Incident: {incident}
Urgency: {urgency}

Communication to {audience}:
Subject: {subject}
Body: {body}

Evaluate:
1. Is the tone appropriate for the audience?
2. Is the information clear and actionable?
3. Does it convey appropriate urgency?

Return JSON: {{"score": <1-5>, "rationale": "<brief explanation>"}}"""
            },
            "triage_judge": {
                "temperature": 0.1,
                "max_tokens": 200,
                "prompt": """Evaluate this complete incident triage. Score 1-5 (5=excellent).

Incident: {incident}

Triage Summary:
- Category: {category} (Urgency: {urgency})
- Impact Score: {impact_score}, Blast Radius: {blast_radius}
- Primary Responder: {responder} (Match Score: {match_score})
- SLA Met: {sla_met}
- Action Items: {action_count}

Evaluate the overall triage quality:
1. Is the classification appropriate?
2. Is the resource assignment logical?
3. Is the response plan comprehensive?

Return JSON: {{"score": <1-5>, "rationale": "<brief explanation>"}}"""
            }
        }

    # Ad-hoc evaluation thresholds
    if "adhoc_thresholds" not in st.session_state:
        st.session_state.adhoc_thresholds = {
            "urgency_threshold": 4,
            "impact_threshold": 7.0
        }


def main():
    """Main application entry point."""
    init_session_state()

    # Sidebar
    with st.sidebar:
        st.title("TriageFlow")
        st.caption("Multi-Agent Incident Triage Configuration")

        st.divider()

        st.markdown("### Navigation")
        st.markdown("""
        Use the pages in the sidebar to:

        1. **Agent Configuration** - Configure agent prompts, models, and parameters
        2. **Tools Configuration** - View and configure agent tools
        3. **Evaluators** - Configure LLM judges and evaluation thresholds
        4. **Run Job** - Select tickets and launch triage jobs
        """)

        st.divider()

        # Quick status
        st.markdown("### Current Settings")
        st.markdown(f"**Provider:** {st.session_state.provider}")
        st.markdown(f"**Model:** {st.session_state.config['models'][st.session_state.provider]}")
        st.markdown(f"**Vertical:** {st.session_state.selected_vertical}")

        if st.session_state.last_job_result:
            st.divider()
            st.markdown("### Last Job")
            job_id = st.session_state.last_job_result.get("job_id", "N/A")
            st.markdown(f"**Job ID:** {job_id}")

    # Main content area
    st.title("TriageFlow Configuration")

    st.markdown("""
    Welcome to the TriageFlow Configuration App. This application allows you to configure
    and launch the multi-agent incident triage pipeline.

    **Getting Started:**

    1. Configure your agents on the **Agent Configuration** page
    2. Review tool settings on the **Tools Configuration** page
    3. Adjust evaluator parameters on the **Evaluators** page
    4. Select tickets and launch a job on the **Run Job** page
    """)

    st.divider()

    # Overview cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("#### Agents")
        st.markdown("4 specialized agents")
        st.caption("Classifier, Impact Assessor, Resource Matcher, Response Drafter")

    with col2:
        st.markdown("#### Tools")
        st.markdown("7 tool functions")
        st.caption("Category lookup, historical search, SLA checks, and more")

    with col3:
        st.markdown("#### Evaluators")
        st.markdown("3 LLM judges")
        st.caption("Classification, Response, and Triage quality scoring")

    with col4:
        st.markdown("#### Verticals")
        st.markdown("4 industry verticals")
        st.caption("Financial Services, Healthcare, Energy, Public Sector")

    st.divider()

    # Configuration summary
    st.markdown("### Current Configuration Summary")

    config = st.session_state.config

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Agent Settings**")
        agent_data = []
        for agent_name in ["classifier", "impact_assessor", "resource_matcher", "response_drafter"]:
            agent = config.get("agents", {}).get(agent_name, {})
            agent_data.append({
                "Agent": agent_name.replace("_", " ").title(),
                "Temperature": agent.get("temperature", "N/A"),
                "Max Tokens": agent.get("max_tokens", "N/A")
            })

        st.dataframe(agent_data, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**Model Configuration**")
        st.markdown(f"- **OpenAI Model:** {config.get('models', {}).get('openai', 'N/A')}")
        st.markdown(f"- **Anthropic Model:** {config.get('models', {}).get('anthropic', 'N/A')}")

        st.markdown("**SLA Configuration**")
        sla = config.get("sla", {})
        st.markdown(f"- Urgency 5: {sla.get('urgency_5', {}).get('response_hours', 'N/A')}h response")
        st.markdown(f"- Urgency 1: {sla.get('urgency_1', {}).get('response_hours', 'N/A')}h response")


if __name__ == "__main__":
    main()
