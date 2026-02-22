"""
TriageFlow - Multi-Agent Incident Triage System

A single-page Streamlit application for running incident triage,
viewing agent responses and reasoning, and adding user evaluations.

This app demonstrates Domino GenAI tracing integration:
1. Import tracing components and initialize
2. Create evaluator functions to extract metrics from outputs
3. Use @add_tracing decorator to create traced functions
4. Run within DominoRun context to capture traces
5. Log human feedback to traces with log_evaluation()

See: https://docs.dominodatalab.com/en/cloud/user_guide/fc1922/set-up-and-run-genai-traces/
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/mnt/code")

# =============================================================================
# TRACING STEP 1: Import and Initialize Tracing Components
# =============================================================================
# Import the key tracing components from the Domino SDK:
# - add_tracing: Decorator that creates traced spans with inputs/outputs
# - init_tracing: Initializes the tracing system
# - DominoRun: Context manager that creates an MLflow run for storing traces
# - log_evaluation: Attaches evaluation scores to specific traces
# - mlflow: Provides autologging for LLM calls
# =============================================================================
try:
    from domino.agents.tracing import add_tracing, init_tracing
    from domino.agents.logging import DominoRun, log_evaluation
    import mlflow
    init_tracing()  # Initialize tracing system
except ImportError as e:
    st.error(
        "Domino SDK is required but not installed. "
        "Please install it with: `pip install dominodatalab[data,aisystems]`"
    )
    st.info(f"Import error: {e}")
    st.stop()

from utils.config_manager import load_base_config, load_sample_tickets, VERTICALS
from src.models import Incident, IncidentSource
from src.agents import classify_incident, assess_impact, match_resources, draft_response
from src.judges import judge_classification, judge_response, judge_triage

# Page configuration
st.set_page_config(
    page_title="TriageFlow",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 { color: #1a1a2e; font-weight: 600; }
    h2 { color: #2d2d44; font-weight: 500; border-bottom: 1px solid #e0e0e0; padding-bottom: 0.5rem; }
    h3 { color: #3d3d5c; font-weight: 500; }
    .stExpander { border: 1px solid #e0e0e0; border-radius: 4px; margin-bottom: 1rem; }
    .stButton > button { border-radius: 4px; font-weight: 500; }
    [data-testid="stMetricValue"] { font-size: 1.5rem; color: #1a1a2e; }
    .stAlert { border-radius: 4px; }
    hr { margin-top: 1.5rem; margin-bottom: 1.5rem; border-color: #e0e0e0; }
</style>
""", unsafe_allow_html=True)

VERTICAL_DISPLAY_NAMES = {
    "financial_services": "Financial Services",
    "healthcare": "Healthcare",
    "energy": "Energy",
    "public_sector": "Public Sector"
}

SOURCE_OPTIONS = ["monitoring", "user_report", "automated_scan", "external_notification", "audit"]


def clean_nan(value):
    """Convert pandas nan/None to None for Pydantic validation."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return value


def init_session_state():
    """Initialize session state."""
    if "config" not in st.session_state:
        st.session_state.config = load_base_config()
    if "provider" not in st.session_state:
        st.session_state.provider = "openai"
    if "demo_vertical" not in st.session_state:
        st.session_state.demo_vertical = "financial_services"
    if "demo_result" not in st.session_state:
        st.session_state.demo_result = None
    if "user_evaluation" not in st.session_state:
        st.session_state.user_evaluation = {}


def initialize_client(provider: str, enable_autolog: bool = False):
    """Initialize LLM client and optionally enable autolog for tracing."""
    if provider == "openai":
        from openai import OpenAI
        if enable_autolog and mlflow:
            mlflow.openai.autolog()
        return OpenAI()
    else:
        from anthropic import Anthropic
        if enable_autolog and mlflow:
            mlflow.anthropic.autolog()
        return Anthropic()


# =============================================================================
# TRACING STEP 2: Define an Evaluator Function
# =============================================================================
# The evaluator extracts metrics from the traced function's output.
# It receives a span object with .outputs containing the function's return value.
# Return a dict of metric_name -> value pairs (numeric or string).
#
# These metrics are automatically:
# - Attached to each trace for filtering in the Trace Explorer
# - Aggregated across traces when using DominoRun with custom_summary_metrics
# =============================================================================
def pipeline_evaluator(span) -> dict:
    """
    Extract metrics from triage pipeline outputs for automatic evaluation.

    Args:
        span: MLflow span object with .outputs containing function return value

    Returns:
        Dict of metric names to values (numeric for aggregation, strings for labels)
    """
    try:
        output = span.outputs or {}

        judge_scores = output.get("judge_scores", {})

        classification = output.get("classification")
        if hasattr(classification, "model_dump"):
            classification = classification.model_dump()
        elif classification is None:
            classification = {}

        impact = output.get("impact")
        if hasattr(impact, "model_dump"):
            impact = impact.model_dump()
        elif impact is None:
            impact = {}

        assignment = output.get("assignment")
        if hasattr(assignment, "model_dump"):
            assignment = assignment.model_dump()
        elif assignment is None:
            assignment = {}

        response = output.get("response")
        if hasattr(response, "model_dump"):
            response = response.model_dump()
        elif response is None:
            response = {}

        metrics = {
            "classification_judge_score": float(judge_scores.get("classification_score", 0.0)),
            "response_judge_score": float(judge_scores.get("response_score", 0.0)),
            "triage_judge_score": float(judge_scores.get("triage_score", 0.0)),
            "combined_quality_score": float(judge_scores.get("combined_score", 0.0)),
            "classification_confidence": float(classification.get("confidence", 0.0)),
            "urgency": int(classification.get("urgency", 0)),
            "impact_score": float(impact.get("impact_score", 0.0)),
            "affected_users_estimate": int(impact.get("affected_users_estimate", 0)),
            "completeness_score": float(response.get("completeness_score", 0.0)),
            "pipeline_success": 1 if output and not output.get("error") else 0,
            "sla_met": 1 if assignment.get("sla_met") else 0,
        }

        primary = assignment.get("primary_responder", {})
        if hasattr(primary, "model_dump"):
            primary = primary.model_dump()
        metrics["resource_match_score"] = float(primary.get("match_score", 0.0))

        category = classification.get("category", "unknown")
        if hasattr(category, "value"):
            category = category.value
        metrics["category"] = str(category)
        metrics["blast_radius"] = str(impact.get("blast_radius", "unknown"))
        metrics["responder"] = str(primary.get("name", "unknown"))

        return metrics

    except Exception as e:
        print(f"Evaluation error: {e}")
        return {
            "classification_judge_score": 0.0,
            "response_judge_score": 0.0,
            "triage_judge_score": 0.0,
            "combined_quality_score": 0.0,
            "pipeline_success": 0
        }


# =============================================================================
# TRACING STEP 3: Create a Traced Function with @add_tracing
# =============================================================================
# The @add_tracing decorator:
# - Creates a parent span that captures function inputs and outputs
# - Nests all LLM calls (from autolog) under this parent span
# - Runs the evaluator on completion to extract metrics
#
# Parameters:
# - name: Name shown in the trace explorer (e.g., "triage_incident")
# - autolog_frameworks: List of LLM frameworks to capture (e.g., ["openai"])
# - evaluator: Function that extracts metrics from the output
# =============================================================================
def create_traced_triage_function(client, provider: str, model: str, config: dict):
    """Create a traced triage function for the given provider."""
    tracing_framework = "openai" if provider == "openai" else provider

    @add_tracing(
        name="triage_incident",           # Span name in trace explorer
        autolog_frameworks=[tracing_framework],  # Capture LLM calls
        evaluator=pipeline_evaluator      # Extract metrics from output
    )
    def triage_incident(incident: Incident) -> dict:
        """
        Run the full triage pipeline for an incident.

        All LLM calls within this function are automatically captured
        as child spans under the 'triage_incident' parent span.
        """
        # Run the 4 agent pipeline
        classification = classify_incident(client, provider, model, incident, config)
        impact = assess_impact(client, provider, model, incident, classification, config)
        assignment = match_resources(client, provider, model, classification, impact, config)
        response = draft_response(client, provider, model, incident, classification, impact, assignment, config)

        triage_output = {
            "classification": classification,
            "impact": impact,
            "assignment": assignment,
            "response": response,
        }

        # Run judge evaluations inside the traced function
        classification_eval = judge_classification(
            client, provider, model, incident.description, classification.model_dump()
        )
        response_evals = judge_response(
            client, provider, model, incident.description, response.model_dump()
        )
        triage_eval = judge_triage(
            client, provider, model, incident.description, {
                "classification": classification.model_dump(),
                "impact": impact.model_dump(),
                "assignment": assignment.model_dump(),
                "response": response.model_dump()
            }
        )

        # Compute scores
        classification_score = classification_eval.get("score", 3) if classification_eval else 3
        triage_score = triage_eval.get("score", 3) if triage_eval else 3
        response_scores = [e.get("score", 3) for e in response_evals] if response_evals else [3]
        response_score = sum(response_scores) / len(response_scores)
        combined_score = (classification_score + response_score + triage_score) / 3

        triage_output["judge_scores"] = {
            "classification_score": classification_score,
            "classification_rationale": classification_eval.get("rationale", "") if classification_eval else "",
            "response_score": response_score,
            "triage_score": triage_score,
            "triage_rationale": triage_eval.get("rationale", "") if triage_eval else "",
            "combined_score": combined_score,
        }

        # Log evaluations explicitly using log_evaluation and capture trace_id for feedback
        trace_id = None
        if mlflow and log_evaluation:
            span = mlflow.get_current_active_span()
            if span:
                trace_id = span.request_id
                class_dict = classification.model_dump()
                impact_dict = impact.model_dump()
                assignment_dict = assignment.model_dump()
                response_dict = response.model_dump()
                primary = assignment_dict.get("primary_responder", {})

                # Log numeric metrics
                log_evaluation(trace_id=trace_id, name="classification_judge_score", value=float(classification_score))
                log_evaluation(trace_id=trace_id, name="response_judge_score", value=float(response_score))
                log_evaluation(trace_id=trace_id, name="triage_judge_score", value=float(triage_score))
                log_evaluation(trace_id=trace_id, name="combined_quality_score", value=round(combined_score, 2))
                log_evaluation(trace_id=trace_id, name="classification_confidence", value=float(class_dict.get("confidence", 0.5)))
                log_evaluation(trace_id=trace_id, name="urgency", value=float(class_dict.get("urgency", 0)))
                log_evaluation(trace_id=trace_id, name="impact_score", value=float(impact_dict.get("impact_score", 0.0)))
                log_evaluation(trace_id=trace_id, name="affected_users_estimate", value=float(impact_dict.get("affected_users_estimate", 0)))
                log_evaluation(trace_id=trace_id, name="completeness_score", value=float(response_dict.get("completeness_score", 0.0)))
                log_evaluation(trace_id=trace_id, name="resource_match_score", value=float(primary.get("match_score", 0.0) if isinstance(primary, dict) else 0.0))
                log_evaluation(trace_id=trace_id, name="pipeline_success", value=1.0)
                log_evaluation(trace_id=trace_id, name="sla_met", value=1.0 if assignment_dict.get("sla_met") else 0.0)

                needs_review = class_dict.get("urgency", 0) >= 4 and impact_dict.get("impact_score", 0) >= 7
                log_evaluation(trace_id=trace_id, name="needs_manual_review", value=1.0 if needs_review else 0.0)

                category = class_dict.get("category", "unknown")
                if hasattr(category, "value"):
                    category = category.value
                log_evaluation(trace_id=trace_id, name="category", value=str(category))
                log_evaluation(trace_id=trace_id, name="blast_radius", value=str(impact_dict.get("blast_radius", "unknown")))
                log_evaluation(trace_id=trace_id, name="responder", value=str(primary.get("name", "unknown") if isinstance(primary, dict) else "unknown"))

        # Store trace_id in output for later human feedback logging
        triage_output["_trace_id"] = trace_id

        return triage_output

    return triage_incident


# =============================================================================
# TRACING STEP 4: Run Within DominoRun Context
# =============================================================================
# DominoRun creates an MLflow run that:
# - Stores all traces from @add_tracing functions
# - Aggregates metrics across traces (mean, median, etc.)
# - Links configuration for reproducibility
#
# The run_id is captured for post-hoc operations like adding human feedback.
# =============================================================================
def run_triage_with_tracing(provider: str, model: str, incident: Incident, config: dict, config_path: str = None):
    """Run triage pipeline with full Domino tracing and metrics."""
    if config_path is None:
        config_path = "/mnt/code/config.yaml"

    # Enable MLflow autologging BEFORE making LLM calls
    # This captures all API calls including request/response details
    if mlflow:
        if provider == "openai":
            mlflow.openai.autolog()  # Captures all OpenAI API calls
        else:
            mlflow.anthropic.autolog()  # Captures all Anthropic API calls

    client = initialize_client(provider, enable_autolog=False)
    triage_incident = create_traced_triage_function(client, provider, model, config)

    # Run within DominoRun context to create MLflow run
    with DominoRun(agent_config_path=config_path) as run:
        result = triage_incident(incident)

        # Get run_id from DominoRun context
        run_id = run.info.run_id if hasattr(run, 'info') else None

        # trace_id is already captured inside create_traced_triage_function
        # Just add run_id for completeness
        result["_run_id"] = run_id
        return result


def render_ticket_selector():
    """Render ticket selection controls."""
    col1, col2 = st.columns([1, 2])

    with col1:
        vertical = st.selectbox(
            "Industry Vertical",
            options=VERTICALS,
            format_func=lambda x: VERTICAL_DISPLAY_NAMES.get(x, x),
            index=VERTICALS.index(st.session_state.demo_vertical) if st.session_state.demo_vertical in VERTICALS else 0
        )
        st.session_state.demo_vertical = vertical

    tickets = load_sample_tickets(vertical)

    with col2:
        ticket_options = {t["ticket_id"]: t for t in tickets}
        selected_ticket_id = st.selectbox(
            "Select Ticket",
            options=list(ticket_options.keys()),
            format_func=lambda x: f"{x} - {ticket_options[x]['description'][:60]}..."
        )

    return ticket_options.get(selected_ticket_id), tickets


def render_ticket_details(ticket: dict):
    """Render detailed view of selected ticket."""
    st.markdown("### Ticket Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Ticket ID:** {ticket.get('ticket_id', 'N/A')}")
    with col2:
        st.markdown(f"**Source:** {ticket.get('source', 'N/A')}")
    with col3:
        st.markdown(f"**Severity:** {ticket.get('initial_severity', 'N/A')}")

    if ticket.get("affected_system"):
        st.markdown(f"**Affected System:** {ticket.get('affected_system')}")

    if ticket.get("reporter"):
        st.markdown(f"**Reporter:** {ticket.get('reporter')}")

    st.markdown("**Description:**")
    description = ticket.get("description", "")
    num_lines = max(5, min(15, len(description) // 80 + description.count('\n') + 2))
    st.text_area(
        "Description",
        value=description,
        height=num_lines * 22,
        disabled=True,
        label_visibility="collapsed"
    )


def render_custom_ticket_form():
    """Render form for custom ticket input."""
    with st.expander("Enter Custom Ticket", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            custom_id = st.text_input("Ticket ID", value="CUSTOM-001", key="custom_ticket_id")
            custom_source = st.selectbox("Source", options=SOURCE_OPTIONS, key="custom_source")
            custom_severity = st.number_input("Initial Severity", min_value=1, max_value=5, value=3, key="custom_severity")

        with col2:
            custom_reporter = st.text_input("Reporter (Optional)", key="custom_reporter")
            custom_system = st.text_input("Affected System (Optional)", key="custom_system")

        custom_description = st.text_area(
            "Description",
            placeholder="Describe the incident in detail...",
            height=120,
            key="custom_description"
        )

        use_custom = st.checkbox("Use this custom ticket instead", key="use_custom_ticket")

        if use_custom and custom_id and custom_description:
            return {
                "ticket_id": custom_id,
                "description": custom_description,
                "source": custom_source,
                "reporter": custom_reporter or None,
                "affected_system": custom_system or None,
                "initial_severity": custom_severity
            }

    return None


def render_agent_results(results: dict, incident: Incident):
    """Render agent responses with reasoning."""
    st.markdown("## Agent Responses")

    classification = results["stages"]["classification"]["result"]
    impact = results["stages"]["impact"]["result"]
    resources = results["stages"]["resources"]["result"]
    response = results["stages"]["response"]["result"]

    st.markdown(f"""
    **Summary:** This **{classification.category.value.replace('_', ' ')}** incident
    with urgency level **{classification.urgency}/5** has an estimated impact score of **{impact.impact_score:.1f}/10**,
    affecting approximately **{impact.affected_users_estimate:,}** users.
    **{resources.primary_responder.name}** has been assigned as the primary responder.
    """)

    st.divider()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Category", classification.category.value.replace("_", " ").title())
    with col2:
        st.metric("Urgency", f"{classification.urgency} / 5")
    with col3:
        st.metric("Impact Score", f"{impact.impact_score:.1f} / 10")
    with col4:
        st.metric("SLA Met", "Yes" if resources.sla_met else "No")
    with col5:
        st.metric("Quality Score", f"{results['combined_quality']:.1f} / 5")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Classification",
        "2. Impact Assessment",
        "3. Resource Assignment",
        "4. Response Plan"
    ])

    with tab1:
        st.markdown("### Classifier Agent")
        st.caption(f"Execution time: {results['stages']['classification']['time']:.2f}s")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Category:** {classification.category.value}")
            st.markdown(f"**Subcategory:** {classification.subcategory}")
            st.markdown(f"**Urgency:** {classification.urgency}/5")
        with col2:
            st.markdown(f"**Confidence:** {classification.confidence:.0%}")
            st.markdown(f"**Affected Domain:** {classification.affected_domain}")

        st.markdown("**Agent Reasoning:**")
        st.info(classification.reasoning)

    with tab2:
        st.markdown("### Impact Assessor Agent")
        st.caption(f"Execution time: {results['stages']['impact']['time']:.2f}s")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Impact Score:** {impact.impact_score:.1f}/10")
            st.markdown(f"**Affected Users (Est.):** {impact.affected_users_estimate:,}")
            st.markdown(f"**Blast Radius:** {impact.blast_radius}")
        with col2:
            if impact.financial_exposure:
                st.markdown(f"**Financial Exposure:** ${impact.financial_exposure:,.2f}")
            st.markdown(f"**Affected Systems:** {len(impact.affected_systems)}")

        if impact.affected_systems:
            with st.expander("Affected Systems", expanded=False):
                for system in impact.affected_systems:
                    st.markdown(f"- {system}")

        if impact.similar_incidents:
            with st.expander("Similar Historical Incidents", expanded=False):
                for inc in impact.similar_incidents[:3]:
                    st.markdown(f"- **{inc.get('incident_id', 'Unknown')}**: {inc.get('summary', 'No summary')}")

        st.markdown("**Agent Reasoning:**")
        st.info(impact.reasoning)

    with tab3:
        st.markdown("### Resource Matcher Agent")
        st.caption(f"Execution time: {results['stages']['resources']['time']:.2f}s")

        primary = resources.primary_responder
        st.markdown("#### Primary Responder")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Name:** {primary.name}")
            st.markdown(f"**Role:** {primary.role}")
        with col2:
            st.markdown(f"**Availability:** {primary.availability}")
            st.markdown(f"**Match Score:** {primary.match_score:.0%}")
        with col3:
            st.markdown(f"**Skills:** {', '.join(primary.skills[:3])}")

        if resources.backup_responders:
            with st.expander(f"Backup Responders ({len(resources.backup_responders)})", expanded=False):
                for backup in resources.backup_responders:
                    st.markdown(f"- **{backup.name}** ({backup.role}) - Match: {backup.match_score:.0%}")

        st.markdown("#### SLA Information")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**SLA Target:** {resources.sla_target_hours} hours")
            st.markdown(f"**SLA Met:** {'Yes' if resources.sla_met else 'No'}")
        with col2:
            if resources.escalation_path:
                st.markdown(f"**Escalation Path:** {' -> '.join(resources.escalation_path)}")

        st.markdown("**Agent Reasoning:**")
        st.info(resources.reasoning)

    with tab4:
        st.markdown("### Response Drafter Agent")
        st.caption(f"Execution time: {results['stages']['response']['time']:.2f}s")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Est. Resolution Time:** {response.estimated_resolution_time}")
            st.markdown(f"**Escalation Required:** {'Yes' if response.escalation_required else 'No'}")
        with col2:
            st.markdown(f"**Completeness Score:** {response.completeness_score:.0%}")
            st.markdown(f"**Action Items:** {len(response.action_items)}")

        if response.communications:
            st.markdown("#### Communications")
            comm_tabs = st.tabs([comm.audience.title() for comm in response.communications])
            for tab, comm in zip(comm_tabs, response.communications):
                with tab:
                    st.markdown(f"**Subject:** {comm.subject}")
                    st.markdown(f"**Urgency Indicator:** {comm.urgency_indicator}")
                    st.text_area(
                        "Message Body",
                        value=comm.body,
                        height=200,
                        disabled=True,
                        key=f"comm_{comm.audience}",
                        label_visibility="collapsed"
                    )

        if response.action_items:
            with st.expander("Action Items", expanded=False):
                for i, item in enumerate(response.action_items, 1):
                    st.markdown(f"{i}. {item}")


def render_llm_judge_results(judges: dict, combined_score: float):
    """Render LLM judge evaluation results."""
    st.markdown("## LLM Judge Evaluation")
    st.markdown("Three LLM judges evaluate the triage output independently, scoring each aspect on a 1-5 scale.")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        class_score = judges.get("classification", {}).get("score", 0)
        st.metric("Classification", f"{class_score} / 5")
        st.caption("Category and urgency accuracy")

    with col2:
        resp_score = judges.get("response", {}).get("score", 0)
        st.metric("Response", f"{resp_score:.1f} / 5")
        st.caption("Communication quality")

    with col3:
        triage_score = judges.get("triage", {}).get("score", 0)
        st.metric("Triage", f"{triage_score} / 5")
        st.caption("Overall completeness")

    with col4:
        st.metric("Combined", f"{combined_score:.1f} / 5")
        st.caption("Average of all judges")

    if combined_score >= 4.0:
        quality_level = "Excellent"
        quality_desc = "The triage output meets high quality standards and can proceed without additional review."
    elif combined_score >= 3.0:
        quality_level = "Good"
        quality_desc = "The triage output is acceptable with minor areas for potential improvement."
    elif combined_score >= 2.0:
        quality_level = "Fair"
        quality_desc = "The triage output may need review before taking action."
    else:
        quality_level = "Needs Review"
        quality_desc = "The triage output should be manually reviewed before action."

    st.markdown(f"**Quality Level:** {quality_level}")
    st.caption(quality_desc)

    st.markdown("#### Judge Rationales")

    with st.expander("Classification Judge", expanded=False):
        st.markdown(judges.get("classification", {}).get("rationale", "No rationale provided"))

    with st.expander("Response Judge", expanded=False):
        st.markdown(judges.get("response", {}).get("rationale", "No rationale provided"))

    with st.expander("Triage Judge", expanded=False):
        st.markdown(judges.get("triage", {}).get("rationale", "No rationale provided"))


def get_feedback_storage_path() -> str:
    """Get the path to store user feedback."""
    project_name = os.environ.get("DOMINO_PROJECT_NAME", "triageflow")
    base_path = f"/mnt/data/{project_name}/feedback"
    os.makedirs(base_path, exist_ok=True)
    return base_path


# =============================================================================
# TRACING STEP 5: Log Human Feedback to Traces
# =============================================================================
# After a trace is created, you can attach additional evaluations using
# log_evaluation(trace_id=..., name=..., value=...).
#
# This is useful for:
# - Adding human feedback scores to production traces
# - Comparing human vs automated evaluations
# - Flagging traces that need review
#
# Human evaluations appear alongside automated metrics in the Trace Explorer.
# =============================================================================
def save_user_feedback(ticket_id: str, evaluation: dict, trace_id: str = None, run_id: str = None):
    """
    Save user feedback to local storage and log to production tracing.

    Args:
        ticket_id: The ticket ID being evaluated
        evaluation: Dictionary containing user evaluation scores and comments
        trace_id: Trace ID for logging to Domino tracing (from @add_tracing span)
        run_id: Run ID for reference (from DominoRun context)
    """
    import json
    from datetime import datetime

    # Add metadata
    feedback_record = {
        "ticket_id": ticket_id,
        "timestamp": datetime.now().isoformat(),
        "evaluator": os.environ.get("DOMINO_USER_NAME", os.environ.get("USER", "anonymous")),
        "trace_id": trace_id,
        "run_id": run_id,
        **evaluation
    }

    # Save to local JSON file
    storage_path = get_feedback_storage_path()
    feedback_file = os.path.join(storage_path, "user_feedback.jsonl")

    with open(feedback_file, "a") as f:
        f.write(json.dumps(feedback_record) + "\n")

    # Log to production tracing if trace_id is available
    if trace_id:
        try:
            # Log numeric scores
            log_evaluation(trace_id=trace_id, name="human_classification_score", value=float(evaluation["classification_score"]))
            log_evaluation(trace_id=trace_id, name="human_impact_score", value=float(evaluation["impact_score"]))
            log_evaluation(trace_id=trace_id, name="human_assignment_score", value=float(evaluation["assignment_score"]))
            log_evaluation(trace_id=trace_id, name="human_response_score", value=float(evaluation["response_score"]))
            log_evaluation(trace_id=trace_id, name="human_combined_score", value=float(evaluation["combined_score"]))

            # Log categorical assessments as numeric values for filtering
            assessment_map = {"Excellent": 5, "Good": 4, "Acceptable": 3, "Needs Improvement": 2, "Poor": 1}
            log_evaluation(trace_id=trace_id, name="human_overall_assessment", value=float(assessment_map.get(evaluation["overall_assessment"], 3)))

            usability_map = {"Yes, as-is": 3, "Yes, with minor edits": 2, "No, needs significant revision": 1}
            log_evaluation(trace_id=trace_id, name="human_usability_score", value=float(usability_map.get(evaluation["would_use"], 2)))

            # Log string labels
            log_evaluation(trace_id=trace_id, name="human_assessment_label", value=evaluation["overall_assessment"])
            log_evaluation(trace_id=trace_id, name="human_usability_label", value=evaluation["would_use"])

            # Flag if human review indicates issues
            human_flagged = evaluation["combined_score"] < 3 or evaluation["overall_assessment"] in ["Needs Improvement", "Poor"]
            log_evaluation(trace_id=trace_id, name="human_flagged_issues", value=1.0 if human_flagged else 0.0)

            return True, "Feedback logged to tracing"
        except Exception as e:
            return False, f"Failed to log to tracing: {e}"

    return True, "Feedback saved locally"


def render_user_evaluation():
    """Render user evaluation form."""
    st.markdown("## Your Evaluation")
    st.markdown("Provide your own assessment of the triage quality.")

    # Show tracing status
    trace_info = st.session_state.demo_result.get("trace_id") if st.session_state.demo_result else None
    if trace_info:
        st.caption(f"Trace ID: {trace_info[:16]}... (feedback will be logged to production tracing)")
    elif st.session_state.demo_result and st.session_state.demo_result.get("tracing_enabled"):
        st.caption("Tracing was enabled - feedback will be logged to production tracing")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Scoring")

        classification_score = st.slider(
            "Classification Accuracy",
            min_value=1,
            max_value=5,
            value=st.session_state.user_evaluation.get("classification_score", 3),
            help="How accurate was the incident categorization and urgency assignment?"
        )

        impact_score = st.slider(
            "Impact Assessment Quality",
            min_value=1,
            max_value=5,
            value=st.session_state.user_evaluation.get("impact_score", 3),
            help="How well did the system assess the incident impact?"
        )

        assignment_score = st.slider(
            "Resource Assignment",
            min_value=1,
            max_value=5,
            value=st.session_state.user_evaluation.get("assignment_score", 3),
            help="Was the responder assignment appropriate?"
        )

        response_score = st.slider(
            "Response Plan Quality",
            min_value=1,
            max_value=5,
            value=st.session_state.user_evaluation.get("response_score", 3),
            help="How useful were the generated communications and action items?"
        )

    with col2:
        st.markdown("#### Feedback")

        overall_assessment = st.selectbox(
            "Overall Assessment",
            options=["Excellent", "Good", "Acceptable", "Needs Improvement", "Poor"],
            index=["Excellent", "Good", "Acceptable", "Needs Improvement", "Poor"].index(
                st.session_state.user_evaluation.get("overall_assessment", "Acceptable")
            )
        )

        would_use = st.radio(
            "Would you use this triage output?",
            options=["Yes, as-is", "Yes, with minor edits", "No, needs significant revision"],
            index=["Yes, as-is", "Yes, with minor edits", "No, needs significant revision"].index(
                st.session_state.user_evaluation.get("would_use", "Yes, with minor edits")
            )
        )

        comments = st.text_area(
            "Additional Comments",
            value=st.session_state.user_evaluation.get("comments", ""),
            height=120,
            placeholder="Any specific feedback on the triage output..."
        )

    user_combined = (classification_score + impact_score + assignment_score + response_score) / 4

    st.divider()

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        st.metric("Your Combined Score", f"{user_combined:.1f} / 5")

    with col2:
        if st.button("Save Evaluation", type="primary"):
            evaluation = {
                "classification_score": classification_score,
                "impact_score": impact_score,
                "assignment_score": assignment_score,
                "response_score": response_score,
                "overall_assessment": overall_assessment,
                "would_use": would_use,
                "comments": comments,
                "combined_score": user_combined
            }

            # Get trace info from session
            trace_id = None
            run_id = None
            ticket_id = "unknown"

            if st.session_state.demo_result:
                trace_id = st.session_state.demo_result.get("trace_id")
                run_id = st.session_state.demo_result.get("run_id")
                incident = st.session_state.demo_result.get("incident")
                if incident:
                    ticket_id = incident.ticket_id

            # Save feedback
            success, message = save_user_feedback(ticket_id, evaluation, trace_id, run_id)

            st.session_state.user_evaluation = evaluation

            if success and trace_id:
                st.success("Evaluation saved and logged to production tracing.")
            elif success:
                st.success("Evaluation saved locally.")
            else:
                st.warning(f"Evaluation saved locally. {message}")


def main():
    """Main application."""
    init_session_state()

    st.title("TriageFlow")
    st.markdown("Multi-agent incident triage system. Select a ticket, run the pipeline, and evaluate the results.")

    st.divider()

    # Configuration
    config = st.session_state.config

    st.markdown("### Configuration")

    model_options = {
        "OpenAI (GPT-4o-mini)": "openai",
        "Anthropic (Claude Sonnet)": "anthropic"
    }

    col1, col2 = st.columns(2)

    with col1:
        selected_model = st.selectbox(
            "Model Provider",
            options=list(model_options.keys()),
            index=0
        )
        provider = model_options[selected_model]
        st.session_state.provider = provider

    with col2:
        model_info = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-sonnet-4-20250514"
        }
        st.text_input("Model", value=model_info[provider], disabled=True)

    model_config = config.get("models", {}).get(provider, {})
    if isinstance(model_config, dict):
        model = model_config.get("name", model_info[provider])
    else:
        model = model_config if model_config else model_info[provider]

    st.divider()

    # Ticket Selection
    st.markdown("## Select Ticket")

    selected_ticket, all_tickets = render_ticket_selector()
    custom_ticket = render_custom_ticket_form()

    active_ticket = custom_ticket if custom_ticket else selected_ticket

    if active_ticket:
        st.divider()
        render_ticket_details(active_ticket)

    st.divider()

    # Run Triage
    st.markdown("## Run Triage")

    col1, col2, col3 = st.columns([1, 1, 3])

    with col1:
        run_button = st.button(
            "Run Triage",
            type="primary",
            use_container_width=True,
            disabled=not active_ticket
        )

    with col2:
        if st.button("Clear Results", type="secondary", use_container_width=True):
            st.session_state.demo_result = None
            st.session_state.user_evaluation = {}
            st.rerun()

    if run_button and active_ticket:
        incident = Incident(
            ticket_id=active_ticket["ticket_id"],
            description=active_ticket["description"],
            source=IncidentSource(active_ticket["source"]),
            reporter=clean_nan(active_ticket.get("reporter")),
            affected_system=clean_nan(active_ticket.get("affected_system")),
            initial_severity=clean_nan(active_ticket.get("initial_severity"))
        )

        # Run triage pipeline with tracing
        with st.spinner("Running triage pipeline..."):
            try:
                result = run_triage_with_tracing(provider, model, incident, config)

                # Extract trace metadata for feedback logging
                trace_id = result.pop("_trace_id", None)
                run_id = result.pop("_run_id", None)

                # Convert traced result to expected format
                final_results = {
                    "stages": {
                        "classification": {"result": result["classification"], "time": 0},
                        "impact": {"result": result["impact"], "time": 0},
                        "resources": {"result": result["assignment"], "time": 0},
                        "response": {"result": result["response"], "time": 0},
                    },
                    "judges": {
                        "classification": {
                            "score": result["judge_scores"]["classification_score"],
                            "rationale": result["judge_scores"]["classification_rationale"]
                        },
                        "response": {
                            "score": result["judge_scores"]["response_score"],
                            "rationale": ""
                        },
                        "triage": {
                            "score": result["judge_scores"]["triage_score"],
                            "rationale": result["judge_scores"]["triage_rationale"]
                        },
                    },
                    "combined_quality": result["judge_scores"]["combined_score"],
                }

                st.session_state.demo_result = {
                    "incident": incident,
                    "results": final_results,
                    "tracing_enabled": True,
                    "trace_id": trace_id,
                    "run_id": run_id
                }
                st.success("Triage completed. View traces in Domino Experiment Manager.")
                st.rerun()

            except Exception as e:
                st.error(f"Triage pipeline failed: {str(e)}")
                st.exception(e)

    # Display Results
    if st.session_state.demo_result:
        st.divider()

        if st.session_state.demo_result.get("tracing_enabled"):
            st.info("Tracing was enabled for this run. View detailed traces in the Domino Experiment Manager.")

        render_agent_results(
            st.session_state.demo_result["results"],
            st.session_state.demo_result["incident"]
        )

        st.divider()
        render_llm_judge_results(
            st.session_state.demo_result["results"]["judges"],
            st.session_state.demo_result["results"]["combined_quality"]
        )

        st.divider()
        render_user_evaluation()


if __name__ == "__main__":
    main()
