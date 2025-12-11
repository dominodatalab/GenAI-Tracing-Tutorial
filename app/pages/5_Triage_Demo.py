"""
Triage Demo Page

Interactive triage demonstration - select a ticket and run the triage pipeline directly.
"""

import streamlit as st
import pandas as pd
import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_manager import load_base_config, load_sample_tickets, VERTICALS

# Add project root for src imports
sys.path.insert(0, "/mnt/code")

from src.models import Incident, IncidentSource
from src.agents import classify_incident, assess_impact, match_resources, draft_response
from src.judges import judge_classification, judge_response, judge_triage

st.set_page_config(
    page_title="Triage Demo - TriageFlow",
    layout="wide"
)

VERTICAL_DISPLAY_NAMES = {
    "financial_services": "Financial Services",
    "healthcare": "Healthcare",
    "energy": "Energy",
    "public_sector": "Public Sector"
}

SOURCE_OPTIONS = ["monitoring", "user_report", "automated_scan", "external_notification", "audit"]


def init_page_state():
    """Initialize page-specific session state."""
    if "config" not in st.session_state:
        st.session_state.config = load_base_config()

    if "provider" not in st.session_state:
        st.session_state.provider = "openai"

    if "demo_vertical" not in st.session_state:
        st.session_state.demo_vertical = "financial_services"

    if "demo_result" not in st.session_state:
        st.session_state.demo_result = None

    if "demo_running" not in st.session_state:
        st.session_state.demo_running = False


def initialize_client(provider: str):
    """Initialize LLM client."""
    if provider == "openai":
        from openai import OpenAI
        return OpenAI()
    else:
        from anthropic import Anthropic
        return Anthropic()


def run_triage_pipeline(client, provider: str, model: str, incident: Incident, config: dict):
    """Run the 4-agent triage pipeline with judges."""
    results = {"stages": {}, "judges": {}}

    # Stage 1: Classification
    yield "stage", "classifier", "Running Classifier Agent..."
    start = time.time()
    classification = classify_incident(client, provider, model, incident, config)
    results["stages"]["classification"] = {
        "result": classification,
        "time": time.time() - start
    }
    yield "result", "classifier", classification

    # Stage 2: Impact Assessment
    yield "stage", "impact_assessor", "Running Impact Assessor Agent..."
    start = time.time()
    impact = assess_impact(client, provider, model, incident, classification, config)
    results["stages"]["impact"] = {
        "result": impact,
        "time": time.time() - start
    }
    yield "result", "impact_assessor", impact

    # Stage 3: Resource Matching
    yield "stage", "resource_matcher", "Running Resource Matcher Agent..."
    start = time.time()
    resources = match_resources(client, provider, model, classification, impact, config)
    results["stages"]["resources"] = {
        "result": resources,
        "time": time.time() - start
    }
    yield "result", "resource_matcher", resources

    # Stage 4: Response Drafting
    yield "stage", "response_drafter", "Running Response Drafter Agent..."
    start = time.time()
    response = draft_response(client, provider, model, incident, classification, impact, resources, config)
    results["stages"]["response"] = {
        "result": response,
        "time": time.time() - start
    }
    yield "result", "response_drafter", response

    # Run Judges
    yield "stage", "judges", "Running Quality Judges..."

    class_dict = classification.model_dump()
    impact_dict = impact.model_dump()
    resources_dict = resources.model_dump()
    response_dict = response.model_dump()

    class_judge = judge_classification(client, provider, incident.description, class_dict)
    results["judges"]["classification"] = class_judge

    comms = response_dict.get("communications", [])
    if comms:
        resp_judge = judge_response(client, provider, incident.description, class_dict.get("urgency", 3), comms[0])
    else:
        resp_judge = {"score": 1, "rationale": "No communications generated"}
    results["judges"]["response"] = resp_judge

    triage_judge = judge_triage(client, provider, incident.description, class_dict, impact_dict, resources_dict, response_dict)
    results["judges"]["triage"] = triage_judge

    # Calculate combined quality score
    combined_quality = (
        class_judge.get("score", 3) +
        resp_judge.get("score", 3) +
        triage_judge.get("score", 3)
    ) / 3
    results["combined_quality"] = combined_quality

    yield "complete", "all", results


def render_ticket_selector():
    """Render ticket selection controls."""
    col1, col2 = st.columns([1, 2])

    with col1:
        vertical = st.selectbox(
            "Industry Vertical",
            options=VERTICALS,
            format_func=lambda x: VERTICAL_DISPLAY_NAMES.get(x, x),
            index=VERTICALS.index(st.session_state.demo_vertical),
            help="Select the industry vertical for sample tickets"
        )
        st.session_state.demo_vertical = vertical

    # Load tickets for selected vertical
    tickets = load_sample_tickets(vertical)

    with col2:
        ticket_options = {t["ticket_id"]: t for t in tickets}
        selected_ticket_id = st.selectbox(
            "Select Ticket",
            options=list(ticket_options.keys()),
            format_func=lambda x: f"{x} - {ticket_options[x]['description'][:60]}..."
        )

    selected_ticket = ticket_options.get(selected_ticket_id)

    return selected_ticket, tickets


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
    st.text_area(
        "Description",
        value=ticket.get("description", ""),
        height=120,
        disabled=True,
        label_visibility="collapsed"
    )


def render_custom_ticket_form():
    """Render form for custom ticket input."""
    with st.expander("Enter Custom Ticket", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            custom_id = st.text_input(
                "Ticket ID",
                value="CUSTOM-001",
                key="custom_ticket_id"
            )
            custom_source = st.selectbox(
                "Source",
                options=SOURCE_OPTIONS,
                key="custom_source"
            )
            custom_severity = st.number_input(
                "Initial Severity",
                min_value=1,
                max_value=5,
                value=3,
                key="custom_severity"
            )

        with col2:
            custom_reporter = st.text_input(
                "Reporter (Optional)",
                key="custom_reporter"
            )
            custom_system = st.text_input(
                "Affected System (Optional)",
                key="custom_system"
            )

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


def render_results(results: dict, incident: Incident):
    """Render the triage results."""
    st.markdown("## Triage Results")

    # Summary metrics
    classification = results["stages"]["classification"]["result"]
    impact = results["stages"]["impact"]["result"]
    resources = results["stages"]["resources"]["result"]
    response = results["stages"]["response"]["result"]

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Category", classification.category.value.replace("_", " ").title())
    with col2:
        st.metric("Urgency", f"{classification.urgency}/5")
    with col3:
        st.metric("Impact Score", f"{impact.impact_score:.1f}/10")
    with col4:
        st.metric("SLA Met", "Yes" if resources.sla_met else "No")
    with col5:
        st.metric("Quality Score", f"{results['combined_quality']:.1f}/5")

    st.divider()

    # Detailed Results in Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Classification",
        "Impact Assessment",
        "Resource Assignment",
        "Response Plan",
        "Quality Evaluation"
    ])

    with tab1:
        render_classification_results(classification)

    with tab2:
        render_impact_results(impact)

    with tab3:
        render_resource_results(resources)

    with tab4:
        render_response_results(response)

    with tab5:
        render_quality_results(results["judges"], results["combined_quality"])


def render_classification_results(classification):
    """Render classification agent results."""
    st.markdown("### Classification")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Category:** {classification.category.value}")
        st.markdown(f"**Subcategory:** {classification.subcategory}")
        st.markdown(f"**Urgency:** {classification.urgency}/5")
    with col2:
        st.markdown(f"**Confidence:** {classification.confidence:.0%}")
        st.markdown(f"**Affected Domain:** {classification.affected_domain}")

    st.markdown("**Reasoning:**")
    st.info(classification.reasoning)


def render_impact_results(impact):
    """Render impact assessment results."""
    st.markdown("### Impact Assessment")

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

    st.markdown("**Reasoning:**")
    st.info(impact.reasoning)


def render_resource_results(resources):
    """Render resource assignment results."""
    st.markdown("### Resource Assignment")

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

    st.markdown("**Reasoning:**")
    st.info(resources.reasoning)


def render_response_results(response):
    """Render response plan results."""
    st.markdown("### Response Plan")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Est. Resolution Time:** {response.estimated_resolution_time}")
        st.markdown(f"**Escalation Required:** {'Yes' if response.escalation_required else 'No'}")
    with col2:
        st.markdown(f"**Completeness Score:** {response.completeness_score:.0%}")
        st.markdown(f"**Action Items:** {len(response.action_items)}")

    # Communications
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

    # Action Items
    if response.action_items:
        with st.expander("Action Items", expanded=False):
            for i, item in enumerate(response.action_items, 1):
                st.markdown(f"{i}. {item}")


def render_quality_results(judges: dict, combined_score: float):
    """Render quality evaluation results."""
    st.markdown("### Quality Evaluation")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        class_score = judges.get("classification", {}).get("score", 0)
        st.metric("Classification Score", f"{class_score}/5")

    with col2:
        resp_score = judges.get("response", {}).get("score", 0)
        st.metric("Response Score", f"{resp_score}/5")

    with col3:
        triage_score = judges.get("triage", {}).get("score", 0)
        st.metric("Triage Score", f"{triage_score}/5")

    with col4:
        st.metric("Combined Score", f"{combined_score:.1f}/5")

    # Interpretation
    if combined_score >= 4.0:
        quality_level = "High Quality"
        quality_desc = "The triage output meets high quality standards."
    elif combined_score >= 3.0:
        quality_level = "Acceptable Quality"
        quality_desc = "The triage output is acceptable but may benefit from review."
    else:
        quality_level = "Needs Review"
        quality_desc = "The triage output should be manually reviewed before action."

    st.markdown(f"**Quality Level:** {quality_level}")
    st.caption(quality_desc)

    # Judge Rationales
    with st.expander("Judge Rationales", expanded=False):
        st.markdown("**Classification Judge:**")
        st.info(judges.get("classification", {}).get("rationale", "No rationale provided"))

        st.markdown("**Response Judge:**")
        st.info(judges.get("response", {}).get("rationale", "No rationale provided"))

        st.markdown("**Triage Judge:**")
        st.info(judges.get("triage", {}).get("rationale", "No rationale provided"))


def main():
    """Main page content."""
    init_page_state()

    st.title("Triage Demo")
    st.markdown("Select a ticket and run the triage pipeline to see real-time results.")

    st.divider()

    # Configuration Status
    config = st.session_state.config
    provider = st.session_state.provider
    model = config.get("models", {}).get(provider, "gpt-4o-mini")

    with st.expander("Current Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Provider:** {provider.title()}")
        with col2:
            st.markdown(f"**Model:** {model}")
        with col3:
            st.markdown("**Status:** Ready")
        st.caption("Configure agents and models on the Agent Configuration page.")

    st.divider()

    # Ticket Selection
    st.markdown("## Select Ticket")

    selected_ticket, all_tickets = render_ticket_selector()
    custom_ticket = render_custom_ticket_form()

    # Use custom ticket if specified
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
            st.rerun()

    if run_button and active_ticket:
        # Create Incident object
        incident = Incident(
            ticket_id=active_ticket["ticket_id"],
            description=active_ticket["description"],
            source=IncidentSource(active_ticket["source"]),
            reporter=active_ticket.get("reporter"),
            affected_system=active_ticket.get("affected_system"),
            initial_severity=active_ticket.get("initial_severity")
        )

        # Initialize client
        try:
            client = initialize_client(provider)
        except Exception as e:
            st.error(f"Failed to initialize {provider} client. Please check your API key configuration.")
            st.exception(e)
            st.stop()

        # Run pipeline with progress
        progress_container = st.container()
        with progress_container:
            st.markdown("### Pipeline Progress")
            progress_bar = st.progress(0)
            status_text = st.empty()

            stages = ["classifier", "impact_assessor", "resource_matcher", "response_drafter", "judges"]
            stage_names = {
                "classifier": "Classifier Agent",
                "impact_assessor": "Impact Assessor Agent",
                "resource_matcher": "Resource Matcher Agent",
                "response_drafter": "Response Drafter Agent",
                "judges": "Quality Judges"
            }

            try:
                final_results = None
                for event_type, stage, data in run_triage_pipeline(client, provider, model, incident, config):
                    if event_type == "stage":
                        status_text.markdown(f"**{stage_names.get(stage, stage)}:** {data}")
                        stage_idx = stages.index(stage) if stage in stages else 0
                        progress_bar.progress((stage_idx) / len(stages))
                    elif event_type == "complete":
                        final_results = data
                        progress_bar.progress(1.0)
                        status_text.markdown("**Complete**")

                if final_results:
                    st.session_state.demo_result = {
                        "incident": incident,
                        "results": final_results
                    }
                    st.success("Triage completed successfully.")
                    st.rerun()

            except Exception as e:
                st.error(f"Triage pipeline failed: {str(e)}")
                st.exception(e)

    # Display Results
    if st.session_state.demo_result:
        st.divider()
        render_results(
            st.session_state.demo_result["results"],
            st.session_state.demo_result["incident"]
        )


if __name__ == "__main__":
    main()
