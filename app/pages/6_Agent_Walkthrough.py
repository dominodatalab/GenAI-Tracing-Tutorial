"""
Agent Walkthrough - Interactive multi-agent triage demonstration.
"""

import streamlit as st
import pandas as pd
import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/mnt/code")

from utils.config_manager import load_base_config, load_sample_tickets, VERTICALS
from src.models import Incident, IncidentSource
from src.agents import classify_incident, assess_impact, match_resources, draft_response, TOOL_FUNCTIONS
from src.guardrails import check_input, check_output, health_check as guardrails_health_check

st.set_page_config(page_title="Agent Walkthrough", layout="wide")

VERTICAL_NAMES = {
    "financial_services": "Financial Services",
    "healthcare": "Healthcare",
    "energy": "Energy",
    "public_sector": "Public Sector",
    "guardrails_test": "Guardrails Test"
}

AGENTS = {
    "classifier": {
        "name": "Classifier Agent",
        "description": "Determines category, subcategory, and urgency level."
    },
    "impact_assessor": {
        "name": "Impact Assessor Agent",
        "description": "Evaluates blast radius, affected users, and financial exposure."
    },
    "resource_matcher": {
        "name": "Resource Matcher Agent",
        "description": "Matches responder skills to incident needs."
    },
    "response_drafter": {
        "name": "Response Drafter Agent",
        "description": "Decides communication strategy and drafts messages."
    }
}


def init_state():
    """Initialize session state."""
    defaults = {
        "config": load_base_config(),
        "provider": "openai",
        "vertical": "financial_services",
        "result": None,
        "trace": [],
        "guardrails": False
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_client(provider: str):
    """Get LLM client."""
    if provider == "openai":
        from openai import OpenAI
        return OpenAI()
    from anthropic import Anthropic
    return Anthropic()


def load_tickets(vertical: str) -> list:
    """Load tickets for vertical."""
    if vertical == "guardrails_test":
        path = "/mnt/code/example-data/guardrails_test_tickets.csv"
        if os.path.exists(path):
            return pd.read_csv(path).to_dict('records')
        return []
    return load_sample_tickets(vertical)


def run_pipeline(client, provider: str, model: str, incident: Incident,
                 config: dict, use_guardrails: bool = False):
    """Run pipeline and capture trace."""
    trace = []
    results = {}

    def add_trace(step, name, input_data):
        trace.append({
            "step": step,
            "name": name,
            "input": input_data,
            "start_time": datetime.now().isoformat()
        })

    def finish_trace(output, duration):
        trace[-1]["output"] = output
        trace[-1]["duration_sec"] = round(duration, 2)

    # Guardrails input check
    if use_guardrails:
        add_trace("guardrails_input", "Input Validation", {"text": incident.description[:300]})
        input_result = check_input(incident.description)
        trace[-1]["output"] = input_result.to_dict()
        trace[-1]["passed"] = input_result.passed

        if not input_result.passed:
            results["blocked"] = True
            results["blocked_reason"] = input_result.blocked_reason
            results["guardrails"] = input_result.to_dict()
            yield "blocked", results, trace
            return

    # Classifier
    add_trace("classifier", "Classifier Agent", {"ticket_id": incident.ticket_id})
    start = time.time()
    classification = classify_incident(client, provider, model, incident, config)
    finish_trace(classification.model_dump(), time.time() - start)
    results["classification"] = classification
    yield "classifier", classification, trace[-1]

    # Impact Assessor
    add_trace("impact_assessor", "Impact Assessor Agent", {"category": classification.category.value})
    start = time.time()
    impact = assess_impact(client, provider, model, incident, classification, config)
    finish_trace(impact.model_dump(), time.time() - start)
    results["impact"] = impact
    yield "impact_assessor", impact, trace[-1]

    # Resource Matcher
    add_trace("resource_matcher", "Resource Matcher Agent", {"urgency": classification.urgency})
    start = time.time()
    resources = match_resources(client, provider, model, classification, impact, config)
    finish_trace(resources.model_dump(), time.time() - start)
    results["resources"] = resources
    yield "resource_matcher", resources, trace[-1]

    # Response Drafter
    add_trace("response_drafter", "Response Drafter Agent", {"blast_radius": impact.blast_radius})
    start = time.time()
    response = draft_response(client, provider, model, incident, classification, impact, resources, config)
    finish_trace(response.model_dump(), time.time() - start)
    results["response"] = response
    yield "response_drafter", response, trace[-1]

    # Guardrails output check
    if use_guardrails and response.communications:
        for comm in response.communications:
            add_trace("guardrails_output", f"Sanitize: {comm.audience}", {})
            output_result = check_output(comm.body)
            trace[-1]["output"] = {"pii_found": output_result.checks.get("pii", {}).get("has_pii", False)}

    results["trace"] = trace
    yield "complete", results, trace


def render_ticket_input():
    """Render ticket selection and custom input."""
    col1, col2 = st.columns([1, 2])

    with col1:
        verticals = VERTICALS + ["guardrails_test"]
        vertical = st.selectbox(
            "Vertical",
            options=verticals,
            format_func=lambda x: VERTICAL_NAMES.get(x, x),
            key="sel_vertical"
        )
        st.session_state.vertical = vertical

    tickets = load_tickets(vertical)

    with col2:
        if tickets:
            options = {t["ticket_id"]: t for t in tickets}
            selected_id = st.selectbox(
                "Ticket",
                options=list(options.keys()),
                format_func=lambda x: f"{x} - {options[x]['description'][:50]}..."
            )
            selected = options.get(selected_id)
        else:
            st.info("No tickets for this vertical.")
            selected = None

    # Custom ticket option
    with st.expander("Or enter custom ticket"):
        custom_desc = st.text_area("Description", placeholder="Describe the incident...", key="custom_desc", height=80)
        if custom_desc:
            selected = {
                "ticket_id": "CUSTOM-001",
                "description": custom_desc,
                "source": "user_report",
                "initial_severity": 3
            }

    return selected


def render_output(results: dict):
    """Render final output summary."""
    if results.get("blocked"):
        st.error(f"Blocked: {results.get('blocked_reason')}")
        with st.expander("Details"):
            st.json(results.get("guardrails", {}))
        return

    c = results["classification"]
    i = results["impact"]
    r = results["resources"]
    resp = results["response"]

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Category", c.category.value.replace("_", " ").title())
    col2.metric("Urgency", f"{c.urgency} / 5")
    col3.metric("Impact", f"{i.impact_score:.1f} / 10")
    col4.metric("Blast Radius", i.blast_radius.title())

    st.divider()

    # Communication Decision
    comm_dec = getattr(resp, 'communication_decision', None)
    if comm_dec:
        st.subheader("Communication Decision")
        cols = st.columns(3)
        cols[0].markdown(f"**External:** {'Yes' if comm_dec.requires_external else 'No'}")
        cols[1].markdown(f"**Slack:** {'Yes' if comm_dec.requires_internal else 'No'}")
        cols[2].markdown(f"**None:** {'Yes' if comm_dec.requires_none else 'No'}")
        st.caption(comm_dec.reasoning)
        st.divider()

    # Communications
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Slack Message")
        slack = getattr(resp, 'slack_messages', [])
        if slack:
            for m in slack:
                st.markdown(f"**#{m.channel}**")
                st.code(m.message, language=None)
        else:
            st.caption("None proposed")

    with col2:
        st.subheader("External Communication")
        if resp.communications:
            for comm in resp.communications:
                st.markdown(f"**To:** {comm.audience}")
                st.markdown(f"**Subject:** {comm.subject}")
                st.code(comm.body[:500], language=None)
        else:
            st.caption("None drafted")

    st.divider()

    # Responder
    st.subheader("Assigned Responder")
    p = r.primary_responder
    cols = st.columns(4)
    cols[0].markdown(f"**{p.name}**")
    cols[1].markdown(p.role)
    cols[2].markdown(p.availability)
    cols[3].markdown(f"SLA: {'Met' if r.sla_met else 'Not Met'}")

    # Actions
    if resp.action_items:
        with st.expander("Action Items"):
            for item in resp.action_items:
                st.markdown(f"- {item}")


def render_trace(trace: list):
    """Render agent trace."""
    if not trace:
        st.caption("Run pipeline to see trace.")
        return

    for i, step in enumerate(trace):
        name = step.get("name", f"Step {i+1}")
        duration = step.get("duration_sec", "")
        blocked = step.get("passed") is False

        label = f"[BLOCKED] {name}" if blocked else name
        if duration:
            label += f" ({duration}s)"

        with st.expander(label, expanded=(i == 0)):
            # Description
            step_key = step.get("step", "")
            if step_key in AGENTS:
                st.caption(AGENTS[step_key]["description"])

            # Input/Output
            tab1, tab2 = st.tabs(["Input", "Output"])
            with tab1:
                st.json(step.get("input", {}))
            with tab2:
                st.json(step.get("output", {}))


def main():
    init_state()

    st.title("Agent Walkthrough")
    st.caption("See how the multi-agent system processes each ticket step by step.")

    # Config bar
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        provider = st.selectbox("Provider", ["openai", "anthropic"], key="prov")
        st.session_state.provider = provider
    with col2:
        config = st.session_state.config
        model_cfg = config.get("models", {}).get(provider, {})
        model = model_cfg.get("name", "gpt-4o-mini") if isinstance(model_cfg, dict) else model_cfg
        st.text_input("Model", value=model, disabled=True)
    with col3:
        try:
            gr_ok = guardrails_health_check()
        except Exception:
            gr_ok = False
        st.session_state.guardrails = st.checkbox(
            "Guardrails" + (" (available)" if gr_ok else " (unavailable)"),
            value=st.session_state.guardrails if gr_ok else False,
            disabled=not gr_ok
        )

    st.divider()

    # Ticket input
    ticket = render_ticket_input()

    if ticket:
        st.caption(f"**{ticket['ticket_id']}** | {ticket.get('source', 'N/A')} | Severity: {ticket.get('initial_severity', 'N/A')}")

    st.divider()

    # Run
    col1, col2 = st.columns([1, 5])
    with col1:
        run = st.button("Run", type="primary", disabled=not ticket)
    with col2:
        if st.button("Clear"):
            st.session_state.result = None
            st.session_state.trace = []
            st.rerun()

    if run and ticket:
        incident = Incident(
            ticket_id=ticket["ticket_id"],
            description=ticket["description"],
            source=IncidentSource(ticket.get("source", "monitoring")),
            reporter=ticket.get("reporter"),
            affected_system=ticket.get("affected_system"),
            initial_severity=ticket.get("initial_severity")
        )

        try:
            client = get_client(provider)
        except Exception as e:
            st.error(f"Client error: {e}")
            st.stop()

        progress = st.progress(0)
        status = st.empty()
        stages = ["classifier", "impact_assessor", "resource_matcher", "response_drafter", "complete"]

        try:
            for stage, result, step in run_pipeline(
                client, provider, model, incident, config,
                use_guardrails=st.session_state.guardrails
            ):
                if stage in ["complete", "blocked"]:
                    progress.progress(1.0)
                    status.empty()
                    st.session_state.result = result
                    st.session_state.trace = result.get("trace", [])
                    st.rerun()
                else:
                    name = AGENTS.get(stage, {}).get("name", stage)
                    status.caption(f"Running: {name}")
                    idx = stages.index(stage) if stage in stages else 0
                    progress.progress((idx + 1) / len(stages))
        except Exception as e:
            st.error(f"Error: {e}")

    # Results
    if st.session_state.result:
        st.divider()
        tab1, tab2 = st.tabs(["Output", "Trace"])
        with tab1:
            render_output(st.session_state.result)
        with tab2:
            render_trace(st.session_state.trace)


if __name__ == "__main__":
    main()
