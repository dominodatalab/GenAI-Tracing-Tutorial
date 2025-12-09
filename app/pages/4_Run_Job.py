"""
Run Job Page

Select tickets and launch triage jobs on Domino.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_manager import (
    load_base_config,
    load_sample_tickets,
    save_config,
    load_job_history,
    save_job_to_history,
    config_to_yaml_string,
    VERTICALS
)
from utils.domino_client import (
    start_triage_job,
    create_job_history_entry,
    get_project_info,
    get_domino_host,
    build_job_url
)

st.set_page_config(
    page_title="Run Job - TriageFlow",
    layout="wide"
)


VERTICAL_DISPLAY_NAMES = {
    "financial_services": "Financial Services",
    "healthcare": "Healthcare",
    "energy": "Energy",
    "public_sector": "Public Sector"
}


def init_page_state():
    """Initialize page-specific session state."""
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

    if "custom_tickets" not in st.session_state:
        st.session_state.custom_tickets = []

    # Initialize Domino host URL - default to blank
    if "domino_host" not in st.session_state:
        st.session_state.domino_host = ""


def render_ticket_table(tickets: list):
    """Render the ticket selection table."""
    if not tickets:
        st.warning("No tickets available for this vertical.")
        return

    # Create DataFrame for display
    df = pd.DataFrame(tickets)

    # Truncate description for display
    if "description" in df.columns:
        df["description_preview"] = df["description"].str[:100] + "..."

    # Select columns to display
    display_cols = ["ticket_id", "description_preview", "source", "initial_severity"]
    display_cols = [c for c in display_cols if c in df.columns]

    st.dataframe(
        df[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "ticket_id": st.column_config.TextColumn("Ticket ID", width="small"),
            "description_preview": st.column_config.TextColumn("Description", width="large"),
            "source": st.column_config.TextColumn("Source", width="small"),
            "initial_severity": st.column_config.NumberColumn("Severity", width="small")
        }
    )


def render_config_summary():
    """Render a summary of current configuration."""
    config = st.session_state.config
    provider = st.session_state.provider

    with st.expander("Configuration Summary", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Provider and Model**")
            st.markdown(f"- Provider: {provider}")
            st.markdown(f"- Model: {config.get('models', {}).get(provider, 'N/A')}")

            st.markdown("**Agent Temperatures**")
            agents = config.get("agents", {})
            for agent_name in ["classifier", "impact_assessor", "resource_matcher", "response_drafter"]:
                agent = agents.get(agent_name, {})
                temp = agent.get("temperature", "N/A")
                st.markdown(f"- {agent_name.replace('_', ' ').title()}: {temp}")

        with col2:
            st.markdown("**Evaluation Thresholds**")
            thresholds = st.session_state.get("adhoc_thresholds", {})
            st.markdown(f"- Urgency threshold: {thresholds.get('urgency_threshold', 4)}")
            st.markdown(f"- Impact threshold: {thresholds.get('impact_threshold', 7.0)}")

            st.markdown("**Tools**")
            tools = config.get("tools", {})
            total_tools = sum(len(t) for t in tools.values())
            st.markdown(f"- Total tools configured: {total_tools}")

        # Full YAML preview
        st.markdown("---")
        st.markdown("**Full Configuration (YAML)**")
        st.code(config_to_yaml_string(config), language="yaml")


def render_pipeline_diagram():
    """Render a diagram showing the pipeline flow based on current configuration."""
    config = st.session_state.config
    provider = st.session_state.provider
    model = config.get("models", {}).get(provider, "N/A")
    agents = config.get("agents", {})
    tools = config.get("tools", {})
    judge_config = st.session_state.get("judge_config", {})

    st.markdown("## Pipeline Overview")
    st.markdown("This diagram shows how your configured agents, tools, and judges will process each ticket.")

    # Get agent configurations
    classifier = agents.get("classifier", {})
    impact = agents.get("impact_assessor", {})
    resource = agents.get("resource_matcher", {})
    response = agents.get("response_drafter", {})

    # Get tool names
    classifier_tools = [t.get("name", "") for t in tools.get("classifier", [])]
    impact_tools = [t.get("name", "") for t in tools.get("impact_assessor", [])]
    resource_tools = [t.get("name", "") for t in tools.get("resource_matcher", [])]
    response_tools = [t.get("name", "") for t in tools.get("response_drafter", [])]

    # Get judge configurations
    class_judge = judge_config.get("classification_judge", {})
    resp_judge = judge_config.get("response_judge", {})
    triage_judge = judge_config.get("triage_judge", {})

    # Build the diagram dynamically to avoid truncation
    diagram_lines = [
        "```",
        "                              TRIAGEFLOW PIPELINE",
        "┌─────────────────────────────────────────────────────────────────────────────────┐",
        "│                                                                                 │",
        "│   ┌─────────────┐                                                               │",
        "│   │   TICKET    │                                                               │",
        "│   │   INPUT     │                                                               │",
        "│   └──────┬──────┘                                                               │",
        "│          │                                                                      │",
        "│          v                                                                      │",
        "│   ┌─────────────────────────────────────────────────────────────────────────┐   │",
        "│   │  AGENT 1: CLASSIFIER                                                    │   │",
        f"│   │    Model: {model}   │   │",
        f"│   │    Temperature: {classifier.get('temperature', 0.3)}    Max Tokens: {classifier.get('max_tokens', 1000)}                                │   │",
        f"│   │    Tools: {', '.join(classifier_tools) if classifier_tools else 'None'}   │   │",
        "│   └──────┬──────────────────────────────────────────────────────────────────┘   │",
        "│          │  Output: Classification (category, urgency, confidence)              │",
        "│          v                                                                      │",
        "│   ┌─────────────────────────────────────────────────────────────────────────┐   │",
        "│   │  AGENT 2: IMPACT ASSESSOR                                               │   │",
        f"│   │    Model: {model}   │   │",
        f"│   │    Temperature: {impact.get('temperature', 0.4)}    Max Tokens: {impact.get('max_tokens', 1500)}                                │   │",
        f"│   │    Tools: {', '.join(impact_tools) if impact_tools else 'None'}   │   │",
        "│   └──────┬──────────────────────────────────────────────────────────────────┘   │",
        "│          │  Output: Impact Assessment (score, blast_radius, affected_users)     │",
        "│          v                                                                      │",
        "│   ┌─────────────────────────────────────────────────────────────────────────┐   │",
        "│   │  AGENT 3: RESOURCE MATCHER                                              │   │",
        f"│   │    Model: {model}   │   │",
        f"│   │    Temperature: {resource.get('temperature', 0.2)}    Max Tokens: {resource.get('max_tokens', 1500)}                                │   │",
        f"│   │    Tools: {', '.join(resource_tools) if resource_tools else 'None'}   │   │",
        "│   └──────┬──────────────────────────────────────────────────────────────────┘   │",
        "│          │  Output: Resource Assignment (responder, SLA, escalation_path)       │",
        "│          v                                                                      │",
        "│   ┌─────────────────────────────────────────────────────────────────────────┐   │",
        "│   │  AGENT 4: RESPONSE DRAFTER                                              │   │",
        f"│   │    Model: {model}   │   │",
        f"│   │    Temperature: {response.get('temperature', 0.7)}    Max Tokens: {response.get('max_tokens', 2000)}                                │   │",
        f"│   │    Tools: {', '.join(response_tools) if response_tools else 'None'}   │   │",
        "│   └──────┬──────────────────────────────────────────────────────────────────┘   │",
        "│          │  Output: Response Plan (communications, action_items)                │",
        "│          v                                                                      │",
        "│   ┌─────────────────────────────────────────────────────────────────────────┐   │",
        "│   │  EVALUATION JUDGES                                                      │   │",
        f"│   │    Classification Judge  (temp: {class_judge.get('temperature', 0.1)}, max_tokens: {class_judge.get('max_tokens', 200)})                    │   │",
        f"│   │    Response Judge        (temp: {resp_judge.get('temperature', 0.1)}, max_tokens: {resp_judge.get('max_tokens', 200)})                    │   │",
        f"│   │    Triage Judge          (temp: {triage_judge.get('temperature', 0.1)}, max_tokens: {triage_judge.get('max_tokens', 200)})                    │   │",
        "│   └──────┬──────────────────────────────────────────────────────────────────┘   │",
        "│          │  Output: Quality Scores (1-5 each)                                   │",
        "│          v                                                                      │",
        "│   ┌─────────────┐                                                               │",
        "│   │   TRIAGE    │                                                               │",
        "│   │   RESULT    │                                                               │",
        "│   └─────────────┘                                                               │",
        "│                                                                                 │",
        "└─────────────────────────────────────────────────────────────────────────────────┘",
        "```",
    ]

    st.code("\n".join(diagram_lines[1:-1]), language=None)

    # Add a summary table below
    with st.expander("Pipeline Configuration Details", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Agents**")
            agent_data = [
                {"Agent": "Classifier", "Temp": classifier.get("temperature", 0.3), "Tokens": classifier.get("max_tokens", 1000), "Tools": len(classifier_tools)},
                {"Agent": "Impact Assessor", "Temp": impact.get("temperature", 0.4), "Tokens": impact.get("max_tokens", 1500), "Tools": len(impact_tools)},
                {"Agent": "Resource Matcher", "Temp": resource.get("temperature", 0.2), "Tokens": resource.get("max_tokens", 1500), "Tools": len(resource_tools)},
                {"Agent": "Response Drafter", "Temp": response.get("temperature", 0.7), "Tokens": response.get("max_tokens", 2000), "Tools": len(response_tools)},
            ]
            st.dataframe(agent_data, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("**Judges**")
            judge_data = [
                {"Judge": "Classification", "Temp": class_judge.get("temperature", 0.1), "Tokens": class_judge.get("max_tokens", 200)},
                {"Judge": "Response", "Temp": resp_judge.get("temperature", 0.1), "Tokens": resp_judge.get("max_tokens", 200)},
                {"Judge": "Triage", "Temp": triage_judge.get("temperature", 0.1), "Tokens": triage_judge.get("max_tokens", 200)},
            ]
            st.dataframe(judge_data, use_container_width=True, hide_index=True)

        st.markdown("**All Tools**")
        all_tools = []
        for agent_name, tool_list in [("Classifier", classifier_tools), ("Impact Assessor", impact_tools),
                                       ("Resource Matcher", resource_tools), ("Response Drafter", response_tools)]:
            for tool in tool_list:
                all_tools.append({"Agent": agent_name, "Tool": tool})
        if all_tools:
            st.dataframe(all_tools, use_container_width=True, hide_index=True)


def render_job_history():
    """Render the job history table with clickable job links."""
    history = load_job_history()

    if not history:
        st.info("No jobs have been submitted yet.")
        return

    # Create DataFrame
    df = pd.DataFrame(history)

    # Format timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Build job URLs if we have a domino_host configured
    domino_host = st.session_state.get("domino_host", "")
    if domino_host and "job_id" in df.columns:
        df["job_link"] = df.apply(
            lambda row: build_job_url(
                row.get("job_id", ""),
                domino_host
            ) if row.get("job_id") else "",
            axis=1
        )

    # Select columns to display
    display_cols = ["timestamp", "job_id", "vertical", "num_tickets", "provider", "status", "user"]
    display_cols = [c for c in display_cols if c in df.columns]

    st.dataframe(
        df[display_cols].head(20),
        use_container_width=True,
        hide_index=True,
        column_config={
            "timestamp": st.column_config.TextColumn("Submitted", width="medium"),
            "job_id": st.column_config.TextColumn("Job ID", width="medium"),
            "vertical": st.column_config.TextColumn("Vertical", width="small"),
            "num_tickets": st.column_config.NumberColumn("Tickets", width="small"),
            "provider": st.column_config.TextColumn("Provider", width="small"),
            "status": st.column_config.TextColumn("Status", width="small"),
            "user": st.column_config.TextColumn("User", width="small")
        }
    )

    # Show clickable links below the table if host is configured
    if domino_host and "job_link" in df.columns:
        with st.expander("Job Links", expanded=False):
            for _, row in df.head(10).iterrows():
                if row.get("job_link"):
                    job_id_short = row.get("job_id", "")[:12] + "..." if row.get("job_id") else "N/A"
                    st.markdown(f"- [{job_id_short}]({row['job_link']}) - {row.get('vertical', '')} ({row.get('timestamp', '')})")
    elif not domino_host:
        st.caption("Configure the Domino URL in settings above to get clickable job links.")


def main():
    """Main page content."""
    init_page_state()

    st.title("Run Triage Job")
    st.markdown("Select tickets and launch a triage job on Domino.")

    st.divider()

    # Ticket Selection Section
    st.markdown("## Ticket Selection")

    col1, col2 = st.columns([2, 1])

    with col1:
        vertical = st.selectbox(
            "Industry Vertical",
            options=VERTICALS,
            format_func=lambda x: VERTICAL_DISPLAY_NAMES.get(x, x),
            index=VERTICALS.index(st.session_state.selected_vertical),
            help="Select the industry vertical for sample tickets"
        )
        st.session_state.selected_vertical = vertical

    with col2:
        # Load tickets for selected vertical
        tickets = load_sample_tickets(vertical)
        total_tickets = len(tickets)

        ticket_mode = st.radio(
            "Tickets to Process",
            options=["all", "limited"],
            format_func=lambda x: f"All ({total_tickets})" if x == "all" else "Limited number",
            horizontal=True
        )

    if ticket_mode == "limited":
        num_tickets = st.slider(
            "Number of Tickets",
            min_value=1,
            max_value=max(total_tickets, 1),
            value=min(5, total_tickets),
            help="Number of tickets to process from the selected vertical"
        )
        st.session_state.num_tickets = num_tickets
    else:
        st.session_state.num_tickets = 0  # 0 means all tickets

    # Display tickets
    st.markdown(f"### Sample Tickets - {VERTICAL_DISPLAY_NAMES.get(vertical, vertical)}")
    render_ticket_table(tickets)

    st.divider()

    # Custom Ticket Section
    with st.expander("Add Custom Ticket (Optional)", expanded=False):
        st.markdown("Enter a custom ticket to include in the triage run.")

        col1, col2 = st.columns(2)

        with col1:
            custom_ticket_id = st.text_input(
                "Ticket ID",
                placeholder="e.g., CUSTOM-001"
            )
            custom_source = st.selectbox(
                "Source",
                options=["monitoring", "user_report", "automated_scan", "external_notification", "audit"]
            )
            custom_severity = st.number_input(
                "Initial Severity",
                min_value=1,
                max_value=5,
                value=3
            )

        with col2:
            custom_reporter = st.text_input(
                "Reporter (Optional)",
                placeholder="e.g., John Smith"
            )
            custom_system = st.text_input(
                "Affected System (Optional)",
                placeholder="e.g., Payment Gateway"
            )

        custom_description = st.text_area(
            "Description",
            placeholder="Describe the incident...",
            height=100
        )

        if st.button("Add Custom Ticket"):
            if custom_ticket_id and custom_description:
                custom_ticket = {
                    "ticket_id": custom_ticket_id,
                    "description": custom_description,
                    "source": custom_source,
                    "reporter": custom_reporter or None,
                    "affected_system": custom_system or None,
                    "initial_severity": custom_severity
                }
                st.session_state.custom_tickets.append(custom_ticket)
                st.success(f"Added custom ticket: {custom_ticket_id}")
            else:
                st.error("Ticket ID and Description are required.")

        # Show custom tickets
        if st.session_state.custom_tickets:
            st.markdown("**Custom Tickets:**")
            for i, ticket in enumerate(st.session_state.custom_tickets):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"- {ticket['ticket_id']}: {ticket['description'][:50]}...")
                with col2:
                    if st.button("Remove", key=f"remove_custom_{i}"):
                        st.session_state.custom_tickets.pop(i)
                        st.rerun()

    st.divider()

    # Pipeline Diagram
    render_pipeline_diagram()

    st.divider()

    # Configuration Summary
    render_config_summary()

    st.divider()

    # Domino Settings Section
    st.markdown("## Domino Settings")

    with st.expander("Configure Domino URL (Optional)", expanded=False):
        st.markdown("""
        Enter your Domino instance URL to enable direct links to submitted jobs.
        This should be the base URL of your Domino deployment (e.g., `https://se-demo.domino.tech`).
        """)

        domino_host = st.text_input(
            "Domino URL",
            value=st.session_state.domino_host,
            placeholder="https://your-domino-instance.domino.tech",
            help="The base URL of your Domino instance (without /api or trailing slash)"
        )

        # Clean up the URL
        if domino_host:
            domino_host = domino_host.rstrip("/")
            for suffix in ["/v4/api", "/api/api", "/api", "/v4"]:
                if domino_host.endswith(suffix):
                    domino_host = domino_host[:-len(suffix)]
                    break

        st.session_state.domino_host = domino_host

        if domino_host:
            project_info = get_project_info()
            example_url = f"{domino_host}/jobs/{project_info['owner']}/{project_info['name']}/[job_id]"
            st.caption(f"Job URLs will be formatted as: {example_url}")

    st.divider()

    # Launch Section
    st.markdown("## Launch Job")

    col1, col2 = st.columns([2, 1])

    with col1:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        default_title = f"TriageFlow-{vertical}-{timestamp}"
        job_title = st.text_input(
            "Job Title",
            value=default_title,
            help="A descriptive name for this job run"
        )

    with col2:
        provider = st.session_state.provider
        st.markdown(f"**Provider:** {provider}")
        num_display = st.session_state.num_tickets if st.session_state.num_tickets > 0 else "All"
        st.markdown(f"**Tickets:** {num_display}")

    # Launch button
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        launch_clicked = st.button("Launch Job", type="primary", use_container_width=True)

    with col2:
        if st.button("Clear History", type="secondary", use_container_width=True):
            # Note: This would clear the history file - implement if needed
            st.info("History clearing not implemented for safety.")

    if launch_clicked:
        with st.spinner("Saving configuration and launching job..."):
            try:
                # Save configuration to dataset
                config_path = save_config(st.session_state.config)
                st.success(f"Configuration saved: {os.path.basename(config_path)}")

                # Launch the job
                job_result = start_triage_job(
                    config_path=config_path,
                    provider=st.session_state.provider,
                    vertical=st.session_state.selected_vertical,
                    num_tickets=st.session_state.num_tickets,
                    title=job_title
                )

                # Create and save history entry with the configured domino_host
                history_entry = create_job_history_entry(
                    job_result=job_result,
                    config_path=config_path,
                    provider=st.session_state.provider,
                    vertical=st.session_state.selected_vertical,
                    num_tickets=st.session_state.num_tickets,
                    domino_host=st.session_state.domino_host
                )
                save_job_to_history(history_entry)

                # Store result in session state
                st.session_state.last_job_result = history_entry

                st.success("Job launched successfully!")

                # Display job info
                st.markdown("### Job Submitted")
                st.markdown(f"- **Job ID:** {history_entry.get('job_id', 'N/A')}")
                st.markdown(f"- **Config File:** {history_entry.get('config_file', 'N/A')}")

                if history_entry.get("job_url"):
                    st.markdown(f"- **View in Domino:** [{history_entry['job_url']}]({history_entry['job_url']})")
                elif st.session_state.domino_host:
                    # Construct URL manually if not in history entry
                    job_id = history_entry.get('job_id')
                    if job_id:
                        job_url = build_job_url(job_id, st.session_state.domino_host)
                        if job_url:
                            st.markdown(f"- **View in Domino:** [{job_url}]({job_url})")
                else:
                    st.caption("Configure the Domino URL in settings above to get direct links to jobs.")

            except Exception as e:
                st.error(f"Failed to launch job: {str(e)}")
                st.exception(e)

    # Display last job result if exists
    if st.session_state.last_job_result and not launch_clicked:
        st.markdown("### Last Submitted Job")
        last_job = st.session_state.last_job_result
        col1, col2, col3 = st.columns(3)
        with col1:
            job_id = last_job.get("job_id", "")
            job_id_display = job_id[:12] + "..." if job_id else "N/A"
            st.metric("Job ID", job_id_display)
        with col2:
            st.metric("Status", last_job.get("status", "N/A"))
        with col3:
            st.metric("Config", last_job.get("config_file", "N/A")[:20] + "...")

        # Show link to job if available
        job_url = last_job.get("job_url")
        if not job_url and st.session_state.domino_host and job_id:
            job_url = build_job_url(job_id, st.session_state.domino_host)
        if job_url:
            st.markdown(f"[View Job in Domino]({job_url})")
        elif job_id:
            st.caption("Configure the Domino URL in settings above to get a direct link.")

    st.divider()

    # Job History Section
    st.markdown("## Job History")

    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("Refresh History"):
            st.rerun()

    render_job_history()


if __name__ == "__main__":
    main()
