"""
Evaluators Page

Configure LLM judges and evaluation thresholds.
"""

import streamlit as st
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_manager import load_base_config

st.set_page_config(
    page_title="Evaluators - TriageFlow",
    layout="wide"
)


# Default judge prompts
DEFAULT_JUDGE_CONFIG = {
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

DEFAULT_ADHOC_THRESHOLDS = {
    "urgency_threshold": 4,
    "impact_threshold": 7.0
}


def init_page_state():
    """Initialize page-specific session state."""
    if "config" not in st.session_state:
        st.session_state.config = load_base_config()

    if "judge_config" not in st.session_state:
        st.session_state.judge_config = DEFAULT_JUDGE_CONFIG.copy()

    if "adhoc_thresholds" not in st.session_state:
        st.session_state.adhoc_thresholds = DEFAULT_ADHOC_THRESHOLDS.copy()


def render_judge_config(judge_key: str, display_name: str, description: str, scoring_criteria: list):
    """Render configuration controls for a single judge."""
    judge = st.session_state.judge_config.get(judge_key, DEFAULT_JUDGE_CONFIG[judge_key])

    with st.expander(f"{display_name}", expanded=False):
        st.caption(description)

        col1, col2 = st.columns(2)

        with col1:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=float(judge.get("temperature", 0.1)),
                step=0.1,
                key=f"{judge_key}_temp",
                help="Lower values produce more consistent scoring (recommended: 0.1)"
            )

        with col2:
            max_tokens = st.number_input(
                "Max Tokens",
                min_value=50,
                max_value=500,
                value=int(judge.get("max_tokens", 200)),
                step=50,
                key=f"{judge_key}_tokens",
                help="Maximum tokens for the judge response"
            )

        prompt = st.text_area(
            "Judge Prompt Template",
            value=judge.get("prompt", ""),
            height=300,
            key=f"{judge_key}_prompt",
            help="The prompt template used to evaluate outputs. Use {placeholders} for dynamic values."
        )

        # Scoring criteria reference
        st.markdown("**Scoring Criteria:**")
        for criterion in scoring_criteria:
            st.markdown(f"- {criterion}")

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Reset to Default", key=f"{judge_key}_reset"):
                st.session_state.judge_config[judge_key] = DEFAULT_JUDGE_CONFIG[judge_key].copy()
                st.rerun()

        # Update config in session state
        st.session_state.judge_config[judge_key] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt": prompt
        }


def main():
    """Main page content."""
    init_page_state()

    st.title("Evaluators and Judges")
    st.markdown("Configure the LLM judges that evaluate pipeline output quality.")

    st.divider()

    # Overview
    st.markdown("""
    The TriageFlow pipeline uses three LLM judges to evaluate the quality of agent outputs:

    1. **Classification Judge** - Evaluates whether incidents are correctly categorized
    2. **Response Judge** - Evaluates the quality of generated communications
    3. **Triage Judge** - Evaluates the overall triage process holistically

    Each judge returns a score from 1-5 (5 = excellent) along with a rationale.
    """)

    # Scoring scale reference
    with st.expander("Scoring Scale Reference", expanded=False):
        st.markdown("""
        | Score | Meaning | Description |
        |-------|---------|-------------|
        | 5 | Excellent | Fully correct, well-reasoned, comprehensive |
        | 4 | Good | Mostly correct with minor improvements possible |
        | 3 | Acceptable | Adequate but with notable gaps or issues |
        | 2 | Suboptimal | Significant issues that affect quality |
        | 1 | Poor | Major errors or completely inappropriate |
        """)

    st.divider()

    # Judge Configurations
    st.markdown("## Judge Configuration")

    # Classification Judge
    render_judge_config(
        "classification_judge",
        "Classification Judge",
        "Evaluates whether the incident was correctly categorized and assigned appropriate urgency.",
        [
            "Is the category appropriate for this incident?",
            "Is the urgency level justified by the description?",
            "Is the reasoning sound and logical?"
        ]
    )

    # Response Judge
    render_judge_config(
        "response_judge",
        "Response Judge",
        "Evaluates the quality of generated communications for different audiences.",
        [
            "Is the tone appropriate for the audience?",
            "Is the information clear and actionable?",
            "Does it convey appropriate urgency?"
        ]
    )

    # Triage Judge
    render_judge_config(
        "triage_judge",
        "Triage Judge",
        "Evaluates the overall triage process and final output holistically.",
        [
            "Is the classification appropriate given the incident?",
            "Is the resource assignment logical and efficient?",
            "Is the response plan comprehensive?"
        ]
    )

    st.divider()

    # Ad-hoc Evaluation Settings
    st.markdown("## Ad-hoc Evaluation Settings")
    st.markdown("""
    These settings control additional evaluations that are computed after the pipeline completes.
    """)

    with st.container(border=True):
        st.markdown("### Manual Review Thresholds")
        st.caption("""
        Incidents are flagged for manual review when they meet BOTH threshold criteria.
        This helps identify high-priority incidents that may need human oversight.
        """)

        col1, col2 = st.columns(2)

        with col1:
            urgency_threshold = st.number_input(
                "Urgency Threshold",
                min_value=1,
                max_value=5,
                value=int(st.session_state.adhoc_thresholds.get("urgency_threshold", 4)),
                help="Incidents with urgency >= this value may be flagged"
            )

        with col2:
            impact_threshold = st.number_input(
                "Impact Score Threshold",
                min_value=0.0,
                max_value=10.0,
                value=float(st.session_state.adhoc_thresholds.get("impact_threshold", 7.0)),
                step=0.5,
                help="Incidents with impact score >= this value may be flagged"
            )

        st.session_state.adhoc_thresholds = {
            "urgency_threshold": urgency_threshold,
            "impact_threshold": impact_threshold
        }

        st.info(
            f"Incidents will be flagged for manual review when urgency >= {urgency_threshold} "
            f"AND impact score >= {impact_threshold}"
        )

    st.divider()

    # Combined Quality Score
    with st.container(border=True):
        st.markdown("### Combined Quality Score")
        st.caption("""
        The combined quality score is computed as the mean of all three judge scores.
        This provides a single metric for overall triage quality.
        """)

        st.markdown("""
        **Formula:** `combined_quality = (classification_score + response_score + triage_score) / 3`

        **Interpretation:**
        - 4.0 - 5.0: High quality triage
        - 3.0 - 3.9: Acceptable quality
        - Below 3.0: May need review
        """)

    st.divider()

    # Actions
    col1, col2, col3 = st.columns([1, 1, 3])

    with col1:
        if st.button("Reset All to Defaults", type="secondary"):
            st.session_state.judge_config = DEFAULT_JUDGE_CONFIG.copy()
            st.session_state.adhoc_thresholds = DEFAULT_ADHOC_THRESHOLDS.copy()
            st.success("Evaluator configuration reset to defaults.")
            st.rerun()

    with col2:
        if st.button("Apply Changes", type="primary"):
            st.success("Changes applied to current session.")


if __name__ == "__main__":
    main()
