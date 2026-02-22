# TriageFlow

A multi-agent system for automated incident triage. The pipeline classifies incoming incidents, assesses impact, assigns responders, and drafts stakeholder communications.

This project demonstrates Domino's GenAI tracing and evaluation capabilities through a production-style pipeline applicable across financial services, healthcare, energy, and public sector.

## Pipeline Overview

Incidents flow through four specialized agents:

1. **Classifier Agent** - Categorizes the incident and assigns urgency (1-5)
2. **Impact Assessor Agent** - Evaluates blast radius, affected users, and financial exposure
3. **Resource Matcher Agent** - Identifies responders based on skills and SLA requirements
4. **Response Drafter Agent** - Generates communications for each stakeholder audience

Each agent uses dedicated tools to query historical data, check resource availability, and apply organizational policies.

## Key Features

### Tracing and Observability

- **Unified Trace Tree** - All agents, tools, and LLM calls appear in one hierarchical trace
- **Automatic Instrumentation** - Uses `@add_tracing` decorator with `mlflow.openai.autolog()` or `mlflow.anthropic.autolog()` frameworks
- **Aggregated Metrics** - `DominoRun` captures statistical summaries (mean, median) across traces

### LLM Judges

Three automated judges evaluate output quality:
- **Classification Judge** - Evaluates category and urgency appropriateness
- **Response Judge** - Assesses communication clarity and tone
- **Triage Judge** - Holistic assessment of the complete triage decision

### Human-in-the-Loop Evaluation

- Users can score triage outputs across four dimensions (classification, impact, assignment, response)
- Feedback is logged to production traces via `log_evaluation()`
- Local backup stored in `/mnt/data/{project}/feedback/user_feedback.jsonl`

### Multi-Vertical Support

Sample incidents for Financial Services, Healthcare, Energy, and Public Sector.

## Setup

1. Add your API key as a Domino user environment variable:
   - Go to **Account Settings** > **User Environment Variables**
   - Add `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`

## Usage

### Notebook

Open `tracing-tutorial.ipynb` for a step-by-step walkthrough:
1. Select provider and vertical
2. Execute the triage pipeline
3. View traces in Domino Experiment Manager

### Deploy as Agent

Deploy TriageFlow as an agentic system with a Streamlit interface for interaction.

**To deploy:**
1. Go to **Deployments > Agents** in Domino
2. Configure the agent with `app/app.sh` as the launch script
3. Click **Deploy**

See [Deploy Agentic Systems](https://docs.dominodatalab.com/en/cloud/user_guide/5aedde/deploy-agentic-systems/) for detailed instructions.

**Features:**
- Select provider (OpenAI/Anthropic) and industry vertical
- Choose from sample tickets or enter custom incidents
- View agent responses with reasoning
- See LLM judge scores and rationales
- Submit human evaluations (logged to traces)

### Scheduled Evaluation Script

The `run_scheduled_evaluation.py` script analyzes traces from a deployed agent and generates an HTML report.

**Setup:** Edit the top of the script with your deployed agent ID:
```python
AGENT_ID = "<REPLACE_WITH_YOUR_AGENT_ID>"
```

**Run daily analysis:**
```bash
python run_scheduled_evaluation.py
# Analyzes traces from the last 24 hours
# Saves report to /mnt/artifacts/daily_report_<timestamp>.html
```

**Batch processing with tracing:**
```bash
python run_scheduled_evaluation.py batch --vertical financial_services -n 10
```

**Add evaluations to existing traces:**
```bash
python run_scheduled_evaluation.py evaluate --run-id <mlflow_run_id>
```

## Project Structure

```
TriageFlow/
├── tracing-tutorial.ipynb      # Tutorial notebook
├── run_triage.py               # Standalone triage script with tracing
├── run_scheduled_evaluation.py # Scheduled evaluation (analyzes last 24h)
├── config.yaml                 # Agent prompts, tools, model configs
├── src/
│   ├── models.py               # Pydantic data models
│   ├── agents.py               # Four triage agents
│   ├── tools.py                # Agent tool functions
│   └── judges.py               # LLM judge evaluators
├── app/
│   ├── app.sh                  # Agent launch script
│   ├── main.py                 # Streamlit interface
│   └── utils/                  # Config management utilities
└── example-data/
    ├── financial_services.csv
    ├── healthcare.csv
    ├── energy.csv
    └── public_sector.csv
```

## Tracing Implementation

### Basic Setup

```python
from domino.agents.tracing import add_tracing, init_tracing
from domino.agents.logging import DominoRun, log_evaluation
import mlflow

# Initialize tracing
init_tracing()

# Enable framework autologging
mlflow.openai.autolog()
```

### Traced Function

```python
@add_tracing(name="triage_incident", autolog_frameworks=["openai"], evaluator=my_evaluator)
def triage_incident(incident):
    # Agent calls are automatically traced
    classification = classify_incident(...)
    impact = assess_impact(...)
    # ...
    return result
```

### Running with DominoRun

```python
with DominoRun(agent_config_path="config.yaml") as run:
    result = triage_incident(incident)
    # Traces are automatically captured and stored
```

### Post-hoc Evaluation

```python
from domino.agents.tracing import search_traces

# Retrieve traces from a completed run
traces = search_traces(run_id=run_id)

# Add evaluations
for trace in traces.data:
    log_evaluation(trace_id=trace.id, name="quality_score", value=4.5)
```

## Documentation

- [Deploy Agentic Systems](https://docs.dominodatalab.com/en/cloud/user_guide/5aedde/deploy-agentic-systems/)
- [Domino GenAI Tracing Guide](https://docs.dominodatalab.com/en/cloud/user_guide/fc1922/set-up-and-run-genai-traces/)
- [Automated GenAI Tracing Blueprint](https://domino.ai/resources/blueprints/automated-genai-tracing-for-agent-and-llm-experimentation)
