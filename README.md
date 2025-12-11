# TriageFlow

A multi-agent system that automatically triages incoming incidents—security alerts, service disruptions, compliance issues, or operational failures—by classifying them, assessing impact, assigning responders, and drafting stakeholder communications.

Demonstrates Domino's GenAI tracing and evaluation capabilities through a realistic, production-style pipeline applicable across financial services, healthcare, energy, and public sector.

## Pipeline

Incidents flow through four specialized agents:

1. **ClassifierAgent** — Categorizes the incident and assigns urgency
2. **ImpactAssessmentAgent** — Evaluates blast radius, affected users, and financial exposure
3. **ResourceMatcherAgent** — Identifies available responders based on skills and SLA requirements
4. **ResponseDrafterAgent** — Generates communications tailored to each stakeholder audience

Each agent uses dedicated tools to query historical data, check resource availability, and apply organizational policies.

## Key Features

**Single Trace Tree** — All agents, tools, and LLM calls appear in one hierarchical trace with proper span types (AGENT, TOOL, CHAT_MODEL).

**LLM Judges** — Three automated judges evaluate output quality:
- Classification Judge — Evaluates category/urgency appropriateness
- Response Judge — Assesses communication clarity and tone
- Triage Judge — Holistic assessment of the entire decision

**Aggregated Metrics** — `DominoRun` captures statistical summaries (mean, median, stdev) across all traces for monitoring quality scores.

**Multi-Vertical Support** — Sample incidents for Financial Services, Healthcare, Energy, and Public Sector.

## Setup

1. Save your API key as a Domino user environment variable:
   - Go to **Account Settings** → **User Environment Variables**
   - Add `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`

## Usage

Open `tracing-tutorial.ipynb` and run the cells sequentially:
1. Select your provider (OpenAI or Anthropic)
2. Select your industry vertical
3. Execute the triage pipeline

View traces in the Domino Experiment Manager.

## Configuration App

A Streamlit application for configuring and launching triage jobs without code.

**To publish the app:**
1. Go to **Deployments > App** in Domino
2. Select `app/app.sh` as the launch script
3. Click **Publish Domino App**

**Features:**
- Configure agent prompts, temperature, and token limits
- View and edit tool configurations
- Adjust LLM judge settings and evaluation thresholds
- Select industry vertical and sample tickets
- View pipeline diagram showing agent/tool flow
- Launch jobs and track history

Configurations are saved to the project dataset with timestamps for reproducibility.

## Project Structure

```
├── tracing-tutorial.ipynb  # Main notebook
├── config.yaml             # Prompts, tool schemas, model configs
├── run_triage.py           # CLI script for running triage
├── src/
│   ├── __init__.py
│   ├── models.py           # Pydantic data models
│   ├── agents.py           # Four triage agents
│   ├── tools.py            # Agent tool functions
│   └── judges.py           # LLM judge evaluators
├── app/
│   ├── app.sh              # Domino app launch script
│   ├── main.py             # Streamlit app entry point
│   ├── run_triage_app.py   # App-launched triage script
│   ├── pages/              # Streamlit pages
│   └── utils/              # Config and Domino client utilities
└── example-data/
    ├── sample_incidents.csv
    ├── financial_services.csv
    ├── healthcare.csv
    ├── energy.csv
    └── public_sector.csv
```
