# Agent Experimentation Guide

This guide covers all parameters to experiment with when optimizing agents, including what to measure and how to structure experiments.

---

## Experimentation Flow

```
Phase 1: Bootstrap Judge (manual labels)
         │
         ▼
Phase 2: Validate Judge (human agreement)
         │
         ▼
Phase 3: Optimize Agents (using validated judge)
         │
         ▼
Phase 4: Refine Judge (on agent failure modes)
         │
         ▼
         └──► Iterate phases 3-4
```

**Key insight**: Optimize judge first. A bad judge means you're optimizing agents toward garbage. A good judge lets you iterate on agents quickly without re-labeling.

---

## Parameters to Experiment With

| Parameter | What It Controls | Typical Range |
|-----------|------------------|---------------|
| **Temperature** | Randomness/creativity | 0.0–1.0 |
| **Top-p (nucleus sampling)** | Cumulative probability cutoff | 0.9–1.0 |
| **Max tokens** | Response length limit | Task-dependent |
| **System prompt** | Persona, constraints, format | Variants |
| **Few-shot examples** | In-context learning | 0–5 examples |
| **Output format** | JSON, XML, free text | Structured vs. natural |
| **Tool/function definitions** | How tools are described | Verbose vs. minimal |
| **Retry/fallback logic** | Error handling strategy | 0–3 retries |

---

## Parameters by Agent Type

### Classifier Agent

| Parameter | Recommended Range | Notes |
|-----------|-------------------|-------|
| Temperature | 0.1–0.3 | Low, you want consistency |
| Few-shot examples | 2–4 | Examples per category dramatically improves edge cases |
| System prompt | Rubric-style | Clear category definitions matter hugely |
| Output format | Strict JSON | Schema enforcement reduces parse failures |

### Reasoning/Analysis Agent (Impact Assessor)

| Parameter | Recommended Range | Notes |
|-----------|-------------------|-------|
| Temperature | 0.3–0.5 | Medium, some exploration helps |
| Chain-of-thought | Test both | "Think step by step" vs. direct answer |
| System prompt | Scoring rubric | Clear criteria for assessment |
| Output format | JSON with reasoning field | Capture the "why" |

### Tool-Using Agent (Resource Matcher)

| Parameter | Recommended Range | Notes |
|-----------|-------------------|-------|
| Temperature | 0.0–0.2 | Low, tool calls need precision |
| Tool descriptions | Verbose with examples | How tools are described matters |
| Max tool calls | 1–5 | Limit runaway tool use |
| Output format | Strict JSON | Tool call schema must be exact |

### Generation Agent (Response Drafter)

| Parameter | Recommended Range | Notes |
|-----------|-------------------|-------|
| Temperature | 0.5–0.8 | Higher, you want natural language |
| Tone instructions | Formal/professional/friendly | Match audience |
| Length guidance | Concise/standard/detailed | Explicit word counts help |
| Few-shot examples | 1–2 | Style examples matter here |

---

## Experiment Grid Configuration

```yaml
agent_experiment:
  classifier:
    model: [local-qwen-3b, gpt-4o-mini]
    temperature: [0.1, 0.2, 0.3]
    few_shot_count: [0, 2, 4]
    system_prompt_variant: [minimal, detailed, rubric]
    output_format: [json_strict, json_flexible]
    
  impact_assessor:
    model: [local-qwen-3b, gpt-4o-mini]
    temperature: [0.2, 0.4, 0.6]
    chain_of_thought: [true, false]
    system_prompt_variant: [scoring_rubric, open_analysis]
    
  resource_matcher:
    model: [local-qwen-3b, gpt-4o-mini]
    temperature: [0.0, 0.1, 0.2]
    tool_description_style: [minimal, verbose, with_examples]
    max_tool_calls: [1, 3, 5]
    
  response_drafter:
    model: [local-qwen-3b, gpt-4o-mini]
    temperature: [0.5, 0.7, 0.9]
    tone: [formal, professional, friendly]
    length_guidance: [concise, standard, detailed]
    few_shot_count: [0, 1, 2]
```

---

## System Prompt Variants

### Classifier: Minimal

```
Classify incidents into: security, performance, access, compliance, other.
Return JSON with category and urgency.
```

### Classifier: Detailed

```
You are an IT incident classifier. Analyze the incident and determine:
1. Category: security (breaches, vulnerabilities), performance (slowdowns, outages), 
   access (login issues, permissions), compliance (policy violations), other
2. Urgency: critical (immediate action), high (same day), medium (this week), low (backlog)

Consider impact scope, affected systems, and business context.
Return JSON: {"category": "...", "urgency": "...", "reasoning": "..."}
```

### Classifier: Rubric

```
Classify this incident using the following rubric:

CATEGORY DEFINITIONS:
- security: Any unauthorized access, data exposure, malware, or vulnerability
- performance: System slowdowns, timeouts, resource exhaustion, outages
- access: Login failures, permission errors, account issues (NOT security breaches)
- compliance: Policy violations, audit findings, regulatory issues
- other: Does not fit above categories

URGENCY CRITERIA:
- critical: Production down, data breach active, >100 users affected
- high: Degraded service, potential breach, 10-100 users affected
- medium: Workaround exists, <10 users affected
- low: Cosmetic, feature request, no immediate impact

Return JSON: {"category": "...", "urgency": "...", "reasoning": "..."}
```

---

## Few-Shot Example Variants

### Minimal (input/output only)

```yaml
examples:
  - input: "Database CPU at 100%, queries timing out"
    output: '{"category": "performance", "urgency": "critical"}'
```

### With Reasoning

```yaml
examples:
  - input: "Database CPU at 100%, queries timing out"
    output: '{"category": "performance", "urgency": "critical", "reasoning": "Resource exhaustion affecting query execution indicates performance issue. Production impact makes this critical."}'
```

### Edge Cases Emphasized

```yaml
examples:
  - input: "User locked out after too many password attempts"
    output: '{"category": "access", "urgency": "medium", "reasoning": "Account lockout is access issue, not security breach. Standard reset process applies."}'
    
  - input: "User locked out, claims they never attempted login"
    output: '{"category": "security", "urgency": "high", "reasoning": "Lockout without user action suggests credential compromise attempt. Escalate for investigation."}'
```

---

## Metrics to Track

### Quality Metrics (from judge)

| Metric | Description |
|--------|-------------|
| `aggregate_score` | Primary weighted quality metric |
| `category_accuracy` | For classifier agent |
| `urgency_accuracy` | For classifier agent |
| `reasoning_quality` | For analytical agents |
| `response_appropriateness` | For generation agents |

### Reliability Metrics

| Metric | Description |
|--------|-------------|
| `json_parse_success_rate` | Did output parse correctly? |
| `schema_compliance_rate` | Did output match expected schema? |
| `retry_rate` | How often did we need to retry? |
| `error_rate` | Exceptions, timeouts, failures |

### Efficiency Metrics

| Metric | Description |
|--------|-------------|
| `latency_ms` | Time to complete |
| `input_tokens` | Prompt size |
| `output_tokens` | Response size |
| `cost_per_call` | $ per invocation |

### Consistency Metrics

| Metric | Description |
|--------|-------------|
| `score_variance` | Same input across multiple runs |
| `output_stability` | How often same input → same output |

---

## Interaction Effects to Watch

Some parameters interact in important ways:

| Combination | Watch For |
|-------------|-----------|
| High temp + strict JSON | More parse failures |
| Few-shot + long prompts | Token limits, increased cost |
| CoT + low temp | Verbose but repetitive reasoning |
| Minimal tool descriptions + complex tools | Wrong tool selection |
| High few-shot count + edge cases | May overfit to examples |

---

## Judge Experiment Configuration

Before optimizing agents, validate your judge:

```yaml
judge_experiment:
  models:
    - gpt-4o-mini
    - gpt-4o
    - claude-sonnet-4-20250514
    
  temperatures: [0.0, 0.1, 0.3]
  
  prompt_styles:
    - direct      # "Is this correct? Answer yes/no."
    - cot         # "Think step by step, then answer yes/no."
    - rubric      # "Use this rubric: [criteria]. Then score."
    - few_shot    # "Here are examples: [good], [bad]. Now evaluate:"
    
  scale_types:
    - binary        # [pass, fail]
    - three_point   # [1, 2, 3]
    - five_point    # [1, 2, 3, 4, 5]
    
  # Exit criteria
  min_accuracy: 0.85
  min_kappa: 0.70
```

### Judge Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| `human_agreement_accuracy` | % match with human labels | >85% |
| `cohen_kappa` | Agreement adjusted for chance | >0.70 |
| `consistency_std` | Variance across repeated runs | <0.1 |
| `pearson_correlation` | For scored criteria | >0.7 |

---

## Ground Truth Data Requirements

You need human-labeled examples to measure judge quality:

```yaml
# test_data/ground_truth_judgments.jsonl
{
  "id": "gt_001",
  "incident_text": "Production database showing 99% CPU...",
  "agent_output": {
    "category": "performance",
    "urgency": "critical",
    "reasoning": "Database CPU saturation affects all services"
  },
  "human_labels": {
    "category_correct": true,
    "urgency_correct": true,
    "reasoning_quality": 3
  },
  "notes": "Clear-cut critical incident"
}
```

**Aim for 50-100 labeled examples covering:**
- Clear correct cases
- Clear incorrect cases
- Edge cases / ambiguous
- Different incident types
- Different urgency levels

---

## Priority Order for Experimentation

### High Impact (Start Here)

1. **System prompt variant** — Usually the biggest lever
2. **Few-shot count** — 0 vs. 2-3 is often dramatic difference
3. **Temperature** — Fine-tune after prompt is stable
4. **Output format strictness** — JSON schema enforcement

### Medium Impact

5. **Chain-of-thought** — For reasoning-heavy agents
6. **Tool descriptions** — For tool-using agents
7. **Tone/length guidance** — For generation agents

### Low Impact (Skip Initially)

- Top-p (leave at 1.0 unless specific issues)
- Frequency/presence penalties (rarely needed)
- Multiple model comparisons (expensive, do after prompt is stable)

---

## Expected Findings (Based on Research)

| Parameter | Likely Finding |
|-----------|----------------|
| Model | GPT-4o-mini ~90% as good as GPT-4o at 10% cost |
| Temperature | 0.0-0.1 best for consistency, 0.5+ for generation |
| Scale | Binary >> 5-point for judge reliability |
| Prompt | CoT improves ~5-10% but 2x tokens |
| Few-shot | Helps edge cases, diminishing returns after 2-3 examples |
| Local vs API | Local Qwen 3B typically 10-15% behind GPT-4o-mini on quality |

---

## Project Structure

```
triageflow/
├── app.sh
├── app.py
│
├── agents/
│   ├── __init__.py
│   ├── classifier.py
│   ├── impact_assessor.py
│   ├── resource_matcher.py
│   └── response_drafter.py
│
├── guardrails/
│   ├── __init__.py
│   ├── checks.py
│   └── models.py
│
├── judges/
│   ├── __init__.py
│   ├── evaluators.py
│   ├── prompts.py
│   └── clients.py
│
├── experiments/
│   ├── run_judge_experiment.py
│   ├── run_agent_experiment.py
│   └── analyze_results.py
│
├── config/
│   ├── agents.yaml
│   ├── guardrails.yaml
│   ├── judges.yaml
│   └── experiment_grid.yaml
│
└── test_data/
    ├── incidents.jsonl
    ├── ground_truth_labels.jsonl
    └── ground_truth_judgments.jsonl
```

---

## Execution Commands

```bash
# Phase 1: Create ground truth (manual, outside Domino)
# ... label 50 examples in test_data/ground_truth_*.jsonl

# Phase 2: Validate judge
domino run experiments/run_judge_experiment.py

# Review results in Domino Experiment Manager
# Update config/judges.yaml with best settings

# Phase 3: Optimize agents
domino run experiments/run_agent_experiment.py

# Review results
# Update config/agents.yaml with best settings

# Phase 4: Add edge cases found during phase 3 to ground truth
# Re-run phase 2 if judge struggled on new cases
```

---

## Final Configuration Output

After experimentation, your configs should look like:

### config/agents.yaml

```yaml
agents:
  classifier:
    model: gpt-4o-mini
    temperature: 0.2
    system_prompt_variant: rubric
    few_shot_count: 3
    output_format: json_strict
    validated_score: 0.92
    
  impact_assessor:
    model: local-qwen-3b
    temperature: 0.4
    chain_of_thought: true
    system_prompt_variant: scoring_rubric
    validated_score: 0.87
    
  resource_matcher:
    model: local-qwen-3b
    temperature: 0.1
    tool_description_style: verbose
    max_tool_calls: 3
    validated_score: 0.89
    
  response_drafter:
    model: gpt-4o-mini
    temperature: 0.7
    tone: professional
    length_guidance: standard
    few_shot_count: 2
    validated_score: 0.85
```

### config/judges.yaml

```yaml
judges:
  classification_judge:
    model: gpt-4o-mini
    temperature: 0.1
    prompt_style: cot
    human_agreement: 0.94
    
  response_judge:
    model: gpt-4o-mini
    temperature: 0.1
    prompt_style: rubric
    human_agreement: 0.88
    
  # Escalation for ambiguous cases
  review_judge:
    model: gpt-4o
    trigger: "aggregate_score < 0.7"
```
