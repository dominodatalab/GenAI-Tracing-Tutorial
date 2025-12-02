import json
import re
import mlflow

CLASSIFICATION_JUDGE_PROMPT = """Evaluate this incident classification. Score 1-5 (5=excellent).

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

RESPONSE_JUDGE_PROMPT = """Evaluate this incident communication. Score 1-5 (5=excellent).

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

TRIAGE_JUDGE_PROMPT = """Evaluate this complete incident triage. Score 1-5 (5=excellent).

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


def call_judge(client, provider: str, prompt: str) -> dict:
    """Call LLM to judge output quality."""
    try:
        if provider == "openai":
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            content = response.choices[0].message.content
        else:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.content[0].text

        match = re.search(r'\{[\s\S]*\}', content)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass
    return {"score": 3, "rationale": "Judge evaluation failed"}


def judge_classification(client, provider: str, incident: str, classification: dict) -> dict:
    """Judge the classification quality."""
    prompt = CLASSIFICATION_JUDGE_PROMPT.format(
        incident=incident,
        category=classification.get("category", "unknown"),
        subcategory=classification.get("subcategory", "unknown"),
        urgency=classification.get("urgency", 3),
        reasoning=classification.get("reasoning", "")
    )
    return call_judge(client, provider, prompt)


def judge_response(client, provider: str, incident: str, urgency: int, communication: dict) -> dict:
    """Judge the response communication quality."""
    prompt = RESPONSE_JUDGE_PROMPT.format(
        incident=incident,
        urgency=urgency,
        audience=communication.get("audience", "unknown"),
        subject=communication.get("subject", ""),
        body=communication.get("body", "")[:500]
    )
    return call_judge(client, provider, prompt)


def judge_triage(client, provider: str, incident: str, classification: dict,
                 impact: dict, resources: dict, response: dict) -> dict:
    """Judge the overall triage quality."""
    primary = resources.get("primary_responder", {})
    if hasattr(primary, "model_dump"):
        primary = primary.model_dump()

    prompt = TRIAGE_JUDGE_PROMPT.format(
        incident=incident,
        category=classification.get("category", "unknown"),
        urgency=classification.get("urgency", 3),
        impact_score=impact.get("impact_score", 5),
        blast_radius=impact.get("blast_radius", "unknown"),
        responder=primary.get("name", "unknown") if isinstance(primary, dict) else "unknown",
        match_score=primary.get("match_score", 0) if isinstance(primary, dict) else 0,
        sla_met=resources.get("sla_met", False),
        action_count=len(response.get("action_items", []))
    )
    return call_judge(client, provider, prompt)
