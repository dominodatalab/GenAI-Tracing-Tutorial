"""
Deployable Judge Implementations

Standalone LLM-as-Judge evaluators that can be:
1. Run during the triage pipeline for quality scoring
2. Deployed as separate services for batch evaluation
3. Used in experiments for comparing model outputs

Each judge is designed to be model-agnostic and can use different LLM providers.
"""

import json
import os
import re
from typing import Dict, Any, Optional, List

from openai import OpenAI
from anthropic import Anthropic


class JudgeConfig:
    """Configuration for a judge instance."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        temperature: float = 0.1,
        max_tokens: int = 300
    ):
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens


class BaseJudge:
    """Base class for all judges."""

    def __init__(self, config: Optional[JudgeConfig] = None):
        self.config = config or JudgeConfig()
        self._client = None

    @property
    def client(self):
        """Lazy initialize the LLM client."""
        if self._client is None:
            if self.config.provider == "anthropic":
                self._client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            else:
                self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return self._client

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM and return the response."""
        if self.config.provider == "anthropic":
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        else:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content

    def _parse_json(self, text: str) -> Dict:
        """Extract JSON from response text."""
        try:
            # Try direct parse first
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON in the text
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                return json.loads(match.group(0))
            return {"error": "Failed to parse JSON", "raw": text}


class ClassificationJudge(BaseJudge):
    """
    Evaluates the quality of incident classification.

    Criteria:
    - Category appropriateness
    - Urgency accuracy
    - Reasoning quality
    - Confidence calibration
    """

    PROMPT_TEMPLATE = """Evaluate this incident classification. Score 1-5 (5=excellent).

Incident: {incident}

Classification:
- Category: {category}
- Subcategory: {subcategory}
- Urgency: {urgency}
- Confidence: {confidence}
- Reasoning: {reasoning}

Evaluate:
1. Is the category appropriate for this incident?
2. Is the urgency level justified by the description?
3. Is the reasoning sound and specific?
4. Is the confidence level calibrated appropriately?

Return JSON:
{{
    "score": <1-5>,
    "category_appropriate": <true/false>,
    "urgency_justified": <true/false>,
    "reasoning_quality": <1-3>,
    "rationale": "<brief explanation>"
}}"""

    def evaluate(
        self,
        incident: str,
        classification: Dict,
        ground_truth: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a classification result.

        Args:
            incident: Original incident description
            classification: Classification output dict
            ground_truth: Optional ground truth for comparison

        Returns:
            Evaluation scores and rationale
        """
        prompt = self.PROMPT_TEMPLATE.format(
            incident=incident,
            category=classification.get("category", "unknown"),
            subcategory=classification.get("subcategory", "unknown"),
            urgency=classification.get("urgency", 0),
            confidence=classification.get("confidence", 0),
            reasoning=classification.get("reasoning", "none")
        )

        response = self._call_llm(prompt)
        result = self._parse_json(response)

        # Add ground truth comparison if available
        if ground_truth:
            result["category_match"] = classification.get("category") == ground_truth.get("expected_category")
            result["urgency_delta"] = abs(
                classification.get("urgency", 0) - ground_truth.get("expected_urgency", 0)
            )

        return result


class ResponseJudge(BaseJudge):
    """
    Evaluates the quality of incident response communications.

    Criteria:
    - Tone appropriateness
    - Information clarity
    - Urgency alignment
    - Actionability
    """

    PROMPT_TEMPLATE = """Evaluate this incident communication. Score 1-5 (5=excellent).

Incident: {incident}
Urgency Level: {urgency}

Communication to {audience}:
Subject: {subject}
Body: {body}

Evaluate:
1. Is the tone appropriate for the audience and urgency?
2. Is the information clear and complete?
3. Does it convey appropriate urgency?
4. Are the next steps actionable?

Return JSON:
{{
    "score": <1-5>,
    "tone_appropriate": <true/false>,
    "information_clear": <true/false>,
    "urgency_conveyed": <true/false>,
    "actionable": <true/false>,
    "rationale": "<brief explanation>"
}}"""

    def evaluate(
        self,
        incident: str,
        urgency: int,
        communication: Dict
    ) -> Dict[str, Any]:
        """
        Evaluate a communication.

        Args:
            incident: Original incident description
            urgency: Urgency level (1-5)
            communication: Communication dict with audience, subject, body

        Returns:
            Evaluation scores and rationale
        """
        prompt = self.PROMPT_TEMPLATE.format(
            incident=incident,
            urgency=urgency,
            audience=communication.get("audience", "unknown"),
            subject=communication.get("subject", ""),
            body=communication.get("body", "")[:1000]  # Truncate long bodies
        )

        response = self._call_llm(prompt)
        return self._parse_json(response)


class TriageJudge(BaseJudge):
    """
    Evaluates the overall quality of a complete triage decision.

    Holistic assessment of the entire pipeline output.
    """

    PROMPT_TEMPLATE = """Evaluate this complete incident triage. Score 1-5 (5=excellent).

Incident: {incident}

Triage Summary:
- Category: {category} (Urgency: {urgency})
- Impact Score: {impact_score}/10, Blast Radius: {blast_radius}
- Primary Responder: {responder} (Match Score: {match_score})
- SLA Met: {sla_met}
- Action Items: {action_count}
- Escalation Required: {escalation}

Evaluate the overall triage quality:
1. Is the classification appropriate for the incident?
2. Is the impact assessment reasonable?
3. Is the resource assignment logical given the incident type?
4. Is the response plan comprehensive?

Return JSON:
{{
    "score": <1-5>,
    "classification_quality": <1-3>,
    "impact_assessment_quality": <1-3>,
    "resource_assignment_quality": <1-3>,
    "response_plan_quality": <1-3>,
    "rationale": "<brief explanation>"
}}"""

    def evaluate(
        self,
        incident: str,
        classification: Dict,
        impact: Dict,
        resources: Dict,
        response: Dict
    ) -> Dict[str, Any]:
        """
        Evaluate a complete triage result.

        Args:
            incident: Original incident description
            classification: Classification dict
            impact: Impact assessment dict
            resources: Resource assignment dict
            response: Response plan dict

        Returns:
            Holistic evaluation scores and rationale
        """
        # Extract primary responder info
        primary = resources.get("primary_responder", {})
        if hasattr(primary, "model_dump"):
            primary = primary.model_dump()

        prompt = self.PROMPT_TEMPLATE.format(
            incident=incident,
            category=classification.get("category", "unknown"),
            urgency=classification.get("urgency", 0),
            impact_score=impact.get("impact_score", 0),
            blast_radius=impact.get("blast_radius", "unknown"),
            responder=primary.get("name", "unknown") if isinstance(primary, dict) else "unknown",
            match_score=primary.get("match_score", 0) if isinstance(primary, dict) else 0,
            sla_met=resources.get("sla_met", False),
            action_count=len(response.get("action_items", [])),
            escalation=response.get("escalation_required", False)
        )

        response_text = self._call_llm(prompt)
        return self._parse_json(response_text)


class CompositeJudge:
    """
    Combines all judges for complete pipeline evaluation.

    Provides a single interface to run all evaluations and compute
    aggregate quality scores.
    """

    def __init__(self, config: Optional[JudgeConfig] = None):
        self.config = config or JudgeConfig()
        self.classification_judge = ClassificationJudge(config)
        self.response_judge = ResponseJudge(config)
        self.triage_judge = TriageJudge(config)

    def evaluate_classification(
        self,
        incident: str,
        classification: Dict,
        ground_truth: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Evaluate classification quality."""
        return self.classification_judge.evaluate(incident, classification, ground_truth)

    def evaluate_response(
        self,
        incident: str,
        urgency: int,
        communication: Dict
    ) -> Dict[str, Any]:
        """Evaluate response communication quality."""
        return self.response_judge.evaluate(incident, urgency, communication)

    def evaluate_triage(
        self,
        incident: str,
        classification: Dict,
        impact: Dict,
        resources: Dict,
        response: Dict
    ) -> Dict[str, Any]:
        """Evaluate overall triage quality."""
        return self.triage_judge.evaluate(incident, classification, impact, resources, response)

    def evaluate_all(
        self,
        incident: str,
        classification: Dict,
        impact: Dict,
        resources: Dict,
        response: Dict,
        ground_truth: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Run all evaluations and compute aggregate scores.

        Returns:
            Dict with all individual scores and combined_quality_score
        """
        results = {
            "classification_eval": self.evaluate_classification(
                incident, classification, ground_truth
            ),
            "triage_eval": self.evaluate_triage(
                incident, classification, impact, resources, response
            ),
            "response_evals": []
        }

        # Evaluate each communication
        communications = response.get("communications", [])
        urgency = classification.get("urgency", 3)
        for comm in communications[:3]:  # Limit to first 3 for efficiency
            if isinstance(comm, dict):
                results["response_evals"].append(
                    self.evaluate_response(incident, urgency, comm)
                )
            elif hasattr(comm, "model_dump"):
                results["response_evals"].append(
                    self.evaluate_response(incident, urgency, comm.model_dump())
                )

        # Calculate combined quality score
        class_score = results["classification_eval"].get("score", 3)
        triage_score = results["triage_eval"].get("score", 3)
        response_scores = [r.get("score", 3) for r in results["response_evals"]]
        avg_response_score = sum(response_scores) / len(response_scores) if response_scores else 3

        combined = (class_score + triage_score + avg_response_score) / 3
        results["combined_quality_score"] = round(combined, 2)

        # Flag for manual review if needed
        results["needs_manual_review"] = (
            urgency >= 4 and
            impact.get("impact_score", 0) >= 7 and
            combined < 3.5
        )

        return results


# Convenience function for quick evaluation
def evaluate_triage_output(
    incident: str,
    classification: Dict,
    impact: Dict,
    resources: Dict,
    response: Dict,
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    Quick evaluation of a triage output.

    Args:
        incident: Original incident description
        classification: Classification dict
        impact: Impact assessment dict
        resources: Resource assignment dict
        response: Response plan dict
        model: Model to use for evaluation

    Returns:
        Evaluation results with combined score
    """
    config = JudgeConfig(model=model)
    judge = CompositeJudge(config)
    return judge.evaluate_all(incident, classification, impact, resources, response)
