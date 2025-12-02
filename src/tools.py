import mlflow
from mlflow.entities import SpanType


@mlflow.trace(span_type=SpanType.TOOL)
def lookup_category_definitions() -> dict:
    """Returns category definitions and classification criteria."""
    return {
        "security": {
            "description": "Unauthorized access, data breaches, malware, security policy violations",
            "subcategories": ["unauthorized_access", "malware", "data_breach", "phishing", "policy_violation"],
            "keywords": ["breach", "unauthorized", "hack", "malware", "virus", "phishing", "credentials", "attack"]
        },
        "operational": {
            "description": "Day-to-day operations and business process issues",
            "subcategories": ["process_failure", "human_error", "resource_shortage", "vendor_issue"],
            "keywords": ["failed", "error", "stuck", "delayed", "missing", "incorrect"]
        },
        "performance": {
            "description": "System slowness, degradation, or capacity issues",
            "subcategories": ["latency", "throughput", "capacity", "timeout"],
            "keywords": ["slow", "timeout", "latency", "degraded", "capacity", "memory", "cpu"]
        },
        "data_integrity": {
            "description": "Data corruption, loss, or inconsistency",
            "subcategories": ["corruption", "data_loss", "sync_failure", "validation_error"],
            "keywords": ["corrupt", "lost", "missing data", "inconsistent", "mismatch", "invalid"]
        },
        "compliance": {
            "description": "Regulatory, audit, or policy compliance issues",
            "subcategories": ["regulatory", "audit_finding", "policy_breach", "certification"],
            "keywords": ["compliance", "audit", "regulation", "GDPR", "HIPAA", "SOX", "violation"]
        },
        "infrastructure": {
            "description": "Hardware, network, or platform infrastructure issues",
            "subcategories": ["hardware_failure", "network_issue", "cloud_service", "database"],
            "keywords": ["server", "network", "database", "cloud", "AWS", "Azure", "outage", "down"]
        },
        "user_access": {
            "description": "Authentication, authorization, or account management issues",
            "subcategories": ["login_issue", "permission_error", "account_locked", "sso_failure"],
            "keywords": ["login", "password", "access denied", "permission", "locked", "SSO", "MFA"]
        }
    }


@mlflow.trace(span_type=SpanType.TOOL)
def lookup_historical_incidents(category: str, subcategory: str, limit: int = 5) -> list[dict]:
    """Returns similar historical incidents."""
    historical_db = [
        {"ticket_id": "INC-2024-1001", "category": "security", "subcategory": "unauthorized_access",
         "impact_score": 8.5, "affected_users": 150, "resolution_hours": 6, "financial_impact": 45000},
        {"ticket_id": "INC-2024-0892", "category": "infrastructure", "subcategory": "database",
         "impact_score": 7.0, "affected_users": 500, "resolution_hours": 4, "financial_impact": 25000},
        {"ticket_id": "INC-2024-0756", "category": "performance", "subcategory": "latency",
         "impact_score": 5.5, "affected_users": 1200, "resolution_hours": 2, "financial_impact": 8000},
        {"ticket_id": "INC-2024-0623", "category": "security", "subcategory": "phishing",
         "impact_score": 6.0, "affected_users": 25, "resolution_hours": 12, "financial_impact": 15000},
        {"ticket_id": "INC-2024-0445", "category": "data_integrity", "subcategory": "sync_failure",
         "impact_score": 7.5, "affected_users": 80, "resolution_hours": 8, "financial_impact": 35000}
    ]
    matches = [inc for inc in historical_db if inc["category"] == category]
    matches.sort(key=lambda x: (x["subcategory"] == subcategory, x["impact_score"]), reverse=True)
    return matches[:limit]


@mlflow.trace(span_type=SpanType.TOOL)
def calculate_impact_score(urgency: int, affected_users: int, affected_systems_count: int,
                           has_financial_impact: bool, has_compliance_implications: bool) -> float:
    """Calculates normalized impact score (0-10)."""
    base_score = (urgency - 1) * 1.0

    if affected_users > 1000:
        user_score = 2.0
    elif affected_users > 100:
        user_score = 1.5
    elif affected_users > 10:
        user_score = 1.0
    else:
        user_score = 0.5

    system_score = min(affected_systems_count * 0.5, 2.0)
    financial_score = 1.0 if has_financial_impact else 0.0
    compliance_score = 1.0 if has_compliance_implications else 0.0

    return min(base_score + user_score + system_score + financial_score + compliance_score, 10.0)


@mlflow.trace(span_type=SpanType.TOOL)
def check_resource_availability(required_skills: list[str], urgency: int) -> list[dict]:
    """Returns available resources matching required skills."""
    resource_pool = [
        {"resource_id": "RES-001", "name": "Alice Chen", "role": "Senior Security Engineer",
         "skills": ["security", "incident_response", "forensics", "network"], "availability": "immediate", "current_load": 2},
        {"resource_id": "RES-002", "name": "Bob Martinez", "role": "Infrastructure Lead",
         "skills": ["infrastructure", "database", "cloud", "networking"], "availability": "within_1h", "current_load": 3},
        {"resource_id": "RES-003", "name": "Carol Williams", "role": "DevOps Engineer",
         "skills": ["infrastructure", "performance", "monitoring", "automation"], "availability": "immediate", "current_load": 1},
        {"resource_id": "RES-004", "name": "David Kim", "role": "Data Engineer",
         "skills": ["data_integrity", "database", "ETL", "compliance"], "availability": "within_4h", "current_load": 4},
        {"resource_id": "RES-005", "name": "Eva Johnson", "role": "Compliance Officer",
         "skills": ["compliance", "audit", "policy", "security"], "availability": "within_1h", "current_load": 2},
        {"resource_id": "RES-006", "name": "Frank Lee", "role": "Support Team Lead",
         "skills": ["user_access", "troubleshooting", "communication", "escalation"], "availability": "immediate", "current_load": 3}
    ]

    availability_scores = {"immediate": 0.2, "within_1h": 0.15, "within_4h": 0.1, "next_business_day": 0.0}

    for resource in resource_pool:
        skill_overlap = len(set(resource["skills"]) & set(required_skills))
        skill_score = skill_overlap / max(len(required_skills), 1)
        availability_bonus = availability_scores.get(resource["availability"], 0)
        load_penalty = resource["current_load"] * 0.05
        resource["match_score"] = min(max(skill_score + availability_bonus - load_penalty, 0), 1.0)

    if urgency >= 4:
        resource_pool = [r for r in resource_pool if r["availability"] in ["immediate", "within_1h"]]

    return sorted(resource_pool, key=lambda x: x["match_score"], reverse=True)


@mlflow.trace(span_type=SpanType.TOOL)
def get_sla_requirements(urgency: int, category: str) -> dict:
    """Returns SLA requirements based on urgency and category."""
    base_sla = {
        5: {"response_hours": 1, "resolution_hours": 4, "escalation_threshold_hours": 2},
        4: {"response_hours": 2, "resolution_hours": 8, "escalation_threshold_hours": 4},
        3: {"response_hours": 4, "resolution_hours": 24, "escalation_threshold_hours": 8},
        2: {"response_hours": 8, "resolution_hours": 48, "escalation_threshold_hours": 24},
        1: {"response_hours": 24, "resolution_hours": 168, "escalation_threshold_hours": 48}
    }
    sla = base_sla.get(urgency, base_sla[3]).copy()

    if category in ["compliance", "security"]:
        sla["response_hours"] = max(sla["response_hours"] * 0.5, 0.5)
        sla["escalation_threshold_hours"] = max(sla["escalation_threshold_hours"] * 0.5, 1)

    return sla


@mlflow.trace(span_type=SpanType.TOOL)
def get_communication_templates(category: str, urgency: int) -> dict:
    """Returns communication templates for different audiences."""
    return {
        "technical_team": {
            "template": "INCIDENT ALERT - {urgency_label}\n\nTicket: {ticket_id}\nCategory: {category}\nAssigned To: {primary_responder}\n\nSummary: {summary}\n\nImmediate Actions Required:\n{action_items}\n\nEscalation Path: {escalation_path}",
            "tone": "direct and technical"
        },
        "management": {
            "template": "INCIDENT NOTIFICATION - {urgency_label}\n\nExecutive Summary:\n{executive_summary}\n\nBusiness Impact:\n- Affected Users: {affected_users}\n- Estimated Financial Impact: {financial_impact}\n- Current Status: {status}\n\nResponse:\n- Lead Responder: {primary_responder}\n- Target Resolution: {resolution_target}",
            "tone": "professional and concise"
        },
        "affected_users": {
            "template": "Service Notice\n\nWe are aware of an issue affecting {affected_service}.\n\nWhat's happening: {user_friendly_description}\n\nWhat you can do: {user_actions}\n\nExpected resolution: {resolution_estimate}",
            "tone": "empathetic and clear"
        }
    }


@mlflow.trace(span_type=SpanType.TOOL)
def get_stakeholder_list(category: str, impact_score: float, blast_radius: str) -> list[dict]:
    """Determines which stakeholders need notification."""
    stakeholders = [{"audience": "technical_team", "required": True, "notification_method": "immediate"}]

    if impact_score >= 5 or blast_radius in ["organization", "external"]:
        stakeholders.append({
            "audience": "management", "required": True,
            "notification_method": "immediate" if impact_score >= 7 else "within_1h"
        })

    if blast_radius in ["department", "organization", "external"]:
        stakeholders.append({
            "audience": "affected_users", "required": True,
            "notification_method": "after_initial_assessment"
        })

    return stakeholders
