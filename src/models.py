from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class IncidentSource(str, Enum):
    MONITORING = "monitoring"
    USER_REPORT = "user_report"
    AUTOMATED_SCAN = "automated_scan"
    EXTERNAL_NOTIFICATION = "external_notification"
    AUDIT = "audit"


class IncidentCategory(str, Enum):
    SECURITY = "security"
    OPERATIONAL = "operational"
    PERFORMANCE = "performance"
    DATA_INTEGRITY = "data_integrity"
    COMPLIANCE = "compliance"
    INFRASTRUCTURE = "infrastructure"
    USER_ACCESS = "user_access"


class Incident(BaseModel):
    ticket_id: str
    description: str
    source: IncidentSource
    timestamp: datetime = Field(default_factory=datetime.now)
    reporter: Optional[str] = None
    affected_system: Optional[str] = None
    initial_severity: Optional[int] = Field(None, ge=1, le=5)


class Classification(BaseModel):
    category: IncidentCategory
    subcategory: str
    urgency: int = Field(..., ge=1, le=5)
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    affected_domain: str


class ImpactAssessment(BaseModel):
    impact_score: float = Field(..., ge=0.0, le=10.0)
    affected_users_estimate: int
    affected_systems: list[str]
    financial_exposure: Optional[float] = None
    similar_incidents: list[dict]
    blast_radius: str
    reasoning: str


class Resource(BaseModel):
    resource_id: str
    name: str
    role: str
    skills: list[str]
    availability: str
    match_score: float = Field(..., ge=0.0, le=1.0)


class ResourceAssignment(BaseModel):
    primary_responder: Resource
    backup_responders: list[Resource]
    sla_target_hours: float
    sla_met: bool
    escalation_path: list[str]
    reasoning: str


class Communication(BaseModel):
    audience: str
    subject: str
    body: str
    urgency_indicator: str


class ResponsePlan(BaseModel):
    communications: list[Communication]
    action_items: list[str]
    estimated_resolution_time: str
    escalation_required: bool
    completeness_score: float = Field(..., ge=0.0, le=1.0)


class TriageResult(BaseModel):
    ticket_id: str
    classification: Classification
    impact: ImpactAssessment
    resources: ResourceAssignment
    response: ResponsePlan
    total_processing_time_seconds: float
    trace_id: Optional[str] = None
