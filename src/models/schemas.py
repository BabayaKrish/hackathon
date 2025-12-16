from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Customer ID")
    query: str = Field(..., description="Natural language query")
    account_ids: Optional[List[str]] = Field(
        default=None, 
        description="Specific accounts to query"
    )
    session_id: Optional[str] = Field(
        default=None, 
        description="Conversation session ID for context"
    )

class IntentResult(BaseModel):
    intent_type: str  # REPORT | PLAN_UPGRADE | PLAN_INFO | UNCLEAR
    confidence: float  # 0.0-1.0
    report_type: Optional[str] = None  # balance, wire, ach, etc.
    plan_type: Optional[str] = None  # bronze, silver, gold, etc.
    action_details: Optional[Dict[str, Any]] = None

class ReportData(BaseModel):
    report_id: str
    report_type: str
    customer_id: str
    account_ids: List[str]
    data: Dict[str, Any]
    record_count: int
    generated_at: datetime
    data_format: str  # JSON, CSV, PDF

class PlanRecommendation(BaseModel):
    current_plan: str
    recommended_plan: str
    reason: str
    monthly_savings: float
    feature_gaps: List[str]  # Features customer needs but doesn't have
    implementation_date: Optional[str] = None

class ChatResponse(BaseModel):
    success: bool
    message: str
    agent: str  # ROOT, REPORT, PLAN, PLAN_INFO
    confidence: float
    intent: Optional[IntentResult] = None
    data: Optional[Dict[str, Any]] = None
    charts: Optional[List[Dict]] = None  # Plotly chart specs
    recommendations: Optional[List[PlanRecommendation]] = None
    error: Optional[str] = None
    request_id: str  # For tracking

class ComplianceCheckResult(BaseModel):
    is_approved: bool
    reason: str
    required_actions: Optional[List[str]] = None
    mfa_required: bool = False

class AuditLog(BaseModel):
    request_id: str
    customer_id: str
    agent_name: str
    action: str
    report_type: Optional[str] = None
    request_timestamp: datetime
    response_time_ms: int
    status: str  # SUCCESS, FAILURE, PARTIAL
    confidence_score: float
    error_message: Optional[str] = None
