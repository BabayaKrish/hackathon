
# ROOT AGENT - Google ADK Implementation (Fixed)
# File: backend/agents/root_agent_adk.py

import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import time
from functools import wraps
import asyncio
import sys
import os

# Add parent directory to path for imports
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common_tools'))
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common_tools'))

from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError as GoogleCloudError
from vertexai.generative_models import GenerativeModel, Tool
import vertexai
from common_tools.AuditTool import async_log_event
# ==================== Configuration & Types ====================

class Intent(str, Enum):
    REPORT_REQUEST = "REPORT_REQUEST"
    PLAN_UPGRADE = "PLAN_UPGRADE"
    PLAN_INFO = "PLAN_INFO"
    BALANCE_CHECK = "BALANCE_CHECK"
    UNKNOWN = "UNKNOWN"


class ComplianceStatus(str, Enum):
    ALLOWED = "ALLOWED"
    DENIED = "DENIED"
    REQUIRES_MFA = "REQUIRES_MFA"


@dataclass
class IntentClassificationResult:
    intent: Intent
    confidence: float
    reasoning: str


@dataclass
class ComplianceCheckResult:
    status: ComplianceStatus
    allowed_features: List[str]
    denied_reason: Optional[str]
    requires_verification: bool


@dataclass
class RoutingDecision:
    target_agent: str
    reasoning: str
    parameters: Dict[str, Any]


@dataclass
class AgentExecutionResult:
    agent_name: str
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    execution_time_ms: float = 0.0


@dataclass
class AggregatedResult:
    intent: Intent
    confidence: float
    compliance_passed: bool
    data: Dict[str, Any]
    next_actions: List[str]
    reasoning: str
    execution_log: List[Dict[str, Any]]


# ==================== Logging & Monitoring ====================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collect and log metrics for observability"""
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.bq_client = bigquery.Client(project=project_id)
        self.metrics = []
    
    def record_execution(self, log_entry: Dict[str, Any]):
        """Write execution log to BigQuery"""
        try:
            table_id = f"{self.project_id}.client_data.agent_audit_log"
            log_entry['created_at'] = datetime.utcnow().isoformat()
            log_entry['log_id'] = f"log_{int(datetime.utcnow().timestamp())}"
            
            errors = self.bq_client.insert_rows_json(
                table_id, [log_entry], skip_invalid_rows=True
            )
            if errors:
                logger.error(f"Failed to insert audit log: {errors}")
        except Exception as e:
            logger.error(f"Error recording metrics: {str(e)}")


# ==================== Tool Decorator ====================

def adk_tool(func):
    """Decorator to mark functions as ADK tools"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    wrapper.is_tool = True
    return wrapper


# ==================== Root Agent - Google ADK Implementation ====================

class RootAgent:
    """
    Financial services routing assistant using Google ADK.
    
    This agent:
    1. Classifies user intent using Vertex AI
    2. Validates compliance based on plan tier
    3. Routes to specialized sub-agents
    4. Aggregates and returns results
    """
    
    def __init__(self, project_id: str, region: str = "us-central1"):
        """Initialize Root Agent with Google ADK"""
        self.project_id = project_id
        self.region = region
        self.bq_client = bigquery.Client(project=project_id)
        self.metrics = MetricsCollector(project_id)
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=region)
        self.model = GenerativeModel("gemini-2.0-flash-exp")
        
        logger.info(f"RootAgent initialized for project {project_id}")
    
    # ==================== Audit Logging Helper ====================
    
    async def _audit_log(
        self,
        agent_name: str,
        caller: str,
        input_text: str,
        output_text: str,
        latency_ms: float,
        safety_result: str = "allowed",
        error_message: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Log event to audit trail using AuditTool.
        
        Args:
            agent_name: Name of the agent performing the action
            caller: Caller context (user_id or function name)
            input_text: Input to the operation
            output_text: Output from the operation
            latency_ms: Execution time in milliseconds
            safety_result: Compliance status (allowed, denied, etc.)
            error_message: Error message if any
            metadata: Additional metadata to log
        """
        if async_log_event is None:
            logger.debug("AuditTool not available, skipping audit logging")
            return
        
        try:
            result = await async_log_event(
                agent_name=agent_name,
                model_name="gemini-2.0-flash-exp",
                caller=caller,
                input_text=input_text,
                output_text=output_text,
                latency_ms=latency_ms,
                safety_result=safety_result,
                error_message=error_message,
                metadata=metadata
            )
            logger.info(f"Audit logged: {result}")
        except Exception as e:
            logger.error(f"Failed to log audit event: {str(e)}")
    
    # ==================== Tool Definitions ====================
    
    @adk_tool
    def classify_intent(self, query: str) -> str:
        """
        TOOL 1: Classify user intent using Vertex AI LLM.
        
        Args:
            query: Natural language user query
        
        Returns:
            JSON string with intent, confidence, and reasoning
            
        Intents:
            - REPORT_REQUEST: User wants to view reports
            - PLAN_UPGRADE: User wants to change plan
            - PLAN_INFO: User asks for plan information
            - BALANCE_CHECK: User wants account balance
        
        Example:
            Input: "Show me my wire status report"
            Output: {"intent": "REPORT_REQUEST", "confidence": 0.95, "reasoning": "..."}
        """
        logger.info(f"Classifying intent for query: {query}")
        
        start_time = time.time()
        
        prompt = f"""
You are a financial services intent classifier. Analyze the following customer query and determine their intent.

Query: "{query}"

Available intents:
1. REPORT_REQUEST - User wants to view or download a report (wire reports, balance reports, transaction history)
2. PLAN_UPGRADE - User wants to change or upgrade their plan to a higher tier
3. PLAN_INFO - User is asking for information about plans, features, pricing, or comparisons
4. BALANCE_CHECK - User wants to know their current account balance or recent transactions

Respond ONLY in valid JSON format with no additional text:
{{
    "intent": "REPORT_REQUEST|PLAN_UPGRADE|PLAN_INFO|BALANCE_CHECK",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of classification"
}}

Rules:
- Confidence must be between 0 and 1
- If unclear or ambiguous, set confidence < 0.8
- Return valid JSON only, no markdown or explanations
"""
        response_text = ""  # ← DEFINE BEFORE TRY BLOCK

        try:
            logger.info(f"Sending prompt to Vertex AI (length: {len(prompt)} chars)")
            response = self.model.generate_content(prompt)

            if not response.text or not response.text.strip():
                logger.warning(f"Empty response from LLM for query: {query}")
                error_output = json.dumps({
                    "status": "error",
                    "intent": "UNKNOWN",
                    "confidence": 0.0,
                    "error": "Empty LLM response"
                })
                # Audit the error
                latency_ms = int((time.time() - start_time) * 1000)
                self._audit_log(
                    agent_name="classify_intent",
                    caller="system",
                    input_text=query,
                    output_text=error_output,
                    latency_ms=latency_ms,
                    safety_result="error",
                    error_message="Empty LLM response"
                )
                return error_output

            logger.info(f"Response text length: {len(response.text) if response.text else 0}")
            logger.info(f"Response text: {response.text[:500] if response.text else 'EMPTY'}")
            response_text = response.text.strip()

            if response_text.startswith("```"):
                # Remove opening fence (```json or ```
                response_text = response_text.lstrip("`").lstrip("json").lstrip()
                # Remove closing fence (```)
                response_text = response_text.rstrip("`").strip()

            result = json.loads(response_text)

            # Ensure valid structure
            intent_value = result.get("intent", "UNKNOWN")
            confidence = float(result.get("confidence", 0.0))
            reasoning = result.get("reasoning", "")

            logger.info(f"Intent: {intent_value} (confidence: {confidence})")

            output = json.dumps({
                "status": "success",
                "intent": intent_value,
                "confidence": confidence,
                "reasoning": reasoning
            })
            
            # Audit the successful classification
            latency_ms = int((time.time() - start_time) * 1000)
            self._audit_log(
                agent_name="classify_intent",
                caller="system",
                input_text=query,
                output_text=output,
                latency_ms=latency_ms,
                safety_result="allowed",
                metadata={"intent": intent_value, "confidence": confidence}
            )
            
            return output
        except Exception as e:
            logger.error(f"Intent classification error: {str(e)}")
            error_output = json.dumps({
                "status": "error",
                "intent": "UNKNOWN",
                "confidence": 0.0,
                "error": str(e)
            })
            
            # Audit the error
            latency_ms = int((time.time() - start_time) * 1000)
            self._audit_log(
                agent_name="classify_intent",
                caller="system",
                input_text=query,
                output_text=error_output,
                latency_ms=latency_ms,
                safety_result="error",
                error_message=str(e)
            )
            
            return error_output
    
    @adk_tool
    def check_compliance(self, user_id: str, intent: str) -> str:
        """
        TOOL 2: Validate user compliance based on plan tier and KYC status.
        
        Args:
            user_id: Unique user identifier
            intent: Classification intent (REPORT_REQUEST, PLAN_UPGRADE, etc.)
        
        Returns:
            JSON string with compliance status and allowed features
            
        Status codes:
            - ALLOWED: User can proceed
            - DENIED: User lacks required plan tier
            - REQUIRES_MFA: Additional verification needed
            
        Access Matrix:
            GOLD:   All features
            SILVER: Reports, balance checks (no upgrades)
            BRONZE: Balance checks only
        """
        logger.info(f"Checking compliance for user {user_id} intent {intent}")
        
        start_time = time.time()
        
        try:
            # Query user profile and plan tier
            query = f"""
            SELECT 
                p.plan_tier,
                p.kyc_verified,
                p.compliance_status,
                COALESCE(ARRAY_AGG(DISTINCT r.feature IGNORE NULLS), []) as allowed_features
            FROM `{self.project_id}.client_data.client_profiles` p
            LEFT JOIN `{self.project_id}.client_data.compliance_rules` r
                ON p.plan_tier = r.tier AND r.is_allowed = true
            WHERE p.user_id = @user_id
            GROUP BY p.plan_tier, p.kyc_verified, p.compliance_status
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                ]
            )
            
            results = list(self.bq_client.query(query, job_config=job_config).result())
            
            if not results:
                error_output = json.dumps({
                    "status": "error",
                    "compliance_status": "DENIED",
                    "reason": "User not found"
                })
                latency_ms = int((time.time() - start_time) * 1000)
                self._audit_log(
                    agent_name="check_compliance",
                    caller=user_id,
                    input_text=f"intent:{intent}",
                    output_text=error_output,
                    latency_ms=latency_ms,
                    safety_result="denied",
                    error_message="User not found"
                )
                return error_output
            
            user_data = results[0]
            plan_tier = user_data.plan_tier
            kyc_verified = user_data.kyc_verified
            allowed_features = user_data.allowed_features or []
            
            # Define access matrix
            access_matrix = {
                "gold": ["REPORT_REQUEST", "PLAN_UPGRADE", "PLAN_INFO", "BALANCE_CHECK"],
                "silver": ["REPORT_REQUEST", "PLAN_INFO", "BALANCE_CHECK"],
                "bronze": ["BALANCE_CHECK", "PLAN_INFO"]
            }
            
            allowed_intents = access_matrix.get(plan_tier.lower(), [])
            intent_allowed = intent in allowed_intents
            
            # Check KYC requirements for sensitive operations
            if intent == "REPORT_REQUEST" and not kyc_verified:
                output = json.dumps({
                    "status": "success",
                    "compliance_status": "REQUIRES_MFA",
                    "reason": "KYC verification required for wire reports",
                    "requires_verification": True,
                    "allowed_features": allowed_features
                })
                latency_ms = int((time.time() - start_time) * 1000)
                self._audit_log(
                    agent_name="check_compliance",
                    caller=user_id,
                    input_text=f"intent:{intent}",
                    output_text=output,
                    latency_ms=latency_ms,
                    safety_result="requires_mfa",
                    metadata={"plan_tier": plan_tier, "kyc_verified": kyc_verified}
                )
                return output
            
            if not intent_allowed:
                output = json.dumps({
                    "status": "success",
                    "compliance_status": "DENIED",
                    "reason": f"Your {plan_tier} plan doesn't include {intent}",
                    "requires_verification": False,
                    "allowed_features": allowed_features
                })
                latency_ms = int((time.time() - start_time) * 1000)
                self._audit_log(
                    agent_name="check_compliance",
                    caller=user_id,
                    input_text=f"intent:{intent}",
                    output_text=output,
                    latency_ms=latency_ms,
                    safety_result="denied",
                    metadata={"plan_tier": plan_tier, "intent": intent}
                )
                return output
            
            output = json.dumps({
                "status": "success",
                "compliance_status": "ALLOWED",
                "plan_tier": plan_tier,
                "allowed_features": allowed_features
            })
            latency_ms = int((time.time() - start_time) * 1000)
            self._audit_log(
                agent_name="check_compliance",
                caller=user_id,
                input_text=f"intent:{intent}",
                output_text=output,
                latency_ms=latency_ms,
                safety_result="allowed",
                metadata={"plan_tier": plan_tier, "intent": intent}
            )
            return output
        
        except GoogleCloudError as e:
            logger.error(f"BigQuery error: {str(e)}")
            error_output = json.dumps({
                "status": "error",
                "compliance_status": "DENIED",
                "error": f"System error: {str(e)}"
            })
            latency_ms = int((time.time() - start_time) * 1000)
            self._audit_log(
                agent_name="check_compliance",
                caller=user_id,
                input_text=f"intent:{intent}",
                output_text=error_output,
                latency_ms=latency_ms,
                safety_result="error",
                error_message=str(e)
            )
            return error_output
    
    @adk_tool
    def route_to_agent(self, intent: str, user_id: str) -> str:
        """
        TOOL 3: Route request to specialized sub-agent.
        
        Args:
            intent: Classified intent
            user_id: User identifier
        
        Returns:
            JSON string with target agent and parameters
            
        Routing Logic:
            REPORT_REQUEST   → report_agent (queries wire_reports table)
            PLAN_UPGRADE     → plan_agent (executes upgrades)
            PLAN_INFO        → plan_info_agent (returns plan details)
            BALANCE_CHECK    → balance_agent (queries transactions)
            UNKNOWN          → general_agent (fallback)
        """
        logger.info(f"Routing {intent} to sub-agent")
        
        start_time = time.time()
        
        agent_map = {
            "REPORT_REQUEST": "report_agent",
            "PLAN_UPGRADE": "plan_agent",
            "PLAN_INFO": "plan_info_agent",
            "BALANCE_CHECK": "balance_agent",
        }
        
        target_agent = agent_map.get(intent, "general_agent")
        
        output = json.dumps({
            "status": "success",
            "target_agent": target_agent,
            "intent": intent,
            "user_id": user_id,
            "parameters": {
                "user_id": user_id,
                "intent": intent,
                "agent_service_url": f"http://localhost:8000/{target_agent}"
            }
        })
        
        latency_ms = int((time.time() - start_time) * 1000)
        self._audit_log(
            agent_name="route_to_agent",
            caller=user_id,
            input_text=f"intent:{intent}",
            output_text=output,
            latency_ms=latency_ms,
            safety_result="allowed",
            metadata={"target_agent": target_agent, "intent": intent}
        )
        
        return output
    
    @adk_tool
    def execute_sub_agent(
        self,
        target_agent: str,
        user_id: str,
        query: str,
        intent: str
    ) -> str:
        """
        TOOL 4: Execute specialized sub-agent.
        
        Args:
            target_agent: Name of sub-agent (report_agent, plan_agent, etc.)
            user_id: User identifier
            query: Original user query
            intent: Classified intent
        
        Returns:
            JSON string with sub-agent results
            
        Sub-agents:
            - report_agent: Fetches wire reports from BigQuery
            - plan_agent: Handles plan upgrade logic
            - plan_info_agent: Returns available plans and pricing
            - balance_agent: Queries account balance and transactions
        """
        logger.info(f"Executing sub-agent: {target_agent}")
        
        start_time = time.time()
        
        try:
            if target_agent == "report_agent":
                data = self._execute_report_agent(user_id, query)
            elif target_agent == "plan_agent":
                data = self._execute_plan_agent(user_id, query)
            elif target_agent == "plan_info_agent":
                data = self._execute_plan_info_agent(query)
            elif target_agent == "balance_agent":
                data = self._execute_balance_agent(user_id)
            else:
                data = {"error": f"Unknown agent: {target_agent}"}
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            output = json.dumps({
                "status": "success",
                "agent": target_agent,
                "data": data,
                "execution_time_ms": elapsed_ms
            })
            
            # Audit the sub-agent execution
            self._audit_log(
                agent_name=target_agent,
                caller=user_id,
                input_text=query,
                output_text=output,
                latency_ms=elapsed_ms,
                safety_result="allowed",
                metadata={"intent": intent, "query": query}
            )
            
            return output
        except Exception as e:
            logger.error(f"Sub-agent execution error: {str(e)}")
            error_output = json.dumps({
                "status": "error",
                "agent": target_agent,
                "error": str(e)
            })
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            self._audit_log(
                agent_name=target_agent,
                caller=user_id,
                input_text=query,
                output_text=error_output,
                latency_ms=elapsed_ms,
                safety_result="error",
                error_message=str(e),
                metadata={"intent": intent}
            )
            
            return error_output
    
    @adk_tool
    def aggregate_results(
        self,
        sub_agent_data: str,
        intent: str,
        confidence: float
    ) -> str:
        """
        TOOL 5: Aggregate results from sub-agents.
        
        Args:
            sub_agent_data: JSON string with sub-agent results
            intent: Classified intent
            confidence: Classification confidence score
        
        Returns:
            JSON string with final aggregated response
            
        Determines next_actions based on intent:
            REPORT_REQUEST: display_report, export_pdf, schedule_weekly
            PLAN_UPGRADE: show_payment, confirm_upgrade
            PLAN_INFO: show_comparison, upgrade_prompt
            BALANCE_CHECK: display_balance, recent_transactions
        """
        logger.info("Aggregating sub-agent results")
        
        start_time = time.time()
        
        try:
            # Determine next actions
            next_actions_map = {
                "REPORT_REQUEST": ["display_report", "export_pdf", "schedule_weekly_report"],
                "PLAN_UPGRADE": ["show_payment_modal", "confirm_upgrade"],
                "PLAN_INFO": ["show_comparison_table", "show_upgrade_prompt"],
                "BALANCE_CHECK": ["display_balance", "show_recent_transactions"],
            }
            
            next_actions = next_actions_map.get(intent, [])
            
            # Parse sub-agent data
            try:
                agent_data = json.loads(sub_agent_data) if isinstance(sub_agent_data, str) else sub_agent_data
            except:
                agent_data = {"raw_data": sub_agent_data}
            
            output = json.dumps({
                "status": "success",
                "intent": intent,
                "confidence": confidence,
                "data": agent_data,
                "next_actions": next_actions,
                "reasoning": f"Processed {intent} with {confidence:.0%} confidence"
            })
            
            # Audit the aggregation
            latency_ms = int((time.time() - start_time) * 1000)
            self._audit_log(
                agent_name="aggregate_results",
                caller="system",
                input_text=f"intent:{intent},confidence:{confidence}",
                output_text=output,
                latency_ms=latency_ms,
                safety_result="allowed",
                metadata={"intent": intent, "confidence": confidence, "next_actions": next_actions}
            )
            
            return output
        except Exception as e:
            logger.error(f"Aggregation error: {str(e)}")
            error_output = json.dumps({
                "status": "error",
                "error": str(e)
            })
            
            latency_ms = int((time.time() - start_time) * 1000)
            self._audit_log(
                agent_name="aggregate_results",
                caller="system",
                input_text=f"intent:{intent}",
                output_text=error_output,
                latency_ms=latency_ms,
                safety_result="error",
                error_message=str(e)
            )
            
            return error_output
    
    # ==================== Sub-Agent Execution Methods ====================
    
    def _execute_report_agent(self, user_id: str, query: str) -> Dict[str, Any]:
        """Execute report agent: fetch wire reports"""
        bq_query = f"""
        SELECT 
            report_id,
            wire_amount,
            destination_bank,
            status,
            report_date
        FROM `{self.project_id}.client_data.wire_reports`
        WHERE user_id = @user_id
        ORDER BY report_date DESC
        LIMIT 10
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("user_id", "STRING", user_id)]
        )
        
        try:
            results = self.bq_client.query(bq_query, job_config=job_config).result()
            reports = [dict(row) for row in results]
            
            for report in reports:
                if hasattr(report.get('report_date'), 'isoformat'):
                    report['report_date'] = report['report_date'].isoformat()
            
            return {
                "reports": reports,
                "count": len(reports),
                "message": f"Retrieved {len(reports)} wire reports"
            }
        except Exception as e:
            logger.error(f"Report agent error: {str(e)}")
            return {"error": str(e), "reports": []}
    
    def _execute_plan_agent(self, user_id: str, query: str) -> Dict[str, Any]:
        """Execute plan agent: handle plan upgrades"""
        return {
            "action": "upgrade_requested",
            "user_id": user_id,
            "new_plan": "gold",
            "current_plan": "silver",
            "message": "Plan upgrade initiated. Awaiting payment confirmation.",
            "next_step": "payment_modal"
        }
    
    def _execute_plan_info_agent(self, query: str) -> Dict[str, Any]:
        """Execute plan info agent: return plan details"""
        bq_query = f"""
        SELECT 
            plan_name,
            tier,
            monthly_price,
            annual_price,
            features
        FROM `{self.project_id}.client_data.plan_offerings`
        ORDER BY monthly_price DESC
        """
        
        try:
            results = self.bq_client.query(bq_query).result()
            plans = [dict(row) for row in results]
            return {"plans": plans, "message": "All available plans retrieved"}
        except Exception as e:
            logger.error(f"Plan info agent error: {str(e)}")
            return {"error": str(e), "plans": []}
    
    def _execute_balance_agent(self, user_id: str) -> Dict[str, Any]:
        """Execute balance agent: fetch account balance and transactions"""
        bq_query = f"""
        SELECT 
            SUM(CASE WHEN transaction_type IN ('deposit', 'interest') THEN amount ELSE -amount END) as balance,
            COUNT(*) as transaction_count,
            MAX(created_at) as last_transaction
        FROM `{self.project_id}.client_data.user_transactions`
        WHERE user_id = @user_id AND created_at >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("user_id", "STRING", user_id)]
        )
        
        try:
            result = list(self.bq_client.query(bq_query, job_config=job_config).result())[0]
            return {
                "balance": float(result.balance or 0),
                "transaction_count": result.transaction_count or 0,
                "currency": "USD",
                "last_transaction": result.last_transaction.isoformat() if result.last_transaction else None
            }
        except Exception as e:
            logger.error(f"Balance agent error: {str(e)}")
            return {"error": str(e), "balance": 0}
    
    # ==================== Public API ====================
    
    async def process_query(self, user_id: str, query: str) -> Dict[str, Any]:
        """
        Main entry point: Process user query through agent pipeline.
        
        Args:
            user_id: Unique user identifier
            query: Natural language user query
        
        Returns:
            Dictionary with:
                - intent: Classified intent
                - confidence: Classification confidence
                - compliance_passed: Whether user has access
                - data: Sub-agent results
                - next_actions: Recommended UI actions
                - execution_log: Full execution trace
                - error: Error message if any
                
        Example:
            >>> agent = RootAgent("my-project")
            >>> result = await agent.process_query("user_001", "Show me wire reports")
            >>> print(result['intent'])  # "REPORT_REQUEST"
        """
        logger.info(f"Processing query for user {user_id}: {query}")
        
        execution_log = []
        overall_start_time = time.time()
        
        try:
            # Step 1: Classify Intent
            step_start = time.time()
            intent_result_str = self.classify_intent(query)
            intent_result = json.loads(intent_result_str)
            intent = intent_result.get("intent", "UNKNOWN")
            confidence = intent_result.get("confidence", 0.0)
            execution_log.append({
                "step": "classify_intent",
                "status": "success",
                "intent": intent,
                "confidence": confidence,
                "timestamp": datetime.utcnow().isoformat(),
                "duration_ms": (time.time() - step_start) * 1000
            })
            logger.info(f"Intent: {intent} (confidence: {confidence})")
            
            # Step 2: Check Compliance
            step_start = time.time()
            compliance_result_str = self.check_compliance(user_id, intent)
            compliance_result = json.loads(compliance_result_str)
            compliance_status = compliance_result.get("compliance_status", "DENIED")
            execution_log.append({
                "step": "check_compliance",
                "status": "success",
                "compliance_status": compliance_status,
                "timestamp": datetime.utcnow().isoformat(),
                "duration_ms": (time.time() - step_start) * 1000
            })
            logger.info(f"Compliance: {compliance_status}")
            
            # Return early if compliance fails
            if compliance_status != "ALLOWED":
                error_result = {
                    "status": "error",
                    "user_id": user_id,
                    "intent": intent,
                    "confidence": confidence,
                    "compliance_passed": False,
                    "error": compliance_result.get("reason", "Compliance check failed"),
                    "execution_log": execution_log
                }
                
                # Audit the compliance failure
                total_latency_ms = int((time.time() - overall_start_time) * 1000)
                await self._audit_log(
                    agent_name="process_query",
                    caller=user_id,
                    input_text=query,
                    output_text=json.dumps(error_result),
                    latency_ms=total_latency_ms,
                    safety_result="denied",
                    error_message=compliance_result.get("reason", "Compliance check failed"),
                    metadata={"intent": intent, "compliance_status": compliance_status}
                )
                
                return error_result
            
            # Step 3: Route to Agent
            step_start = time.time()
            routing_result_str = self.route_to_agent(intent, user_id)
            routing_result = json.loads(routing_result_str)
            target_agent = routing_result.get("target_agent", "general_agent")
            execution_log.append({
                "step": "route_to_agent",
                "status": "success",
                "target_agent": target_agent,
                "timestamp": datetime.utcnow().isoformat(),
                "duration_ms": (time.time() - step_start) * 1000
            })
            logger.info(f"Routed to: {target_agent}")
            
            # Step 4: Execute Sub-Agent
            step_start = time.time()
            exec_result_str = self.execute_sub_agent(target_agent, user_id, query, intent)
            exec_result = json.loads(exec_result_str)
            sub_agent_data = exec_result.get("data", {})
            execution_log.append({
                "step": "execute_agent",
                "status": exec_result.get("status"),
                "agent": target_agent,
                "timestamp": datetime.utcnow().isoformat(),
                "duration_ms": exec_result.get("execution_time_ms", (time.time() - step_start) * 1000)
            })
            logger.info(f"Sub-agent executed: {target_agent}")
            
            # Step 5: Aggregate Results
            step_start = time.time()
            agg_result_str = self.aggregate_results(
                json.dumps(sub_agent_data),
                intent,
                confidence
            )
            agg_result = json.loads(agg_result_str)
            execution_log.append({
                "step": "aggregate_results",
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "duration_ms": (time.time() - step_start) * 1000
            })
            
            # Record metrics
            self.metrics.record_execution({
                "user_id": user_id,
                "intent": intent,
                "confidence": confidence,
                "agent_name": target_agent,
                "execution_time_ms": sum(log.get("duration_ms", 0) for log in execution_log),
                "success": True,
                "error_message": None
            })
            
            # Return final response
            final_response = {
                "status": "success",
                "user_id": user_id,
                "intent": intent,
                "confidence": confidence,
                "compliance_passed": True,
                "agent_used": target_agent,
                "data": agg_result.get("data", {}),
                "next_actions": agg_result.get("next_actions", []),
                "reasoning": agg_result.get("reasoning", ""),
                "execution_log": execution_log,
                "error": None
            }
            
            # Audit the final successful response
            total_latency_ms = int((time.time() - overall_start_time) * 1000)
            self._audit_log(
                agent_name="process_query",
                caller=user_id,
                input_text=query,
                output_text=json.dumps(final_response),
                latency_ms=total_latency_ms,
                safety_result="allowed",
                metadata={
                    "intent": intent,
                    "confidence": confidence,
                    "agent_used": target_agent,
                    "compliance_passed": True
                }
            )
            
            return final_response
        
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            
            self.metrics.record_execution({
                "user_id": user_id,
                "intent": "UNKNOWN",
                "confidence": 0.0,
                "agent_name": "unknown",
                "execution_time_ms": 0,
                "success": False,
                "error_message": str(e)
            })
            
            error_result = {
                "status": "error",
                "user_id": user_id,
                "intent": None,
                "confidence": 0.0,
                "compliance_passed": False,
                "error": str(e),
                "execution_log": execution_log
            }
            
            # Audit the error response
            total_latency_ms = int((time.time() - overall_start_time) * 1000)
            self._audit_log(
                agent_name="process_query",
                caller=user_id,
                input_text=query,
                output_text=json.dumps(error_result),
                latency_ms=total_latency_ms,
                safety_result="error",
                error_message=str(e)
            )
            
            return error_result


# ==================== ADK App Deployment (Optional) ====================

def create_app():
    """
    Create ADK app for deployment to Vertex AI Agent Engine.
    
    Usage:
        from vertexai import agent_engines
        app = create_app()
        agent_engines.create(agent_engine=app, ...)
    """
    try:
        from vertexai import agent_engines
        
        agent = RootAgent(project_id="your-project-id")
        
        app = agent_engines.AdkApp(
            agent=agent,
            enable_tracing=True
        )
        
        return app
    except ImportError:
        logger.warning("agent_engines not available. Skipping ADK app creation.")
        return None
    
 # ==================== FACTORY FUNCTION ====================

def create_root_agent(project_id: str, region: str = "us-central1") -> RootAgent:
    """
    Factory function to create and initialize a Root Agent instance.
    
    This function handles initialization of all dependencies:
    - Google Cloud credentials
    - Vertex AI connection  
    - BigQuery client
    - Metrics collector
    
    Args:
        project_id (str): GCP project ID (e.g., "ccibt-hack25ww7-743")
        region (str): GCP region (default: "us-central1")
    
    Returns:
        RootAgent: Fully initialized Root Agent instance
    
    Raises:
        ValueError: If project_id is invalid
        Exception: If initialization fails
    
    Example:
        >>> agent = create_root_agent("my-project-id", "us-central1")
        >>> result = await agent.process_query("user_001", "What is my balance?")
    """
    if not project_id or not isinstance(project_id, str):
        raise ValueError("project_id must be a non-empty string")
    
    logger.info(f"Creating Root Agent for project: {project_id}")
    
    try:
        # Create RootAgent instance (which initializes all services)
        agent = RootAgent(project_id=project_id, region=region)
        logger.info(f"✅ Root Agent created successfully")
        return agent
    
    except Exception as e:
        logger.error(f"Failed to create Root Agent: {str(e)}", exc_info=True)
        raise


# ==================== EXPORTS ====================

__all__ = [
    'RootAgent',
    'Intent',
    'ComplianceStatus',
    'IntentClassificationResult',
    'ComplianceCheckResult',
    'RoutingDecision',
    'AgentExecutionResult',
    'MetricsCollector',
    'create_root_agent'
]
   