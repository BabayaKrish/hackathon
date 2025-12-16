
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
import concurrent.futures

from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError as GoogleCloudError
import vertexai
from vertexai.generative_models import GenerativeModel, Tool

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
            
            # Only include basic fields that are likely to exist in the audit log table
            basic_log_entry = {
                'created_at': datetime.utcnow().isoformat(),
                'log_id': f"log_{int(datetime.utcnow().timestamp())}",
                'user_id': log_entry.get('user_id', 'unknown'),
                'intent': log_entry.get('intent', 'unknown'),
                'agent_name': log_entry.get('agent_name', 'unknown'),
                'status': log_entry.get('status', False),
                'error_message': log_entry.get('error_message', ''),
                'execution_time_ms': log_entry.get('execution_time_ms', 'test')
            }
            
            errors = self.bq_client.insert_rows_json(
                table_id, [basic_log_entry], skip_invalid_rows=True
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
        #self.metrics = MetricsCollector(project_id)
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=region)
        self.model = GenerativeModel("gemini-2.0-flash-exp")
        
        logger.info(f"RootAgent initialized for project {project_id}")
    
    # ==================== Tool Definitions ====================
    
    @adk_tool
    def classify_intent(self, query: str) -> str:
        """
        TOOL 1: Classify user intent using Vertex AI LLM.
        
        Args:
            query (str): Natural language user query
            
        Returns:
            str: JSON string with intent, confidence, and reasoning
        """
        
        logger.info(f"Classifying intent for query: {query}")
        
        # Define valid intents to match our enum
        valid_intents = ["REPORT_REQUEST", "PLAN_UPGRADE", "PLAN_INFO", "BALANCE_CHECK"]
        
        prompt = f"""You are a financial services intent classifier. Analyze the following customer query and determine their intent.

    Query: {query}

    Available intents:
    1. REPORT_REQUEST - User wants to view or download a report (wire reports, balance reports, transaction history)
    2. PLAN_UPGRADE - User wants to change or upgrade their plan to a higher tier
    3. PLAN_INFO - User is asking for information about plans, features, pricing, or comparisons
    4. BALANCE_CHECK - User wants to know their current account balance or recent transactions

    Respond ONLY in valid JSON format with no additional text:
    {{"intent": "REPORT_REQUEST|PLAN_UPGRADE|PLAN_INFO|BALANCE_CHECK", "confidence": 0.0-1.0, "reasoning": "Brief explanation"}}

    Rules:
    - Confidence must be between 0 and 1
    - If unclear or ambiguous, set confidence to 0.6-0.8
    - Return valid JSON only, no markdown or explanations"""
        
        max_retries = 3
        base_delay = 1.0  # seconds
        
        # Add a small delay to help with rate limiting
        time.sleep(0.5)
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Sending prompt to Vertex AI (attempt {attempt + 1}/{max_retries}, length: {len(prompt)} chars)")
                
                # Add timeout to prevent hanging
                def call_vertex_ai():
                    return self.model.generate_content(prompt)
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(call_vertex_ai)
                    try:
                        response = future.result(timeout=30.0)  # 30 second timeout
                    except concurrent.futures.TimeoutError:
                        logger.warning(f"Vertex AI call timed out on attempt {attempt + 1}")
                        if attempt < max_retries - 1:
                            time.sleep(base_delay * (2 ** attempt))  # Exponential backoff
                            continue
                        else:
                            return json.dumps({
                                "status": "error",
                                "intent": "UNKNOWN",
                                "confidence": 0.0,
                                "error": "Vertex AI timeout after all retries"
                            })
                
                # CHECK IF EMPTY FIRST
                if not response.text or not response.text.strip():
                    logger.warning(f"Empty response from LLM for query: {query}")
                    if attempt < max_retries - 1:
                        time.sleep(base_delay * (2 ** attempt))
                        continue
                    return json.dumps({
                        "status": "error",
                        "intent": "UNKNOWN",
                        "confidence": 0.0,
                        "error": "Empty LLM response after all retries"
                    })
                
                logger.info(f"Response text length: {len(response.text)}")
                logger.info(f"Response text: {response.text[:500]}")
                
                # ASSIGN TO response_text
                response_text = response.text.strip()
                
                # STRIP MARKDOWN IF PRESENT
                if response_text.startswith("```"):
                    logger.info("Response has markdown fences, removing them")
                    response_text = response_text.lstrip("`").lstrip("json").lstrip()
                    response_text = response_text.rstrip("`").strip()
                    logger.info(f"After fence removal: {response_text[:200]}")
                
                # PARSE CLEANED TEXT
                result = json.loads(response_text)
                intent_value = result.get("intent", "UNKNOWN")
                confidence = float(result.get("confidence", 0.0))
                reasoning = result.get("reasoning", "")
                
                # Validate intent is in our allowed list
                if intent_value not in valid_intents:
                    logger.warning(f"Invalid intent returned: {intent_value}, expected one of {valid_intents}")
                    if attempt < max_retries - 1:
                        time.sleep(base_delay * (2 ** attempt))
                        continue
                    intent_value = "UNKNOWN"
                    confidence = 0.0
                
                logger.info(f"Intent: {intent_value} (confidence: {confidence})")
                
                return json.dumps({
                    "status": "success",
                    "intent": intent_value,
                    "confidence": confidence,
                    "reasoning": reasoning
                })
            
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error on attempt {attempt + 1}: {str(e)}")
                logger.error(f"Raw response text: '{response_text}'")
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                return json.dumps({
                    "status": "error",
                    "intent": "UNKNOWN",
                    "confidence": 0.0,
                    "error": f"Invalid JSON response after all retries: {str(e)}"
                })
            
            except Exception as e:
                logger.error(f"Intent classification error on attempt {attempt + 1}: {str(e)}", exc_info=True)
                
                # Check if it's a quota exceeded error
                if "RESOURCE_EXHAUSTED" in str(e) or "Quota exceeded" in str(e):
                    logger.warning(f"Vertex AI quota exceeded on attempt {attempt + 1}, using fallback logic")
                    # For quota exceeded, try to classify based on keywords as fallback
                    query_lower = query.lower()
                    if any(word in query_lower for word in ['balance', 'how much', 'current']):
                        fallback_intent = "BALANCE_CHECK"
                        fallback_confidence = 0.8
                    elif any(word in query_lower for word in ['report', 'wire', 'transfer']):
                        fallback_intent = "REPORT_REQUEST"
                        fallback_confidence = 0.8
                    elif any(word in query_lower for word in ['upgrade', 'change plan', 'gold', 'silver']):
                        fallback_intent = "PLAN_UPGRADE"
                        fallback_confidence = 0.8
                    elif any(word in query_lower for word in ['plan', 'pricing', 'features', 'what is']):
                        fallback_intent = "PLAN_INFO"
                        fallback_confidence = 0.8
                    else:
                        fallback_intent = "UNKNOWN"
                        fallback_confidence = 0.5
                    
                    return json.dumps({
                        "status": "success",
                        "intent": fallback_intent,
                        "confidence": fallback_confidence,
                        "reasoning": f"Fallback classification due to API quota limit: {str(e)}",
                        "fallback": True
                    })
                
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                return json.dumps({
                    "status": "error",
                    "intent": "UNKNOWN",
                    "confidence": 0.0,
                    "error": str(e)
                })
        
        # This should never be reached, but just in case
        return json.dumps({
            "status": "error",
            "intent": "UNKNOWN",
            "confidence": 0.0,
            "error": "Maximum retries exceeded"
        })
    
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
        
        try:
            # Query user profile and plan tier
            query = f"""
            SELECT 
                p.plan_tier,
                p.kyc_verified,
                p.compliance_status
            FROM `{self.project_id}.client_data.client_profiles` p
            WHERE p.user_id = @user_id
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                ]
            )
            
            results = list(self.bq_client.query(query, job_config=job_config).result())
            
            logger.info(f"User profile query returned {len(results)} rows")
            if not results:
                return json.dumps({
                    "status": "error",
                    "compliance_status": "DENIED",
                    "reason": "User not found"
                })
            
            user_data = results[0]
            plan_tier = user_data.plan_tier
            kyc_verified = user_data.kyc_verified
            
            # Convert kyc_verified to boolean if it's a string
            if isinstance(kyc_verified, str):
                kyc_verified = kyc_verified.lower() in ('true', '1', 'yes')
            
            # Define access matrix (simplified - no need for complex JOIN)
            access_matrix = {
                "gold": ["REPORT_REQUEST", "PLAN_UPGRADE", "PLAN_INFO", "BALANCE_CHECK"],
                "silver": ["REPORT_REQUEST", "PLAN_INFO", "BALANCE_CHECK"],
                "bronze": ["BALANCE_CHECK", "PLAN_INFO"]
            }
            
            allowed_intents = access_matrix.get(plan_tier.lower() if plan_tier else "", [])
            intent_allowed = intent in allowed_intents
            
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
                return json.dumps({
                    "status": "success",
                    "compliance_status": "REQUIRES_MFA",
                    "reason": "KYC verification required for wire reports",
                    "requires_verification": True,
                    "allowed_features": allowed_intents
                })
            
            if not intent_allowed:
                return json.dumps({
                    "status": "success",
                    "compliance_status": "DENIED",
                    "reason": f"Your {plan_tier} plan doesn't include {intent}",
                    "requires_verification": False,
                    "allowed_features": allowed_intents
                })
            
            return json.dumps({
                "status": "success",
                "compliance_status": "ALLOWED",
                "plan_tier": plan_tier,
                "allowed_features": allowed_intents
            })
        
        except GoogleCloudError as e:
            logger.error(f"BigQuery error: {str(e)}")
            return json.dumps({
                "status": "error",
                "compliance_status": "DENIED",
                "error": f"System error: {str(e)}"
            })
    
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
        
        agent_map = {
            "REPORT_REQUEST": "report_agent",
            "PLAN_UPGRADE": "plan_agent",
            "PLAN_INFO": "plan_info_agent",
            "BALANCE_CHECK": "balance_agent",
        }
        
        target_agent = agent_map.get(intent, "general_agent")
        
        return json.dumps({
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
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            return json.dumps({
                "status": "success",
                "agent": target_agent,
                "data": data,
                "execution_time_ms": elapsed_ms
            })
        except Exception as e:
            logger.error(f"Sub-agent execution error: {str(e)}")
            return json.dumps({
                "status": "error",
                "agent": target_agent,
                "error": str(e)
            })
    
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
            
            return json.dumps({
                "status": "success",
                "intent": intent,
                "confidence": confidence,
                "data": agent_data,
                "next_actions": next_actions,
                "reasoning": f"Processed {intent} with {confidence:.0%} confidence"
            })
        except Exception as e:
            logger.error(f"Aggregation error: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": str(e)
            })
    
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
            reports = []
            for row in results:
                report_dict = dict(row)
                # Convert Decimal objects to floats for JSON serialization
                if 'wire_amount' in report_dict and hasattr(report_dict['wire_amount'], '__float__'):
                    report_dict['wire_amount'] = float(report_dict['wire_amount'])
                # Convert timestamp to ISO format
                if hasattr(report_dict.get('report_date'), 'isoformat'):
                    report_dict['report_date'] = report_dict['report_date'].isoformat()
                reports.append(report_dict)
            
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
            plans = []
            for row in results:
                plan_dict = dict(row)
                # Convert Decimal objects to floats for JSON serialization
                if 'monthly_price' in plan_dict and hasattr(plan_dict['monthly_price'], '__float__'):
                    plan_dict['monthly_price'] = float(plan_dict['monthly_price'])
                if 'annual_price' in plan_dict and hasattr(plan_dict['annual_price'], '__float__'):
                    plan_dict['annual_price'] = float(plan_dict['annual_price'])
                plans.append(plan_dict)
            
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
        WHERE user_id = @user_id AND created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
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
                return {
                    "status": "error",
                    "user_id": user_id,
                    "intent": intent,
                    "confidence": confidence,
                    "compliance_passed": False,
                    "error": compliance_result.get("reason", "Compliance check failed"),
                    "execution_log": execution_log
                }
            
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
            #self.metrics.record_execution({
            #    "user_id": user_id,
            #    "intent": intent,
            #    "confidence": confidence,
            #    "agent_name": target_agent,
            #    "execution_time_ms": sum(log.get("duration_ms", 0) for log in execution_log),
            #    "success": True,
            #    "error_message": None
            #})
            
            # Return final response
            return {
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
        
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)

            #self.metrics.record_execution({
            #    "user_id": user_id,
            #    "intent": "UNKNOWN",
            #    "confidence": 0.0,
            #    "agent_name": "unknown",
            #    "execution_time_ms": 0,
            #    "success": False,
            #    "error_message": str(e)
            #})
            
            return {
                "status": "error",
                "user_id": user_id,
                "intent": None,
                "confidence": 0.0,
                "compliance_passed": False,
                "error": str(e),
                "execution_log": execution_log
            }


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
    # 'MetricsCollector',
    'create_root_agent'
]
   