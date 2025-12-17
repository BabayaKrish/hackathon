
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
import asyncio
import sys
import os
from agents.plan_info_agent import PlanInfoAgent

from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError as GoogleCloudError
from vertexai.generative_models import GenerativeModel, Tool
from .report_agent import ReportAgent
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
            table_id = f"{self.project_id}.client_data.agent_audit_logs"
            
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
    
    def __init__(self, project_id: str, region: str = "us-central1", plan_info_agent=None):
        """Initialize Root Agent with Google ADK"""
        self.project_id = project_id
        self.region = region
        self.bq_client = bigquery.Client(project=project_id)
        vertexai.init(project=project_id, location=region)
        self.model = GenerativeModel("gemini-2.0-flash-exp")
        self.plan_info_agent = plan_info_agent  # <-- add this line
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
            query (str): Natural language user query
            
        Returns:
            str: JSON string with intent, confidence, and reasoning
        """
        
        logger.info(f"Classifying intent for query: {query}")

        start_time = time.time()

        # Define valid intents to match our enum (including OUT_OF_SCOPE)
        REPORT_TYPES = ["balance_report", "wire_details", "intraday_balance", "running_ledger"]
        valid_intents = REPORT_TYPES + ["PLAN_UPGRADE", "PLAN_INFO", "BALANCE_CHECK", "OUT_OF_SCOPE"]
        # Use a regular triple-quoted string and .format(query=query) to safely inject the query.
        improved_prompt =  """You are a STRICT financial services intent classifier for a banking/financial services platform.

YOUR SOLE JOB: Classify customer queries into ONE of these 7 financial intents ONLY.
DO NOT answer questions. DO NOT provide information. DO NOT engage in conversation.
DO NOT hallucinate or make assumptions.

**CRITICAL RULE**: If a query does NOT match ANY of the 7 intents below, you MUST respond with:
{{"intent": "OUT_OF_SCOPE", "confidence": 1.0, "reasoning": "Query is not related to financial services or account management"}}

**AVAILABLE INTENTS** (ONLY these 7 + OUT_OF_SCOPE):

1. BALANCE_CHECK - User wants to know their CURRENT ACCOUNT BALANCE or RECENT TRANSACTIONS
    Examples: "What is my balance?", "Show me recent transactions", "How much do I have?"
   
2. BALANCE_REPORT - User wants to VIEW or DOWNLOAD a complete BALANCE REPORT/STATEMENT
    Examples: "Download my statement", "Show me a balance report", "I need a monthly report"
   
3. INTRADAY_BALANCE - User wants to view REAL-TIME or CURRENT-MOMENT balance updates
    Examples: "What's my balance right now?", "Real-time balance", "Current balance update"
   
4. RUNNING_LEDGER - User wants to view the COMPLETE TRANSACTION LOG or DETAILED LEDGER
    Examples: "Show me all transactions", "Display the full ledger", "Transaction history"
   
5. WIRE_DETAILS - User wants to VIEW or DOWNLOAD WIRE TRANSFER details/instructions
    Examples: "Show wire transfer details", "How do I send a wire?", "Wire instructions"
   
6. PLAN_INFO - User is asking SPECIFICALLY about PLANS, FEATURES, PRICING, or PLAN COMPARISONS
    ONLY for: plan features, pricing details, plan comparison, upgrade options
    DO NOT classify general questions here
    Examples: "What features does the gold plan have?", "How much is the premium plan?", "Compare plans"
   
7. PLAN_UPGRADE - User explicitly wants to CHANGE their PLAN TIER (UPGRADE or DOWNGRADE)
   UPGRADE: Switching to a HIGHER-COST or MORE-FEATURED plan
   DOWNGRADE: Switching to a LOWER-COST or LESS-FEATURED plan
   Examples: "Upgrade my plan", "Switch to premium", "I want a higher tier", "Downgrade to silver", "Switch to basic plan", "I want a cheaper plan"

8. OUT_OF_SCOPE - Query is NOT related to financial services (STRICT DEFAULT for anything else)
    Examples: "What is your name?", "What is AI?", "Hello", "Who are you?", "Tell me a joke"

**REJECTION EXAMPLES** (Do NOT classify these as PLAN_INFO):
- "What is your name?" → OUT_OF_SCOPE (greeting/personal question)
- "What is AI?" → OUT_OF_SCOPE (general knowledge question)
- "Who are you?" → OUT_OF_SCOPE (identity question)
- "Can you help me?" → OUT_OF_SCOPE (vague/non-financial)
- "Hello" → OUT_OF_SCOPE (greeting)
- "Tell me about pricing" (without mention of plans) → OUT_OF_SCOPE
- "How does banking work?" → OUT_OF_SCOPE (educational, not account-specific)

**CONFIDENCE SCORING RULES**:
- 0.95-1.0: Clear, unambiguous financial query matching one intent
- 0.85-0.94: Strong match with minor ambiguity
- 0.75-0.84: Reasonable match but some ambiguity
- 0.65-0.74: Weak match, consider OUT_OF_SCOPE instead
- Below 0.65: MUST use OUT_OF_SCOPE instead

**OUTPUT FORMAT** (STRICT):
Return ONLY valid JSON with NO markdown fences, NO explanations, NO additional text:
{{"intent": "BALANCE_CHECK|BALANCE_REPORT|INTRADAY_BALANCE|RUNNING_LEDGER|WIRE_DETAILS|PLAN_INFO|PLAN_UPGRADE|OUT_OF_SCOPE", "confidence": 0.0-1.0, "reasoning": "Brief 3-sentence explanation"}}

**VALIDATION RULES**:
1. Confidence must be a number between 0.0 and 1.0
2. Intent must be ONE of the 8 options above
3. Reasoning must be 1-2 sentences maximum
4. Return ONLY JSON - no markdown, no code fences, no text
5. If unclear, default to OUT_OF_SCOPE with confidence 0.85-1.0

**WHEN TO USE OUT_OF_SCOPE** (Use liberally):
✓ Greeting/personal questions
✓ General knowledge questions
✓ Requests for information not in the 7 intents
✓ Vague or ambiguous queries
✓ Philosophical or conversational questions
✓ Anything that feels like "general chitchat"

Do NOT try to be helpful by answering beyond your scope. 
Do NOT assume intent from ambiguous queries.
Do NOT extend PLAN_INFO to cover general questions.
Respond STRICTLY per these rules.
CLASSIFY THE FOLLOWING QUERY:
Query: {query}  
Respond ONLY in valid JSON format and reasoning about the selection of intent."""
        
        prompt = improved_prompt.format(query=query)    
        
        max_retries = 3
        base_delay = 1.0  # seconds
        
        # Add a small delay to help with rate limiting
        time.sleep(0.5)
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Sending prompt to Vertex AI (attempt {attempt + 1}/{max_retries}, length: {len(prompt)} chars)")
                logger.info(f"Prompt content: {prompt}")
                # Add timeout to prevent hanging
                def call_vertex_ai():
                    return self.model.generate_content(prompt)
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(call_vertex_ai)
                    try:
                        response = future.result(timeout=30.0) 
                        logger.info(f"Received response from Vertex AI as response: {response}")
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
                intent_value = intent_value.lower()  # Normalize to lowercase
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
    async def check_compliance(self, user_id: str, intent: str) -> str:
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
            
            logger.info(f"User {user_id} has plan tier {plan_tier}, KYC verified: {kyc_verified}")
            # Define access matrix (simplified - no need for complex JOIN)
            REPORT_TYPES = ["balance_report", "wire_details", "intraday_balance", "running_ledger"]
            access_matrix = {
                "gold": REPORT_TYPES + ["PLAN_UPGRADE", "PLAN_INFO", "BALANCE_CHECK"],
                "silver": ["balance_report", "intraday_balance", "running_ledger", "PLAN_INFO", "BALANCE_CHECK"],
                "bronze": ["balance_report", "intraday_balance", "BALANCE_CHECK", "PLAN_INFO"]
            }
            
            allowed_intents = access_matrix.get(plan_tier.lower() if plan_tier else "", [])
            intent_allowed = intent in allowed_intents
            
            # Check KYC requirements for all report types
            if intent in REPORT_TYPES and not kyc_verified:
                return json.dumps({
                    "status": "success",
                    "compliance_status": "REQUIRES_MFA",
                    "reason": f"KYC verification required for {intent}",
                    "requires_verification": True,
                    "allowed_features": allowed_intents
                })

            if not intent_allowed:
                return json.dumps({
                    "status": "success",
                    "compliance_status": "DENIED",
                    "reason": f"Your {plan_tier} plan doesn't include {intent}. Allowed report types: {', '.join(REPORT_TYPES)}",
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
    async def route_to_agent(self, intent: str, user_id: str) -> str:
        """
        TOOL 3: Route request to specialized sub-agent.
        
        Args:
            intent: Classified intent
            user_id: User identifier
        
        Returns:
            JSON string with target agent and parameters
            
        Routing Logic:
            REPORT_REQUEST   → report_agent (queries wire_reports table)
            PLAN_INFO        → plan_info_agent (returns plan details)
            BALANCE_CHECK    → balance_agent (queries transactions)
            UNKNOWN          → general_agent (fallback)
        """
        logger.info(f"Routing {intent} to sub-agent")
        
        start_time = time.time()

        agent_map = {
            "REPORT_REQUEST": "report_agent",
            "PLAN_INFO": "plan_info_agent",
            "PLAN_UPGRADE": "plan_info_agent",
            "BALANCE_CHECK": "balance_agent",
        }

        target_agent = agent_map.get(intent, "general_agent")
        REPORT_TYPES = ["balance_report", "wire_details", "intraday_balance", "running_ledger"]

        if intent in REPORT_TYPES:
            target_agent = "report_agent"
        else:
            agent_map = {
                "PLAN_INFO": "plan_info_agent",
                "PLAN_UPGRADE": "plan_info_agent",
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
        await self._audit_log(
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
    async def execute_sub_agent(
        self,
        target_agent: str,
        user_id: str,
        query: str,
        intent: str
    ) -> str:
        """
        TOOL 4: Execute specialized sub-agent.
        
        Args:
            target_agent: Name of sub-agent (report_agent, etc.)
            user_id: User identifier
            query: Original user query
            intent: Classified intent
        
        Returns:
            JSON string with sub-agent results
            
        Sub-agents:
            - report_agent: Fetches wire reports from BigQuery
            - plan_info_agent: Returns available plans and pricing
            - balance_agent: Queries account balance and transactions
        """
        logger.info(f"Executing sub-agent: {target_agent}")
        REPORT_TYPES = ["balance_report", "wire_details", "intraday_balance", "running_ledger"]

        start_time = time.time()
        
        try:
            if target_agent == "report_agent":
                agent = ReportAgent('ccibt-hack25ww7-743', 'us-central1')  # ✅ Instantiate agent
                result = await agent.process_report_request(user_id, intent) # ✅ Delegate
                data = result
            elif target_agent == "plan_info_agent":
                agent = PlanInfoAgent('ccibt-hack25ww7-743', 'us-central1')  # Instantiate PlanInfoAgent
                data = agent.process_plan_upgrade_request(query, user_id)  # Call the method to get plan features
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
            await self._audit_log(
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

            #elapsed_ms = int((time.time() - start_time) * 1000)
            #self._audit_log(
            #    agent_name=target_agent,
            #    caller=user_id,
            #    input_text=query,
            #    output_text=error_output,
            #   latency_ms=elapsed_ms,
            #    safety_result="error",
            #    error_message=str(e),
            #    metadata={"intent": intent}
            #)

            return error_output

    @adk_tool
    async def aggregate_results(
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
            
                # ✅ FORMAT DATA BASED ON INTENT
            formatted_data = self._format_agent_response(intent, agent_data)

            output = json.dumps({
                "status": "success",
                "intent": intent,
                "confidence": confidence,
                "data": agent_data,
                "formatted_response": formatted_data,  # ← Add human-readable version
                "next_actions": next_actions,
                "reasoning": self._get_intent_reasoning(intent, agent_data)
            })

            # Audit the aggregation
            latency_ms = int((time.time() - start_time) * 1000)
            await self._audit_log(
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
            await self._audit_log(
                agent_name="aggregate_results",
                caller="system",
                input_text=f"intent:{intent}",
                output_text=error_output,
                latency_ms=latency_ms,
                safety_result="error",
                error_message=str(e)
            )

            return error_output


    def _format_agent_response(self, intent: str, agent_data: Dict) -> str:
        """Format agent response for user display"""

        if intent == "PLAN_INFO":
            return self._format_plan_info(agent_data)
        elif intent == "BALANCECHECK":
            return self._format_balance_info(agent_data)
        elif intent == "REPORTREQUEST":
            return self._format_report_info(agent_data)
        elif intent == "PLANUPGRADE":
            return self._format_upgrade_info(agent_data)
        else:
            return str(agent_data)

    def _format_plan_info(self, data: Dict) -> str:
        """Format plan information for display"""

        if not data or "plans" not in data:
            return "No plan information available."

        plans = data.get("plans", [])
        if not plans:
            return "No plans found."

        formatted = "📋 **Available Plans**\n\n"

        for plan in plans:
            plan_name = plan.get("plan_name", "Unknown Plan")
            tier = plan.get("tier", "")
            monthly = plan.get("monthly_price")
            annual = plan.get("annual_price")
            features = plan.get("features", "")

            # Fix: handle None values for monthly/annual
            try:
                monthly_str = f"${float(monthly):.2f}/month" if monthly is not None else "N/A"
            except Exception:
                monthly_str = "N/A"
            try:
                annual_str = f"${float(annual):.2f}/year" if annual is not None else "N/A"
            except Exception:
                annual_str = "N/A"

            formatted += f"### {plan_name} ({tier.upper()})\n"
            formatted += f"**Pricing:**\n"
            formatted += f"- Monthly: {monthly_str}\n"
            formatted += f"- Annual: {annual_str}\n"

            if features:
                formatted += f"**Features:**\n"
                feature_list = [f.strip() for f in features.split(",")]
                for feature in feature_list:
                    formatted += f"- {feature}\n"

            formatted += "\n"

        return formatted


    def _format_balance_info(self, data: Dict) -> str:
        """Format balance information for display"""

        balance = data.get("balance", 0)
        currency = data.get("currency", "USD")
        transaction_count = data.get("transaction_count", 0)
        last_transaction = data.get("last_transaction", "Never")

        return f"""💰 **Account Balance**

**Current Balance:** {currency} {balance:,.2f}
**Recent Transactions:** {transaction_count} in last 90 days
**Last Transaction:** {last_transaction}
"""


    def _format_report_info(self, data: Dict) -> str:
        """Format report information for display"""

        reports = data.get("reports", [])
        if not reports:
            return "No reports available."

        formatted = f"📊 **Available Reports** ({len(reports)} found)\n\n"

        for report in reports:
            report_id = report.get("report_id", "N/A")
            amount = report.get("wire_amount", 0)
            destination = report.get("destination_bank", "N/A")
            status = report.get("status", "pending")
            date = report.get("report_date", "N/A")

            formatted += f"**Report #{report_id}** - {status.upper()}\n"
            formatted += f"- Amount: ${amount:,.2f}\n"
            formatted += f"- Destination: {destination}\n"
            formatted += f"- Date: {date}\n\n"

        return formatted


    def _format_upgrade_info(self, data: Dict) -> str:
        """Format upgrade information for display"""

        action = data.get("action", "upgrade_requested")
        current_plan = data.get("current_plan", "Unknown")
        new_plan = data.get("new_plan", "Unknown")
        message = data.get("message", "Processing your request...")

        return f"""🚀 **Plan Upgrade**

    **Current Plan:** {current_plan.upper()}
    **Upgrade To:** {new_plan.upper()}
    **Status:** {message}
    **Next Step:** {data.get("next_step", "Contact support")}
    """


    def _get_intent_reasoning(self, intent: str, data: Dict) -> str:
        """Generate reasoning based on intent and data"""

        reasoning_map = {
            "PLAN_INFO": f"Retrieved {len(data.get('plans', []))} available plans",
            "BALANCECHECK": f"Current account balance is {data.get('currency', 'USD')} {data.get('balance', 0):,.2f}",
            "REPORTREQUEST": f"Found {len(data.get('reports', []))} available reports",
            "PLANUPGRADE": f"Upgrade from {data.get('current_plan', 'current')} to {data.get('new_plan', 'selected')} plan",
        }

        return reasoning_map.get(intent, "Request processed successfully")

    def _format_agent_response(self, intent: str, agent_data: Dict) -> str:
        """Format agent response for user display"""

        if intent == "PLAN_INFO":
            return self._format_plan_info(agent_data)
        elif intent == "BALANCECHECK":
            return self._format_balance_info(agent_data)
        elif intent == "REPORTREQUEST":
            return self._format_report_info(agent_data)
        elif intent == "PLANUPGRADE":
            return self._format_upgrade_info(agent_data)
        else:
            return str(agent_data)

    def _format_plan_info(self, data: Dict) -> str:
        """Format plan information for display"""

        if not data or "plans" not in data:
            return "No plan information available."

        plans = data.get("plans", [])
        if not plans:
            return "No plans found."

        formatted = "📋 **Available Plans**\n\n"

        for plan in plans:
            plan_name = plan.get("plan_name", "Unknown Plan")
            tier = plan.get("tier", "")
            monthly = plan.get("monthly_price")
            annual = plan.get("annual_price")
            features = plan.get("features", "")

            # Fix: handle None values for monthly/annual
            try:
                monthly_str = f"${float(monthly):.2f}/month" if monthly is not None else "N/A"
            except Exception:
                monthly_str = "N/A"
            try:
                annual_str = f"${float(annual):.2f}/year" if annual is not None else "N/A"
            except Exception:
                annual_str = "N/A"

            formatted += f"### {plan_name} ({tier.upper()})\n"
            formatted += f"**Pricing:**\n"
            formatted += f"- Monthly: {monthly_str}\n"
            formatted += f"- Annual: {annual_str}\n"

            if features:
                formatted += f"**Features:**\n"
                feature_list = [f.strip() for f in features.split(",")]
                for feature in feature_list:
                    formatted += f"- {feature}\n"

            formatted += "\n"

        return formatted


    def _format_balance_info(self, data: Dict) -> str:
        """Format balance information for display"""

        balance = data.get("balance", 0)
        currency = data.get("currency", "USD")
        transaction_count = data.get("transaction_count", 0)
        last_transaction = data.get("last_transaction", "Never")

        return f"""💰 **Account Balance**

**Current Balance:** {currency} {balance:,.2f}
**Recent Transactions:** {transaction_count} in last 90 days
**Last Transaction:** {last_transaction}
"""


    def _format_report_info(self, data: Dict) -> str:
        """Format report information for display"""

        reports = data.get("reports", [])
        if not reports:
            return "No reports available."

        formatted = f"📊 **Available Reports** ({len(reports)} found)\n\n"

        for report in reports:
            report_id = report.get("report_id", "N/A")
            amount = report.get("wire_amount", 0)
            destination = report.get("destination_bank", "N/A")
            status = report.get("status", "pending")
            date = report.get("report_date", "N/A")

            formatted += f"**Report #{report_id}** - {status.upper()}\n"
            formatted += f"- Amount: ${amount:,.2f}\n"
            formatted += f"- Destination: {destination}\n"
            formatted += f"- Date: {date}\n\n"

        return formatted


    def _format_upgrade_info(self, data: Dict) -> str:
        """Format upgrade information for display"""

        action = data.get("action", "upgrade_requested")
        current_plan = data.get("current_plan", "Unknown")
        new_plan = data.get("new_plan", "Unknown")
        message = data.get("message", "Processing your request...")

        return f"""🚀 **Plan Upgrade**

    **Current Plan:** {current_plan.upper()}
    **Upgrade To:** {new_plan.upper()}
    **Status:** {message}
    **Next Step:** {data.get("next_step", "Contact support")}
    """


    def _get_intent_reasoning(self, intent: str, data: Dict) -> str:
        """Generate reasoning based on intent and data"""

        reasoning_map = {
            "PLAN_INFO": f"Retrieved {len(data.get('plans', []))} available plans",
            "BALANCECHECK": f"Current account balance is {data.get('currency', 'USD')} {data.get('balance', 0):,.2f}",
            "REPORTREQUEST": f"Found {len(data.get('reports', []))} available reports",
            "PLANUPGRADE": f"Upgrade from {data.get('current_plan', 'current')} to {data.get('new_plan', 'selected')} plan",
        }

        return reasoning_map.get(intent, "Request processed successfully")
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
    async def _execute_plan_info_agent(self, query: str) -> Dict[str, Any]:
        if self.plan_info_agent is None:
            logger.error("PlanInfoAgent not initialized")
            return {"error": "PlanInfoAgent not available"}
        # Await the coroutine directly
        return await self.plan_info_agent.process_info_request(query)

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

            # If the query is out of scope, return a friendly message immediately
            if intent == "OUT_OF_SCOPE":
                reasoning = intent_result.get("reasoning", "This question is outside the scope of this assistant.")
                out_of_scope_response = {
                    "status": "success",
                    "user_id": user_id,
                    "intent": intent,
                    "confidence": confidence,
                    "compliance_passed": False,
                    "agent_used": None,
                    "data": {},
                    "next_actions": [],
                    "reasoning": reasoning,
                    "execution_log": execution_log,
                    "error": None
                }

                total_latency_ms = int((time.time() - overall_start_time) * 1000)
                await self._audit_log(
                    agent_name="process_query",
                    caller=user_id,
                    input_text=query,
                    output_text=json.dumps(out_of_scope_response),
                    latency_ms=total_latency_ms,
                    safety_result="allowed",
                    metadata={
                        "intent": intent,
                        "confidence": confidence,
                        "agent_used": None,
                        "compliance_passed": False
                    }
                )

                return out_of_scope_response
            
            # Step 2: Check Compliance
            step_start = time.time()
            compliance_result_str = await self.check_compliance(user_id, intent)
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
            routing_result_str = await self.route_to_agent(intent, user_id)
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
            logger.info(f"Executing sub-agent: {target_agent}")
            logger.info(f"Parameters for logging: user_id={user_id}, query={query}, intent={intent}")
            exec_result_str = await self.execute_sub_agent(target_agent, user_id, query, intent)
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
            agg_result_str = await self.aggregate_results(
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
            await self._audit_log(
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

            #self.metrics.record_execution({
            #    "user_id": user_id,
            #    "intent": "UNKNOWN",
            #    "confidence": 0.0,
            #    "agent_name": "unknown",
            #    "execution_time_ms": 0,
            #    "success": False,
            #    "error_message": str(e)
            #})
            
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
            await self._audit_log(
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
    if not project_id or not isinstance(project_id, str):
        raise ValueError("project_id must be a non-empty string")
    
    logger.info(f"Creating Root Agent for project: {project_id}")
    
    try:
        # Create PlanInfoAgent instance
        plan_info_agent = PlanInfoAgent(project_id=project_id, region=region)
        # Pass it to RootAgent
        agent = RootAgent(project_id=project_id, region=region, plan_info_agent=plan_info_agent)
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
   