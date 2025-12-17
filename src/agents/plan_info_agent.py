# Plan Information Agent - Plan Details & Support
# File: backend/agents/plan_info_agent.py

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import time
import concurrent.futures

from google.cloud import bigquery
from common_tools.AuditTool import async_log_event
import vertexai
from vertexai.generative_models import GenerativeModel

# ==================== Configuration ====================

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ==================== Plan Information Agent ====================

class PlanInfoAgent:
    """
    Plan Information Agent
    
    Responsibilities:
    - Provide detailed plan features and capabilities
    - Display pricing information
    - Compare plans side-by-side
    - Answer FAQ questions
    - Explain feature entitlements
    
    Tools:
    - retrieve_plan_features: Get all plan features
    - get_pricing: Fetch pricing tiers
    - build_comparison: Compare plans
    - retrieve_faq: Get knowledge base
    - explain_entitlements: Map access rules
    
    Prompt Template:
    "You are a customer success specialist. Answer questions about:
    1. Plan features and capabilities
    2. Pricing and billing models
    3. Feature availability by tier
    4. Upgrade paths and transitions
    5. Frequently asked questions"
    """

    def _convert_decimal(self, obj):
        """
        Recursively convert Decimal objects to float in dicts/lists.
        """
        import decimal
        if isinstance(obj, dict):
            return {k: self._convert_decimal(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_decimal(i) for i in obj]
        elif isinstance(obj, decimal.Decimal):
            return float(obj)
        else:
            return obj
    
    def __init__(self, project_id: str, region: str = "us-central1"):
        """Initialize Plan Info Agent"""
        self.project_id = project_id
        self.region = region
        self.bq_client = bigquery.Client(project=project_id)

        # Retry configuration for Vertex AI calls
        self.max_retries = 3
        self.base_delay = 1.0  # seconds
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=region)
        self.model = GenerativeModel("gemini-2.0-flash-exp")
        
        # Define plan details
        self.plans = {
            "bronze": {
                "name": "Bronze",
                "description": "Perfect for small businesses getting started",
                "price_monthly": 29,
                "price_annual": 290,
                "features": [
                    "Up to 500 monthly transactions",
                    "Basic account balance reports",
                    "Mobile app access",
                    "Email support",
                    "7-day transaction history",
                    "Basic security (2FA)"
                ]
            },
            "silver": {
                "name": "Silver",
                "description": "Ideal for growing businesses",
                "price_monthly": 99,
                "price_annual": 990,
                "features": [
                    "Up to 5,000 monthly transactions",
                    "Advanced reporting (9 report types)",
                    "API access",
                    "Mobile app + web dashboard",
                    "Phone + Email support",
                    "90-day transaction history",
                    "Advanced security (SSO, MFA)",
                    "Custom alerts",
                    "Scheduled reports"
                ]
            },
            "gold": {
                "name": "Gold",
                "description": "Enterprise-grade for large operations",
                "price_monthly": 299,
                "price_annual": 2990,
                "features": [
                    "Unlimited transactions",
                    "All reports + custom reports",
                    "Full API access + webhooks",
                    "Multiple user accounts",
                    "24/7 dedicated support",
                    "7-year transaction history",
                    "Enterprise security (SSO, MFA, IP restrictions)",
                    "Custom dashboards",
                    "Real-time alerts",
                    "SLA guarantee (99.9% uptime)",
                    "Priority onboarding",
                    "Quarterly business reviews"
                ]
            }
        }
        
        # FAQ database
        self.faqs = {
            "billing": [
                {
                    "question": "How am I billed?",
                    "answer": "You're billed monthly on the same date each month. Annual plans are billed upfront and include a 10% discount."
                },
                {
                    "question": "Can I change my plan anytime?",
                    "answer": "Yes! You can upgrade or downgrade your plan anytime. Changes take effect immediately."
                },
                {
                    "question": "Is there a contract?",
                    "answer": "No contracts required. You can cancel anytime with no penalties."
                }
            ],
            "features": [
                {
                    "question": "What reports are available in Silver?",
                    "answer": "Silver includes: Balance Reports, Wire Details, ACH Inbound, Intraday Balance, Expanded Details, Statements, Deposit Details, Check Images, and Running Ledger."
                },
                {
                    "question": "Can I create custom reports?",
                    "answer": "Custom reports are available in the Gold plan. Contact our team for details on customization options."
                },
                {
                    "question": "How far back can I see transaction history?",
                    "answer": "Bronze: 7 days, Silver: 90 days, Gold: 7 years"
                }
            ],
            "api": [
                {
                    "question": "Is API access included?",
                    "answer": "API access is included in Silver and Gold plans. It's not available in Bronze."
                },
                {
                    "question": "What's the API rate limit?",
                    "answer": "Silver: 1,000 requests/day, Gold: Unlimited with prioritization"
                },
                {
                    "question": "Is there API documentation?",
                    "answer": "Yes! Comprehensive API documentation is available in your dashboard. We also offer developer support."
                }
            ],
            "support": [
                {
                    "question": "What support is included?",
                    "answer": "Bronze: Email support, Silver: Phone + Email, Gold: 24/7 dedicated support team"
                },
                {
                    "question": "What's the response time?",
                    "answer": "Bronze: 24-48 hours, Silver: 4-8 hours, Gold: 1 hour (24/7)"
                },
                {
                    "question": "Is onboarding included?",
                    "answer": "Basic onboarding is included. Gold includes priority onboarding with a dedicated specialist."
                }
            ]
        }
        
        logger.info(f"PlanInfoAgent initialized for project {project_id}")

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
    
    def retrieve_a_plan_features(self) -> Dict[str, Any]:
        """
        TOOL 1: Retrieve detailed plan features
        
        Returns:
            Dictionary with all plan features
        """
        logger.info("Retrieving plan features")
        try:
            query = f"""
                SELECT 
                    plan_name,
                    tier,
                    monthly_price,
                    annual_price,
                    features
                FROM `{self.project_id}.client_data.plan_offerings`
                ORDER BY monthly_price
            """
            results = self.bq_client.query(query).result()
            plans = [self._convert_decimal(dict(row)) for row in results]
            return {
                "status": "success",
                "plans": plans,
                "total_plans": len(plans)
            }
        except Exception as e:
            logger.error(f"Feature retrieval error: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "plans": []
            }
    
    async def get_pricing(self) -> Dict[str, Any]:
        """
        TOOL 2: Get pricing information
        
        Returns:
            Dictionary with pricing details and discounts
        """
        logger.info("Retrieving pricing information")
        start_time = time.time()
        input_text = "{}"
        caller = "anonymous_user"

        try:
            pricing_data = {}
            for plan_name, plan_details in self.plans.items():
                annual_savings = (plan_details["price_monthly"] * 12) - plan_details["price_annual"]
                discount_pct = (annual_savings / (plan_details["price_monthly"] * 12)) * 100
                
                pricing_data[plan_name] = {
                    "monthly": plan_details["price_monthly"],
                    "annual": plan_details["price_annual"],
                    "savings_with_annual": annual_savings,
                    "discount_percentage": discount_pct,
                    "cost_per_transaction_monthly": plan_details["price_monthly"] / 500,  # Approx
                    "currency": "USD"
                }
            
            output = {
                "status": "success",
                "pricing": pricing_data,
                "payment_methods": [
                    "Credit Card (Visa, MasterCard, American Express)",
                    "Bank Transfer (ACH)",
                    "Wire Transfer"
                ],
                "discounts": {
                    "annual_commitment": "10% off annual plans",
                    "enterprise": "Contact sales for custom pricing"
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            latency_ms = (time.time() - start_time) * 1000
            await self._audit_log("plan_info_agent.get_pricing", caller, input_text, json.dumps(output), latency_ms)
            return output
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Pricing retrieval error: {str(e)}")
            error_output = {
                "status": "error",
                "error": str(e)
            }
            await self._audit_log(
                "plan_info_agent.get_pricing",
                caller,
                input_text,
                json.dumps(error_output),
                latency_ms,
                safety_result="error",
                error_message=str(e)
            )
            return error_output
    
    async def build_comparison(self) -> Dict[str, Any]:
        """
        TOOL 3: Build side-by-side plan comparison
        
        Returns:
            Dictionary with comparison matrix
        """
        logger.info("Building plan comparison")
        start_time = time.time()
        input_text = "{}"
        caller = "anonymous_user"
        
        try:
            # Define comparison dimensions
            comparison_dimensions = {
                "Pricing": {
                    "bronze": "$29/month",
                    "silver": "$99/month",
                    "gold": "$299/month"
                },
                "Monthly Transactions": {
                    "bronze": "500",
                    "silver": "5,000",
                    "gold": "Unlimited"
                },
                "Report Types": {
                    "bronze": "3",
                    "silver": "9",
                    "gold": "9 + Custom"
                },
                "API Access": {
                    "bronze": "❌",
                    "silver": "✅",
                    "gold": "✅ Unlimited"
                },
                "Custom Reports": {
                    "bronze": "❌",
                    "silver": "❌",
                    "gold": "✅"
                },
                "Custom Dashboards": {
                    "bronze": "❌",
                    "silver": "❌",
                    "gold": "✅"
                },
                "Mobile App": {
                    "bronze": "✅",
                    "silver": "✅",
                    "gold": "✅"
                },
                "Web Dashboard": {
                    "bronze": "Basic",
                    "silver": "Advanced",
                    "gold": "Full"
                },
                "Support": {
                    "bronze": "Email",
                    "silver": "Phone + Email",
                    "gold": "24/7 Dedicated"
                },
                "Response Time": {
                    "bronze": "24-48 hours",
                    "silver": "4-8 hours",
                    "gold": "1 hour"
                },
                "History Retention": {
                    "bronze": "7 days",
                    "silver": "90 days",
                    "gold": "7 years"
                },
                "Security": {
                    "bronze": "2FA",
                    "silver": "SSO + MFA",
                    "gold": "SSO + MFA + IP restrictions"
                },
                "SLA Guarantee": {
                    "bronze": "❌",
                    "silver": "❌",
                    "gold": "✅ 99.9%"
                }
            }
            
            output = {
                "status": "success",
                "comparison": comparison_dimensions,
                "recommendation": "Gold for enterprise needs, Silver for growing businesses, Bronze to get started",
                "upgrade_path": "Bronze → Silver → Gold"
            }
            latency_ms = (time.time() - start_time) * 1000
            await self._audit_log("plan_info_agent.build_comparison", caller, input_text, json.dumps(output), latency_ms)
            return output
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Comparison building error: {str(e)}")
            error_output = {
                "status": "error",
                "error": str(e)
            }
            await self._audit_log(
                "plan_info_agent.build_comparison",
                caller,
                input_text,
                json.dumps(error_output),
                latency_ms,
                safety_result="error",
                error_message=str(e)
            )
            return error_output
    
    async def retrieve_faq(self, category: str = None) -> Dict[str, Any]:
        """
        TOOL 4: Retrieve FAQ by category
        
        Args:
            category: FAQ category (billing, features, api, support, or None for all)
        
        Returns:
            Dictionary with FAQs
        """
        logger.info(f"Retrieving FAQs for category: {category}")
        start_time = time.time()
        input_text = json.dumps({"category": category})
        caller = "anonymous_user"

        try:
            if category and category.lower() in self.faqs:
                faqs = {category.lower(): self.faqs[category.lower()]}
            else:
                faqs = self.faqs
            
            output = {
                "status": "success",
                "faqs": faqs,
                "categories": list(self.faqs.keys()),
                "total_questions": sum(len(v) for v in faqs.values()),
                "timestamp": datetime.utcnow().isoformat()
            }
            latency_ms = (time.time() - start_time) * 1000
            await self._audit_log("plan_info_agent.retrieve_faq", caller, input_text, json.dumps(output), latency_ms, metadata={"category": category})
            return output
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"FAQ retrieval error: {str(e)}")
            error_output = {
                "status": "error",
                "error": str(e)
            }
            await self._audit_log(
                "plan_info_agent.retrieve_faq",
                caller,
                input_text,
                json.dumps(error_output),
                latency_ms,
                safety_result="error",
                error_message=str(e),
                metadata={"category": category}
            )
            return error_output
    
    async def explain_entitlements(self, plan_tier: str) -> Dict[str, Any]:
        """
        TOOL 5: Explain what features are included in a plan
        
        Args:
            plan_tier: Plan tier (bronze, silver, gold)
        
        Returns:
            Dictionary with detailed entitlements
        """
        logger.info(f"Explaining entitlements for {plan_tier}")
        start_time = time.time()
        input_text = json.dumps({"plan_tier": plan_tier})
        caller = "anonymous_user"
        
        try:
            plan = self.plans.get(plan_tier.lower())
            
            if not plan:
                error_output = {
                    "status": "error",
                    "error": f"Plan '{plan_tier}' not found"
                }
                latency_ms = (time.time() - start_time) * 1000
                await self._audit_log(
                    "plan_info_agent.explain_entitlements",
                    caller,
                    input_text,
                    json.dumps(error_output),
                    latency_ms,
                    safety_result="error",
                    error_message=f"Plan '{plan_tier}' not found",
                    metadata={"plan_tier": plan_tier}
                )
                return error_output
            
            # Map features to capabilities
            entitlements = {
                "reports": {
                    "bronze": ["balance_report", "intraday_balance", "statements"],
                    "silver": ["balance_report", "wire_details", "ach_inbound", "intraday_balance",
                              "expanded_details", "statements", "deposit_details", "check_images", "running_ledger"],
                    "gold": ["balance_report", "wire_details", "ach_inbound", "intraday_balance",
                            "expanded_details", "statements", "deposit_details", "check_images", "running_ledger", "custom_reports"]
                },
                "api": {
                    "bronze": False,
                    "silver": True,
                    "gold": True
                },
                "users": {
                    "bronze": 1,
                    "silver": 5,
                    "gold": 999
                },
                "alerts": {
                    "bronze": False,
                    "silver": True,
                    "gold": True
                },
                "integrations": {
                    "bronze": 0,
                    "silver": 5,
                    "gold": 999
                }
            }
            
            tier_lower = plan_tier.lower()
            
            output = {
                "status": "success",
                "plan": plan_tier,
                "entitlements": {
                    "reports": entitlements["reports"].get(tier_lower, []),
                    "api_access": entitlements["api"].get(tier_lower, False),
                    "max_users": entitlements["users"].get(tier_lower, 1),
                    "alerts_enabled": entitlements["alerts"].get(tier_lower, False),
                    "integrations_allowed": entitlements["integrations"].get(tier_lower, 0),
                    "custom_dashboards": tier_lower == "gold",
                    "dedicated_support": tier_lower == "gold",
                    "sso_enabled": tier_lower != "bronze",
                    "mfa_enabled": tier_lower != "bronze"
                },
                "plan_details": plan
            }
            latency_ms = (time.time() - start_time) * 1000
            await self._audit_log("plan_info_agent.explain_entitlements", caller, input_text, json.dumps(output), latency_ms, metadata={"plan_tier": plan_tier})
            return output
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Entitlement explanation error: {str(e)}")
            error_output = {
                "status": "error",
                "error": str(e)
            }
            await self._audit_log(
                "plan_info_agent.explain_entitlements",
                caller,
                input_text,
                json.dumps(error_output),
                latency_ms,
                safety_result="error",
                error_message=str(e),
                metadata={"plan_tier": plan_tier}
            )
            return error_output
    
    async def process_info_request(self, query: str) -> Dict[str, Any]:
        """
        Main Plan Info Agent entry point

        Args:
            query: User question about plans

        Returns:
            Comprehensive answer with relevant plan information
        """
        logger.info(f"Processing plan info request: {query}")

        execution_log = []
        start_time = time.time()
        input_text = json.dumps({"query": query})
        caller = "anonymous_user"

        try:
            # Step 1: Analyze query to determine intent
            query_lower = query.lower()

            # Step 2: Retrieve relevant information based on intent
            data = None

            # Check for pricing questions
            if any(word in query_lower for word in ["price", "cost", "pricing", "how much", "expensive"]):
                pricing_result = await self.get_pricing()
                execution_log.append({"step": "get_pricing", "status": "success"})
                data = pricing_result

            # Check for comparison questions
            elif any(word in query_lower for word in ["compare", "difference", "vs", "versus", "better"]):
                comparison_result = await self.build_comparison()
                execution_log.append({"step": "build_comparison", "status": "success"})
                data = comparison_result

            # Check for feature questions (RAG: retrieve from compliance_rules)
            elif any(word in query_lower for word in ["feature", "include", "included", "what's in", "contains", "reports"]):
                # Try to extract plan tier from the query, default to 'bronze'
                plan_mentioned = None
                for plan in ["bronze", "silver", "gold"]:
                    if plan in query_lower:
                        plan_mentioned = plan
                        break
                if not plan_mentioned:
                    plan_mentioned = "bronze"  # or handle as error/default

                # RAG: Retrieve features from compliance_rules
                features_result = await self.retrieve_compliant_features(plan_mentioned)
                execution_log.append({"step": "retrieve_compliant_features", "status": "success"})
                data = features_result

            # Check for FAQ questions
            elif any(word in query_lower for word in ["faq", "frequently asked", "help", "question", "support"]):
                faq_result = await self.retrieve_faq()
                execution_log.append({"step": "retrieve_faq", "status": "success"})
                data = faq_result

            # Check for entitlement questions
            elif any(word in query_lower for word in ["can i", "am i", "entitle", "access", "allowed"]):
                # Extract plan name if mentioned
                plan_mentioned = None
                for plan in ["bronze", "silver", "gold"]:
                    if plan in query_lower:
                        plan_mentioned = plan
                        break

                if plan_mentioned:
                    entitlements_result = await self.explain_entitlements(plan_mentioned)
                    execution_log.append({"step": "explain_entitlements", "status": "success"})
                    data = entitlements_result
                else:
                    # Default to showing all plans
                    features_result = await self.retrieve_plan_features()
                    execution_log.append({"step": "retrieve_plan_features", "status": "success"})
                    data = features_result

            # Default: Get all plan features
            else:
                features_result = await self.retrieve_plan_features()
                execution_log.append({"step": "retrieve_plan_features", "status": "success"})
                data = features_result

            elapsed_time = (time.time() - start_time) * 1000

            output = {
                "status": "success",
                "query": query,
                "data": data,
                "execution_log": execution_log,
                "total_execution_time_ms": elapsed_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            await self._audit_log("plan_info_agent.process_info_request", caller, input_text, json.dumps(output), elapsed_time, metadata={"query": query})
            return output

        except Exception as e:
            elapsed_time = (time.time() - start_time) * 1000
            logger.error(f"Info request processing error: {str(e)}")
            error_output = {
                "status": "error",
                "error": str(e),
                "execution_log": execution_log
            }
            await self._audit_log(
                "plan_info_agent.process_info_request",
                caller,
                input_text,
                json.dumps(error_output),
                elapsed_time,
                safety_result="error",
                error_message=str(e),
                metadata={"query": query}
            )
            return error_output

    async def retrieve_compliant_features(self, plan_tier: str) -> Dict[str, Any]:
        """
        Retrieve features for a plan tier based on compliance_rules table.
        Only returns features where is_allowed is TRUE or Optional.
        """
        logger.info(f"Retrieving compliant features for {plan_tier}")
        start_time = time.time()
        input_text = json.dumps({"plan_tier": plan_tier})
        caller = "anonymous_user"

        try:
            query = f"""
                SELECT feature, is_allowed
                FROM `{self.project_id}.client_data.compliance_rules`
                WHERE LOWER(tier) = '{plan_tier.lower()}'
                AND (UPPER(is_allowed) = 'TRUE' OR UPPER(is_allowed) = 'OPTIONAL')
            """
            results = self.bq_client.query(query).result()
            features = []
            for row in results:
                features.append({
                    "feature": row.feature,
                    "is_allowed": row.is_allowed
                })

            output = {
                "status": "success",
                "plan_tier": plan_tier,
                "features": features,
                "total_features": len(features),
                "timestamp": datetime.utcnow().isoformat()
            }
            latency_ms = (time.time() - start_time) * 1000
            # ...existing code...
            await self._audit_log(
                "plan_info_agent.retrieve_compliant_features",
                caller,
                input_text,
                json.dumps(output),
                int(latency_ms),  # Ensure integer type
                metadata={"plan_tier": plan_tier}
            )
# ...existing code...
            # await self._audit_log("plan_info_agent.retrieve_compliant_features", caller, input_text, json.dumps(output), latency_ms, metadata={"plan_tier": plan_tier})
            return output
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Compliant feature retrieval error: {str(e)}")
            error_output = {
                "status": "error",
                "error": str(e)
            }
            await self._audit_log(
                "plan_info_agent.retrieve_compliant_features",
                caller,
                input_text,
                json.dumps(error_output),
                latency_ms,
                safety_result="error",
                error_message=str(e),
                metadata={"plan_tier": plan_tier}
            )
            return error_output

    def retrieve_all_plan_features(self) -> Dict[str, Any]:
        """
        TOOL 1: Retrieve all plan features from BigQuery
        
        Returns:
            Dictionary with all available plans
        """
        logger.info("Retrieving all plan features")
        try:
            query = f"""
                SELECT 
                    plan_name,
                    tier,
                    monthly_price,
                    annual_price,
                    features
                FROM `{self.project_id}.client_data.plan_offerings`
                ORDER BY monthly_price
            """
            results = self.bq_client.query(query).result()
            plans = [self._convert_decimal(dict(row)) for row in results]
            logger.info(f"Successfully retrieved {len(plans)} plans")
            return {
                "status": "success",
                "plans": plans,
                "total_plans": len(plans)
            }
        except Exception as e:
            logger.error(f"Feature retrieval error: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "plans": []
            }

    def retrieve_plan_features(self) -> Dict[str, Any]:
        """Backward-compatible wrapper to retrieve plan features.

        This method matches the name used by existing callers and
        simply delegates to ``retrieve_all_plan_features``.
        """
        return self.retrieve_all_plan_features()

    def retrieve_specific_plan(self, plan_tier: str) -> Dict[str, Any]:
        """
        TOOL 2: Retrieve specific plan details by tier
        
        Args:
            plan_tier: Plan tier name (e.g., 'Bronze', 'Silver', 'Gold')
        
        Returns:
            Dictionary with specific plan details
        """
        logger.info(f"Retrieving plan details for tier: {plan_tier}")
        try:
            query = f"""
                SELECT 
                    plan_name,
                    tier,
                    monthly_price,
                    annual_price,
                    features
                FROM `{self.project_id}.client_data.plan_offerings`
                WHERE UPPER(tier) = UPPER('{plan_tier}')
                LIMIT 1
            """
            results = self.bq_client.query(query).result()
            rows = list(results)
            
            if not rows:
                logger.warning(f"No plan found for tier: {plan_tier}")
                return {
                    "status": "error",
                    "error": f"Plan tier '{plan_tier}' not found",
                    "plan": None
                }
            
            plan = self._convert_decimal(dict(rows[0]))
            logger.info(f"Successfully retrieved plan: {plan_tier}")
            return {
                "status": "success",
                "plan": plan,
                "found": True
            }
        except Exception as e:
            logger.error(f"Plan retrieval error for tier {plan_tier}: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "plan": None
            }

    def get_upgrade_suggestions(self, current_tier: str) -> Dict[str, Any]:
        """
        TOOL 3: Get upgrade suggestions based on current plan tier
        
        Args:
            current_tier: Current plan tier (e.g., 'Bronze')
        
        Returns:
            Dictionary with upgrade suggestions
        """
        logger.info(f"Getting upgrade suggestions for tier: {current_tier}")
        try:
            query = f"""
                SELECT 
                    plan_name,
                    tier,
                    monthly_price,
                    annual_price,
                    features
                FROM `{self.project_id}.client_data.plan_offerings`
                WHERE UPPER(tier) > UPPER('{current_tier}')
                ORDER BY monthly_price ASC
            """
            results = self.bq_client.query(query).result()
            upgrade_plans = [self._convert_decimal(dict(row)) for row in results]
            
            if not upgrade_plans:
                logger.info(f"No upgrade options available for tier: {current_tier}")
                return {
                    "status": "success",
                    "current_tier": current_tier,
                    "upgrade_options": [],
                    "message": "You are on the highest tier plan"
                }
            
            logger.info(f"Found {len(upgrade_plans)} upgrade options for {current_tier}")
            return {
                "status": "success",
                "current_tier": current_tier,
                "upgrade_options": upgrade_plans,
                "total_options": len(upgrade_plans)
            }
        except Exception as e:
            logger.error(f"Upgrade suggestions error for tier {current_tier}: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "upgrade_options": []
            }

    def classify_plan_intent(self, query: str) -> Dict[str, Any]:
        """
        Use Vertex AI to classify user intent and extract plan information
        
        Args:
            query: User's query string
        
        Returns:
            Dictionary with intent classification and extracted plan info
        """
        prompt = f"""You are a financial services plan assistant. Analyze the following customer query and determine their intent.

Query: {query}

Available intents:
1. PLAN_INFO - User is asking for information about plans, features, pricing, or comparisons
2. PLAN_UPGRADE - User wants to change or upgrade their plan to a higher tier
3. PLAN_DETAILS - User wants detailed information about a specific plan
4. CURRENT_PLAN - User wants to know their current plan details

Extract any mentioned plan tier if present (e.g., 'Bronze', 'Silver', 'Gold', 'No Plan').

Respond ONLY in valid JSON format with no additional text:
{{"intent": "PLAN_INFO|PLAN_UPGRADE|PLAN_DETAILS|CURRENT_PLAN", "confidence": 0.0-1.0, "mentioned_tier": "tier_name_or_null", "reasoning": "Brief explanation"}}

Rules:
- Confidence must be between 0 and 1
- If unclear or ambiguous, set confidence to 0.6-0.8
- mentioned_tier should be null if no plan tier is mentioned
- Return valid JSON only, no markdown or explanations"""

        logger.info(f"Classifying plan intent for query: {query[:100]}...")
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Calling Vertex AI for intent classification (attempt {attempt + 1}/{self.max_retries})")
                
                def call_vertex_ai():
                    return self.model.generate_content(prompt)
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(call_vertex_ai)
                    try:
                        response = future.result(timeout=30.0)
                        
                        # Parse response
                        response_text = response.text.strip()
                        logger.info(f"Vertex AI response: {response_text}")

                        try:
                            # First try direct JSON parse
                            result = json.loads(response_text)
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON parsing error in classify_plan_intent on attempt {attempt + 1}: {e}")
                            # Try to extract JSON object from within markdown fences or extra text
                            start = response_text.find("{")
                            end = response_text.rfind("}")
                            if start != -1 and end != -1 and end > start:
                                json_candidate = response_text[start:end + 1]
                                logger.info("Attempting to parse JSON candidate from response text")
                                result = json.loads(json_candidate)
                            else:
                                raise

                        result["status"] = "success"
                        
                        logger.info(f"Intent classified: {result.get('intent')} with confidence {result.get('confidence')}")
                        return result
                        
                    except concurrent.futures.TimeoutError:
                        logger.warning(f"Vertex AI call timed out on attempt {attempt + 1}")
                        if attempt < self.max_retries - 1:
                            time.sleep(self.base_delay * (2 ** attempt))
                            continue
                        else:
                            return {
                                "status": "error",
                                "intent": "UNKNOWN",
                                "confidence": 0.0,
                                "error": "Vertex AI timeout after all retries"
                            }
                            
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.base_delay * (2 ** attempt))
                    continue
                else:
                    return {
                        "status": "error",
                        "intent": "UNKNOWN",
                        "confidence": 0.0,
                        "error": f"Failed to parse Vertex AI response: {str(e)}"
                    }
            except Exception as e:
                logger.error(f"Intent classification error on attempt {attempt + 1}: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.base_delay * (2 ** attempt))
                    continue
                else:
                    return {
                        "status": "error",
                        "intent": "UNKNOWN",
                        "confidence": 0.0,
                        "error": str(e)
                    }

    def generate_upgrade_recommendation(self, current_tier: str, upgrade_options: list, user_query: str) -> Dict[str, Any]:
        """
        Use Vertex AI to generate personalized upgrade recommendations
        
        Args:
            current_tier: User's current plan tier
            upgrade_options: List of available upgrade plans
            user_query: Original user query
        
        Returns:
            Dictionary with AI-generated recommendations
        """
        plans_text = "\n".join([
            f"- {plan['plan_name']} ({plan['tier']}): ${plan['monthly_price']}/month, ${plan['annual_price']}/year\n  Features: {plan['features']}"
            for plan in upgrade_options
        ])
        
        prompt = f"""You are a financial services advisor. Based on the user's current plan and available options, provide a personalized upgrade recommendation.

Current Plan: {current_tier}
User Query: {user_query}

Available Upgrade Options:
{plans_text}

Provide a JSON response with:
1. recommended_plan - The best upgrade option
2. key_benefits - Top 3 benefits of upgrading
3. cost_analysis - Monthly and annual savings/cost comparison
4. reasoning - Why this upgrade matches their needs

Respond in JSON format only:
{{"recommended_plan": "plan_name", "key_benefits": ["benefit1", "benefit2", "benefit3"], "cost_analysis": "detailed comparison", "reasoning": "explanation"}}"""

        logger.info(f"Generating upgrade recommendation for {current_tier}")
        
        for attempt in range(self.max_retries):
            try:
                def call_vertex_ai():
                    return self.model.generate_content(prompt)
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(call_vertex_ai)
                    response = future.result(timeout=30.0)

                response_text = response.text.strip()
                logger.info(f"Vertex AI upgrade response: {response_text}")

                try:
                    # First try direct JSON parse
                    result = json.loads(response_text)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error in generate_upgrade_recommendation on attempt {attempt + 1}: {e}")
                    # Try to extract JSON object from within markdown fences or extra text
                    start = response_text.find("{")
                    end = response_text.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        json_candidate = response_text[start:end + 1]
                        logger.info("Attempting to parse JSON candidate from response text")
                        result = json.loads(json_candidate)
                    else:
                        raise

                result["status"] = "success"
                logger.info(f"Recommendation generated: {result.get('recommended_plan')}")
                return result

            except Exception as e:
                logger.error(f"Recommendation generation error on attempt {attempt + 1}: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.base_delay * (2 ** attempt))
                    continue
                else:
                    return {
                        "status": "error",
                        "error": str(e),
                        "recommended_plan": None
                    }

    def process_plan_upgrade_request(self, user_query: str, user_id: str) -> Dict[str, Any]:
        """
        Main orchestrator: Process user query and return upgrade recommendation
        
        Args:
            user_query: User's question/request
            user_id: User identifier
        
        Returns:
            Complete upgrade recommendation with options
        """
        logger.info(f"Processing plan upgrade request for user {user_id}: {user_query}")
        
        try:
            # Step 1: Classify intent
            intent_result = self.classify_plan_intent(user_query)
            if intent_result.get("status") == "error":
                logger.warning(f"Intent classification failed: {intent_result.get('error')}")
                intent_result["mentioned_tier"] = None

            mentioned_tier = intent_result.get("mentioned_tier")
            logger.info(f"Mentioned plan tier in classification result: {mentioned_tier}")
            intent = intent_result.get("intent", "UNKNOWN")
            
            # Step 2: Get all plans for context
            all_plans = self.retrieve_all_plan_features()
            
            # Step 3: If specific tier mentioned, get details
            specific_plan = None
            if mentioned_tier:
                specific_plan = self.retrieve_specific_plan(mentioned_tier)
                logger.info(f"Retrieved specific plan for tier: {mentioned_tier}")
            
            # Step 4: Get upgrade suggestions
            upgrade_suggestions = self.get_upgrade_suggestions(mentioned_tier or "Bronze")
            
            # Step 5: Generate AI recommendation
            recommendation = None
            if upgrade_suggestions.get("upgrade_options"):
                recommendation = self.generate_upgrade_recommendation(
                    mentioned_tier or "Bronze",
                    upgrade_suggestions["upgrade_options"],
                    user_query
                )
            
            # Compile final response
            response = {
                "status": "success",
                "user_id": user_id,
                "intent": intent,
                "confidence": intent_result.get("confidence", 0),
                "mentioned_tier": mentioned_tier,
                "all_plans": all_plans.get("plans", []),
                "current_plan_details": specific_plan,
                "upgrade_options": upgrade_suggestions.get("upgrade_options", []),
                "ai_recommendation": recommendation,
                "next_actions": ["show_comparison_table", "show_upgrade_prompt"]
            }
            
            logger.info(f"Successfully processed upgrade request for user {user_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error in process_plan_upgrade_request: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "user_id": user_id
            }   

# ==================== Exports ====================

__all__ = ['PlanInfoAgent']
