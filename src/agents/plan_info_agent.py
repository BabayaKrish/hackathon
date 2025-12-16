# Plan Information Agent - Plan Details & Support
# File: backend/agents/plan_info_agent.py

import json
import logging
from typing import Dict, Any, List
from datetime import datetime
import time

from google.cloud import bigquery
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
    
    def __init__(self, project_id: str, region: str = "us-central1"):
        """Initialize Plan Info Agent"""
        self.project_id = project_id
        self.region = region
        self.bq_client = bigquery.Client(project=project_id)
        
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
    
    def retrieve_plan_features(self) -> Dict[str, Any]:
        """
        TOOL 1: Retrieve detailed plan features
        
        Returns:
            Dictionary with all plan features
        """
        logger.info("Retrieving all plan features")
        
        try:
            return {
                "status": "success",
                "plans": self.plans,
                "total_plans": len(self.plans),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Feature retrieval error: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_pricing(self) -> Dict[str, Any]:
        """
        TOOL 2: Get pricing information
        
        Returns:
            Dictionary with pricing details and discounts
        """
        logger.info("Retrieving pricing information")
        
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
            
            return {
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
        except Exception as e:
            logger.error(f"Pricing retrieval error: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def build_comparison(self) -> Dict[str, Any]:
        """
        TOOL 3: Build side-by-side plan comparison
        
        Returns:
            Dictionary with comparison matrix
        """
        logger.info("Building plan comparison")
        
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
            
            return {
                "status": "success",
                "comparison": comparison_dimensions,
                "recommendation": "Gold for enterprise needs, Silver for growing businesses, Bronze to get started",
                "upgrade_path": "Bronze → Silver → Gold"
            }
        except Exception as e:
            logger.error(f"Comparison building error: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def retrieve_faq(self, category: str = None) -> Dict[str, Any]:
        """
        TOOL 4: Retrieve FAQ by category
        
        Args:
            category: FAQ category (billing, features, api, support, or None for all)
        
        Returns:
            Dictionary with FAQs
        """
        logger.info(f"Retrieving FAQs for category: {category}")
        
        try:
            if category and category.lower() in self.faqs:
                faqs = {category.lower(): self.faqs[category.lower()]}
            else:
                faqs = self.faqs
            
            return {
                "status": "success",
                "faqs": faqs,
                "categories": list(self.faqs.keys()),
                "total_questions": sum(len(v) for v in faqs.values()),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"FAQ retrieval error: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def explain_entitlements(self, plan_tier: str) -> Dict[str, Any]:
        """
        TOOL 5: Explain what features are included in a plan
        
        Args:
            plan_tier: Plan tier (bronze, silver, gold)
        
        Returns:
            Dictionary with detailed entitlements
        """
        logger.info(f"Explaining entitlements for {plan_tier}")
        
        try:
            plan = self.plans.get(plan_tier.lower())
            
            if not plan:
                return {
                    "status": "error",
                    "error": f"Plan '{plan_tier}' not found"
                }
            
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
            
            return {
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
        except Exception as e:
            logger.error(f"Entitlement explanation error: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
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
        
        try:
            # Step 1: Analyze query to determine intent
            query_lower = query.lower()
            
            # Step 2: Retrieve relevant information based on intent
            
            # Check for pricing questions
            if any(word in query_lower for word in ["price", "cost", "pricing", "how much", "expensive"]):
                pricing_result = self.get_pricing()
                execution_log.append({"step": "get_pricing", "status": "success"})
                data = pricing_result
            
            # Check for comparison questions
            elif any(word in query_lower for word in ["compare", "difference", "vs", "versus", "better"]):
                comparison_result = self.build_comparison()
                execution_log.append({"step": "build_comparison", "status": "success"})
                data = comparison_result
            
            # Check for feature questions
            elif any(word in query_lower for word in ["feature", "include", "included", "what's in", "contains", "reports"]):
                features_result = self.retrieve_plan_features()
                execution_log.append({"step": "retrieve_plan_features", "status": "success"})
                data = features_result
            
            # Check for FAQ questions
            elif any(word in query_lower for word in ["faq", "frequently asked", "help", "question", "support"]):
                faq_result = self.retrieve_faq()
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
                    entitlements_result = self.explain_entitlements(plan_mentioned)
                    execution_log.append({"step": "explain_entitlements", "status": "success"})
                else:
                    # Default to showing all plans
                    features_result = self.retrieve_plan_features()
                    execution_log.append({"step": "retrieve_plan_features", "status": "success"})
                data = features_result
            
            # Default: Get all plan features
            else:
                features_result = self.retrieve_plan_features()
                execution_log.append({"step": "retrieve_plan_features", "status": "success"})
                data = features_result
            
            elapsed_time = (time.time() - start_time) * 1000
            
            return {
                "status": "success",
                "query": query,
                "data": data,
                "execution_log": execution_log,
                "total_execution_time_ms": elapsed_time,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Info request processing error: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "execution_log": execution_log
            }


# ==================== Exports ====================

__all__ = ['PlanInfoAgent']
