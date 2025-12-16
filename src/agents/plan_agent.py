# Plan Agent - Usage Analysis & Recommendations
# File: backend/agents/plan_agent.py

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

# ==================== Plan Agent ====================

class PlanAgent:
    """
    Plan Recommendation Agent
    
    Responsibilities:
    - Analyze customer usage patterns
    - Calculate plan fit scores
    - Compute ROI for upgrades
    - Detect upsell opportunities
    - Execute plan changes
    - Simulate billing scenarios
    
    Tools:
    - analyze_usage: Extract usage metrics and patterns
    - calculate_fit: Find best matching plans
    - calculate_roi: Compute upgrade ROI
    - detect_upsell: Identify upgrade opportunities
    - execute_upgrade: Process plan changes
    - simulate_billing: Estimate future costs
    
    Prompt Template:
    "You are a financial services advisor. Analyze customer usage and:
    1. Identify current plan limitations
    2. Calculate usage patterns
    3. Recommend better matching plans with ROI
    4. Explain upgrade benefits
    5. Provide implementation rationale"
    """
    
    def __init__(self, project_id: str, region: str = "us-central1"):
        """Initialize Plan Agent"""
        self.project_id = project_id
        self.region = region
        self.bq_client = bigquery.Client(project=project_id)
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=region)
        self.model = GenerativeModel("gemini-2.0-flash-exp")
        
        # Define plan tiers
        self.plan_tiers = {
            "bronze": {
                "name": "Bronze",
                "price": 29,
                "transaction_limit": 500,
                "api_access": False,
                "custom_reports": False,
                "support": "Email"
            },
            "silver": {
                "name": "Silver",
                "price": 99,
                "transaction_limit": 5000,
                "api_access": True,
                "custom_reports": False,
                "support": "Phone + Email"
            },
            "gold": {
                "name": "Gold",
                "price": 299,
                "transaction_limit": 999999,
                "api_access": True,
                "custom_reports": True,
                "support": "24/7 Dedicated"
            }
        }
        
        logger.info(f"PlanAgent initialized for project {project_id}")
    
    def analyze_usage(self, user_id: str) -> Dict[str, Any]:
        """
        TOOL 1: Extract and analyze usage metrics
        
        Args:
            user_id: Customer identifier
        
        Returns:
            Dictionary with usage metrics and patterns
        """
        logger.info(f"Analyzing usage for user {user_id}")
        
        try:
            # Query usage metrics from past 90 days
            query = f"""
                WITH monthly_usage AS (
                    SELECT 
                        DATE_TRUNC(transaction_date, MONTH) as month,
                        COUNT(*) as transaction_count,
                        SUM(CASE WHEN transaction_type = 'wire' THEN 1 ELSE 0 END) as wire_count,
                        SUM(CASE WHEN transaction_type = 'ach' THEN 1 ELSE 0 END) as ach_count,
                        SUM(CASE WHEN transaction_type = 'check' THEN 1 ELSE 0 END) as check_count,
                        SUM(amount) as total_volume,
                        AVG(amount) as avg_transaction_amount
                    FROM `{self.project_id}.client_data.user_transactions`
                    WHERE user_id = @user_id
                        AND transaction_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
                    GROUP BY month
                )
                SELECT 
                    *,
                    LAG(transaction_count) OVER (ORDER BY month) as prev_month_count,
                    ROUND(100.0 * (transaction_count - LAG(transaction_count) OVER (ORDER BY month)) 
                        / LAG(transaction_count) OVER (ORDER BY month), 2) as month_over_month_growth
                FROM monthly_usage
                ORDER BY month DESC
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                ]
            )
            
            results = list(self.bq_client.query(query, job_config=job_config).result())
            
            # Calculate aggregate metrics
            total_transactions = sum(r.transaction_count for r in results) if results else 0
            total_volume = sum(r.total_volume for r in results) if results else 0
            avg_monthly_transactions = total_transactions / len(results) if results else 0
            
            # Get current plan
            plan_query = f"""
                SELECT plan_tier FROM `{self.project_id}.client_data.client_profiles`
                WHERE user_id = @user_id
            """
            plan_results = list(self.bq_client.query(plan_query, job_config=job_config).result())
            current_plan = plan_results[0].plan_tier if plan_results else "bronze"
            
            # Convert dates to ISO strings
            usage_data = []
            for r in results:
                usage_data.append({
                    "month": r.month.isoformat() if hasattr(r.month, 'isoformat') else str(r.month),
                    "transaction_count": r.transaction_count,
                    "wire_count": r.wire_count,
                    "ach_count": r.ach_count,
                    "check_count": r.check_count,
                    "total_volume": float(r.total_volume) if r.total_volume else 0,
                    "avg_transaction_amount": float(r.avg_transaction_amount) if r.avg_transaction_amount else 0
                })
            
            return {
                "status": "success",
                "user_id": user_id,
                "current_plan": current_plan,
                "usage_summary": {
                    "total_transactions_90d": total_transactions,
                    "avg_monthly_transactions": avg_monthly_transactions,
                    "total_volume_90d": float(total_volume) if total_volume else 0,
                    "peak_month_transactions": max((r.transaction_count for r in results), default=0),
                    "analysis_period_days": 90
                },
                "monthly_breakdown": usage_data
            }
        except Exception as e:
            logger.error(f"Usage analysis error: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "usage_summary": {}
            }
    
    def calculate_fit(self, usage_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        TOOL 2: Calculate plan fit scores
        
        Args:
            usage_metrics: Usage data from analyze_usage
        
        Returns:
            Dictionary with plan fit scores and recommendations
        """
        logger.info("Calculating plan fit scores")
        
        try:
            avg_monthly = usage_metrics.get("usage_summary", {}).get("avg_monthly_transactions", 0)
            peak_monthly = usage_metrics.get("usage_summary", {}).get("peak_month_transactions", 0)
            
            # Calculate fit score for each plan
            fit_scores = {}
            for plan_name, plan_details in self.plan_tiers.items():
                limit = plan_details["transaction_limit"]
                
                # Score based on utilization
                avg_utilization = (avg_monthly / limit * 100) if limit > 0 else 0
                peak_utilization = (peak_monthly / limit * 100) if limit > 0 else 0
                
                # Fit score: prefer 60-80% utilization
                if peak_utilization > 100:
                    fit_score = 0  # Over limit
                elif avg_utilization < 10:
                    fit_score = 50  # Underutilized
                elif 60 <= avg_utilization <= 80:
                    fit_score = 100  # Perfect fit
                elif avg_utilization > 80:
                    fit_score = 80  # Near capacity
                else:
                    fit_score = 70  # Good fit
                
                fit_scores[plan_name] = {
                    "fit_score": fit_score,
                    "avg_utilization_percent": avg_utilization,
                    "peak_utilization_percent": peak_utilization,
                    "sufficient_capacity": peak_utilization <= 100,
                    "has_headroom": peak_utilization < 80
                }
            
            # Find recommended plan
            recommended_plan = max(fit_scores.items(), key=lambda x: x[1]["fit_score"])[0]
            
            return {
                "status": "success",
                "fit_analysis": fit_scores,
                "recommended_plan": recommended_plan,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Fit calculation error: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def calculate_roi(
        self,
        current_plan: str,
        recommended_plan: str,
        monthly_transactions: float
    ) -> Dict[str, Any]:
        """
        TOOL 3: Calculate ROI for plan upgrade
        
        Args:
            current_plan: Current plan tier
            recommended_plan: Recommended plan tier
            monthly_transactions: Average monthly transactions
        
        Returns:
            Dictionary with ROI analysis
        """
        logger.info(f"Calculating ROI: {current_plan} -> {recommended_plan}")
        
        try:
            current_details = self.plan_tiers.get(current_plan, {})
            recommended_details = self.plan_tiers.get(recommended_plan, {})
            
            current_price = current_details.get("price", 0)
            recommended_price = recommended_details.get("price", 0)
            
            # Calculate monthly cost difference
            monthly_cost_increase = recommended_price - current_price
            annual_cost_increase = monthly_cost_increase * 12
            
            # Estimate benefits
            api_benefit = 50 if (recommended_details.get("api_access") and not current_details.get("api_access")) else 0
            reporting_benefit = 100 if (recommended_details.get("custom_reports") and not current_details.get("custom_reports")) else 0
            support_benefit = 75  # Better support tier
            
            total_benefits = api_benefit + reporting_benefit + support_benefit
            
            # Calculate ROI
            roi_percentage = ((total_benefits - monthly_cost_increase) / monthly_cost_increase * 100) if monthly_cost_increase > 0 else 0
            
            return {
                "status": "success",
                "current_plan": current_plan,
                "recommended_plan": recommended_plan,
                "cost_analysis": {
                    "current_monthly_cost": current_price,
                    "recommended_monthly_cost": recommended_price,
                    "monthly_increase": monthly_cost_increase,
                    "annual_increase": annual_cost_increase
                },
                "benefits": {
                    "api_access": api_benefit,
                    "custom_reports": reporting_benefit,
                    "support_tier": support_benefit,
                    "total_monthly_benefit": total_benefits
                },
                "roi_analysis": {
                    "roi_percentage": roi_percentage,
                    "payback_period_months": (monthly_cost_increase / total_benefits) if total_benefits > 0 else 999,
                    "break_even_month": 2 if roi_percentage > 0 else None
                }
            }
        except Exception as e:
            logger.error(f"ROI calculation error: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def detect_upsell(
        self,
        current_plan: str,
        usage_metrics: Dict[str, Any],
        fit_scores: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        TOOL 4: Detect upsell opportunities
        
        Args:
            current_plan: Current plan tier
            usage_metrics: Usage data
            fit_scores: Plan fit scores
        
        Returns:
            Dictionary with upsell opportunities
        """
        logger.info(f"Detecting upsell opportunities for {current_plan}")
        
        try:
            peak_utilization = usage_metrics.get("usage_summary", {}).get("peak_month_transactions", 0)
            current_limit = self.plan_tiers.get(current_plan, {}).get("transaction_limit", 1)
            peak_utilization_pct = (peak_utilization / current_limit * 100) if current_limit > 0 else 0
            
            opportunities = []
            urgency = "low"
            
            # Check if approaching limits
            if peak_utilization_pct >= 80:
                opportunities.append({
                    "type": "capacity_warning",
                    "message": f"Approaching plan limit ({peak_utilization_pct:.1f}% utilization)",
                    "impact": "high"
                })
                urgency = "critical"
            elif peak_utilization_pct >= 70:
                opportunities.append({
                    "type": "capacity_warning",
                    "message": f"Nearing plan limit ({peak_utilization_pct:.1f}% utilization)",
                    "impact": "medium"
                })
                urgency = "high"
            
            # Check API access need
            if current_plan in ["bronze", "silver"] and peak_utilization_pct >= 50:
                opportunities.append({
                    "type": "api_access",
                    "message": "High transaction volume - consider API access",
                    "impact": "medium"
                })
            
            # Check reporting needs
            if current_plan == "bronze":
                opportunities.append({
                    "type": "reporting",
                    "message": "Upgrade to access advanced reporting features",
                    "impact": "low"
                })
            
            return {
                "status": "success",
                "current_plan": current_plan,
                "opportunities": opportunities,
                "urgency": urgency,
                "total_opportunities": len(opportunities),
                "recommendation": "immediate_upgrade" if urgency == "critical" else "consider_upgrade" if urgency == "high" else "monitor"
            }
        except Exception as e:
            logger.error(f"Upsell detection error: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def execute_upgrade(
        self,
        user_id: str,
        from_plan: str,
        to_plan: str
    ) -> Dict[str, Any]:
        """
        TOOL 5: Execute plan upgrade
        
        Args:
            user_id: Customer identifier
            from_plan: Current plan
            to_plan: New plan
        
        Returns:
            Dictionary with upgrade confirmation
        """
        logger.info(f"Executing upgrade: {from_plan} -> {to_plan} for user {user_id}")
        
        try:
            # Update plan in database
            update_query = f"""
                UPDATE `{self.project_id}.client_data.client_profiles`
                SET 
                    plan_tier = @new_plan,
                    plan_updated_at = CURRENT_TIMESTAMP(),
                    upgrade_history = ARRAY_CONCAT(upgrade_history, [STRUCT(
                        @old_plan as from_plan,
                        @new_plan as to_plan,
                        CURRENT_TIMESTAMP() as timestamp
                    )])
                WHERE user_id = @user_id
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                    bigquery.ScalarQueryParameter("new_plan", "STRING", to_plan),
                    bigquery.ScalarQueryParameter("old_plan", "STRING", from_plan),
                ]
            )
            
            self.bq_client.query(update_query, job_config=job_config).result()
            
            # Log to audit table
            audit_query = f"""
                INSERT INTO `{self.project_id}.client_data.upgrade_audit_log`
                (user_id, from_plan, to_plan, timestamp, status)
                VALUES (@user_id, @from_plan, @to_plan, CURRENT_TIMESTAMP(), 'completed')
            """
            
            self.bq_client.query(audit_query, job_config=job_config).result()
            
            new_plan_details = self.plan_tiers.get(to_plan, {})
            
            return {
                "status": "success",
                "confirmation": {
                    "upgrade_id": f"upg_{user_id}_{datetime.utcnow().timestamp()}",
                    "user_id": user_id,
                    "from_plan": from_plan,
                    "to_plan": to_plan,
                    "effective_date": datetime.utcnow().isoformat(),
                    "new_monthly_cost": new_plan_details.get("price", 0)
                }
            }
        except Exception as e:
            logger.error(f"Upgrade execution error: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def simulate_billing(
        self,
        plan_tier: str,
        months: int = 12
    ) -> Dict[str, Any]:
        """
        TOOL 6: Simulate future billing
        
        Args:
            plan_tier: Plan to simulate
            months: Number of months to project
        
        Returns:
            Dictionary with billing projection
        """
        logger.info(f"Simulating billing for {plan_tier} over {months} months")
        
        try:
            plan_details = self.plan_tiers.get(plan_tier, {})
            monthly_cost = plan_details.get("price", 0)
            
            # Create monthly breakdown
            monthly_breakdown = []
            total_cost = 0
            
            for month in range(1, months + 1):
                cost = monthly_cost
                
                # Add discount for annual commitment (if applicable)
                if months >= 12 and month >= 12:
                    discount = 0.1  # 10% discount
                    cost = cost * (1 - discount)
                
                monthly_breakdown.append({
                    "month": month,
                    "cost": cost,
                    "cumulative_cost": total_cost + cost
                })
                total_cost += cost
            
            annual_cost = monthly_cost * 12
            annual_savings_with_commitment = (monthly_cost * 0.9) * 12 if months >= 12 else 0
            
            return {
                "status": "success",
                "plan": plan_tier,
                "projection": {
                    "months": months,
                    "monthly_cost": monthly_cost,
                    "annual_cost": annual_cost,
                    "total_projection_cost": total_cost,
                    "annual_savings_with_commitment": annual_savings_with_commitment
                },
                "monthly_breakdown": monthly_breakdown
            }
        except Exception as e:
            logger.error(f"Billing simulation error: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def process_plan_request(
        self,
        user_id: str,
        current_plan: str
    ) -> Dict[str, Any]:
        """
        Main Plan Agent entry point
        
        Args:
            user_id: Customer identifier
            current_plan: Current plan tier
        
        Returns:
            Complete plan analysis with recommendations
        """
        logger.info(f"Processing plan request for user {user_id}")
        
        execution_log = []
        start_time = time.time()
        
        try:
            # Step 1: Analyze usage
            usage_result = self.analyze_usage(user_id)
            execution_log.append({
                "step": "analyze_usage",
                "status": usage_result.get("status")
            })
            usage_metrics = usage_result
            
            # Step 2: Calculate fit
            fit_result = self.calculate_fit(usage_metrics)
            execution_log.append({
                "step": "calculate_fit",
                "status": fit_result.get("status")
            })
            recommended_plan = fit_result.get("recommended_plan")
            fit_scores = fit_result.get("fit_analysis", {})
            
            # Step 3: Calculate ROI
            roi_result = self.calculate_roi(
                current_plan,
                recommended_plan,
                usage_metrics.get("usage_summary", {}).get("avg_monthly_transactions", 0)
            )
            execution_log.append({
                "step": "calculate_roi",
                "status": roi_result.get("status")
            })
            
            # Step 4: Detect upsell
            upsell_result = self.detect_upsell(current_plan, usage_metrics, fit_scores)
            execution_log.append({
                "step": "detect_upsell",
                "status": upsell_result.get("status")
            })
            
            # Step 5: Simulate billing
            billing_result = self.simulate_billing(recommended_plan)
            execution_log.append({
                "step": "simulate_billing",
                "status": billing_result.get("status")
            })
            
            elapsed_time = (time.time() - start_time) * 1000
            
            return {
                "status": "success",
                "user_id": user_id,
                "current_plan": current_plan,
                "usage_analysis": usage_result,
                "plan_fit": fit_result,
                "roi_analysis": roi_result,
                "upsell_opportunities": upsell_result,
                "billing_projection": billing_result,
                "recommendation": {
                    "recommended_plan": recommended_plan,
                    "should_upgrade": recommended_plan != current_plan,
                    "urgency": upsell_result.get("urgency", "low"),
                    "roi_percentage": roi_result.get("roi_analysis", {}).get("roi_percentage", 0)
                },
                "execution_log": execution_log,
                "total_execution_time_ms": elapsed_time
            }
        
        except Exception as e:
            logger.error(f"Plan request processing error: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "execution_log": execution_log
            }


    def get_all_clients_with_plans(self) -> Dict[str, Any]:
        """
        Retrieve all clients and their current plans from client_profiles.

        Returns:
            Dictionary with status and a list of clients with their plans.
        """
        logger.info("Retrieving all clients and their plans")
        try:
            query = f"""
                SELECT user_id, plan_tier
                FROM `{self.project_id}.client_data.client_profiles`
            """
            results = self.bq_client.query(query).result()
            clients = []
            for row in results:
                clients.append({
                    "user_id": row.user_id,
                    "plan_tier": row.plan_tier
                })
            return {
                "status": "success",
                "clients": clients,
                "total_clients": len(clients)
            }
        except Exception as e:
            logger.error(f"Error retrieving clients: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "clients": []
            }
# ==================== Exports ====================

__all__ = ['PlanAgent']
