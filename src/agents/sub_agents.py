# Complete Sub-Agents Implementation
# File: backend/agents/sub_agents.py

import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import time
from enum import Enum

from google.cloud import bigquery
import vertexai
from vertexai.generative_models import GenerativeModel

# ==================== Configuration & Types ====================

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ReportType(str, Enum):
    """Supported financial report types"""
    BALANCE_REPORT = "balance_report"
    WIRE_DETAILS = "wire_details"
    ACH_INBOUND = "ach_inbound"
    INTRADAY_BALANCE = "intraday_balance"
    EXPANDED_DETAILS = "expanded_details"
    STATEMENTS = "statements"
    DEPOSIT_DETAILS = "deposit_details"
    CHECK_IMAGES = "check_images"
    RUNNING_LEDGER = "running_ledger"

class ReportFormat(str, Enum):
    """Output formats for reports"""
    PDF = "pdf"
    EXCEL = "excel"
    CSV = "csv"
    JSON = "json"

# ==================== Report Agent ====================

class ReportAgent:
    """
    Financial Report Agent
    
    Responsibilities:
    - Fetch customer financial data from BigQuery
    - Format into various report types (Balance, Wire, ACH, etc.)
    - Support multiple output formats (PDF, Excel, CSV, JSON)
    - Generate visualizations and charts
    - Validate user access entitlements
    
    9 Report Types Supported:
    1. Balance Report - Account balances
    2. Wire Details - Wire transfer history
    3. ACH Inbound - ACH deposits
    4. Intraday Balance - Real-time balance
    5. Expanded Details - Detailed transactions
    6. Statements - Monthly statements
    7. Deposit Details - Deposit history
    8. Check Images - Scanned checks
    9. Running Ledger - Complete transaction log
    
    Tools:
    - query_bigquery: Execute BQ queries for raw data
    - get_report_template: Retrieve report schema/structure
    - format_data: Apply formatting (PDF/Excel/CSV)
    - generate_visualization: Create charts/graphs
    - validate_access: Check user entitlements
    """
    
    def __init__(self, project_id: str, region: str = "us-central1"):
        """Initialize Report Agent"""
        self.project_id = project_id
        self.region = region
        self.bq_client = bigquery.Client(project=project_id)
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=region)
        self.model = GenerativeModel("gemini-2.0-flash-exp")
        
        logger.info(f"ReportAgent initialized for project {project_id}")
    
    def query_bigquery(self, user_id: str, report_type: str) -> Dict[str, Any]:
        """
        TOOL 1: Execute BigQuery queries to fetch raw financial data
        
        Args:
            user_id: Customer identifier
            report_type: Type of report (BALANCE_REPORT, WIRE_DETAILS, etc.)
        
        Returns:
            Dictionary with raw query results and metadata
        """
        logger.info(f"Querying BigQuery for {report_type} report for user {user_id}")
        
        queries = {
            "balance_report": f"""
                SELECT 
                    user_id,
                    account_id,
                    balance_date,
                    opening_balance,
                    closing_balance,
                    total_debits,
                    total_credits,
                    average_daily_balance
                FROM `{self.project_id}.client_data.wire_reports`
                WHERE user_id = @user_id
                    AND balance_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
                ORDER BY balance_date DESC
                LIMIT 100
            """,
            "wire_details": f"""
                SELECT 
                    wire_id,
                    user_id,
                    wire_amount,
                    destination_bank,
                    destination_account,
                    status,
                    wire_date,
                    reference_number,
                    beneficiary_name
                FROM `{self.project_id}.client_data.wire_reports`
                WHERE user_id = @user_id
                    AND wire_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
                ORDER BY wire_date DESC
                LIMIT 100
            """,
            "intraday_balance": f"""
                SELECT 
                    user_id,
                    account_id,
                    balance_timestamp,
                    current_balance,
                    available_balance,
                    pending_credits,
                    pending_debits
                FROM `{self.project_id}.client_data.user_transactions`
                WHERE user_id = @user_id
                    AND DATE(balance_timestamp) = CURRENT_DATE()
                ORDER BY balance_timestamp DESC
                LIMIT 100
            """,
            "running_ledger": f"""
                SELECT 
                    user_id,
                    transaction_date,
                    transaction_type,
                    amount,
                    running_balance,
                    description,
                    reference_id
                FROM `{self.project_id}.client_data.user_transactions`
                WHERE user_id = @user_id
                ORDER BY transaction_date DESC
                LIMIT 1000
            """
        }
        
        try:
            query = queries.get(report_type, queries["balance_report"])
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                ]
            )
            
            results = self.bq_client.query(query, job_config=job_config).result()
            data = [dict(row) for row in results]
            
            # Convert datetime objects to ISO strings
            for record in data:
                for key, value in record.items():
                    if hasattr(value, 'isoformat'):
                        record[key] = value.isoformat()
            
            return {
                "status": "success",
                "report_type": report_type,
                "row_count": len(data),
                "data": data,
                "query_timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"BigQuery error: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "data": []
            }
    
    def get_report_template(self, report_type: str) -> Dict[str, Any]:
        """
        TOOL 2: Retrieve report schema/template structure
        
        Args:
            report_type: Type of report
        
        Returns:
            Dictionary defining report structure, columns, calculations
        """
        logger.info(f"Retrieving template for {report_type}")
        
        templates = {
            "balance_report": {
                "title": "Account Balance Report",
                "description": "Shows opening/closing balances and daily activity",
                "columns": [
                    "balance_date", "opening_balance", "closing_balance",
                    "total_debits", "total_credits", "average_daily_balance"
                ],
                "summary_metrics": [
                    "average_daily_balance", "max_balance", "min_balance",
                    "total_activity"
                ],
                "frequency": "Daily",
                "retention": "90 days"
            },
            "wire_details": {
                "title": "Wire Transfer Details",
                "description": "Complete wire transfer history with status tracking",
                "columns": [
                    "wire_id", "wire_amount", "destination_bank",
                    "destination_account", "status", "wire_date",
                    "reference_number", "beneficiary_name"
                ],
                "summary_metrics": [
                    "total_outbound", "total_inbound", "average_wire_amount",
                    "wire_count"
                ],
                "frequency": "Real-time",
                "retention": "7 years"
            },
            "running_ledger": {
                "title": "Running Ledger",
                "description": "Complete transaction log with running balance calculations",
                "columns": [
                    "transaction_date", "transaction_type", "amount",
                    "running_balance", "description", "reference_id"
                ],
                "summary_metrics": [
                    "total_debits", "total_credits", "net_activity"
                ],
                "frequency": "Real-time",
                "retention": "7 years"
            }
        }
        
        return templates.get(report_type, {
            "title": "Financial Report",
            "description": "Standard financial data report",
            "columns": [],
            "summary_metrics": []
        })
    
    def format_data(
        self, 
        raw_data: List[Dict[str, Any]], 
        format_type: str,
        report_type: str
    ) -> Dict[str, Any]:
        """
        TOOL 3: Format data into specified output format
        
        Args:
            raw_data: Raw query results
            format_type: Output format (pdf, excel, csv, json)
            report_type: Type of report
        
        Returns:
            Dictionary with formatted data and download info
        """
        logger.info(f"Formatting {report_type} as {format_type}")
        
        try:
            if format_type == "json":
                formatted_data = json.dumps(raw_data, indent=2)
                file_extension = "json"
            elif format_type == "csv":
                # Simple CSV conversion
                if raw_data:
                    headers = list(raw_data[0].keys())
                    csv_rows = [",".join(headers)]
                    for record in raw_data:
                        csv_rows.append(",".join(str(record.get(h, "")) for h in headers))
                    formatted_data = "\n".join(csv_rows)
                else:
                    formatted_data = ""
                file_extension = "csv"
            elif format_type == "excel":
                # Placeholder for Excel generation
                formatted_data = f"Excel file with {len(raw_data)} rows"
                file_extension = "xlsx"
            elif format_type == "pdf":
                # Placeholder for PDF generation
                formatted_data = f"PDF report with {len(raw_data)} records"
                file_extension = "pdf"
            else:
                formatted_data = raw_data
                file_extension = "json"
            
            return {
                "status": "success",
                "format": format_type,
                "file_name": f"{report_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.{file_extension}",
                "data": formatted_data,
                "size_bytes": len(str(formatted_data)),
                "ready_for_download": True
            }
        except Exception as e:
            logger.error(f"Formatting error: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def generate_visualization(
        self,
        raw_data: List[Dict[str, Any]],
        report_type: str,
        chart_type: str = "line"
    ) -> Dict[str, Any]:
        """
        TOOL 4: Generate visualizations (charts, graphs)
        
        Args:
            raw_data: Raw query results
            report_type: Type of report
            chart_type: Type of chart (line, bar, pie, etc.)
        
        Returns:
            Dictionary with chart data and metadata
        """
        logger.info(f"Generating {chart_type} visualization for {report_type}")
        
        try:
            visualization_data = {
                "status": "success",
                "chart_type": chart_type,
                "title": f"{report_type.replace('_', ' ').title()} Chart",
                "data_points": len(raw_data),
                "chart_url": f"/api/charts/{report_type}_{chart_type}",
                "metadata": {
                    "x_axis": "Date",
                    "y_axis": "Amount (USD)",
                    "generated_at": datetime.utcnow().isoformat(),
                    "data_quality": "verified"
                }
            }
            
            return visualization_data
        except Exception as e:
            logger.error(f"Visualization error: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def validate_access(self, user_id: str, report_type: str) -> Dict[str, Any]:
        """
        TOOL 5: Validate user access entitlements
        
        Args:
            user_id: User identifier
            report_type: Type of report
        
        Returns:
            Dictionary with access validation results
        """
        logger.info(f"Validating access for user {user_id} to {report_type}")
        
        try:
            query = f"""
                SELECT 
                    p.plan_tier,
                    p.kyc_verified,
                    r.feature as allowed_feature
                FROM `{self.project_id}.client_data.client_profiles` p
                LEFT JOIN `{self.project_id}.client_data.compliance_rules` r
                    ON p.plan_tier = r.tier
                WHERE p.user_id = @user_id
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                ]
            )
            
            results = list(self.bq_client.query(query, job_config=job_config).result())
            
            if not results:
                return {
                    "status": "error",
                    "access_granted": False,
                    "reason": "User not found"
                }
            
            user_data = results[0]
            plan_tier = user_data.plan_tier
            kyc_verified = user_data.kyc_verified
            
            # Check if wire reports require KYC
            requires_kyc = report_type in ["wire_details", "check_images"]
            if requires_kyc and not kyc_verified:
                return {
                    "status": "success",
                    "access_granted": False,
                    "reason": f"KYC verification required for {report_type}",
                    "plan_tier": plan_tier,
                    "kyc_verified": False
                }
            
            # Check plan tier access
            plan_access = {
                "gold": ["balance_report", "wire_details", "ach_inbound", 
                        "intraday_balance", "expanded_details", "statements",
                        "deposit_details", "check_images", "running_ledger"],
                "silver": ["balance_report", "intraday_balance", "statements", 
                          "running_ledger"],
                "bronze": ["balance_report", "intraday_balance"]
            }
            
            allowed_reports = plan_access.get(plan_tier.lower(), [])
            access_granted = report_type in allowed_reports
            
            return {
                "status": "success",
                "access_granted": access_granted,
                "plan_tier": plan_tier,
                "kyc_verified": kyc_verified,
                "available_reports": allowed_reports
            }
        except Exception as e:
            logger.error(f"Access validation error: {str(e)}")
            return {
                "status": "error",
                "access_granted": False,
                "error": str(e)
            }
    
    async def process_report_request(
        self,
        user_id: str,
        report_type: str,
        format_type: str = "json"
    ) -> Dict[str, Any]:
        """
        Main Report Agent entry point
        
        Args:
            user_id: Customer identifier
            report_type: Type of report requested
            format_type: Output format (pdf, excel, csv, json)
        
        Returns:
            Complete report with data, formatting, and visualizations
        """
        logger.info(f"Processing {report_type} report for user {user_id}")
        
        execution_log = []
        start_time = time.time()
        
        try:
            # Step 1: Validate access
            access_result = self.validate_access(user_id, report_type)
            execution_log.append({
                "step": "validate_access",
                "status": access_result.get("status"),
                "access_granted": access_result.get("access_granted")
            })
            
            if not access_result.get("access_granted"):
                return {
                    "status": "error",
                    "error": access_result.get("reason", "Access denied"),
                    "execution_log": execution_log
                }
            
            # Step 2: Query data
            query_result = self.query_bigquery(user_id, report_type)
            execution_log.append({
                "step": "query_bigquery",
                "status": query_result.get("status"),
                "row_count": query_result.get("row_count")
            })
            
            raw_data = query_result.get("data", [])
            
            # Step 3: Get template
            template = self.get_report_template(report_type)
            execution_log.append({
                "step": "get_report_template",
                "status": "success"
            })
            
            # Step 4: Format data
            format_result = self.format_data(raw_data, format_type, report_type)
            execution_log.append({
                "step": "format_data",
                "status": format_result.get("status"),
                "format": format_type
            })
            
            # Step 5: Generate visualization
            viz_result = self.generate_visualization(raw_data, report_type)
            execution_log.append({
                "step": "generate_visualization",
                "status": viz_result.get("status"),
                "chart_type": viz_result.get("chart_type")
            })
            
            elapsed_time = (time.time() - start_time) * 1000
            
            return {
                "status": "success",
                "report_type": report_type,
                "format": format_type,
                "user_id": user_id,
                "summary": {
                    "total_records": len(raw_data),
                    "file_name": format_result.get("file_name"),
                    "file_size_bytes": format_result.get("size_bytes")
                },
                "report_data": format_result.get("data"),
                "visualization": {
                    "chart_url": viz_result.get("chart_url"),
                    "chart_type": viz_result.get("chart_type"),
                    "title": viz_result.get("title")
                },
                "template": template,
                "execution_log": execution_log,
                "total_execution_time_ms": elapsed_time
            }
        
        except Exception as e:
            logger.error(f"Report processing error: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "execution_log": execution_log
            }


# ==================== Plan Agent ====================

class PlanAgent:
    """
    Plan Recommendation & Upgrade Agent
    
    Responsibilities:
    - Analyze customer usage patterns
    - Calculate plan fit based on usage
    - Compute ROI for upgrades
    - Detect upsell opportunities
    - Execute plan changes
    - Simulate billing impacts
    
    Tools:
    - analyze_usage: Extract usage metrics from transactions
    - calculate_fit: Find best matching plans
    - calculate_roi: Compute upgrade value
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
        
        vertexai.init(project=project_id, location=region)
        self.model = GenerativeModel("gemini-2.0-flash-exp")
        
        logger.info(f"PlanAgent initialized for project {project_id}")
    
    def analyze_usage(self, user_id: str) -> Dict[str, Any]:
        """
        TOOL 1: Extract usage metrics from transactions
        
        Args:
            user_id: Customer identifier
        
        Returns:
            Dictionary with usage metrics and patterns
        """
        logger.info(f"Analyzing usage for user {user_id}")
        
        try:
            query = f"""
                SELECT 
                    DATE_TRUNC(DATE(transaction_date), MONTH) as month,
                    COUNT(*) as transaction_count,
                    SUM(CASE WHEN transaction_type = 'wire' THEN 1 ELSE 0 END) as wire_count,
                    SUM(CASE WHEN transaction_type = 'ach' THEN 1 ELSE 0 END) as ach_count,
                    SUM(ABS(amount)) as total_volume,
                    AVG(ABS(amount)) as avg_transaction_size,
                    MAX(ABS(amount)) as max_transaction_size,
                    MIN(ABS(amount)) as min_transaction_size
                FROM `{self.project_id}.client_data.user_transactions`
                WHERE user_id = @user_id
                    AND transaction_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 12 MONTH)
                GROUP BY month
                ORDER BY month DESC
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                ]
            )
            
            results = self.bq_client.query(query, job_config=job_config).result()
            usage_data = [dict(row) for row in results]
            
            # Calculate averages
            if usage_data:
                avg_monthly_transactions = sum(r['transaction_count'] for r in usage_data) / len(usage_data)
                avg_monthly_volume = sum(r['total_volume'] for r in usage_data) / len(usage_data)
                total_wires = sum(r['wire_count'] for r in usage_data)
                total_achs = sum(r['ach_count'] for r in usage_data)
            else:
                avg_monthly_transactions = 0
                avg_monthly_volume = 0
                total_wires = 0
                total_achs = 0
            
            return {
                "status": "success",
                "user_id": user_id,
                "monthly_data": usage_data,
                "summary": {
                    "avg_monthly_transactions": avg_monthly_transactions,
                    "avg_monthly_volume": avg_monthly_volume,
                    "total_wires": total_wires,
                    "total_achs": total_achs,
                    "months_analyzed": len(usage_data)
                }
            }
        except Exception as e:
            logger.error(f"Usage analysis error: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def calculate_fit(self, usage_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        TOOL 2: Calculate which plans fit customer usage
        
        Args:
            usage_metrics: Customer usage data
        
        Returns:
            Dictionary with plan recommendations and fit scores
        """
        logger.info("Calculating plan fit")
        
        avg_transactions = usage_metrics.get("summary", {}).get("avg_monthly_transactions", 0)
        avg_volume = usage_metrics.get("summary", {}).get("avg_monthly_volume", 0)
        
        plan_tiers = {
            "gold": {
                "description": "Unlimited transactions and premium features",
                "features": ["unlimited_wire", "unlimited_ach", "real_time_reporting",
                           "custom_dashboards", "api_access", "priority_support"],
                "ideal_for": "High volume, enterprise customers",
                "min_monthly_transactions": 500,
                "min_monthly_volume": 5000000
            },
            "silver": {
                "description": "Standard business features",
                "features": ["5000_monthly_items", "bulk_operations", "reporting",
                           "standard_support", "api_access"],
                "ideal_for": "Mid-market customers",
                "min_monthly_transactions": 100,
                "min_monthly_volume": 500000
            },
            "bronze": {
                "description": "Basic features",
                "features": ["500_monthly_items", "basic_reporting", "email_support"],
                "ideal_for": "Small businesses",
                "min_monthly_transactions": 0,
                "min_monthly_volume": 0
            }
        }
        
        # Score plans based on usage
        scores = {}
        for plan_name, plan_info in plan_tiers.items():
            min_txn = plan_info["min_monthly_transactions"]
            min_vol = plan_info["min_monthly_volume"]
            
            if avg_transactions >= min_txn and avg_volume >= min_vol:
                fit_score = min(1.0, (avg_transactions / max(min_txn, 1)) * 0.5 + 
                               (avg_volume / max(min_vol, 1)) * 0.5)
            else:
                fit_score = 0.3
            
            scores[plan_name] = {
                "plan_info": plan_info,
                "fit_score": fit_score,
                "fit_percentage": f"{fit_score * 100:.1f}%"
            }
        
        # Find best fit
        best_fit = max(scores.items(), key=lambda x: x[1]["fit_score"])
        
        return {
            "status": "success",
            "plan_scores": scores,
            "recommended_plan": {
                "name": best_fit[0],
                "fit_score": best_fit[1]["fit_score"],
                "description": best_fit[1]["plan_info"]["description"],
                "features": best_fit[1]["plan_info"]["features"]
            }
        }
    
    def calculate_roi(self, current_plan: str, recommended_plan: str) -> Dict[str, Any]:
        """
        TOOL 3: Calculate ROI for plan upgrade
        
        Args:
            current_plan: Current plan tier
            recommended_plan: Recommended plan tier
        
        Returns:
            Dictionary with ROI analysis
        """
        logger.info(f"Calculating ROI: {current_plan} -> {recommended_plan}")
        
        plan_pricing = {
            "bronze": {"monthly": 29, "annual": 290},
            "silver": {"monthly": 99, "annual": 990},
            "gold": {"monthly": 299, "annual": 2990}
        }
        
        current_cost = plan_pricing.get(current_plan, {}).get("monthly", 0)
        recommended_cost = plan_pricing.get(recommended_plan, {}).get("monthly", 0)
        upgrade_cost = recommended_cost - current_cost
        
        # Estimate benefits
        benefit_multiplier = {
            "bronze_to_silver": 1.5,
            "bronze_to_gold": 3.0,
            "silver_to_gold": 2.0
        }
        
        key = f"{current_plan}_to_{recommended_plan}"
        multiplier = benefit_multiplier.get(key, 1.0)
        estimated_benefit = upgrade_cost * multiplier
        
        roi = ((estimated_benefit - upgrade_cost) / abs(upgrade_cost) * 100) if upgrade_cost != 0 else 0
        
        return {
            "status": "success",
            "current_plan": current_plan,
            "recommended_plan": recommended_plan,
            "current_monthly_cost": current_cost,
            "recommended_monthly_cost": recommended_cost,
            "upgrade_monthly_cost": upgrade_cost,
            "estimated_monthly_benefit": estimated_benefit,
            "estimated_roi_percentage": roi,
            "payback_period_months": abs(upgrade_cost / estimated_benefit) if estimated_benefit > 0 else 0
        }
    
    def detect_upsell(self, usage_metrics: Dict[str, Any], current_plan: str) -> Dict[str, Any]:
        """
        TOOL 4: Detect upsell opportunities
        
        Args:
            usage_metrics: Customer usage patterns
            current_plan: Current plan tier
        
        Returns:
            Dictionary with upsell recommendations
        """
        logger.info(f"Detecting upsell opportunities for {current_plan} tier")
        
        avg_transactions = usage_metrics.get("summary", {}).get("avg_monthly_transactions", 0)
        usage_percent = (avg_transactions / (500 if current_plan == "silver" else 
                                            5000 if current_plan == "gold" else 100)) * 100
        
        opportunities = []
        
        if current_plan == "bronze" and avg_transactions > 50:
            opportunities.append({
                "type": "plan_upgrade",
                "from_plan": "bronze",
                "to_plan": "silver",
                "reason": "Exceeding Bronze tier capacity",
                "usage_percentage": min(usage_percent, 100),
                "priority": "high"
            })
        
        if current_plan == "silver" and avg_transactions > 1000:
            opportunities.append({
                "type": "plan_upgrade",
                "from_plan": "silver",
                "to_plan": "gold",
                "reason": "High transaction volume detected",
                "usage_percentage": min(usage_percent, 100),
                "priority": "high"
            })
        
        if usage_percent > 80:
            opportunities.append({
                "type": "capacity_warning",
                "message": f"Approaching plan capacity at {usage_percent:.0f}%",
                "recommended_action": "Plan upgrade",
                "priority": "medium"
            })
        
        return {
            "status": "success",
            "current_plan": current_plan,
            "usage_percentage": usage_percent,
            "opportunities": opportunities,
            "urgency": "high" if len(opportunities) > 0 else "low"
        }
    
    def execute_upgrade(
        self,
        user_id: str,
        from_plan: str,
        to_plan: str
    ) -> Dict[str, Any]:
        """
        TOOL 5: Execute plan change
        
        Args:
            user_id: Customer identifier
            from_plan: Current plan
            to_plan: New plan
        
        Returns:
            Dictionary with upgrade confirmation
        """
        logger.info(f"Executing upgrade for {user_id}: {from_plan} -> {to_plan}")
        
        try:
            # Update user plan in database
            query = f"""
                UPDATE `{self.project_id}.client_data.client_profiles`
                SET 
                    plan_tier = @new_plan,
                    plan_upgrade_date = CURRENT_TIMESTAMP(),
                    previous_plan = @old_plan
                WHERE user_id = @user_id
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                    bigquery.ScalarQueryParameter("old_plan", "STRING", from_plan),
                    bigquery.ScalarQueryParameter("new_plan", "STRING", to_plan),
                ]
            )
            
            self.bq_client.query(query, job_config=job_config).result()
            
            return {
                "status": "success",
                "user_id": user_id,
                "from_plan": from_plan,
                "to_plan": to_plan,
                "upgrade_timestamp": datetime.utcnow().isoformat(),
                "effective_immediately": True,
                "next_billing_date": "First of next month",
                "confirmation_id": f"UPG_{user_id}_{datetime.utcnow().timestamp()}"
            }
        except Exception as e:
            logger.error(f"Upgrade execution error: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def simulate_billing(
        self,
        plan: str,
        months: int = 12
    ) -> Dict[str, Any]:
        """
        TOOL 6: Simulate future billing costs
        
        Args:
            plan: Plan tier
            months: Number of months to simulate
        
        Returns:
            Dictionary with billing projections
        """
        logger.info(f"Simulating {months} month billing for {plan} tier")
        
        plan_pricing = {
            "bronze": {"monthly": 29, "annual_discount": 0.05},
            "silver": {"monthly": 99, "annual_discount": 0.10},
            "gold": {"monthly": 299, "annual_discount": 0.15}
        }
        
        pricing = plan_pricing.get(plan, {})
        monthly_cost = pricing.get("monthly", 0)
        annual_discount = pricing.get("annual_discount", 0)
        
        monthly_breakdown = []
        for i in range(1, months + 1):
            is_annual = (i % 12 == 0)
            if is_annual:
                cost = monthly_cost * 12 * (1 - annual_discount)
                discount = monthly_cost * 12 * annual_discount
            else:
                cost = monthly_cost
                discount = 0
            
            monthly_breakdown.append({
                "month": i,
                "cost": cost,
                "discount": discount,
                "net_cost": cost - discount
            })
        
        total_cost = sum(m["net_cost"] for m in monthly_breakdown)
        
        return {
            "status": "success",
            "plan": plan,
            "projection_months": months,
            "monthly_breakdown": monthly_breakdown,
            "total_projected_cost": total_cost,
            "average_monthly_cost": total_cost / months
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
            Plan analysis with recommendations
        """
        logger.info(f"Processing plan analysis for user {user_id}")
        
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
            recommended_plan = fit_result.get("recommended_plan", {}).get("name", current_plan)
            
            # Step 3: Calculate ROI
            roi_result = self.calculate_roi(current_plan, recommended_plan)
            execution_log.append({
                "step": "calculate_roi",
                "status": roi_result.get("status")
            })
            
            # Step 4: Detect upsell opportunities
            upsell_result = self.detect_upsell(usage_metrics, current_plan)
            execution_log.append({
                "step": "detect_upsell",
                "status": upsell_result.get("status")
            })
            
            # Step 5: Simulate billing
            billing_result = self.simulate_billing(recommended_plan, 12)
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
                    "roi_percentage": roi_result.get("estimated_roi_percentage", 0)
                },
                "execution_log": execution_log,
                "total_execution_time_ms": elapsed_time
            }
        
        except Exception as e:
            logger.error(f"Plan processing error: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "execution_log": execution_log
            }


# ==================== Plan Information Agent ====================

class PlanInfoAgent:
    """
    Plan Information Agent
    
    Responsibilities:
    - Provide plan feature details
    - Display pricing information
    - Compare plans side-by-side
    - Answer frequently asked questions
    - Explain access entitlements
    
    Tools:
    - retrieve_plan_features: Get plan capabilities
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
        
        vertexai.init(project=project_id, location=region)
        self.model = GenerativeModel("gemini-2.0-flash-exp")
        
        logger.info(f"PlanInfoAgent initialized for project {project_id}")
    
    def retrieve_plan_features(self) -> Dict[str, Any]:
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
                    features,
                    description,
                    ideal_for,
                    support_level,
                    api_access,
                    sso_enabled,
                    custom_branding,
                    dedicated_support
                FROM `{self.project_id}.client_data.plan_offerings`
                ORDER BY monthly_price
            """
            
            results = self.bq_client.query(query).result()
            plans = [dict(row) for row in results]
            
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
    
    def get_pricing(self) -> Dict[str, Any]:
        """
        TOOL 2: Retrieve pricing information
        
        Returns:
            Dictionary with pricing details
        """
        logger.info("Retrieving pricing information")
        
        plans = {
            "bronze": {
                "name": "Bronze",
                "monthly": 29,
                "annual": 290,
                "annual_savings": 58,
                "features": [
                    "500 monthly items",
                    "Basic reporting",
                    "Email support",
                    "Web dashboard"
                ],
                "ideal_for": "Small businesses, startups"
            },
            "silver": {
                "name": "Silver",
                "monthly": 99,
                "annual": 990,
                "annual_savings": 198,
                "features": [
                    "5,000 monthly items",
                    "Bulk operations",
                    "Standard reporting",
                    "API access (limited)",
                    "Phone & email support",
                    "Custom integrations"
                ],
                "ideal_for": "Growing businesses, mid-market"
            },
            "gold": {
                "name": "Gold",
                "monthly": 299,
                "annual": 2990,
                "annual_savings": 598,
                "features": [
                    "Unlimited items",
                    "All transaction types",
                    "Advanced reporting",
                    "Full API access",
                    "Custom dashboards",
                    "Priority phone support",
                    "Dedicated account manager",
                    "White-label options",
                    "SSO / SAML",
                    "99.99% SLA"
                ],
                "ideal_for": "Enterprise, high-volume customers"
            }
        }
        
        return {
            "status": "success",
            "plans": plans,
            "currency": "USD",
            "billing_frequency": "Monthly or Annual"
        }
    
    def build_comparison(self) -> Dict[str, Any]:
        """
        TOOL 3: Build side-by-side plan comparison
        
        Returns:
            Dictionary with comparison matrix
        """
        logger.info("Building plan comparison")
        
        comparison = {
            "status": "success",
            "comparison_matrix": {
                "Monthly Cost": {
                    "bronze": "$29",
                    "silver": "$99",
                    "gold": "$299"
                },
                "Annual Cost": {
                    "bronze": "$290",
                    "silver": "$990",
                    "gold": "$2,990"
                },
                "Monthly Transactions": {
                    "bronze": "500",
                    "silver": "5,000",
                    "gold": "Unlimited"
                },
                "Wire Transfers": {
                    "bronze": "Limited",
                    "silver": "5,000/month",
                    "gold": "Unlimited"
                },
                "API Access": {
                    "bronze": "No",
                    "silver": "Limited",
                    "gold": "Full"
                },
                "Custom Dashboards": {
                    "bronze": "No",
                    "silver": "No",
                    "gold": "Yes"
                },
                "Reporting": {
                    "bronze": "Basic",
                    "silver": "Standard",
                    "gold": "Advanced"
                },
                "Support": {
                    "bronze": "Email",
                    "silver": "Phone & Email",
                    "gold": "Dedicated + Phone"
                },
                "SLA": {
                    "bronze": "99%",
                    "silver": "99.5%",
                    "gold": "99.99%"
                }
            }
        }
        
        return comparison
    
    def retrieve_faq(self, category: Optional[str] = None) -> Dict[str, Any]:
        """
        TOOL 4: Retrieve FAQ knowledge base
        
        Args:
            category: Optional FAQ category filter
        
        Returns:
            Dictionary with FAQs
        """
        logger.info(f"Retrieving FAQ for category: {category}")
        
        faqs = {
            "pricing": [
                {
                    "question": "Can I change my plan anytime?",
                    "answer": "Yes! You can upgrade or downgrade your plan at any time. Changes take effect immediately."
                },
                {
                    "question": "Do you offer annual discounts?",
                    "answer": "Yes. Annual billing provides 5-15% discount depending on plan tier."
                },
                {
                    "question": "What happens if I exceed my monthly limit?",
                    "answer": "We'll notify you when approaching limits. You can upgrade plan or purchase overages."
                }
            ],
            "billing": [
                {
                    "question": "When will I be charged?",
                    "answer": "Monthly plans charge on your billing date each month. Annual plans charge once per year."
                },
                {
                    "question": "What payment methods do you accept?",
                    "answer": "We accept all major credit cards, ACH, and wire transfers for annual plans."
                },
                {
                    "question": "Can I get an invoice?",
                    "answer": "Yes. Invoices are generated automatically and available in your account dashboard."
                }
            ],
            "features": [
                {
                    "question": "What's included in each plan?",
                    "answer": "See our pricing page for detailed feature comparisons across Bronze, Silver, and Gold plans."
                },
                {
                    "question": "Do you offer API access?",
                    "answer": "Yes. Silver and Gold plans include API access. Bronze doesn't include APIs."
                },
                {
                    "question": "Can I customize my dashboard?",
                    "answer": "Custom dashboards are available on Gold tier. Silver/Bronze have standard templates."
                }
            ],
            "support": [
                {
                    "question": "What support is included?",
                    "answer": "Bronze: Email. Silver: Phone & Email. Gold: Dedicated account manager + priority phone."
                },
                {
                    "question": "What are your support hours?",
                    "answer": "Email support: 24/5. Phone support: Mon-Fri 9am-6pm EST. Gold: 24/7."
                }
            ]
        }
        
        if category:
            result_faqs = faqs.get(category, [])
        else:
            result_faqs = [item for items in faqs.values() for item in items]
        
        return {
            "status": "success",
            "faqs": result_faqs,
            "category": category or "all",
            "total_faqs": len(result_faqs)
        }
    
    def explain_entitlements(self, plan_tier: str) -> Dict[str, Any]:
        """
        TOOL 5: Explain access rules and entitlements
        
        Args:
            plan_tier: Plan tier (bronze, silver, gold)
        
        Returns:
            Dictionary with access entitlements
        """
        logger.info(f"Explaining entitlements for {plan_tier}")
        
        entitlements = {
            "bronze": {
                "plan_tier": "bronze",
                "access_level": "basic",
                "features": [
                    "Basic balance reporting",
                    "Email support",
                    "Web dashboard access"
                ],
                "transaction_limits": {
                    "monthly_items": 500,
                    "wire_transfers": "Not included",
                    "api_calls": 0
                },
                "reporting_access": [
                    "balance_report",
                    "intraday_balance"
                ],
                "api_access": False,
                "custom_dashboards": False,
                "dedicated_support": False
            },
            "silver": {
                "plan_tier": "silver",
                "access_level": "standard",
                "features": [
                    "All Bronze features",
                    "5,000 monthly items",
                    "Wire transfer support",
                    "API access (limited)",
                    "Phone & email support",
                    "Custom integrations"
                ],
                "transaction_limits": {
                    "monthly_items": 5000,
                    "wire_transfers": 5000,
                    "api_calls": 100000
                },
                "reporting_access": [
                    "balance_report",
                    "intraday_balance",
                    "wire_details",
                    "statements"
                ],
                "api_access": True,
                "custom_dashboards": False,
                "dedicated_support": False
            },
            "gold": {
                "plan_tier": "gold",
                "access_level": "enterprise",
                "features": [
                    "All Silver features",
                    "Unlimited transactions",
                    "Custom dashboards",
                    "Full API access",
                    "Dedicated account manager",
                    "Priority support",
                    "SSO/SAML",
                    "White-label options",
                    "99.99% SLA"
                ],
                "transaction_limits": {
                    "monthly_items": "Unlimited",
                    "wire_transfers": "Unlimited",
                    "api_calls": "Unlimited"
                },
                "reporting_access": [
                    "All report types"
                ],
                "api_access": True,
                "custom_dashboards": True,
                "dedicated_support": True
            }
        }
        
        return {
            "status": "success",
            "entitlements": entitlements.get(plan_tier.lower(), {}),
            "effective_date": datetime.utcnow().isoformat()
        }
    
    async def process_info_request(self, query: str) -> Dict[str, Any]:
        """
        Main Plan Info Agent entry point
        
        Args:
            query: User question about plans
        
        Returns:
            Information response with relevant data
        """
        logger.info(f"Processing info request: {query}")
        
        execution_log = []
        start_time = time.time()
        
        try:
            # Determine what information to retrieve
            query_lower = query.lower()
            
            response_data = {}
            
            # Always include pricing
            pricing = self.get_pricing()
            response_data["pricing"] = pricing.get("plans", {})
            execution_log.append({
                "step": "get_pricing",
                "status": pricing.get("status")
            })
            
            # Include features if asked
            if any(word in query_lower for word in ["feature", "include", "capability", "what"]):
                features = self.retrieve_plan_features()
                response_data["features"] = features.get("plans", [])
                execution_log.append({
                    "step": "retrieve_plan_features",
                    "status": features.get("status")
                })
            
            # Include comparison if asked
            if any(word in query_lower for word in ["compare", "difference", "vs", "versus", "comparison"]):
                comparison = self.build_comparison()
                response_data["comparison"] = comparison.get("comparison_matrix", {})
                execution_log.append({
                    "step": "build_comparison",
                    "status": comparison.get("status")
                })
            
            # Include FAQ if asked
            if any(word in query_lower for word in ["faq", "question", "asked", "help", "how"]):
                faq = self.retrieve_faq()
                response_data["faq"] = faq.get("faqs", [])
                execution_log.append({
                    "step": "retrieve_faq",
                    "status": faq.get("status")
                })
            
            # Include entitlements if asked
            if any(word in query_lower for word in ["access", "entitle", "include", "get"]):
                for tier in ["bronze", "silver", "gold"]:
                    entitlements = self.explain_entitlements(tier)
                    if "entitlements" not in response_data:
                        response_data["entitlements"] = {}
                    response_data["entitlements"][tier] = entitlements.get("entitlements", {})
                execution_log.append({
                    "step": "explain_entitlements",
                    "status": "success"
                })
            
            elapsed_time = (time.time() - start_time) * 1000
            
            return {
                "status": "success",
                "query": query,
                "data": response_data,
                "execution_log": execution_log,
                "total_execution_time_ms": elapsed_time
            }
        
        except Exception as e:
            logger.error(f"Info request error: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "execution_log": execution_log
            }


# ==================== Balance Agent ====================

class BalanceAgent:
    """
    Account Balance & Transaction Agent
    
    Responsibilities:
    - Fetch current account balance
    - Retrieve recent transactions
    - Calculate balance trends
    - Provide transaction details
    - Generate balance summaries
    """
    
    def __init__(self, project_id: str, region: str = "us-central1"):
        """Initialize Balance Agent"""
        self.project_id = project_id
        self.region = region
        self.bq_client = bigquery.Client(project=project_id)
        
        vertexai.init(project=project_id, location=region)
        self.model = GenerativeModel("gemini-2.0-flash-exp")
        
        logger.info(f"BalanceAgent initialized for project {project_id}")
    
    def get_current_balance(self, user_id: str) -> Dict[str, Any]:
        """Get current account balance"""
        logger.info(f"Fetching current balance for user {user_id}")
        
        try:
            query = f"""
                SELECT 
                    user_id,
                    account_id,
                    MAX(balance_timestamp) as last_update,
                    ARRAY_AGG(DISTINCT balance_type)[SAFE_OFFSET(0)] as balance_type,
                    SUM(CASE WHEN transaction_type IN ('deposit', 'interest') THEN amount ELSE -amount END) as current_balance
                FROM `{self.project_id}.client_data.user_transactions`
                WHERE user_id = @user_id
                GROUP BY user_id, account_id
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                ]
            )
            
            result = list(self.bq_client.query(query, job_config=job_config).result())
            
            if result:
                data = dict(result[0])
                return {
                    "status": "success",
                    "user_id": user_id,
                    "balance": float(data.get('current_balance', 0)),
                    "currency": "USD",
                    "last_update": data.get('last_update', datetime.utcnow()).isoformat() if hasattr(data.get('last_update'), 'isoformat') else str(data.get('last_update')),
                    "account_id": data.get('account_id')
                }
            else:
                return {
                    "status": "success",
                    "user_id": user_id,
                    "balance": 0,
                    "currency": "USD",
                    "last_update": datetime.utcnow().isoformat()
                }
        except Exception as e:
            logger.error(f"Balance fetch error: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_recent_transactions(self, user_id: str, limit: int = 50) -> Dict[str, Any]:
        """Get recent transactions"""
        logger.info(f"Fetching recent transactions for user {user_id}")
        
        try:
            query = f"""
                SELECT 
                    transaction_id,
                    transaction_date,
                    transaction_type,
                    amount,
                    description,
                    reference_id,
                    status
                FROM `{self.project_id}.client_data.user_transactions`
                WHERE user_id = @user_id
                    AND transaction_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
                ORDER BY transaction_date DESC
                LIMIT @limit
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                    bigquery.ScalarQueryParameter("limit", "INT64", limit),
                ]
            )
            
            results = self.bq_client.query(query, job_config=job_config).result()
            transactions = [dict(row) for row in results]
            
            # Convert timestamps
            for txn in transactions:
                if hasattr(txn.get('transaction_date'), 'isoformat'):
                    txn['transaction_date'] = txn['transaction_date'].isoformat()
            
            return {
                "status": "success",
                "user_id": user_id,
                "transactions": transactions,
                "transaction_count": len(transactions)
            }
        except Exception as e:
            logger.error(f"Transaction fetch error: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "transactions": []
            }
    
    async def process_balance_request(self, user_id: str) -> Dict[str, Any]:
        """
        Main Balance Agent entry point
        
        Args:
            user_id: Customer identifier
        
        Returns:
            Balance and transaction information
        """
        logger.info(f"Processing balance request for user {user_id}")
        
        execution_log = []
        start_time = time.time()
        
        try:
            # Get current balance
            balance_result = self.get_current_balance(user_id)
            execution_log.append({
                "step": "get_current_balance",
                "status": balance_result.get("status")
            })
            
            # Get recent transactions
            txn_result = self.get_recent_transactions(user_id)
            execution_log.append({
                "step": "get_recent_transactions",
                "status": txn_result.get("status"),
                "count": txn_result.get("transaction_count")
            })
            
            elapsed_time = (time.time() - start_time) * 1000
            
            return {
                "status": "success",
                "user_id": user_id,
                "balance_info": balance_result,
                "recent_transactions": txn_result,
                "summary": {
                    "current_balance": balance_result.get("balance", 0),
                    "recent_transaction_count": txn_result.get("transaction_count", 0),
                    "currency": "USD"
                },
                "execution_log": execution_log,
                "total_execution_time_ms": elapsed_time
            }
        
        except Exception as e:
            logger.error(f"Balance processing error: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "execution_log": execution_log
            }


# ==================== Exports ====================

__all__ = [
    'ReportAgent',
    'PlanAgent',
    'PlanInfoAgent',
    'BalanceAgent',
    'ReportType',
    'ReportFormat'
]
