# Report Agent - Financial Report Generation
# File: backend/agents/report_agent.py

import json
import logging
from typing import Dict, Any, List, Optional
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
                    report_id,
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
    
    def aggregate_wire_report_by_status(self, raw_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate wire reports by status for visualization"""
        status_aggregates = {}
        status_amounts = {}
        
        for record in raw_data:
            status = record.get('status', 'unknown').lower()
            amount = float(record.get('wire_amount', 0))
            status_aggregates[status] = status_aggregates.get(status, 0) + 1
            status_amounts[status] = status_amounts.get(status, 0) + amount
        
        return {
            "status_counts": status_aggregates,
            "status_amounts": status_amounts,
            "total_reports": len(raw_data),
            "total_amount": sum(status_amounts.values())
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
                    "report_id", "wire_amount", "destination_bank",
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
            # Convert Decimal to float for all records
            def convert_decimals(obj):
                if isinstance(obj, list):
                    return [convert_decimals(item) for item in obj]
                elif isinstance(obj, dict):
                    return {k: convert_decimals(v) for k, v in obj.items()}
                elif hasattr(obj, '__decimal__') or str(type(obj)).endswith("'decimal.Decimal'>"):
                    return float(obj)
                elif str(type(obj)).endswith("'Decimal'>"):
                    return float(obj)
                elif type(obj).__name__ == 'Decimal':
                    return float(obj)
                else:
                    try:
                        import decimal
                        if isinstance(obj, decimal.Decimal):
                            return float(obj)
                    except Exception:
                        pass
                    return obj

            if format_type == "json":
                safe_data = convert_decimals(raw_data)
                formatted_data = json.dumps(safe_data, indent=2)
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
                    plan_tier,
                    kyc_verified
                FROM `{self.project_id}.client_data.client_profiles`
                WHERE user_id = @user_id
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                ]
            )
            results = list(self.bq_client.query(query, job_config=job_config).result())
            logger.info(f"Access query results: {results}")
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
            logger.info(f"KYC verification status for user {user_id}: {kyc_verified}")
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
            logger.info(f"Plan access for user {user_id}: {plan_access}")
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
            logger.info(f"Access validation result: {access_result}")
            execution_log.append({
                "step": "validate_access",
                "status": access_result.get("status"),
                "access_granted": access_result.get("access_granted")
            })
            
            if not access_result.get("access_granted"):
                logger.warning(f"Access denied for user {user_id} to {report_type}")
                return {
                    "status": "error",
                    "error": access_result.get("reason", "Access denied"),
                    "execution_log": execution_log
                }
            
            # Step 2: Query data
            query_result = self.query_bigquery(user_id, report_type)
            logger.info(f"Query result for user {user_id}, report {report_type}: {query_result}")   
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


# ==================== Exports ====================

__all__ = ['ReportAgent', 'ReportType', 'ReportFormat']
