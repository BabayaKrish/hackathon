# Sub-Agents Package Initialization
# File: backend/agents/__init__.py

"""
Multi-Agent Financial Services Platform
4 Specialized Sub-Agents

Each agent is responsible for a specific domain:
- ReportAgent: Financial report generation (9 report types)
- PlanAgent: Usage analysis and plan recommendations
- PlanInfoAgent: Plan details, pricing, and support
- BalanceAgent: Account balance and transactions
"""

from .report_agent import ReportAgent, ReportType, ReportFormat
from .plan_agent import PlanAgent
from .plan_info_agent import PlanInfoAgent
from .balance_agent import BalanceAgent
from .root_agent import RootAgent,create_root_agent

__all__ = [
    'ReportAgent',
    'create_root_agent',
    'RootAgent',
    'ReportType',
    'ReportFormat',
    'PlanAgent',
    'PlanInfoAgent',
    'BalanceAgent',
]

__version__ = '1.0.0'
__author__ = 'Client Data Access Team'

# ==================== Module Documentation ====================

"""
AGENTS OVERVIEW:

1. REPORT AGENT (report_agent.py)
   Purpose: Generate, format, and visualize financial reports
   Tools: 5 (query_bigquery, get_report_template, format_data, 
              generate_visualization, validate_access)
   Reports: 9 types (Balance, Wire, ACH, Intraday, Expanded, 
                     Statements, Deposits, Checks, Ledger)
   Formats: PDF, Excel, CSV, JSON
   
   Example:
       agent = ReportAgent(project_id="my-project")
       result = await agent.process_report_request(
           user_id="user_001",
           report_type="wire_details",
           format_type="pdf"
       )

2. PLAN AGENT (plan_agent.py)
   Purpose: Analyze usage and recommend plan upgrades
   Tools: 6 (analyze_usage, calculate_fit, calculate_roi,
             detect_upsell, execute_upgrade, simulate_billing)
   Tiers: Bronze ($29), Silver ($99), Gold ($299)
   
   Example:
       agent = PlanAgent(project_id="my-project")
       result = await agent.process_plan_request(
           user_id="user_001",
           current_plan="silver"
       )

3. PLAN INFO AGENT (plan_info_agent.py)
   Purpose: Provide plan details and answer questions
   Tools: 5 (retrieve_plan_features, get_pricing, build_comparison,
             retrieve_faq, explain_entitlements)
   
   Example:
       agent = PlanInfoAgent(project_id="my-project")
       result = await agent.process_info_request(
           query="What's in the Gold plan?"
       )

4. BALANCE AGENT (balance_agent.py)
   Purpose: Get account balance and recent transactions
   Tools: 2 + helper (get_current_balance, get_recent_transactions,
                       get_balance_trends)
   
   Example:
       agent = BalanceAgent(project_id="my-project")
       result = await agent.process_balance_request(
           user_id="user_001"
       )

USAGE IN FastAPI:

from agents import ReportAgent, PlanAgent, PlanInfoAgent, BalanceAgent

PROJECT_ID = "your-gcp-project"
report_agent = ReportAgent(project_id=PROJECT_ID)
plan_agent = PlanAgent(project_id=PROJECT_ID)
plan_info_agent = PlanInfoAgent(project_id=PROJECT_ID)
balance_agent = BalanceAgent(project_id=PROJECT_ID)

@app.post("/report")
async def generate_report(request: ReportRequest):
    return await report_agent.process_report_request(
        user_id=request.user_id,
        report_type=request.report_type,
        format_type=request.format_type
    )

@app.post("/plan")
async def analyze_plan(request: PlanRequest):
    return await plan_agent.process_plan_request(
        user_id=request.user_id,
        current_plan=request.current_plan
    )

@app.post("/plan-info")
async def get_plan_info(request: PlanInfoRequest):
    return await plan_info_agent.process_info_request(
        query=request.query
    )

@app.post("/balance")
async def get_balance(request: BalanceRequest):
    return await balance_agent.process_balance_request(
        user_id=request.user_id
    )
"""
