# FastAPI Backend - Multi-Agent System Integration
# File: backend/main.py or backend/fastapi-backend.py

import json
import logging
from datetime import datetime
from typing import Dict, Any
import time

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import all agents
from agents import (
    ReportAgent,
    RootAgent,
    create_root_agent,
    ReportType,
    ReportFormat,
    PlanAgent,
    PlanInfoAgent,
    BalanceAgent
)

# ==================== Logging Configuration ====================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== FastAPI App Configuration ====================

app = FastAPI(
    title="Client Data Access Agent",
    description="Multi-agent financial services platform with 4 specialized agents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ==================== CORS Configuration ====================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Project Configuration ====================

PROJECT_ID = "ccibt-hack25ww7-743"  # SET THIS TO YOUR GCP PROJECT ID
REGION = "us-central1"

# ==================== Initialize All Agents ====================

logger.info("Initializing agents...")

try:
    report_agent = ReportAgent(project_id=PROJECT_ID, region=REGION)
    logger.info("✅ ReportAgent initialized")
except Exception as e:
    logger.error(f"❌ ReportAgent initialization failed: {str(e)}")
    report_agent = None

try:
    plan_agent = PlanAgent(project_id=PROJECT_ID, region=REGION)
    logger.info("✅ PlanAgent initialized")
except Exception as e:
    logger.error(f"❌ PlanAgent initialization failed: {str(e)}")
    plan_agent = None

try:
    plan_info_agent = PlanInfoAgent(project_id=PROJECT_ID, region=REGION)
    logger.info("✅ PlanInfoAgent initialized")
except Exception as e:
    logger.error(f"❌ PlanInfoAgent initialization failed: {str(e)}")
    plan_info_agent = None

try:
    balance_agent = BalanceAgent(project_id=PROJECT_ID, region=REGION)
    logger.info("✅ BalanceAgent initialized")
except Exception as e:
    logger.error(f"❌ BalanceAgent initialization failed: {str(e)}")
    balance_agent = None

# ==================== Initialize Root Agent ====================

try:
    root_agent = create_root_agent(PROJECT_ID, REGION)
    logger.info("✅ RootAgent initialized")
except Exception as e:
    logger.error(f"❌ RootAgent initialization failed!")
    logger.error(f"   Error: {type(e).__name__}: {str(e)}", exc_info=True)
    root_agent = None




class ReportRequest(BaseModel):
    """Request model for report generation"""
    user_id: str
    report_type: str
    format_type: str = "json"
    
    class Config:
        example = {
            "user_id": "user_001",
            "report_type": "wire_details",
            "format_type": "json"
        }

class PlanRequest(BaseModel):
    """Request model for plan analysis"""
    user_id: str
    current_plan: str
    
    class Config:
        example = {
            "user_id": "user_001",
            "current_plan": "silver"
        }

class PlanInfoRequest(BaseModel):
    """Request model for plan information"""
    query: str
    
    class Config:
        example = {
            "query": "What's in the Gold plan?"
        }

class BalanceRequest(BaseModel):
    """Request model for balance information"""
    user_id: str
    
    class Config:
        example = {
            "user_id": "user_001"
        }

class ProcessQueryRequest(BaseModel):
    """Request model for root agent (process any query)"""
    user_id: str
    query: str
    
    class Config:
        example = {
            "user_id": "user_001",
            "query": "Show me my wire reports"
        }

# ==================== Middleware ====================

@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    """Add correlation ID for request tracking"""
    correlation_id = request.headers.get(
        "X-Correlation-ID",
        f"corr_{int(time.time() * 1000)}"
    )
    request.state.correlation_id = correlation_id
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id
    return response

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = time.time()
    logger.info(f"→ {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    elapsed_time = (time.time() - start_time) * 1000
    logger.info(f"← {request.method} {request.url.path} {response.status_code} ({elapsed_time:.2f}ms)")
    
    return response

# ==================== Health & Status Endpoints ====================

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint
    
    Returns: Status of all agents and services
    """
    return {
        "status": "healthy",
        "service": "Client Data Access Agent",
        "version": "1.0.0",
        "project_id": PROJECT_ID,
        "region": REGION,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/agents/status", tags=["Admin"])
async def get_agents_status():
    """
    Get status of all agents
    
    Returns: Health status of each agent
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "agents": {
            "report_agent": {
                "name": "Report Agent",
                "status": "operational" if report_agent else "unavailable",
                "tools": 5,
                "reports": 9,
                "formats": ["pdf", "excel", "csv", "json"],
                "description": "Generate financial reports (9 types)"
            },
            "plan_agent": {
                "name": "Plan Agent",
                "status": "operational" if plan_agent else "unavailable",
                "tools": 6,
                "tiers": 3,
                "description": "Analyze usage and recommend plan upgrades"
            },
            "plan_info_agent": {
                "name": "Plan Information Agent",
                "status": "operational" if plan_info_agent else "unavailable",
                "tools": 5,
                "categories": 4,
                "description": "Provide plan details, pricing, and support"
            },
            "balance_agent": {
                "name": "Balance Agent",
                "status": "operational" if balance_agent else "unavailable",
                "tools": 3,
                "description": "Get account balance and transactions"
            },
            "root_agent": {
                "name": "Root Agent",
                "status": "operational" if root_agent else "unavailable",
                "tools": 5,
                "description": "Intelligent query router with LLM-based intent classification"
            },
        }
    }

# ==================== Report Agent Endpoints ====================

@app.post("/report", tags=["Report Agent"])
async def generate_report(request: ReportRequest):
    """
    Generate Financial Report
    
    Args:
        user_id: Customer identifier
        report_type: Type of report (balance_report, wire_details, ach_inbound, etc.)
        format_type: Output format (json, pdf, excel, csv)
    
    Returns:
        Complete report with data, formatting, and visualizations
    
    Report Types:
    - balance_report: Account balances
    - wire_details: Wire transfer history
    - ach_inbound: ACH deposits
    - intraday_balance: Real-time balance
    - expanded_details: Detailed transactions
    - statements: Monthly statements
    - deposit_details: Deposit history
    - check_images: Scanned checks
    - running_ledger: Complete transaction log
    
    Example:
        POST /report
        {
            "user_id": "user_001",
            "report_type": "wire_details",
            "format_type": "json"
        }
    """
    if not report_agent:
        raise HTTPException(status_code=503, detail="Report Agent unavailable")
    
    logger.info(f"Generating {request.report_type} report for {request.user_id}")
    
    try:
        result = await report_agent.process_report_request(
            user_id=request.user_id,
            report_type=request.report_type,
            format_type=request.format_type
        )
        return result
    except Exception as e:
        logger.error(f"Report generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/report-types", tags=["Report Agent"])
async def get_report_types():
    """
    Get all available report types
    
    Returns: List of supported report types with descriptions
    """
    report_types = [
        {
            "type": "balance_report",
            "name": "Balance Report",
            "description": "Account balances and daily activity"
        },
        {
            "type": "wire_details",
            "name": "Wire Transfer Details",
            "description": "Complete wire transfer history"
        },
        {
            "type": "ach_inbound",
            "name": "ACH Inbound",
            "description": "ACH deposit history"
        },
        {
            "type": "intraday_balance",
            "name": "Intraday Balance",
            "description": "Real-time account balance"
        },
        {
            "type": "expanded_details",
            "name": "Expanded Details",
            "description": "Detailed transaction information"
        },
        {
            "type": "statements",
            "name": "Statements",
            "description": "Monthly account statements"
        },
        {
            "type": "deposit_details",
            "name": "Deposit Details",
            "description": "Deposit transaction history"
        },
        {
            "type": "check_images",
            "name": "Check Images",
            "description": "Scanned check images"
        },
        {
            "type": "running_ledger",
            "name": "Running Ledger",
            "description": "Complete transaction log with running balances"
        }
    ]
    
    return {
        "status": "success",
        "report_types": report_types,
        "total": len(report_types)
    }

@app.get("/format-types", tags=["Report Agent"])
async def get_format_types():
    """
    Get all available output formats
    
    Returns: List of supported output formats
    """
    formats = [
        {"format": "json", "description": "JSON format"},
        {"format": "pdf", "description": "PDF document"},
        {"format": "excel", "description": "Excel spreadsheet"},
        {"format": "csv", "description": "CSV format"}
    ]
    
    return {
        "status": "success",
        "formats": formats,
        "total": len(formats)
    }

# ==================== Plan Agent Endpoints ====================

@app.post("/plan", tags=["Plan Agent"])
async def analyze_plan(request: PlanRequest):
    """
    Analyze Plan Fit and Recommend Upgrades
    
    Args:
        user_id: Customer identifier
        current_plan: Current plan tier (bronze, silver, gold)
    
    Returns:
        Usage analysis, plan fit, ROI, and recommendations
    
    Plan Tiers:
    - bronze: $29/month, 500 transactions
    - silver: $99/month, 5,000 transactions
    - gold: $299/month, unlimited transactions
    
    Example:
        POST /plan
        {
            "user_id": "user_001",
            "current_plan": "silver"
        }
    """
    if not plan_agent:
        raise HTTPException(status_code=503, detail="Plan Agent unavailable")
    
    logger.info(f"Analyzing plan for user {request.user_id}")
    
    try:
        result = await plan_agent.process_plan_request(
            user_id=request.user_id,
            current_plan=request.current_plan
        )
        return result
    except Exception as e:
        logger.error(f"Plan analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/plans", tags=["Plan Agent"])
async def get_plans():
    """
    Get all available plans with details
    
    Returns: List of all plan tiers with features and pricing
    """
    plans = [
        {
            "tier": "bronze",
            "name": "Bronze",
            "price": 29,
            "billing": "monthly",
            "transactions": "500/month",
            "features": [
                "Basic balance reports",
                "Email support",
                "7-day history",
                "Mobile app access"
            ]
        },
        {
            "tier": "silver",
            "name": "Silver",
            "price": 99,
            "billing": "monthly",
            "transactions": "5,000/month",
            "features": [
                "All 9 report types",
                "API access",
                "Phone + Email support",
                "90-day history",
                "Advanced security"
            ]
        },
        {
            "tier": "gold",
            "name": "Gold",
            "price": 299,
            "billing": "monthly",
            "transactions": "Unlimited",
            "features": [
                "All reports + custom reports",
                "Full API access",
                "24/7 dedicated support",
                "7-year history",
                "Enterprise security",
                "Custom dashboards",
                "SLA guarantee (99.9%)"
            ]
        }
    ]
    
    return {
        "status": "success",
        "plans": plans,
        "total": len(plans)
    }

# ==================== Plan Info Agent Endpoints ====================

@app.post("/plan-info", tags=["Plan Info Agent"])
async def get_plan_info(request: PlanInfoRequest):
    """
    Get Plan Information
    
    Args:
        query: Question about plans (features, pricing, comparison, etc.)
    
    Returns:
        Relevant plan information based on query
    
    Example Questions:
    - "What's in the Gold plan?"
    - "Compare Silver and Gold"
    - "How much does Silver cost?"
    - "What API features are available?"
    - "How do I upgrade my plan?"
    
    Example:
        POST /plan-info
        {
            "query": "What features are in the Gold plan?"
        }
    """
    if not plan_info_agent:
        raise HTTPException(status_code=503, detail="Plan Info Agent unavailable")
    
    logger.info(f"Processing plan info query: {request.query}")
    
    try:
        result = await plan_info_agent.process_info_request(
            query=request.query
        )
        return result
    except Exception as e:
        logger.error(f"Plan info query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/plan-comparison", tags=["Plan Info Agent"])
async def get_plan_comparison():
    """
    Get Side-by-Side Plan Comparison
    
    Returns: Detailed comparison matrix of all plans
    """
    if not plan_info_agent:
        raise HTTPException(status_code=503, detail="Plan Info Agent unavailable")
    
    logger.info("Generating plan comparison")
    
    try:
        result = plan_info_agent.build_comparison()
        return result
    except Exception as e:
        logger.error(f"Plan comparison error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/plan-pricing", tags=["Plan Info Agent"])
async def get_plan_pricing():
    """
    Get Plan Pricing Information
    
    Returns: Detailed pricing for all plans
    """
    if not plan_info_agent:
        raise HTTPException(status_code=503, detail="Plan Info Agent unavailable")
    
    logger.info("Retrieving plan pricing")
    
    try:
        result = plan_info_agent.get_pricing()
        return result
    except Exception as e:
        logger.error(f"Plan pricing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/plan-faqs", tags=["Plan Info Agent"])
async def get_plan_faqs(category: str = None):
    """
    Get Plan FAQs
    
    Args:
        category: FAQ category (billing, features, api, support)
    
    Returns: Frequently asked questions by category
    """
    if not plan_info_agent:
        raise HTTPException(status_code=503, detail="Plan Info Agent unavailable")
    
    logger.info(f"Retrieving FAQs for category: {category}")
    
    try:
        result = plan_info_agent.retrieve_faq(category=category)
        return result
    except Exception as e:
        logger.error(f"FAQ retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Balance Agent Endpoints ====================

@app.post("/balance", tags=["Balance Agent"])
async def get_balance(request: BalanceRequest):
    """
    Get Account Balance and Recent Transactions
    
    Args:
        user_id: Customer identifier
    
    Returns:
        Current balance, recent transactions, trends, and summary
    
    Example:
        POST /balance
        {
            "user_id": "user_001"
        }
    """
    if not balance_agent:
        raise HTTPException(status_code=503, detail="Balance Agent unavailable")
    
    logger.info(f"Retrieving balance for user {request.user_id}")
    
    try:
        result = await balance_agent.process_balance_request(
            user_id=request.user_id
        )
        return result
    except Exception as e:
        logger.error(f"Balance retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/balance/current", tags=["Balance Agent"])
async def get_current_balance(request: BalanceRequest):
    """
    Get Current Balance Only
    
    Args:
        user_id: Customer identifier
    
    Returns: Current account balance information
    """
    if not balance_agent:
        raise HTTPException(status_code=503, detail="Balance Agent unavailable")
    
    logger.info(f"Retrieving current balance for user {request.user_id}")
    
    try:
        result = balance_agent.get_current_balance(user_id=request.user_id)
        return result
    except Exception as e:
        logger.error(f"Current balance error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/balance/transactions", tags=["Balance Agent"])
async def get_transactions(
    request: BalanceRequest,
    limit: int = 50,
    days_back: int = 90
):
    """
    Get Recent Transactions
    
    Args:
        user_id: Customer identifier
        limit: Number of transactions to retrieve (default: 50)
        days_back: Number of days to look back (default: 90)
    
    Returns: Transaction history with summary
    """
    if not balance_agent:
        raise HTTPException(status_code=503, detail="Balance Agent unavailable")
    
    logger.info(f"Retrieving transactions for user {request.user_id}")
    
    try:
        result = balance_agent.get_recent_transactions(
            user_id=request.user_id,
            limit=limit,
            days_back=days_back
        )
        return result
    except Exception as e:
        logger.error(f"Transaction retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/balance/trends", tags=["Balance Agent"])
async def get_balance_trends(
    request: BalanceRequest,
    days_back: int = 30
):
    """
    Get Balance Trends
    
    Args:
        user_id: Customer identifier
        days_back: Number of days to analyze (default: 30)
    
    Returns: Balance trend analysis over time
    """
    if not balance_agent:
        raise HTTPException(status_code=503, detail="Balance Agent unavailable")
    
    logger.info(f"Analyzing balance trends for user {request.user_id}")
    
    try:
        result = balance_agent.get_balance_trends(
            user_id=request.user_id,
            days_back=days_back
        )
        return result
    except Exception as e:
        logger.error(f"Trend analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Error Handlers ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error": exc.detail,
            "correlation_id": getattr(request.state, "correlation_id", None),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": "Internal server error",
            "correlation_id": getattr(request.state, "correlation_id", None),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# ==================== Startup/Shutdown Events ====================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("=" * 60)
    logger.info("🚀 Client Data Access Agent Starting")
    logger.info("=" * 60)
    logger.info(f"Project ID: {PROJECT_ID}")
    logger.info(f"Region: {REGION}")
    logger.info("")
    logger.info("✅ ReportAgent - Ready" if report_agent else "❌ ReportAgent - Failed")
    logger.info("✅ PlanAgent - Ready" if plan_agent else "❌ PlanAgent - Failed")
    logger.info("✅ PlanInfoAgent - Ready" if plan_info_agent else "❌ PlanInfoAgent - Failed")
    logger.info("✅ BalanceAgent - Ready" if balance_agent else "❌ BalanceAgent - Failed")
    logger.info("")
    logger.info("📚 Documentation: http://localhost:8080/docs")
    logger.info("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Client Data Access Agent...")

# ==================== Root Endpoint ====================

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API documentation
    
    Returns: Information about the API
    """
    return {
        "status": "running",
        "service": "Client Data Access Agent",
        "version": "1.0.0",
        "description": "Multi-agent financial services platform",
        "endpoints": {
            "docs": "/docs (Swagger UI)",
            "redoc": "/redoc (ReDoc)",
            "health": "/health",
            "agents": "/agents/status",
            "reports": {
                "generate": "POST /report",
                "types": "GET /report-types",
                "formats": "GET /format-types"
            },
            "plans": {
                "analyze": "POST /plan",
                "list": "GET /plans",
                "info": "POST /plan-info",
                "comparison": "GET /plan-comparison",
                "pricing": "GET /plan-pricing",
                "faqs": "GET /plan-faqs"
            },
            "balance": {
                "full": "POST /balance",
                "current": "POST /balance/current",
                "transactions": "POST /balance/transactions",
                "trends": "POST /balance/trends"
            }
        },
        "agents": [
            "ReportAgent (5 tools, 9 reports)",
            "PlanAgent (6 tools, 3 tiers)",
            "PlanInfoAgent (5 tools, features/pricing/faq)",
            "BalanceAgent (3 tools, balance/transactions)"
        ]
    }

@app.post("/process-query", tags=["Root Agent"])
async def process_query(request: ProcessQueryRequest):
    """
    Main entry point - Process natural language query through Root Agent
    
    The Root Agent will:
    1. Classify user intent (using LLM)
    2. Check compliance
    3. Route to appropriate sub-agent
    4. Aggregate and return results
    
    Args:
        user_id: Customer identifier
        query: Natural language query
    
    Returns:
        Complete response with intent, agent used, and data
    
    Example:
        POST /process-query
        {
          "user_id": "user001",
          "query": "What is my balance?"
        }
    """
    
    correlation_id = f"root_agent_{int(time.time() * 1000)}"
    
    logger.info(f"[{correlation_id}] ========== ROOT AGENT PROCESSING START ==========")
    logger.info(f"[{correlation_id}] User ID: {request.user_id}")
    logger.info(f"[{correlation_id}] Query: {request.query}")
    
    if not root_agent:
        logger.error(f"[{correlation_id}] ❌ Root Agent unavailable")
        raise HTTPException(status_code=503, detail="Root Agent unavailable")
    
    logger.info(f"[{correlation_id}] ✅ Root Agent is operational")
    logger.info(f"[{correlation_id}] ⏳ Calling: root_agent.process_query()")
    
    try:
        start_time = time.time()
        result = await root_agent.process_query(request.user_id, request.query)
        elapsed = (time.time() - start_time) * 1000
        
        logger.info(f"[{correlation_id}] ✅ Root Agent returned successfully")
        logger.info(f"[{correlation_id}] Intent: {result.get('intent')}")
        logger.info(f"[{correlation_id}] Agent Used: {result.get('agent_used')}")
        logger.info(f"[{correlation_id}] Compliance Passed: {result.get('compliance_passed')}")
        logger.info(f"[{correlation_id}] Total Execution Time: {elapsed:.2f}ms")
        logger.info(f"[{correlation_id}] ========== ROOT AGENT PROCESSING END ==========")
        
        return result
    
    except Exception as e:
        logger.error(f"[{correlation_id}] ❌ Error in Root Agent processing: {str(e)}")
        logger.error(f"[{correlation_id}] ========== ROOT AGENT PROCESSING FAILED ==========")
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.get("/clients-with-plans", tags=["Plan Agent"])
async def clients_with_plans():
    """
    Get all clients and their current plans

    Returns: List of clients with their plan tiers
    """
    if not plan_agent:
        raise HTTPException(status_code=503, detail="Plan Agent unavailable")
    try:
        result = plan_agent.get_all_clients_with_plans()
        return result
    except Exception as e:
        logger.error(f"Error fetching clients with plans: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
# ==================== Main Entry Point ====================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("🚀 Starting FastAPI Server")
    print("=" * 60)
    print("📍 Local: http://localhost:8080")
    print("📚 Docs: http://localhost:8080/docs")
    print("🔍 ReDoc: http://localhost:8080/redoc")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
