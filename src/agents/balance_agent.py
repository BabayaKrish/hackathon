# Balance Agent - Account Balance & Transactions
# File: backend/agents/balance_agent.py

import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta, timezone
import time
import asyncio

try:
    from google.cloud import bigquery
except Exception:
    bigquery = None

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel
except Exception:
    vertexai = None

    class GenerativeModel:
        def __init__(self, *args, **kwargs):
            pass

# ==================== Configuration ====================

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ==================== Balance Agent ====================

class BalanceAgent:
    """
    Balance Information Agent
    
    Responsibilities:
    - Retrieve current account balance
    - Fetch recent transaction history
    - Provide balance trends
    - Calculate summary statistics
    - Monitor account activity
    """
    
    def __init__(self, project_id: str, region: str = "us-central1"):
        """Initialize Balance Agent"""
        self.project_id = project_id
        self.region = region
        self.bq_client = bigquery.Client(project=project_id)
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=region)
        self.model = GenerativeModel("gemini-2.0-flash-exp")
        
        logger.info(f"BalanceAgent initialized for project {project_id}")
    
    def get_current_balance(self, user_id: str) -> Dict[str, Any]:
        """TOOL 1: Get current account balance"""
        logger.info(f"Retrieving current balance for user {user_id}")
        
        try:
            query = f"""
            SELECT
                user_id,
                MAX(running_balance) as balance,
                MAX(created_at) as last_update
            FROM `{self.project_id}.client_data.user_transactions`
            WHERE user_id = @user_id
            GROUP BY user_id
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                ]
            )
            
            results = list(self.bq_client.query(query, job_config=job_config).result())
            
            if not results:
                return {
                    "status": "success",
                    "user_id": user_id,
                    "balance": 0.0,
                    "currency": "USD",
                    "last_update": datetime.now(timezone.utc).isoformat(),
                    "note": "No transactions found for user"
                }
            
            data = results[0]
            return {
                "status": "success",
                "user_id": user_id,
                "balance": float(data['balance']) if data['balance'] else 0.0,
                "currency": "USD",
                "last_update": data['last_update'].isoformat() if hasattr(data.last_update, 'isoformat') else str(data.last_update),
                "transaction_count": len(results)
            }
        
        except Exception as e:
            logger.error(f"Balance retrieval error: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "user_id": user_id,
                "balance": 0.0
            }
    
    def get_recent_transactions(
        self,
        user_id: str,
        limit: int = 50,
        days_back: int = 90
    ) -> Dict[str, Any]:
        """TOOL 2: Get recent transactions"""
        logger.info(f"Retrieving recent transactions for user {user_id}")
        
        try:
            # FIXED: Removed broken CTE syntax, simplified to direct query
            query = f"""
            SELECT 
                transaction_id,
                user_id,
                transaction_type,
                amount,
                description,
                status,
                reference_id,
                counterparty,
                running_balance,
                transaction_date,
                COUNT(*) OVER() as total_matching_transactions
            FROM `{self.project_id}.client_data.user_transactions`
            WHERE user_id = @user_id
                AND transaction_date >= DATE_SUB(CURRENT_DATE(), INTERVAL @days_back DAY)
            ORDER BY transaction_date DESC
            LIMIT @limit
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                    bigquery.ScalarQueryParameter("limit", "INT64", limit),
                    bigquery.ScalarQueryParameter("days_back", "INT64", days_back),
                ]
            )
            
            results = list(self.bq_client.query(query, job_config=job_config).result())
            
            # Format transactions
            transactions = []
            for row in results:
                transactions.append({
                    "transaction_id": row.transaction_id,
                    "date": row.transaction_date.isoformat() if hasattr(row.transaction_date, 'isoformat') else str(row.transaction_date),
                    "type": row.transaction_type,
                    "amount": float(row.amount) if row.amount else 0,
                    "description": row.description,
                    "status": row.status,
                    "reference_id": row.reference_id,
                    "counterparty": row.counterparty,
                    "running_balance": float(row.running_balance) if row.running_balance else 0
                })
            
            # Calculate summary statistics
            total_count = len(results)
            total_debits = sum(t["amount"] for t in transactions if t["type"] in ["wire", "ach", "check"])
            total_credits = sum(t["amount"] for t in transactions if t["type"] in ["deposit", "transfer_in"])
            
            return {
                "status": "success",
                "user_id": user_id,
                "transactions": transactions,
                "summary": {
                    "transaction_count": total_count,
                    "total_debits": total_debits,
                    "total_credits": total_credits,
                    "net_activity": total_credits - total_debits,
                    "period_days": days_back
                },
                "breakdown": {
                    "wire_transfers": len([t for t in transactions if t["type"] == "wire"]),
                    "ach_transfers": len([t for t in transactions if t["type"] == "ach"]),
                    "deposits": len([t for t in transactions if t["type"] == "deposit"]),
                    "checks": len([t for t in transactions if t["type"] == "check"])
                }
            }
        except Exception as e:
            logger.error(f"Transaction retrieval error: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "transactions": []
            }
    
    def get_balance_trends(self, user_id: str, days_back: int = 30) -> Dict[str, Any]:
        """Helper method to get balance trends over time"""
        logger.info(f"Analyzing balance trends for user {user_id}")
        
        try:
            # FIXED: Use transaction_date and running_balance from actual table
            query = f"""
            SELECT 
                DATE(transaction_date) as date,
                AVG(running_balance) as avg_daily_balance,
                MIN(running_balance) as min_daily_balance,
                MAX(running_balance) as max_daily_balance,
                COUNT(*) as update_count
            FROM `{self.project_id}.client_data.user_transactions`
            WHERE user_id = @user_id
                AND transaction_date >= DATE_SUB(CURRENT_DATE(), INTERVAL @days_back DAY)
            GROUP BY date
            ORDER BY date DESC
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                    bigquery.ScalarQueryParameter("days_back", "INT64", days_back),
                ]
            )
            
            results = list(self.bq_client.query(query, job_config=job_config).result())
            
            trends = []
            for row in results:
                trends.append({
                    "date": row.date.isoformat() if hasattr(row.date, 'isoformat') else str(row.date),
                    "average": float(row.avg_daily_balance) if row.avg_daily_balance else 0,
                    "minimum": float(row.min_daily_balance) if row.min_daily_balance else 0,
                    "maximum": float(row.max_daily_balance) if row.max_daily_balance else 0
                })
            
            if trends:
                overall_trend = "increasing" if trends[0]["average"] > trends[-1]["average"] else "decreasing"
                volatility = max([t["maximum"] - t["minimum"] for t in trends]) if trends else 0
            else:
                overall_trend = "stable"
                volatility = 0
            
            return {
                "status": "success",
                "user_id": user_id,
                "trends": trends,
                "analysis": {
                    "overall_trend": overall_trend,
                    "average_volatility": volatility,
                    "days_analyzed": len(trends),
                    "period_days": days_back
                }
            }
        except Exception as e:
            logger.error(f"Trend analysis error: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "trends": []
            }
    
    async def process_balance_request(self, user_id: str) -> Dict[str, Any]:
        """Main Balance Agent entry point"""
        logger.info(f"Processing balance request for user {user_id}")
        
        execution_log = []
        start_time = time.time()
        
        try:
            # Step 1: Get current balance
            balance_result = self.get_current_balance(user_id)
            execution_log.append({
                "step": "get_current_balance",
                "status": balance_result.get("status")
            })
            
            if balance_result.get("status") != "success":
                return {
                    "status": "error",
                    "error": balance_result.get("error"),
                    "execution_log": execution_log
                }
            
            # Step 2: Get recent transactions
            transactions_result = self.get_recent_transactions(user_id, limit=50, days_back=90)
            execution_log.append({
                "step": "get_recent_transactions",
                "status": transactions_result.get("status"),
                "transaction_count": len(transactions_result.get("transactions", []))
            })
            
            # Step 3: Get balance trends
            trends_result = self.get_balance_trends(user_id, days_back=30)
            execution_log.append({
                "step": "get_balance_trends",
                "status": trends_result.get("status"),
                "days_analyzed": trends_result.get("analysis", {}).get("days_analyzed", 0)
            })
            
            elapsed_time = (time.time() - start_time) * 1000
            
            # FIXED: Use correct balance_result structure (no nested "balance_info")
            return {
                "status": "success",
                "user_id": user_id,
                "current_balance": balance_result.get("balance", 0),
                "balance": balance_result.get("balance", 0),
                "last_update": balance_result.get("last_update"),
                "recent_transactions": {
                    "transactions": transactions_result.get("transactions", []),
                    "summary": transactions_result.get("summary", {}),
                    "breakdown": transactions_result.get("breakdown", {})
                },
                "trends": {
                    "data": trends_result.get("trends", []),
                    "analysis": trends_result.get("analysis", {})
                },
                "summary": {
                    "current_balance": balance_result.get("balance", 0),
                    "recent_transactions_count": len(transactions_result.get("transactions", [])),
                    "trend_direction": trends_result.get("analysis", {}).get("overall_trend", "stable")
                },
                "execution_log": execution_log,
                "total_execution_time_ms": elapsed_time,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        except Exception as e:
            logger.error(f"Balance request processing error: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "execution_log": execution_log
            }


# ==================== Eval / Testcases ====================
class MockBalanceAgent(BalanceAgent):
    """Lightweight mock agent that overrides external calls for local evals."""
    def __init__(self):
        # Do not initialize external clients
        self.project_id = "mock_project"
        self.region = "local"

    def get_current_balance(self, user_id: str) -> Dict[str, Any]:
        if user_id == "user_no_tx":
            return {
                "status": "success",
                "user_id": user_id,
                "balance": 0.0,
                "currency": "USD",
                "last_update": datetime.now(timezone.utc).isoformat(),
                "note": "No transactions found for user"
            }

        # default mocked balance
        return {
            "status": "success",
            "user_id": user_id,
            "balance": 1234.56,
            "currency": "USD",
            "last_update": datetime.now(timezone.utc).isoformat()
        }

    def get_recent_transactions(self, user_id: str, limit: int = 50, days_back: int = 90) -> Dict[str, Any]:
        if user_id == "user_no_tx":
            return {"status": "success", "user_id": user_id, "transactions": [], "summary": {}, "breakdown": {}}

        # provide a few mocked transactions for evaluation
        now = datetime.now(timezone.utc)
        transactions = [
            {
                "transaction_id": f"tx-{i}",
                "date": (now - timedelta(days=i)).isoformat(),
                "type": "deposit" if i % 2 == 0 else "wire",
                "amount": 100.0 * (i + 1),
                "description": f"Mock transaction {i}",
                "status": "completed",
                "reference_id": f"ref-{i}",
                "counterparty": "counterparty",
                "running_balance": 1000.0 + 100.0 * i
            }
            for i in range(min(limit, 5))
        ]

        total_debits = sum(t["amount"] for t in transactions if t["type"] in ["wire", "ach", "check"])
        total_credits = sum(t["amount"] for t in transactions if t["type"] in ["deposit", "transfer_in"])

        return {
            "status": "success",
            "user_id": user_id,
            "transactions": transactions,
            "summary": {
                "transaction_count": len(transactions),
                "total_debits": total_debits,
                "total_credits": total_credits,
                "net_activity": total_credits - total_debits,
                "period_days": days_back
            },
            "breakdown": {
                "wire_transfers": len([t for t in transactions if t["type"] == "wire"]),
                "ach_transfers": 0,
                "deposits": len([t for t in transactions if t["type"] == "deposit"]),
                "checks": 0
            }
        }

    def get_balance_trends(self, user_id: str, days_back: int = 30) -> Dict[str, Any]:
        if user_id == "user_trend_dec":
            # decreasing trend
            trends = [
                {"date": (datetime.now(timezone.utc) - timedelta(days=i)).date().isoformat(), "average": 1000.0 - i * 10, "minimum": 900.0 - i * 10, "maximum": 1100.0 - i * 10}
                for i in range(5)
            ]
            overall_trend = "decreasing"
        else:
            # default increasing/stable trend
            trends = [
                {"date": (datetime.now(timezone.utc) - timedelta(days=i)).date().isoformat(), "average": 1000.0 + i * 10, "minimum": 900.0 + i * 10, "maximum": 1100.0 + i * 10}
                for i in range(5)
            ]
            overall_trend = "increasing"

        volatility = max([t["maximum"] - t["minimum"] for t in trends]) if trends else 0

        return {
            "status": "success",
            "user_id": user_id,
            "trends": trends,
            "analysis": {
                "overall_trend": overall_trend,
                "average_volatility": volatility,
                "days_analyzed": len(trends),
                "period_days": days_back
            }
        }


async def run_eval_cases() -> None:
    """Run a set of eval cases using the MockBalanceAgent and print concise summaries."""
    agent = MockBalanceAgent()

    test_users = ["user_no_tx", "user_normal", "user_trend_dec"]

    for uid in test_users:
        print(f"\n=== Eval for {uid} ===")
        result = await agent.process_balance_request(uid)
        print(f"status: {result.get('status')}")
        print(f"current_balance: {result.get('current_balance')}")
        print(f"recent_transactions_count: {result.get('summary', {}).get('recent_transactions_count')}")
        print(f"trend_direction: {result.get('summary', {}).get('trend_direction')}")


if __name__ == "__main__":
    try:
        asyncio.run(run_eval_cases())
    except Exception as e:
        print(f"Eval run failed: {e}")

# ==================== Exports ====================
__all__ = ['BalanceAgent']
