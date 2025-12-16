import os
import asyncio
import importlib.util
import pytest

# Load balance_agent module directly to avoid importing package-level modules
here = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(here, "..", "src", "agents", "balance_agent.py"))
spec = importlib.util.spec_from_file_location("balance_agent", module_path)
balance_agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(balance_agent)

MockBalanceAgent = balance_agent.MockBalanceAgent


@pytest.mark.parametrize(
    "user,expected_balance,expected_tx_count,expected_trend",
    [
        ("user_no_tx", 0.0, 0, "increasing"),
        ("user_normal", 1234.56, 5, "increasing"),
        ("user_trend_dec", 1234.56, 5, "decreasing"),
    ],
)
def test_process_balance_request(user, expected_balance, expected_tx_count, expected_trend):
    agent = MockBalanceAgent()
    result = asyncio.run(agent.process_balance_request(user))

    assert result["status"] == "success"
    assert result.get("current_balance") == expected_balance
    assert result.get("summary", {}).get("recent_transactions_count") == expected_tx_count
    assert result.get("summary", {}).get("trend_direction") == expected_trend


def test_recent_transactions_structure():
    agent = MockBalanceAgent()
    result = asyncio.run(agent.process_balance_request("user_normal"))

    recent = result.get("recent_transactions", {})
    assert "transactions" in recent
    assert isinstance(recent["transactions"], list)
    assert all("transaction_id" in t for t in recent["transactions"])
