# tests/test_streamlit_app.py
import sys
import os
from unittest.mock import MagicMock, patch

# Add the project root to the Python path to allow importing streamlit_app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock streamlit before it's imported by the app
# This prevents errors from streamlit trying to connect to a server
st_mock = MagicMock()
sys.modules['streamlit'] = st_mock
sys.modules['st_aggrid'] = MagicMock()
sys.modules['plotly'] = MagicMock()
sys.modules['altair'] = MagicMock()


# Now that streamlit is mocked, we can import the function
from streamlit_app import detect_intent

# Define a mock session state object
mock_session_state = MagicMock()
mock_session_state.user_id = 'silver_001'
mock_session_state.current_plan = 'SILVER'

@patch('streamlit.session_state', mock_session_state)
def test_detect_intent_balance_check():
    """TC-F-01: Test balance check intent."""
    query = "what is my account balance"
    endpoint, body = detect_intent(query)
    assert endpoint == "/balance"
    assert body == {"user_id": "silver_001"}

@patch('streamlit.session_state', mock_session_state)
def test_detect_intent_wire_report():
    """TC-F-02: Test wire report intent."""
    query = "show me my wire transfers"
    endpoint, body = detect_intent(query)
    assert endpoint == "/report"
    assert body == {
        "user_id": "test_user",
        "report_type": "wire_details",
        "format_type": "json"
    }

@patch('streamlit.session_state', mock_session_state)
def test_detect_intent_plan_info():
    """TC-F-03: Test plan info intent."""
    query = "what is in the gold plan?"
    endpoint, body = detect_intent(query)
    assert endpoint == "/plan-info"
    assert body == {"query": query}

@patch('streamlit.session_state', mock_session_state)
def test_detect_intent_plan_upgrade():
    """TC-F-04: Test plan upgrade intent."""
    query = "I want to upgrade my plan"
    endpoint, body = detect_intent(query)
    assert endpoint == "/plan"
    assert body == {
        "user_id": "silver_001",
        "current_plan": "silver"
    }

@patch('streamlit.session_state', mock_session_state)
def test_detect_intent_keyword_ambiguity():
    """TC-F-05: Test keyword ambiguity."""
    # This query contains "balance" and "plan".
    # According to the function's logic, "plan" is checked first.
    query = "how much does it cost to balance my plan?"
    endpoint, body = detect_intent(query)
    assert endpoint == "/plan-info"
    assert body == {"query": query}

@patch('streamlit.session_state', mock_session_state)
def test_detect_intent_default_case():
    """Test default intent when no keywords are matched."""
    query = "tell me a joke"
    endpoint, body = detect_intent(query)
    assert endpoint == "/balance"
    assert body == {"user_id": "silver_001"}

@patch('streamlit.session_state', mock_session_state)
def test_detect_intent_transaction_query():
    """Test transaction query intent."""
    query = "show my recent activity"
    endpoint, body = detect_intent(query)
    assert endpoint == "/balance/transactions"
    assert body == {
        "user_id": "silver_001",
        "limit": 50,
        "days_back": 90
    }
