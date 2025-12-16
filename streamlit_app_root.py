# frontend/app.py - FIXED VERSION WITH ROUTING
import streamlit as st
import requests
import json
from datetime import datetime
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Configuration
API_BASE_URL = "http://localhost:8080"

st.set_page_config(
    page_title="Data Access Plans Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_plan' not in st.session_state:
    st.session_state.current_plan = 'SILVER'
if 'user_id' not in st.session_state:
    st.session_state.user_id = 'demo_user'

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("## 👤 User Profile")
    
    new_user_id = st.text_input(
        "User ID",
        value=st.session_state.user_id,
        key="user_id_input"
    )
    if new_user_id != st.session_state.user_id:
        st.session_state.user_id = new_user_id
        st.session_state.chat_history = []
    
    new_plan = st.selectbox(
        "Plan",
        ["BRONZE", "SILVER", "GOLD"],
        index=["BRONZE", "SILVER", "GOLD"].index(st.session_state.current_plan),
        key="plan_selector"
    )
    if new_plan != st.session_state.current_plan:
        st.session_state.current_plan = new_plan
    
    if st.button("Clear Chat"):
        st.session_state.chat_history = []

# ===== MAIN CHAT =====
st.markdown("# 💬 Financial Assistant")
st.caption(f"👤 User: **{st.session_state.user_id}** | Plan: **{st.session_state.current_plan}**")

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def send_to_root_agent(user_input: str) -> Dict:
    """Send query to Root Agent for intelligent routing"""
    logger.info(f"[streamlit] Sending to Root Agent: {user_input}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/process-query",
            json={
                "user_id": st.session_state.user_id,
                "query": user_input
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"[streamlit] ✅ Root Agent responded: {result.get('intent')}")
            return result
        else:
            logger.error(f"[streamlit] ❌ Root Agent error: {response.status_code}")
            return {"status": "error", "error": response.text}
    
    except Exception as e:
        logger.error(f"[streamlit] ❌ Failed to call Root Agent: {str(e)}")
        return {"status": "error", "error": str(e)}


def send_to_root_agent(user_input: str) -> Dict:
    """Send query to Root Agent for intelligent routing"""
    logger.info(f"[streamlit] Sending to Root Agent: {user_input}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/process-query",
            json={
                "user_id": st.session_state.user_id,
                "query": user_input
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"[streamlit] ✅ Root Agent responded: {result.get('intent')}")
            return result
        else:
            logger.error(f"[streamlit] ❌ Root Agent error: {response.status_code}")
            return {"status": "error", "error": response.text}
    
    except Exception as e:
        logger.error(f"[streamlit] ❌ Failed to call Root Agent: {str(e)}")
        return {"status": "error", "error": str(e)}


# ===== INTENT ROUTING FUNCTION =====
def detect_intent(query: str) -> tuple:
    """
    Detect user intent and route to correct endpoint
    Returns: (endpoint, request_body)
    """
    query_lower = query.lower()
    
    # Wire transfer queries
    if any(word in query_lower for word in ["wire", "transfer"]):
        return "/report", {
            "user_id": st.session_state.user_id,
            "report_type": "wire_details",
            "format_type": "json"
        }
    
    # Balance queries
    elif any(word in query_lower for word in ["balance", "account balance", "how much"]) and "plan" not in query_lower:
        return "/balance", {
            "user_id": st.session_state.user_id
        }
    
    # Transaction queries
    elif any(word in query_lower for word in ["transaction", "activity", "recent"]):
        return "/balance/transactions", {
            "user_id": st.session_state.user_id,
            "limit": 50,
            "days_back": 90
        }
    
    # Plan queries
    elif any(word in query_lower for word in ["plan", "gold", "silver", "bronze", "feature", "price", "cost"]):
        return "/plan-info", {
            "query": query
        }
    
    # Upgrade queries
    elif any(word in query_lower for word in ["upgrade", "upgrade to", "recommend"]):
        return "/plan", {
            "user_id": st.session_state.user_id,
            "current_plan": st.session_state.current_plan.lower()
        }
    
    # Default: balance
    else:
        return "/balance", {
            "user_id": st.session_state.user_id
        }

# ===== CHAT INPUT & PROCESSING =====
user_input = st.chat_input(
    "Ask about balance, plans, wire transfers...",
    key="chat_input"
)

if user_input:
    # Add user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Detect intent and call appropriate endpoint
    with st.spinner("Processing..."):
        try:
            result = send_to_root_agent(user_input)
            
            if result.get("status") == "success":
                intent = result.get("intent")
                agent_used = result.get("agent_used")
                data = result.get("data", {})
                
                logger.info(f"[streamlit] Received from {agent_used}: {intent}")
                
                # Format response based on intent/agent
                if agent_used == "balance_agent":
                    balance = data.get("balance", 0)
                    transactions = data.get("transaction_count", 0)
                    assistant_response = f"""
**Your Account Balance: ${balance:,.2f}**

**Recent Activity:**
- Total Transactions: {transactions}
- Plan: {st.session_state.current_plan}

✅ Use 'show my wire transfers' for wire details
✅ Use 'what's in Gold plan' for plan info
"""
                
                elif agent_used == "report_agent":
                    # Wire reports formatting
                    assistant_response = "Wire reports go here..."
                
                elif agent_used == "plan_info_agent":
                    # Plan info formatting
                    assistant_response = data.get("response", "")
                
                else:
                    assistant_response = str(data)
                
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                
                with st.chat_message("assistant"):
                    st.markdown(assistant_response)
            
            else:
                error_msg = f"Error: {result.get('error', 'Unknown error')}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.error(error_msg)
        
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
            with st.chat_message("assistant"):
                st.error(error_msg)
                

# Footer
st.markdown("---")
st.markdown("*Powered by Multi-Agent AI | FastAPI + Streamlit + BigQuery*")
