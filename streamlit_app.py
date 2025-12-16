# frontend/app.py - FIXED VERSION WITH ROUTING
import streamlit as st
import requests
import json
from datetime import datetime

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
            endpoint, request_body = detect_intent(user_input)
            
            # Special handling for POST requests with/without params
            if endpoint == "/balance/transactions":
                response = requests.post(
                    f"{API_BASE_URL}{endpoint}",
                    json=request_body,
                    timeout=10
                )
            elif endpoint == "/plan-info":
                response = requests.post(
                    f"{API_BASE_URL}{endpoint}",
                    json=request_body,
                    timeout=10
                )
            else:
                response = requests.post(
                    f"{API_BASE_URL}{endpoint}",
                    json=request_body,
                    timeout=10
                )
            
            if response.status_code == 200:
                result = response.json()
                
                # Format response based on endpoint
                if endpoint == "/report":
                    # Wire report formatting
                    if result.get("status") == "success":
                        data = result.get("report_data", [])
                        if isinstance(data, str):
                            data = json.loads(data) if data.startswith("[") else data
                        
                        if isinstance(data, list) and len(data) > 0:
                            assistant_response = "**Wire Transfer Details:**\n\n"
                            for i, wire in enumerate(data, 1):
                                if isinstance(wire, dict):
                                    wire_amount = wire.get('wire_amount', wire.get('amount', 'N/A'))
                                    destination = wire.get('destination_bank', 'N/A')
                                    status = wire.get('status', 'N/A')
                                    assistant_response += f"{i}. Amount: ${wire_amount} → {destination} ({status})\n"
                        else:
                            assistant_response = f"Wire Details Found: {len(data)} records"
                    else:
                        assistant_response = f"Error: {result.get('error', 'Unknown error')}"
                
                elif endpoint == "/balance":
                    # Balance formatting
                    if result.get("status") == "success":
                        balance = result.get('balance', 'N/A')
                        transactions = result.get('transaction_count', 0)
                        formatted_balance = f"{balance:,.2f}" if isinstance(balance, (int, float)) else balance
                        assistant_response = f"""
**Your Account Balance:** ${formatted_balance}

**Recent Activity:**
- Total Transactions: {transactions}
- Plan: {st.session_state.current_plan}

✅ Use 'show my wire transfers' for wire details
✅ Use 'what's in Gold plan' for plan info
"""
                    else:
                        assistant_response = f"Error: {result.get('error', 'Unknown')}"
                
                elif endpoint == "/balance/transactions":
                    # Transaction formatting
                    if result.get("status") == "success":
                        transactions = result.get('transactions', [])
                        assistant_response = "**Recent Transactions:**\n\n"
                        for i, txn in enumerate(transactions[:5], 1):
                            txn_type = txn.get('transaction_type', 'N/A')
                            amount = txn.get('amount', 'N/A')
                            assistant_response += f"{i}. {txn_type}: ${amount}\n"
                    else:
                        assistant_response = f"Error: {result.get('error', 'Unknown')}"
                
                elif endpoint == "/plan-info":
                    # Plan info formatting
                    # Case 1: The agent returned a text answer (LLM response)
                    if "response" in result or "answer" in result:
                        assistant_response = result.get("response", result.get("answer"))
                    
                    # Case 2: The agent returned structured plan data (Dictionary)
                    elif "data" in result and "plans" in result["data"]:
                        plans = result["data"]["plans"]
                        lines = ["Here are the plan details:\n"]
                        
                        # Loop through each plan found in the data
                        for plan_key, p in plans.items():
                            lines.append(f"### 🏷️ {p.get('name', plan_key).title()} Plan")
                            lines.append(f"**Price**: ${p.get('price_monthly')}/mo")
                            lines.append(f"_{p.get('description', '')}_")
                            if "features" in p:
                                lines.append("**Features:**")
                                for feature in p["features"]:
                                    lines.append(f"- {feature}")
                            lines.append("---")
                        
                        assistant_response = "\n".join(lines)
                    
                    # Case 3: Fallback
                    else:
                        assistant_response = str(result)
                
                elif endpoint == "/plan":
                    # Plan analysis formatting
                    if result.get("status") == "success":
                        recommendation = result.get('recommendation', 'N/A')
                        assistant_response = f"**Plan Recommendation:**\n\n{recommendation}"
                    else:
                        assistant_response = f"Error: {result.get('error', 'Unknown')}"
                
                else:
                    assistant_response = json.dumps(result, indent=2)
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": assistant_response
                })
                
                with st.chat_message("assistant"):
                    st.markdown(assistant_response)
            
            elif response.status_code == 404:
                error_msg = f"❌ Endpoint not found: {endpoint}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.error(error_msg)
            
            else:
                error_msg = f"Error {response.status_code}: {response.text}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.error(error_msg)
        
        except requests.exceptions.ConnectionError:
            error_msg = "❌ Cannot connect to backend on port 8080"
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
