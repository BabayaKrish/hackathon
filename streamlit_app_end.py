# frontend/app.py - WITH INLINE PAYMENT FORM (Hide/Show on Click)

import streamlit as st
import requests
import json
import pandas as pd
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
if 'selected_upgrade_plan' not in st.session_state:
    st.session_state.selected_upgrade_plan = None
if 'show_payment_form' not in st.session_state:
    st.session_state.show_payment_form = False
if 'upgrade_success' not in st.session_state:
    st.session_state.upgrade_success = False

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
        "Current Plan",
        ["BRONZE", "SILVER", "GOLD"],
        index=["BRONZE", "SILVER", "GOLD"].index(st.session_state.current_plan),
        key="plan_selector"
    )
    if new_plan != st.session_state.current_plan:
        st.session_state.current_plan = new_plan
    
    st.divider()
    
    if st.button("🔄 Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# ===== HEADER =====
st.markdown("# 💬 Financial Assistant & Plan Management")
st.caption(f"👤 User: **{st.session_state.user_id}** | Current Plan: **{st.session_state.current_plan}**")

# ===== TABS (Only Chat and Plan Comparison now) =====
tab1, tab2 = st.tabs(["💬 Chat Assistant", "📊 Plan Comparison"])

# Helper function (define outside all tabs)
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

# =====================================================================
# TAB 1: CHAT ASSISTANT
# =====================================================================
with tab1:
    st.markdown("## Chat with Your Financial Assistant")
    
    # Display chat history (INSIDE tab - this is allowed)
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ✅ CHAT INPUT IS HERE - OUTSIDE ALL TABS (NO INDENTATION)
user_input = st.chat_input(
    "Ask about balance, plans, wire transfers...",
    key="chat_input"
)

# Process chat input (still outside tabs)
if user_input:
    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })
    
    # Re-render to show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Process with Root Agent
    with st.spinner("Processing..."):
        try:
            result = send_to_root_agent(user_input)
            
            if result.get("status") == "success":
                intent = result.get("intent")
                agent_used = result.get("agent_used")
                data = result.get("data", {})
                confidence = result.get("confidence", 0.0)
                
                logger.info(f"[streamlit] Intent: {intent}, Agent: {agent_used}, Confidence: {confidence}")
                
                # Format response based on agent
                if agent_used == "balance_agent":
                    balance = data.get("balance", 0)
                    transactions = data.get("transaction_count", 0)
                    assistant_response = f"""
**💰 Your Account Balance: ${balance:,.2f}**

**📈 Recent Activity:**
- Total Transactions (90 days): {transactions}
- Current Plan: **{st.session_state.current_plan}**

**💡 Quick Tips:**
✅ Use 'show my wire transfers' for wire details
✅ Use 'what's in Gold plan' for plan comparison
✅ Visit the 'Plan Comparison' tab to upgrade your plan
"""
                
                elif agent_used == "report_agent":
                    reports = data.get("reports", [])
                    assistant_response = f"""
**📋 Wire Transfer Details**

{json.dumps(reports, indent=2)}

For more detailed reports, visit the reports section.
"""
                
                elif agent_used == "plan_info_agent":
                    plans = data.get("plans", [])
                    assistant_response = f"""
**📊 Plan Information**

{json.dumps(plans, indent=2)}

Visit the **Plan Comparison** tab to see a detailed comparison and upgrade.
"""
                
                else:
                    assistant_response = str(data)
                
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                
                with st.chat_message("assistant"):
                    st.markdown(assistant_response)
            
            elif result.get("status") == "low_confidence":
                error_msg = f"🤔 I'm not sure about that ({result.get('confidence', 0)*100:.0f}% confident). Could you rephrase your question?"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.warning(error_msg)
            
            else:
                error_msg = f"❌ Error: {result.get('error', 'Unknown error')}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.error(error_msg)
        
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
            with st.chat_message("assistant"):
                st.error(error_msg)
    
    st.rerun()

# =====================================================================
# TAB 2: PLAN COMPARISON
# =====================================================================
with tab2:
    st.markdown("## 📊 Plan Comparison & Upgrade Options")
    
    # Plan comparison data
    plans_data = {
        "Bronze": {
            "price_monthly": "$29",
            "price_annual": "$290",
            "transactions": "500/month",
            "reports": "3 basic",
            "support": "Email",
            "history": "7 days",
            "features": [
                "Up to 500 monthly transactions",
                "Basic account balance reports",
                "Mobile app access",
                "Email support",
                "7-day transaction history",
                "2-Factor Authentication"
            ]
        },
        "Silver": {
            "price_monthly": "$99",
            "price_annual": "$990",
            "transactions": "5,000/month",
            "reports": "9 types",
            "support": "Phone & Email",
            "history": "90 days",
            "features": [
                "Up to 5,000 monthly transactions",
                "Advanced reporting (9 report types)",
                "API access",
                "Mobile app + Web dashboard",
                "Phone & Email support",
                "90-day transaction history",
                "Advanced security (SSO, MFA)",
                "Custom alerts",
                "Scheduled reports"
            ]
        },
        "Gold": {
            "price_monthly": "$299",
            "price_annual": "$2,990",
            "transactions": "Unlimited",
            "reports": "9 + Custom",
            "support": "24/7 Dedicated",
            "history": "7 years",
            "features": [
                "Unlimited transactions",
                "All reports + custom reports",
                "Full API access with webhooks",
                "Multiple user accounts",
                "24/7 dedicated support team",
                "7-year transaction history",
                "Enterprise security",
                "Custom dashboards",
                "Real-time alerts",
                "SLA guarantee (99.9%)",
                "Priority onboarding",
                "Quarterly business reviews"
            ]
        }
    }
    
    # Create comparison table
    st.subheader("Quick Comparison")
    
    comparison_df = pd.DataFrame({
        "Feature": [
            "Monthly Price", "Annual Price", "Transactions/Month",
            "Reports Available", "Support", "History Retention"
        ],
        "Bronze": [
            plans_data["Bronze"]["price_monthly"],
            plans_data["Bronze"]["price_annual"],
            plans_data["Bronze"]["transactions"],
            plans_data["Bronze"]["reports"],
            plans_data["Bronze"]["support"],
            plans_data["Bronze"]["history"]
        ],
        "Silver": [
            plans_data["Silver"]["price_monthly"],
            plans_data["Silver"]["price_annual"],
            plans_data["Silver"]["transactions"],
            plans_data["Silver"]["reports"],
            plans_data["Silver"]["support"],
            plans_data["Silver"]["history"]
        ],
        "Gold": [
            plans_data["Gold"]["price_monthly"],
            plans_data["Gold"]["price_annual"],
            plans_data["Gold"]["transactions"],
            plans_data["Gold"]["reports"],
            plans_data["Gold"]["support"],
            plans_data["Gold"]["history"]
        ]
    })
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Plan cards (without border parameter)
    st.subheader("Plan Details")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🥉 Bronze")
        st.markdown(f"**{plans_data['Bronze']['price_monthly']}/mo**")
        st.info("⭐ Most Reasonable")
        st.markdown("---")
        for feature in plans_data['Bronze']['features']:
            st.markdown(f"✓ {feature}")
        st.markdown("---")
        if st.session_state.current_plan != "BRONZE":
            if st.button("⬆️ Upgrade to Bronze", key="bronze_upgrade"):
                st.session_state.selected_upgrade_plan = "BRONZE"
                st.session_state.show_payment_form = True
                st.rerun()
        else:
            st.markdown("**✅ Current Plan**")
    
    with col2:
        st.markdown("### 🥈 Silver (Popular)")
        st.markdown(f"**{plans_data['Silver']['price_monthly']}/mo**")
        st.info("⭐ Most popular")
        st.markdown("---")
        for feature in plans_data['Silver']['features']:
            st.markdown(f"✓ {feature}")
        st.markdown("---")
        if st.session_state.current_plan != "SILVER":
            if st.button("⬆️ Upgrade to Silver", key="silver_upgrade"):
                st.session_state.selected_upgrade_plan = "SILVER"
                st.session_state.show_payment_form = True
                st.rerun()
        else:
            st.markdown("**✅ Current Plan**")
    
    with col3:
        st.markdown("### 🥇 Gold (Enterprise)")
        st.markdown(f"**{plans_data['Gold']['price_monthly']}/mo**")
        st.success("🚀 Enterprise")
        st.markdown("---")
        for feature in plans_data['Gold']['features']:
            st.markdown(f"✓ {feature}")
        st.markdown("---")
        if st.session_state.current_plan != "GOLD":
            if st.button("⬆️ Upgrade to Gold", key="gold_upgrade"):
                st.session_state.selected_upgrade_plan = "GOLD"
                st.session_state.show_payment_form = True
                st.rerun()
        else:
            st.markdown("**✅ Current Plan**")

# =====================================================================
# PAYMENT FORM (Show/Hide on Upgrade Click)
# =====================================================================
if st.session_state.show_payment_form:
    st.markdown("---")
    st.markdown("## 💳 Complete Your Upgrade")
    
    # Show selected plan
    selected_plan = st.session_state.selected_upgrade_plan
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"📊 Upgrading to: **{selected_plan}** Plan")
    
    with col2:
        if st.button("❌ Cancel Upgrade"):
            st.session_state.show_payment_form = False
            st.session_state.selected_upgrade_plan = None
            st.rerun()
    
    st.markdown("### Enter Payment Details")
    
    # Payment form in columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        account_number = st.text_input(
            "Account Number",
            placeholder="Enter your account number",
            type="password",
            key="account_number"
        )
        
        card_holder_name = st.text_input(
            "Card Holder Name",
            placeholder="John Doe",
            key="cardholder_name"
        )
        
        email = st.text_input(
            "Email Address",
            placeholder="john@example.com",
            key="email"
        )
    
    with col2:
        expiry_date = st.text_input(
            "Expiry Date (MM/YY)",
            placeholder="12/25",
            key="expiry_date"
        )
        
        cvv = st.text_input(
            "CVV",
            placeholder="123",
            type="password",
            key="cvv"
        )
        
        amount = st.text_input(
            "Amount",
            placeholder="99.00",
            key="amount",
            disabled=True,
            value="99.00 (Sample)"
        )
    
    # Submit button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("✅ Complete Payment", key="submit_payment"):
            # Validate form
            if not all([account_number, card_holder_name, email, expiry_date, cvv]):
                st.error("❌ Please fill in all fields")
            else:
                # Call your payment API
                with st.spinner("Processing payment..."):
                    try:
                        # Replace with your actual API endpoint
                        payment_data = {
                            "user_id": st.session_state.user_id,
                            "from_plan": st.session_state.current_plan,
                            "to_plan": selected_plan,
                            "account_number": account_number,
                            "cardholder_name": card_holder_name,
                            "email": email,
                            "expiry_date": expiry_date,
                            "cvv": cvv
                        }
                        
                        # Example: Uncomment and modify for your API
                        # response = requests.post(
                        #     f"{API_BASE_URL}/process-payment",
                        #     json=payment_data,
                        #     timeout=30
                        # )
                        
                        # For now, simulate success
                        response_status = 200  # response.status_code
                        
                        if response_status == 200:
                            # Update current plan
                            st.session_state.current_plan = selected_plan
                            st.session_state.show_payment_form = False
                            st.session_state.upgrade_success = True
                            st.rerun()
                        else:
                            st.error("❌ Payment processing failed. Please try again.")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

# =====================================================================
# SUCCESS MESSAGE
# =====================================================================
if st.session_state.upgrade_success:
    st.success("🎉 Upgrade successful! Your plan has been updated.")
    st.balloons()
    
    # Reset success flag after showing
    if st.button("Close"):
        st.session_state.upgrade_success = False
        st.rerun()

# =====================================================================
# FOOTER
# =====================================================================
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📊 Features**")
    st.markdown("- Real-time balance tracking")
    st.markdown("- Plan comparison")
    st.markdown("- Easy upgrades")

with col2:
    st.markdown("**🔒 Security**")
    st.markdown("- End-to-end encryption")
    st.markdown("- Bank-level security")
    st.markdown("- Multi-factor auth")

with col3:
    st.markdown("**💬 Support**")
    st.markdown("- 24/7 dedicated team")
    st.markdown("- Email support")
    st.markdown("- Phone support")

st.markdown("*Powered by Supernova GenAI Hackthon Team*")