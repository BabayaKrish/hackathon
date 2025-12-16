# frontend/app.py - FIXED VERSION (Chat Input Outside Tabs)

import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any
import logging
import time

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
if 'plan_comparison_data' not in st.session_state:
    st.session_state.plan_comparison_data = None
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0
if 'last_request_time' not in st.session_state:
    st.session_state.last_request_time = 0

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

# ===== TABS =====
tab1, tab2, tab3 = st.tabs(["💬 Chat Assistant", "📊 Plan Comparison", "💳 Upgrade & Payment"])

# =====================================================================
# TAB 1: CHAT ASSISTANT
# =====================================================================
with tab1:
    st.markdown("## Chat with Your Financial Assistant")
    
    # Display chat history
    chat_container = st.container(height=400)
    with chat_container:
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
    
    # Chat input and processing - OUTSIDE tabs but inside tab1
    user_input = st.chat_input(
        "Ask about balance, plans, wire transfers...",
        key="chat_input"
    )
    
    if user_input:
        # Check rate limiting
        current_time = time.time()
        time_since_last_request = current_time - st.session_state.last_request_time
        if time_since_last_request < 2.0:  # Minimum 2 seconds between requests
            st.warning("⏳ Please wait a moment before sending another message.")
            st.stop()
        
        st.session_state.last_request_time = current_time
        
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
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
                    error = result.get('error', 'Unknown error')
                    if "quota" in error.lower() or "rate limit" in error.lower():
                        error_msg = f"⚠️ **API Rate Limit Reached**\n\nThe AI service is currently at capacity. Please wait a moment and try again.\n\n*Your request is being processed with fallback logic.*"
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                        with st.chat_message("assistant"):
                            st.warning(error_msg)
                    else:
                        error_msg = f"❌ Error: {error}"
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
            "savings": "Save $58/year",
            "transactions": "500/month",
            "reports": "3 basic",
            "api_access": "❌",
            "custom_reports": "❌",
            "support": "Email",
            "history": "7 days",
            "security": "2FA",
            "sla": "❌",
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
            "savings": "Save $198/year",
            "transactions": "5,000/month",
            "reports": "9 types",
            "api_access": "✅",
            "custom_reports": "❌",
            "support": "Phone & Email",
            "history": "90 days",
            "security": "SSO, MFA",
            "sla": "❌",
            "features": [
                "Up to 5,000 monthly transactions",
                "Advanced reporting (9 report types)",
                "API access",
                "Mobile app + Web dashboard",
                "Phone & Email support (4-8 hours)",
                "90-day transaction history",
                "Advanced security (SSO, MFA)",
                "Custom alerts",
                "Scheduled reports"
            ]
        },
        "Gold": {
            "price_monthly": "$299",
            "price_annual": "$2,990",
            "savings": "Save $588/year",
            "transactions": "Unlimited",
            "reports": "9 + Custom",
            "api_access": "✅",
            "custom_reports": "✅",
            "support": "24/7 Dedicated",
            "history": "7 years",
            "security": "Enterprise",
            "sla": "99.9%",
            "features": [
                "Unlimited transactions",
                "All reports + custom reports",
                "Full API access with webhooks",
                "Multiple user accounts",
                "24/7 dedicated support team",
                "7-year transaction history",
                "Enterprise security (SSO, MFA, IP restrictions)",
                "Custom dashboards",
                "Real-time alerts",
                "SLA guarantee (99.9% uptime)",
                "Priority onboarding",
                "Quarterly business reviews"
            ]
        }
    }
    
    # Create comparison table
    st.subheader("Quick Comparison")
    
    comparison_df = pd.DataFrame({
        "Feature": [
            "Monthly Price",
            "Annual Price",
            "Transactions/Month",
            "Reports Available",
            "API Access",
            "Custom Reports",
            "Support",
            "History Retention",
            "Security",
            "SLA Guarantee"
        ],
        "Bronze": [
            plans_data["Bronze"]["price_monthly"],
            plans_data["Bronze"]["price_annual"],
            plans_data["Bronze"]["transactions"],
            plans_data["Bronze"]["reports"],
            plans_data["Bronze"]["api_access"],
            plans_data["Bronze"]["custom_reports"],
            plans_data["Bronze"]["support"],
            plans_data["Bronze"]["history"],
            plans_data["Bronze"]["security"],
            plans_data["Bronze"]["sla"]
        ],
        "Silver": [
            plans_data["Silver"]["price_monthly"],
            plans_data["Silver"]["price_annual"],
            plans_data["Silver"]["transactions"],
            plans_data["Silver"]["reports"],
            plans_data["Silver"]["api_access"],
            plans_data["Silver"]["custom_reports"],
            plans_data["Silver"]["support"],
            plans_data["Silver"]["history"],
            plans_data["Silver"]["security"],
            plans_data["Silver"]["sla"]
        ],
        "Gold": [
            plans_data["Gold"]["price_monthly"],
            plans_data["Gold"]["price_annual"],
            plans_data["Gold"]["transactions"],
            plans_data["Gold"]["reports"],
            plans_data["Gold"]["api_access"],
            plans_data["Gold"]["custom_reports"],
            plans_data["Gold"]["support"],
            plans_data["Gold"]["history"],
            plans_data["Gold"]["security"],
            plans_data["Gold"]["sla"]
        ]
    })
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Detailed plan cards
    st.subheader("Detailed Plan Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🥉 Bronze Plan")
        st.markdown(f"**{plans_data['Bronze']['price_monthly']}/month**")
        st.markdown(f"*or {plans_data['Bronze']['price_annual']}/year ({plans_data['Bronze']['savings']})*")
        
        with st.container(border=True):
            for feature in plans_data['Bronze']['features']:
                st.markdown(f"✓ {feature}")
            
            st.divider()
            
            if st.session_state.current_plan != "BRONZE":
                if st.button("⬆️ Upgrade to Bronze", key="upgrade_bronze", use_container_width=True):
                    st.session_state.selected_upgrade_plan = "BRONZE"
                    st.session_state.show_payment_form = True
                    st.session_state.active_tab = 2
                    st.rerun()
            else:
                st.markdown("**✅ Current Plan**", help="You are currently on the Bronze plan")
    
    with col2:
        st.markdown("### 🥈 Silver Plan (Popular)")
        st.markdown(f"**{plans_data['Silver']['price_monthly']}/month**")
        st.markdown(f"*or {plans_data['Silver']['price_annual']}/year ({plans_data['Silver']['savings']})*")
        
        with st.container(border=True):
            st.info("⭐ Most popular choice for growing businesses")
            for feature in plans_data['Silver']['features']:
                st.markdown(f"✓ {feature}")
            
            st.divider()
            
            if st.session_state.current_plan != "SILVER":
                if st.button("⬆️ Upgrade to Silver", key="upgrade_silver", use_container_width=True):
                    st.session_state.selected_upgrade_plan = "SILVER"
                    st.session_state.show_payment_form = True
                    st.session_state.active_tab = 2
                    st.rerun()
            else:
                st.markdown("**✅ Current Plan**", help="You are currently on the Silver plan")
    
    with col3:
        st.markdown("### 🥇 Gold Plan (Enterprise)")
        st.markdown(f"**{plans_data['Gold']['price_monthly']}/month**")
        st.markdown(f"*or {plans_data['Gold']['price_annual']}/year ({plans_data['Gold']['savings']})*")
        
        with st.container(border=True):
            st.success("🚀 Enterprise-grade solution")
            for feature in plans_data['Gold']['features']:
                st.markdown(f"✓ {feature}")
            
            st.divider()
            
            if st.session_state.current_plan != "GOLD":
                if st.button("⬆️ Upgrade to Gold", key="upgrade_gold", use_container_width=True):
                    st.session_state.selected_upgrade_plan = "GOLD"
                    st.session_state.show_payment_form = True
                    st.session_state.active_tab = 2
                    st.rerun()
            else:
                st.markdown("**✅ Current Plan**", help="You are currently on the Gold plan")

# =====================================================================
# TAB 3: UPGRADE & PAYMENT
# =====================================================================
with tab3:
    st.markdown("## 💳 Plan Upgrade & Payment")
    
    if st.session_state.selected_upgrade_plan:
        selected_plan = st.session_state.selected_upgrade_plan
        plan_info = plans_data[selected_plan]
        
        st.success(f"✅ You selected: **{selected_plan} Plan**")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📋 Plan Summary")
            st.markdown(f"""
- **Plan**: {selected_plan}
- **Monthly Price**: {plan_info['price_monthly']}
- **Annual Price**: {plan_info['price_annual']}
- **Transactions**: {plan_info['transactions']}
- **Support**: {plan_info['support']}
            """)
        
        with col2:
            st.markdown("### 💰 Billing Options")
            billing_choice = st.radio(
                "Choose billing cycle",
                ["Monthly", "Annual"],
                key="billing_choice"
            )
            
            if billing_choice == "Monthly":
                price = plan_info['price_monthly'].replace('$', '')
                billing_period = "month"
            else:
                price = plan_info['price_annual'].replace('$', '')
                billing_period = "year"
            
            st.info(f"**Total: {billing_choice} ${price} per {billing_period}**")
        
        st.divider()
        
        # Payment form
        st.markdown("### 🔐 Payment Information")
        
        with st.form("payment_form", border=True):
            st.markdown("**Account Information**")
            
            col1, col2 = st.columns(2)
            with col1:
                account_number = st.text_input(
                    "Account Number",
                    placeholder="Enter your account number",
                    type="password",
                    key="account_number"
                )
            
            with col2:
                routing_number = st.text_input(
                    "Routing Number",
                    placeholder="Enter your routing number",
                    type="password",
                    key="routing_number"
                )
            
            st.markdown("**Billing Address**")
            
            col1, col2 = st.columns(2)
            with col1:
                first_name = st.text_input(
                    "First Name",
                    placeholder="John",
                    key="first_name"
                )
            
            with col2:
                last_name = st.text_input(
                    "Last Name",
                    placeholder="Doe",
                    key="last_name"
                )
            
            col1, col2 = st.columns(2)
            with col1:
                street_address = st.text_input(
                    "Street Address",
                    placeholder="123 Main St",
                    key="street_address"
                )
            
            with col2:
                city = st.text_input(
                    "City",
                    placeholder="New York",
                    key="city"
                )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                state = st.text_input(
                    "State",
                    placeholder="NY",
                    key="state",
                    max_chars=2
                )
            
            with col2:
                zip_code = st.text_input(
                    "ZIP Code",
                    placeholder="10001",
                    key="zip_code"
                )
            
            with col3:
                country = st.text_input(
                    "Country",
                    placeholder="USA",
                    key="country"
                )
            
            st.markdown("**Confirmation**")
            
            agree_terms = st.checkbox(
                "I agree to the terms and conditions",
                key="agree_terms"
            )
            
            agree_payment = st.checkbox(
                f"I confirm upgrading to {selected_plan} plan at {billing_choice} billing",
                key="agree_payment"
            )
            
            # Submit button
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                submitted = st.form_submit_button(
                    "✅ Complete Upgrade",
                    type="primary",
                    use_container_width=True,
                    disabled=not (agree_terms and agree_payment and account_number and routing_number)
                )
            
            if submitted:
                if not all([first_name, last_name, street_address, city, state, zip_code]):
                    st.error("❌ Please fill in all fields")
                else:
                    # Process payment
                    payment_data = {
                        "user_id": st.session_state.user_id,
                        "current_plan": st.session_state.current_plan,
                        "new_plan": selected_plan,
                        "billing_cycle": billing_choice,
                        "amount": price,
                        "account_number": account_number,
                        "routing_number": routing_number,
                        "first_name": first_name,
                        "last_name": last_name,
                        "street_address": street_address,
                        "city": city,
                        "state": state,
                        "zip_code": zip_code,
                        "country": country,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    try:
                        # Call backend payment endpoint
                        response = requests.post(
                            f"{API_BASE_URL}/upgrade-plan",
                            json=payment_data,
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            st.success(f"✅ **Plan upgrade successful!** Your plan has been upgraded to {selected_plan}.")
                            st.balloons()
                            
                            # Update current plan
                            st.session_state.current_plan = selected_plan
                            st.session_state.selected_upgrade_plan = None
                            st.session_state.show_payment_form = False
                            
                            st.info(f"🎉 Welcome to the {selected_plan} plan! Your benefits are now active.")
                            
                            # Log transaction
                            logger.info(f"[Payment] Upgrade successful: {st.session_state.user_id} -> {selected_plan}")
                        else:
                            st.error(f"❌ Payment failed: {response.text}")
                    
                    except Exception as e:
                        st.error(f"❌ Error processing payment: {str(e)}")
    
    else:
        st.info("👈 Select a plan from the **Plan Comparison** tab to upgrade")
        
        st.markdown("### Current Plan: **{}**".format(st.session_state.current_plan))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Compare Plans", use_container_width=True):
                st.session_state.active_tab = 1
                st.rerun()
        
        with col2:
            if st.button("Chat with Assistant", use_container_width=True):
                st.session_state.active_tab = 0
                st.rerun()
        
        with col3:
            st.button("View Reports", use_container_width=True, disabled=True)

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

st.markdown("*Powered by Multi-Agent AI | FastAPI + Streamlit + Vertex AI*")