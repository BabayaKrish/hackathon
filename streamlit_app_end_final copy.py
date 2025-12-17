# frontend/streamlit_app_end_final.py - Financial Assistant (FIXED)
# FIX: Moved st.chat_input() OUTSIDE the with tab1: block

import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any
import logging
import plotly.express as px
import plotly.graph_objects as go

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


# ===== Initialize session state =====
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_plan' not in st.session_state:
    st.session_state.current_plan = 'GOLD'
if 'user_id' not in st.session_state:
    st.session_state.user_id = 'user_001'
if 'selected_upgrade_plan' not in st.session_state:
    st.session_state.selected_upgrade_plan = None
if 'show_payment_form' not in st.session_state:
    st.session_state.show_payment_form = False
if 'upgrade_success' not in st.session_state:
    st.session_state.upgrade_success = False
if 'report_data' not in st.session_state:
    st.session_state.report_data = None
if 'report_type' not in st.session_state:
    st.session_state.report_type = 'transactions_today'
if 'last_processed_input' not in st.session_state:
    st.session_state.last_processed_input = ""

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
st.markdown("# 💬 AgenticVantage Assist")
st.caption(f"👤 User: **{st.session_state.user_id}** | Current Plan: **{st.session_state.current_plan}**")

# ===== TABS =====
tab1, tab2, tab3 = st.tabs(["💬 Chat Assistant", "📊 Plan Comparison", "📈 Reports"])

def display_additional_features_in_chat(response: Dict[str, Any]) -> str:
    """
    Format additional features response for chat display, with dummy pricing.
    Handles both list of strings and single comma-separated string.
    """
    features = response.get("additional_features", [])
    markdown = ""
    markdown += "✨ **Additional Features You Can Add**\n\n"
    # Handle if features is a single string (comma-separated)
    if isinstance(features, str):
        features = [f.strip() for f in features.split(",") if f.strip()]
    # Remove empty or 1-char features
    features = [f for f in features if isinstance(f, str) and len(f.strip()) > 2]
    if not features:
        markdown += "No additional features available at this time."
        return markdown
    # Limit to 20 features for display
    for idx, feat in enumerate(features[:20], 1):
        price = 10 * idx
        markdown += f"\n• {feat} — **${price}/mo**"
    if len(features) > 20:
        markdown += f"\n...and {len(features)-20} more features."
    markdown += "\n\nTo add a feature, contact support or use the upgrade options."
    return markdown.strip()

def display_plan_upgrade_in_chat(response: Dict[str, Any]) -> str:
    """
    Format plan upgrade response as markdown for chat display,
    showing the requested plan details FIRST, then AI recommendation
    and other context.
    """
    logger.info(f"[streamlit] Formatting plan info response for chat display")
    logger.info(f"[streamlit] Response keys: {response}")
    if response.get("status") != "success":
        return f"❌ **Error Processing Request**\n\nError: {response.get('error', 'Unknown error')}"

    # ---- Extract core fields ----
    confidence = response.get("confidence", 0.0)
    mentioned_tier = (response.get("mentioned_tier") or "").strip()
    all_plans = response.get("all_plans", [])
    current_plan_details = response.get("current_plan_details", {})
    upgrade_options = response.get("upgrade_options", [])
    ai_recommendation = response.get("ai_recommendation", {})
    next_actions = response.get("next_actions", [])

    markdown = ""
    # If additional_features is present and non-empty, show them as a special block
    additional_features = response.get("additional_features", [])
    if additional_features:
        return display_additional_features_in_chat(response)

    # ============================================================
    # 1. PRIMARY: Requested plan details (e.g. Silver)
    #    - if user asked "Get the plan details for silver"
    #    - use mentioned_tier first, then fall back to current_plan_details
    # ============================================================

    # Try to find the requested tier in all_plans
    requested_plan = None
    if mentioned_tier:
        for p in all_plans:
            if str(p.get("tier", "")).lower() == mentioned_tier.lower() or \
               str(p.get("plan_name", "")).lower().startswith(mentioned_tier.lower()):
                requested_plan = p
                break

    # If not found from all_plans, fall back to current_plan_details.plan
    if not requested_plan and current_plan_details and current_plan_details.get("status") == "success":
        requested_plan = current_plan_details.get("plan")

    # Header
    markdown += "📋 **Plan Information**\n\n"

    if requested_plan:
        name = requested_plan.get("plan_name", "Unknown Plan")
        tier = requested_plan.get("tier", "N/A")
        monthly = requested_plan.get("monthly_price")
        annual = requested_plan.get("annual_price")
        features = requested_plan.get("features", "")

        tier_emoji = "🥇" if tier.lower() == "gold" else \
                     "🥈" if tier.lower() == "silver" else \
                     "🥉" if tier.lower() == "bronze" else "🆓"

        markdown += f"### {tier_emoji} **{name}** ({tier})\n\n"
        markdown += "**💰 Pricing:**\n"
        if monthly is not None:
            markdown += f"• Monthly: **${monthly:.2f}**\n"
        else:
            markdown += "• Monthly: **Free**\n"
        if annual is not None:
            markdown += f"• Annual: **${annual:.2f}**"
            if monthly:
                savings = (monthly * 12) - annual
                if savings > 0:
                    savings_pct = (savings / (monthly * 12)) * 100
                    markdown += f" (Save **${savings:.2f}** / year - {savings_pct:.0f}% off)"
            markdown += "\n"
        else:
            markdown += "• Annual: **Free**\n"

        if features:
            feats = [f.strip() for f in features.split(",") if f.strip()]
            if tier.lower() == "no plan":
                markdown += "\n**✨ Additional Features You Can Add:**\n"
                for f in feats:
                    markdown += f"• {f}\n"
            else:
                markdown += "\n**✨ Other optional Features Included that you can add to the account:**\n"
                for f in feats:
                    markdown += f"• {f}\n"
        markdown += "\n"
    else:
        markdown += "Could not find specific plan details for your request.\n\n"

    # ============================================================
    # 2. SECONDARY: AI-powered recommendation (optional)
    # ============================================================

    if ai_recommendation and ai_recommendation.get("status") == "success":
        markdown += "---\n\n"
        markdown += "### 🤖 AI-Powered Recommendation\n\n"
        recommended_plan = ai_recommendation.get("recommended_plan")
        key_benefits = ai_recommendation.get("key_benefits", [])
        cost_analysis = ai_recommendation.get("cost_analysis", "")
        reasoning = ai_recommendation.get("reasoning", "")

        if recommended_plan:
            markdown += f"**Recommended Plan:** `{recommended_plan}`\n\n"
        if reasoning:
            markdown += f"**Why:** {reasoning}\n\n"
        if key_benefits:
            markdown += "**🎯 Key Benefits:**\n"
            for b in key_benefits:
                markdown += f"✨ {b}\n"
            markdown += "\n"
        if cost_analysis:
            markdown += f"**💳 Cost Breakdown:** {cost_analysis}\n\n"

    # ============================================================
    # 3. TERTIARY: Upgrade options (if any)
    # ============================================================

    if upgrade_options:
        markdown += "---\n\n"
        markdown += "### ⬆️ Available Upgrade Options\n\n"
        for idx, up in enumerate(upgrade_options, 1):
            pname = up.get("plan_name", "Unknown")
            tier = up.get("tier", "N/A")
            monthly = up.get("monthly_price")
            annual = up.get("annual_price")
            feats = up.get("features", "")
            tier_emoji = "🥇" if tier.lower() == "gold" else \
                         "🥈" if tier.lower() == "silver" else "🥉"

            price_str = ""
            if monthly is not None:
                price_str = f"${monthly:.2f}/mo"
            if annual is not None:
                if price_str:
                    price_str += f" or ${annual:.2f}/year"
                else:
                    price_str = f"${annual:.2f}/year"

            markdown += f"{idx}. {tier_emoji} **{pname}** - {price_str}\n"
            if feats:
                flist = [f.strip() for f in feats.split(",") if f.strip()]
                markdown += f"   • Features: {', '.join(flist[:2])}"
                if len(flist) > 2:
                    markdown += f", +{len(flist) - 2} more"
                markdown += "\n"
            markdown += "\n"

    # ============================================================
    # 4. QUATERNARY: All plans table (overview)
    # ============================================================

    if all_plans and len(all_plans) > 1:
        markdown += "---\n\n"
        markdown += "### 📊 Complete Plans Overview\n\n"
        markdown += "| Plan | Tier | Monthly | Annual | Features |\n"
        markdown += "|------|------|---------|--------|----------|\n"
        for p in all_plans:
            pname = p.get("plan_name", "Unknown")
            tier = p.get("tier", "N/A")
            monthly = p.get("monthly_price")
            annual = p.get("annual_price")
            feats = p.get("features", "")
            monthly_str = f"${monthly:.2f}" if monthly else "Free"
            annual_str = f"${annual:.2f}" if annual else "Free"
            fcount = len(feats.split(",")) if feats else 0
            markdown += f"| {pname} | {tier} | {monthly_str} | {annual_str} | {fcount} |\n"
        markdown += "\n"

    # ============================================================
    # 5. Next actions
    # ============================================================

    if next_actions:
        markdown += "---\n\n"
        markdown += "### 💡 Next Steps\n\n"
        for action in next_actions:
            if action == "show_comparison_table":
                markdown += "🔹 **View Detailed Comparison** - See side-by-side feature comparison in the Plan Comparison tab\n"
            elif action == "show_upgrade_prompt":
                markdown += "🔹 **Choose & Upgrade** - Select a plan from the recommendations above\n"
            else:
                markdown += f"🔹 **{action.replace('_', ' ').title()}**\n"
        markdown += "\n💬 *Need help? Ask me any questions about the plans!*"

    return markdown.strip()


# ===== Helper function =====
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
            logger.info(f"[streamlit] Response keys: {list(result.keys())}")
            return result
        else:
            logger.error(f"[streamlit] ❌ Root Agent error: {response.status_code}")
            return {"status": "error", "error": response.text}
    except Exception as e:
        logger.error(f"[streamlit] ❌ Failed to call Root Agent: {str(e)}")
        return {"status": "error", "error": str(e)}

def generate_report(user_id: str, report_type: str, time_period: str = "today") -> Dict:
    """Generate a report from backend"""
    logger.info(f"[streamlit] Generating report: {report_type}")
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate-report",
            json={
                "user_id": user_id,
                "report_type": report_type,
                "time_period": time_period
            },
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            logger.info(f"[streamlit] ✅ Report generated successfully")
            return result
        else:
            logger.error(f"[streamlit] ❌ Report generation error: {response.status_code}")
            return {"status": "error", "error": response.text}
    except Exception as e:
        logger.error(f"[streamlit] ❌ Failed to generate report: {str(e)}")
        return {"status": "error", "error": str(e)}

# ✅ CRITICAL FIX: st.chat_input() OUTSIDE ALL TABS
# This must be at the main level, not inside any tab
user_input = st.chat_input(
    "Ask about balance, plans, wire transfers...",
    key="chat_input"
)

# =====================================================================
# TAB 1: CHAT ASSISTANT
# =====================================================================
with tab1:
    st.markdown("## Chat with Your Financial Assistant")
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Process chat input (from OUTSIDE the tab)
    if user_input and user_input != st.session_state.last_processed_input:
        st.session_state.last_processed_input = user_input
        
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Process with Root Agent
        with st.spinner("Processing..."):
            try:
                result = send_to_root_agent(user_input)
                handled = False
                if result.get("status") == "success":
                    assistant_response = ""
                    intent = result.get("intent")
                    agent_used = result.get("agent_used")
                    data = result.get("data", {})
                    confidence = result.get("confidence", 0.0)
                    reasoning = result.get("reasoning", "")
                    
                    logger.info(f"[streamlit] Intent: {intent}, Agent: {agent_used}, Confidence: {confidence}")
                    
                    # ✅ HANDLE OUT_OF_SCOPE QUERIES
                    if intent == "OUT_OF_SCOPE":
                        handled = True
                        reasoning_str = str(reasoning) if not isinstance(reasoning, str) else reasoning
                        assistant_response = f"""🤔 **I'm Not Sure About That**

{reasoning_str}

💡 **I can help you with:**
✅ Showing wire transfers, balance report, intraday balance and ledger and reports
✅ Comparing and managing plans
✅ Answering questions about features and pricing

Feel free to ask me about any of these topics!"""
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": assistant_response
                        })
                        with st.chat_message("assistant"):
                            st.markdown(assistant_response)
                    
                    # ✅ FORMAT RESPONSE BASED ON AGENT TYPE
                    elif agent_used == "balance_agent":
                        balance = data.get("balance", 0)
                        transactions = data.get("transaction_count", 0)
                        assistant_response = f"""💰 **Your Account Balance: ${balance:,.2f}**

📈 **Recent Activity:**

💡 **Quick Tips:**
✅ Use 'show my wire transfers' for wire details
✅ Use 'what's in Gold plan' for plan comparison
✅ Visit the 'Reports' tab for detailed analytics"""
                    
                    elif agent_used == "report_agent":
                        handled = True
                        reports = []
                        if isinstance(data, list):
                            reports = data
                        elif "report_data" in data:
                            try:
                                reports = json.loads(data["report_data"])
                            except Exception as e:
                                reports = []
                        elif "reports" in data:
                            reports = data.get("reports", [])
                        logger.info(f"[streamlit] Report Agent data: {data}")
                        if reports:
                            df = pd.DataFrame(reports)
                            # Balance Report
                            if all(col in df.columns for col in ["balance_date", "opening_balance", "closing_balance"]):
                                assistant_response = f"📊 **Balance Report** ({len(reports)} records)\n\n"
                                with st.chat_message("assistant"):
                                    st.markdown(assistant_response)
                                    st.dataframe(df, use_container_width=True)
                                    import plotly.graph_objects as go
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(x=df["balance_date"], y=df["opening_balance"], mode="lines+markers", name="Opening Balance"))
                                    fig.add_trace(go.Scatter(x=df["balance_date"], y=df["closing_balance"], mode="lines+markers", name="Closing Balance"))
                                    fig.update_layout(title="Opening & Closing Balances Over Time", xaxis_title="Date", yaxis_title="Balance", height=500, template="plotly_white")
                                    st.plotly_chart(fig, use_container_width=True)
                            # Intraday Balance Report
                            elif all(col in df.columns for col in ["balance_timestamp", "current_balance", "available_balance"]):
                                assistant_response = f"📊 **Intraday Balance Report** ({len(reports)} records)\n\n"
                                with st.chat_message("assistant"):
                                    st.markdown(assistant_response)
                                    st.dataframe(df, use_container_width=True)
                                    # Bar chart for current/available balance over time
                                    fig = go.Figure()
                                    fig.add_trace(go.Bar(x=df["balance_timestamp"], y=df["current_balance"], name="Current Balance"))
                                    fig.add_trace(go.Bar(x=df["balance_timestamp"], y=df["available_balance"], name="Available Balance"))
                                    fig.update_layout(barmode="group", title="Intraday Balances Over Time", xaxis_title="Timestamp", yaxis_title="Balance", height=500, template="plotly_white")
                                    st.plotly_chart(fig, use_container_width=True)
                            # Running Ledger Report
                            elif all(col in df.columns for col in ["transaction_type", "amount"]):
                                assistant_response = f"📊 **Running Ledger** ({len(reports)} records)\n\n"
                                with st.chat_message("assistant"):
                                    st.markdown(assistant_response)
                                    st.dataframe(df, use_container_width=True)
                                    # Bar chart: total amount by transaction_type
                                    bar_df = df.groupby("transaction_type")["amount"].sum().reset_index()
                                    fig = px.bar(bar_df, x="transaction_type", y="amount", title="Total Amount by Transaction Type", color="transaction_type")
                                    st.plotly_chart(fig, use_container_width=True)
                                    # Pie chart: amount by transaction_type
                                    if bar_df.shape[0] > 1:
                                        fig2 = px.pie(bar_df, names="transaction_type", values="amount", title="Amount Distribution by Transaction Type")
                                        st.plotly_chart(fig2, use_container_width=True)
                            # Default: Wire Transfer Report
                            else:
                                display_cols = [
                                    col for col in [
                                        "wire_amount", "status", "wire_date", "destination_account", "destination_bank", "beneficiary_name", "reference_number"
                                    ] if col in df.columns
                                ]
                                pretty_df = df[display_cols] if display_cols else df
                                assistant_response = f"📊 **Wire Transfer Reports** ({len(reports)} found)\n\n"
                                with st.chat_message("assistant"):
                                    st.markdown(assistant_response)
                                    st.dataframe(pretty_df, use_container_width=True)
                                    if df.shape[0] > 0 and "wire_amount" in df.columns and "status" in df.columns:
                                        fig = px.pie(df, names="status", values="wire_amount", title="Wire Amount Distribution by Status")
                                        st.plotly_chart(fig, use_container_width=True)
                            # Do not return here; just end the block to prevent double rendering
                        else:
                            if not handled:
                                assistant_response = "No wire reports found."
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": assistant_response
                                })
                                with st.chat_message("assistant"):
                                    st.markdown(assistant_response)

                    elif agent_used == "plan_agent":
                        # Simple confirmation message for plan changes (upgrade/downgrade)
                        action = data.get("action", "plan_change")
                        current_plan = data.get("current_plan", "current")
                        new_plan = data.get("new_plan", "selected")
                        message = data.get("message", "Processing your request...")
                        next_step = data.get("next_step", "payment_modal")

                        assistant_response = f"""🔄 **Plan Change Request**\n\n"""
                        assistant_response += f"**From:** `{current_plan}` → **To:** `{new_plan}`\n\n"
                        assistant_response += f"**Status:** {message}\n"
                        assistant_response += f"**Next Step:** {next_step.replace('_', ' ').title()}"

                    elif agent_used == "plan_info_agent":
                                            # If additional_features present, show them differently
                                            if data.get("additional_features"):
                                                assistant_response = display_additional_features_in_chat(data)
                                            else:
                                                assistant_response = display_plan_upgrade_in_chat(data)
                    
                    # Only add to history and render if not handled above
                    if not handled:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": assistant_response
                        })
                        with st.chat_message("assistant"):
                            st.markdown(assistant_response)
                            if 'reports' in locals() and reports:
                                st.dataframe(pd.DataFrame(reports))
                
                elif result.get("status") == "low_confidence":
                    error_msg = f"🤔 I'm not sure about that ({result.get('confidence', 0)*100:.0f}% confident). Could you rephrase?"
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
                logger.error(f"[streamlit] Exception: {error_msg}")
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.error(error_msg)
        
        #st.rerun()

# =====================================================================
# TAB 2: PLAN COMPARISON
# =====================================================================
with tab2:
    logger.info("[streamlit] Plan Comparison tab selected")
    print("[streamlit] Plan Comparison tab selected")
    st.markdown("## 📊 Plan Comparison & Upgrade Options")
    
    # Fetch plan info from API (define and call inside tab block)
    import requests
    def fetch_plans_data():
        print("[streamlit] fetch_plans_data CALLED")
        try:
            response = requests.get(f"{API_BASE_URL}/plans", timeout=10)
            logger.info(f"[streamlit] Fetched plans data, status code: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                logger.info(f"[streamlit] Fetched plans data: {result}")
                plans = result.get("plans", [])
                # Map API response to the required format
                plans_data = {}
                for plan in plans:
                    raw_name = plan.get("name", "").lower()
                    # Normalize to canonical keys
                    if "bronze" in raw_name:
                        key = "Bronze"
                    elif "silver" in raw_name:
                        key = "Silver"
                    elif "gold" in raw_name:
                        key = "Gold"
                    else:
                        continue  # skip unknown plans
                    price = plan.get("price", 0)
                    price_monthly = f"${price}" if price is not None else "N/A"
                    price_annual = f"${plan.get('price_annual', price*12)}" if plan.get('price_annual') else (f"${price*12}" if price else "N/A")
                    transactions = plan.get("transactions", "")
                    features = plan.get("features", [])
                    if isinstance(features, str):
                        features = [f.strip() for f in features.split(",") if f.strip()]
                    reports = plan.get("reports", "") if "reports" in plan else ""
                    support = plan.get("support", "") if "support" in plan else ""
                    history = plan.get("history", "") if "history" in plan else ""
                    plans_data[key] = {
                        "price_monthly": price_monthly,
                        "price_annual": price_annual,
                        "transactions": transactions,
                        "reports": reports,
                        "support": support,
                        "history": history,
                        "features": features
                    }
                return plans_data
            else:
                return {}
        except Exception as e:
            return {}

    # Call fetch_plans_data when tab is selected
    plans_data = fetch_plans_data()
    
    comparison_df = pd.DataFrame({
        "Feature": ["Monthly Price", "Annual Price", "Transactions/Month", "Reports Available", "Support", "History Retention"],
        "Bronze": ["$29", "$290", "500/month", "3 basic", "Email", "7 days"],
        "Silver": ["$99", "$990", "5,000/month", "9 types", "Phone & Email", "90 days"],
        "Gold": ["$299", "$2,990", "Unlimited", "9 + Custom", "24/7 Dedicated", "7 years"]
    })
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)

    # Track which upgrade button was clicked
    if 'show_payment_form' not in st.session_state:
        st.session_state.show_payment_form = False
    if 'selected_upgrade_plan' not in st.session_state:
        st.session_state.selected_upgrade_plan = None

    def show_payment(plan):
        st.session_state.show_payment_form = True
        st.session_state.selected_upgrade_plan = plan

    with col1:
        st.markdown("### 🥉 Bronze")
        bronze = plans_data.get('Bronze', plans_data.get('bronze', {}))
        if bronze:
            st.markdown(f"**{bronze.get('price_monthly', 'N/A')}/mo**")
            st.info("⭐ Affordable")
            for feature in bronze.get('features', []):
                st.markdown(f"✓ {feature}")
        else:
            st.warning("Bronze plan data not available.")
        if st.session_state.current_plan != "BRONZE":
            if st.button("⬆️ Switch to Bronze", key="bronze_upgrade"):
                show_payment('Bronze')
        else:
            st.markdown("**✅ Current Plan**")

    with col2:
        st.markdown("### 🥈 Silver (Popular)")
        silver = plans_data.get('Silver', plans_data.get('silver', {}))
        if silver:
            st.markdown(f"**{silver.get('price_monthly', 'N/A')}/mo**")
            st.info("⭐ Most popular")
            for feature in silver.get('features', []):
                st.markdown(f"✓ {feature}")
        else:
            st.warning("Silver plan data not available.")
        if st.session_state.current_plan != "SILVER":
            if st.button("⬆️ Switch to Silver", key="silver_upgrade"):
                show_payment('Silver')
        else:
            st.markdown("**✅ Current Plan**")

    with col3:
        st.markdown("### 🥇 Gold (Enterprise)")
        gold = plans_data.get('Gold', plans_data.get('gold', {}))
        if gold:
            st.markdown(f"**{gold.get('price_monthly', 'N/A')}/mo**")
            st.info("⭐ Most features")
            for feature in gold.get('features', []):
                st.markdown(f"✓ {feature}")
        else:
            st.warning("Gold plan data not available.")
        if st.session_state.current_plan != "GOLD":
            if st.button("⬆️ Switch to Gold", key="gold_upgrade"):
                show_payment('Gold')
        else:
            st.markdown("**✅ Current Plan**")

    # Show payment form if upgrade button was clicked
    if st.session_state.show_payment_form and st.session_state.selected_upgrade_plan:
        st.markdown(f"""
            <div>        
                <h3 style='text-align:center; margin-bottom: 1.5rem;'>💳 Billing Details for <span style='color:#0072C6'>{st.session_state.selected_upgrade_plan} Plan</span></h3>
            </div>
        """, unsafe_allow_html=True)
        # Use session state to persist form values
        if 'payment_form_data' not in st.session_state:
            st.session_state.payment_form_data = {
                'name': '',
                'account_number': '',
                'card_number': '',
                'exp_date': '',
                'cvv': ''
            }
        form_data = st.session_state.payment_form_data
        with st.form(key="billing_form"):
            form_data['name'] = st.text_input("👤 Name on Account", value=form_data['name'], placeholder="Full Name")
            form_data['account_number'] = st.text_input("🏦 Account Number", value=form_data['account_number'], placeholder="1234567890")
            st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)
            st.markdown("<b>Card Details</b>", unsafe_allow_html=True)
            card_col1, card_col2 = st.columns([2,1])
            with card_col1:
                form_data['card_number'] = st.text_input("💳 Card Number", value=form_data['card_number'], placeholder="1234 5678 9012 3456", max_chars=19)
            with card_col2:
                form_data['exp_date'] = st.text_input("Exp. (MM/YY)", value=form_data['exp_date'], placeholder="MM/YY", max_chars=5)
            cvv_col, _ = st.columns([1,2])
            with cvv_col:
                form_data['cvv'] = st.text_input("CVV", value=form_data['cvv'], placeholder="123", type="password", max_chars=4)
            st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("💸 Proceed Payment", use_container_width=True)
            cancel = st.form_submit_button("Cancel", use_container_width=True)
            # Validation: all fields required
            missing_fields = not all([
                form_data['name'].strip(),
                form_data['account_number'].strip(),
                form_data['card_number'].strip(),
                form_data['exp_date'].strip(),
                form_data['cvv'].strip()
            ])
            if submitted:
                if missing_fields:
                    st.warning("Please fill in all required fields.")
                    st.session_state.show_confirm_popup = False
                else:
                    st.session_state.show_confirm_popup = True
            elif cancel:
                st.session_state.show_payment_form = False
                st.session_state.selected_upgrade_plan = None
                st.session_state.payment_form_data = {
                    'name': '',
                    'account_number': '',
                    'card_number': '',
                    'exp_date': '',
                    'cvv': ''
                }
                st.session_state.show_confirm_popup = False

        # Show confirmation popup if requested
        if st.session_state.get('show_confirm_popup', False):
            st.info("Are you sure you want to proceed to payment?")
            col_ok, col_cancel = st.columns([1,1])
            with col_ok:
                if st.button("OK", key="confirm_payment_btn"):
                    st.success("✅ Payment processed! (Demo)")
                    st.session_state.show_payment_form = False
                    st.session_state.selected_upgrade_plan = None
                    st.session_state.payment_form_data = {
                        'name': '',
                        'account_number': '',
                        'card_number': '',
                        'exp_date': '',
                        'cvv': ''
                    }
                    st.session_state.show_confirm_popup = False
            with col_cancel:
                if st.button("Cancel", key="cancel_payment_btn_popup"):
                    st.session_state.show_confirm_popup = False

# =====================================================================
# TAB 3: REPORTS
# =====================================================================
with tab3:
    st.markdown("## 📈 Agentic Vantage Assist")
    st.markdown("Generate and visualize reports based on your financial activity")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        report_type = st.selectbox(
            "📊 Select Report Type",
            [
                ("Transactions Today", "transactions_today"),
                ("Daily Transactions", "transactions_daily"),
                ("Wire Transfers", "wire_transfers"),
                ("Account Balance", "account_balance"),
                ("Monthly Summary", "monthly_summary")
            ],
            format_func=lambda x: x[0],
            key="report_type_selector"
        )
    
    with col2:
        time_period = st.selectbox(
            "⏰ Time Period",
            [
                ("Today", "today"),
                ("This Week", "week"),
                ("This Month", "month"),
                ("Last 90 Days", "90days")
            ],
            format_func=lambda x: x[0],
            key="time_period_selector"
        )
    
    with col3:
        chart_type = st.selectbox(
            "📉 Chart Type",
            [
                ("Bar Chart", "bar"),
                ("Line Chart", "line"),
                ("Pie Chart", "pie"),
                ("Area Chart", "area")
            ],
            format_func=lambda x: x[0],
            key="chart_type_selector"
        )
    
    if st.button("🔄 Generate Report", use_container_width=True, key="generate_report_btn"):
        with st.spinner("📊 Generating your report..."):
            report_result = generate_report(
                user_id=st.session_state.user_id,
                report_type=report_type[1],
                time_period=time_period[1]
            )
            
            if report_result.get("status") == "success":
                st.session_state.report_data = report_result
                st.success("✅ Report generated successfully!")
            else:
                st.error(f"❌ Error: {report_result.get('error', 'Unknown error')}")
    
    st.divider()
    
    if st.session_state.report_data:
        report_data = st.session_state.report_data
        
        st.markdown(f"### {report_data.get('title', 'Report')}")
        st.markdown(f"**Generated:** {report_data.get('generated_at', 'N/A')}")
        
        metrics = report_data.get("data", {})
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            st.metric("Total Transactions", metrics.get("total_transactions", 0))
        
        with metric_cols[1]:
            st.metric("Total Amount", f"${metrics.get('total_amount', 0):,.2f}")
        
        with metric_cols[2]:
            st.metric("Average", f"${metrics.get('avg_transaction', 0):,.2f}")
        
        with metric_cols[3]:
            st.metric("Highest", f"${metrics.get('max_transaction', 0):,.2f}")
        
        st.divider()
        
        try:
            chart_data = report_data.get("chart_data", [])
            if chart_data:
                df = pd.DataFrame(chart_data)
                
                if chart_type[1] == "bar":
                    fig = px.bar(df, x=df.columns[0], y=df.columns[1],
                                title=f"{report_type[0]} - Bar Chart",
                                color=df.columns[1], color_continuous_scale="viridis")
                elif chart_type[1] == "line":
                    fig = px.line(df, x=df.columns[0], y=df.columns[1],
                                 title=f"{report_type[0]} - Line Chart", markers=True)
                elif chart_type[1] == "pie":
                    fig = px.pie(df, names=df.columns[0], values=df.columns[1],
                                title=f"{report_type[0]} - Distribution")
                else:  # area
                    fig = px.area(df, x=df.columns[0], y=df.columns[1],
                                 title=f"{report_type[0]} - Area Chart")
                
                fig.update_layout(height=500, hovermode="x unified", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error rendering chart: {str(e)}")
    else:
        st.info("👆 Select report parameters and click 'Generate Report' to view analytics")



# =====================================================================
# FOOTER
# =====================================================================
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📊 Features**")
    st.markdown("- Real-time balance tracking")
    st.markdown("- Plan comparison")
    st.markdown("- Financial reports")

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

st.markdown("*Powered by Supernova GenAI Hackathon Team*")