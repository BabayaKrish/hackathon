import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any
import logging
import plotly.express as px
import plotly.graphobjects as go

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8080"

# ============================================================================
# PAGE CONFIGURATION & THEMING
# ============================================================================
st.set_page_config(
    page_title="AgenticVantage - Premium Financial Assistant",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* Premium color scheme */
    :root {
        --primary: #00D4FF;      /* Cyan/Teal */
        --secondary: #FF006E;    /* Hot Pink */
        --accent: #FFD60A;       /* Golden Yellow */
        --dark-bg: #0A0E27;      /* Deep Navy */
        --light-bg: #1A1F3A;     /* Dark Blue */
        --success: #00C853;      /* Vibrant Green */
        --danger: #FF3D00;       /* Vibrant Red */
    }
    
    /* Main background */
    body {
        background: linear-gradient(135deg, #0A0E27 0%, #1A1F3A 50%, #2D1B69 100%) !important;
        color: #E0E0E0 !important;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0A0E27 0%, #1A1F3A 50%, #2D1B69 100%) !important;
    }
    
    /* Header styling */
    h1 {
        background: linear-gradient(135deg, #00D4FF, #FFD60A) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-weight: 900 !important;
        font-size: 2.5em !important;
        letter-spacing: -2px !important;
        text-shadow: 0 4px 20px rgba(0, 212, 255, 0.3) !important;
    }
    
    h2 {
        color: #00D4FF !important;
        border-bottom: 3px solid #FF006E !important;
        padding-bottom: 10px !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    h3 {
        color: #FFD60A !important;
        font-weight: 700 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A1F3A 0%, #2D1B69 100%) !important;
        border-right: 2px solid #00D4FF !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #1A1F3A 0%, #2D1B69 100%) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00D4FF, #00B8D4) !important;
        color: #0A0E27 !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 12px 24px !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.6) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) !important;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(26, 31, 58, 0.8) !important;
        color: #E0E0E0 !important;
        border: 2px solid #00D4FF !important;
        border-radius: 8px !important;
        padding: 12px 15px !important;
        font-weight: 500 !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stNumberInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #FF006E !important;
        box-shadow: 0 0 15px rgba(255, 0, 110, 0.4) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px !important;
        border-bottom: 2px solid #00D4FF !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(255, 0, 110, 0.1)) !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 15px 20px !important;
        font-weight: 700 !important;
        color: #00D4FF !important;
        border: 2px solid transparent !important;
        transition: all 0.3s ease !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00D4FF, #FF006E) !important;
        color: #0A0E27 !important;
        border-bottom: 3px solid #FFD60A !important;
    }
    
    /* Chat messages */
    .stChatMessage {
        background: rgba(26, 31, 58, 0.6) !important;
        border-left: 4px solid #00D4FF !important;
        border-radius: 12px !important;
        padding: 15px !important;
        margin: 10px 0 !important;
    }
    
    .stChatMessage[data-testid="user-message"] {
        border-left-color: #FF006E !important;
        background: rgba(45, 27, 105, 0.5) !important;
    }
    
    /* Cards and containers */
    .stInfo, .stWarning, .stError, .stSuccess {
        border-radius: 12px !important;
        border-left: 4px solid #00D4FF !important;
        background: rgba(26, 31, 58, 0.6) !important;
        padding: 15px !important;
        font-weight: 500 !important;
    }
    
    .stSuccess {
        border-left-color: #00C853 !important;
    }
    
    .stError {
        border-left-color: #FF3D00 !important;
    }
    
    .stWarning {
        border-left-color: #FFD60A !important;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(255, 0, 110, 0.1)) !important;
        border: 2px solid #00D4FF !important;
        border-radius: 12px !important;
        padding: 20px !important;
        text-align: center !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background: rgba(26, 31, 58, 0.8) !important;
        border: 2px solid #00D4FF !important;
        border-radius: 12px !important;
        padding: 15px !important;
    }
    
    /* Divider */
    hr {
        border-top: 2px solid #00D4FF !important;
        margin: 20px 0 !important;
    }
    
    /* Plan cards */
    .plan-card {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.05), rgba(255, 0, 110, 0.05)) !important;
        border: 2px solid #00D4FF !important;
        border-radius: 15px !important;
        padding: 25px !important;
        transition: all 0.3s ease !important;
    }
    
    .plan-card:hover {
        transform: translateY(-5px) !important;
        border-color: #FF006E !important;
        box-shadow: 0 8px 30px rgba(0, 212, 255, 0.2) !important;
    }
    
    .plan-card.popular {
        border: 3px solid #FFD60A !important;
        background: linear-gradient(135deg, rgba(255, 214, 10, 0.1), rgba(255, 0, 110, 0.1)) !important;
    }
    
    /* Text styling */
    p, .stMarkdown {
        color: #E0E0E0 !important;
    }
    
    a {
        color: #00D4FF !important;
        text-decoration: none !important;
        font-weight: 600 !important;
    }
    
    a:hover {
        color: #FFD60A !important;
        text-decoration: underline !important;
    }
    
    /* Spinner */
    .spinner {
        border: 4px solid rgba(0, 212, 255, 0.3) !important;
        border-top: 4px solid #00D4FF !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_plan" not in st.session_state:
    st.session_state.current_plan = "GOLD"
if "user_id" not in st.session_state:
    st.session_state.user_id = "user_001"
if "selected_upgrade_plan" not in st.session_state:
    st.session_state.selected_upgrade_plan = None
if "show_payment_form" not in st.session_state:
    st.session_state.show_payment_form = False
if "upgrade_success" not in st.session_state:
    st.session_state.upgrade_success = False
if "report_data" not in st.session_state:
    st.session_state.report_data = None
if "report_type" not in st.session_state:
    st.session_state.report_type = "transactions_today"
if "last_processed_input" not in st.session_state:
    st.session_state.last_processed_input = ""

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### 👤 User Profile")
    new_user_id = st.text_input("🔑 User ID", value=st.session_state.user_id, key="user_id_input")
    if new_user_id != st.session_state.user_id:
        st.session_state.user_id = new_user_id
    
    new_plan = st.selectbox(
        "📊 Current Plan",
        ["BRONZE", "SILVER", "GOLD"],
        index=["BRONZE", "SILVER", "GOLD"].index(st.session_state.current_plan),
        key="plan_selector"
    )
    if new_plan != st.session_state.current_plan:
        st.session_state.current_plan = new_plan
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    st.divider()
    
    st.markdown("""
    ### 🎯 Quick Info
    **User:** `{}`
    
    **Plan:** `{}`
    
    **Status:** ✅ Active
    """.format(st.session_state.user_id, st.session_state.current_plan))
    
    st.markdown("""
    ---
    ### 💡 Need Help?
    • Ask about balance and transactions
    • Compare plans
    • View reports
    • Manage subscriptions
    """)

# ============================================================================
# MAIN HEADER
# ============================================================================
st.markdown("""
<div style="text-align: center; padding: 20px; margin-bottom: 30px;">
    <h1>💎 AgenticVantage Assist</h1>
    <p style="font-size: 18px; color: #00D4FF; margin-top: 10px; font-weight: 600;">
        Your Premium Financial Intelligence Platform
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS (Keep all existing helper functions)
# ============================================================================

def display_additional_features_in_chat_response(response: Dict[str, Any]) -> str:
    """Format additional features response for chat display"""
    features = response.get("additional_features", [])
    markdown = "### 🎁 Additional Features You Can Add\n"
    
    if isinstance(features, str):
        features = [f.strip() for f in features.split(",") if f.strip()]
    features = [f for f in features if isinstance(f, str) and len(f.strip()) > 1]
    
    if not features:
        markdown += "No additional features available at this time.\n"
        return markdown
    
    for idx, feat in enumerate(features[:20], 1):
        price = 10 * idx
        markdown += f"- **{feat}** - ${price}/mo\n"
    
    if len(features) > 20:
        markdown += f"...and {len(features) - 20} more features.\n"
        markdown += "_Add a feature, contact support or use the upgrade options._\n"
    
    return markdown.strip()

def display_plan_upgrade_in_chat_response(response: Dict[str, Any]) -> str:
    """Format plan upgrade response as markdown for chat display"""
    logger.info(f"streamlit: Formatting plan info response for chat display")
    logger.info(f"streamlit: Response keys: {response}")
    
    if response.get("status") != "success":
        return f"❌ Error Processing Request: {response.get('error', 'Unknown error')}"
    
    confidence = response.get("confidence", 0.0)
    mentioned_tier = response.get("mentioned_tier", "").strip()
    all_plans = response.get("all_plans", [])
    current_plan_details = response.get("current_plan_details", {})
    upgrade_options = response.get("upgrade_options", [])
    ai_recommendation = response.get("ai_recommendation", {})
    next_actions = response.get("next_actions", [])
    
    markdown = "### 📋 Plan Information\n"
    
    requested_plan = None
    if mentioned_tier:
        for p in all_plans:
            if str(p.get("tier", "")).lower() == mentioned_tier.lower() or str(p.get("plan_name", "")).lower().startswith(mentioned_tier.lower()):
                requested_plan = p
                break
    
    if not requested_plan and current_plan_details and current_plan_details.get("status") == "success":
        requested_plan = current_plan_details.get("plan")
    
    if requested_plan:
        name = requested_plan.get("plan_name", "Unknown Plan")
        tier = requested_plan.get("tier", "N/A")
        monthly = requested_plan.get("monthly_price")
        annual = requested_plan.get("annual_price")
        features = requested_plan.get("features", [])
        
        tier_emoji = "🥇" if tier.lower() == "gold" else ("🥈" if tier.lower() == "silver" else ("🥉" if tier.lower() == "bronze" else "📦"))
        
        markdown += f"**{tier_emoji} {name}** ({tier})\n\n"
        markdown += "#### 💰 Pricing\n"
        if monthly is not None:
            markdown += f"- **Monthly:** ${monthly:.2f}\n"
        else:
            markdown += "- **Monthly:** Free\n"
        
        if annual is not None:
            markdown += f"- **Annual:** ${annual:.2f}\n"
            if monthly:
                savings = (monthly * 12) - annual
                if savings > 0:
                    savings_pct = (savings / (monthly * 12)) * 100
                    markdown += f"- **Save:** ${savings:.2f}/year ({savings_pct:.0f}% off)\n"
        else:
            markdown += "- **Annual:** Free\n"
        
        if features:
            feats = [f.strip() for f in (features if isinstance(features, list) else features.split(","))]
            markdown += f"\n#### ✨ Features\n"
            if tier.lower() == "no plan":
                markdown += "**Additional Features You Can Add:**\n"
                for f in feats:
                    markdown += f"- {f}\n"
            else:
                markdown += "**Other Optional Features Included:**\n"
                for f in feats:
                    markdown += f"- {f}\n"
    else:
        markdown += "Could not find specific plan details for your request.\n"
    
    if ai_recommendation and ai_recommendation.get("status") == "success":
        markdown += "\n---\n### 🤖 AI-Powered Recommendation\n"
        recommended_plan = ai_recommendation.get("recommended_plan")
        key_benefits = ai_recommendation.get("key_benefits", [])
        cost_analysis = ai_recommendation.get("cost_analysis")
        reasoning = ai_recommendation.get("reasoning")
        
        if recommended_plan:
            markdown += f"**Recommended Plan:** {recommended_plan}\n"
        if reasoning:
            markdown += f"**Why:** {reasoning}\n"
        if key_benefits:
            markdown += f"\n**Key Benefits:**\n"
            for b in key_benefits:
                markdown += f"- {b}\n"
        if cost_analysis:
            markdown += f"\n**Cost Breakdown:** {cost_analysis}\n"
    
    if upgrade_options:
        markdown += "\n---\n### 🚀 Available Upgrade Options\n"
        for idx, up in enumerate(upgrade_options, 1):
            pname = up.get("plan_name", "Unknown")
            tier = up.get("tier", "N/A")
            monthly = up.get("monthly_price")
            annual = up.get("annual_price")
            feats = up.get("features", "")
            
            tier_emoji = "🥇" if tier.lower() == "gold" else ("🥈" if tier.lower() == "silver" else ("🥉" if tier.lower() == "bronze" else "📦"))
            
            price_str = ""
            if monthly is not None:
                price_str = f"${monthly:.2f}/mo"
            if annual is not None:
                if price_str:
                    price_str += f" or ${annual:.2f}/year"
                else:
                    price_str = f"${annual:.2f}/year"
            
            markdown += f"{idx}. **{tier_emoji} {pname}** - {price_str}\n"
            if feats:
                flist = [f.strip() for f in (feats if isinstance(feats, list) else feats.split(","))]
                markdown += f"   _Features: {', '.join(flist[:2])}"
                if len(flist) > 2:
                    markdown += f" + {len(flist) - 2} more_\n"
                else:
                    markdown += "_\n"
    
    if all_plans and len(all_plans) > 1:
        markdown += "\n---\n### 📊 Complete Plans Overview\n\n"
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
    
    if next_actions:
        markdown += "\n---\n### 📍 Next Steps\n"
        for action in next_actions:
            if action == "show_comparison_table":
                markdown += "- **View Detailed Comparison** - See side-by-side feature comparison in the Plan Comparison tab\n"
            elif action == "show_upgrade_prompt":
                markdown += "- **Choose Upgrade** - Select a plan from the recommendations above\n"
            else:
                markdown += f"- {action.replace('_', ' ').title()}\n"
    
    markdown += "\n**Need help?** Ask me any questions about the plans!\n"
    
    return markdown.strip()

def send_to_root_agent(user_input: str) -> Dict:
    """Send query to Root Agent for intelligent routing"""
    logger.info(f"streamlit: Sending to Root Agent: {user_input}")
    try:
        response = requests.post(
            f"{API_BASE_URL}/process-query",
            json={"user_id": st.session_state.user_id, "query": user_input},
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            logger.info(f"streamlit: Root Agent responded: {result.get('intent')}")
            logger.info(f"streamlit: Response keys: {list(result.keys())}")
            return result
        else:
            logger.error(f"streamlit: Root Agent error: {response.status_code}")
            return {"status": "error", "error": response.text}
    except Exception as e:
        logger.error(f"streamlit: Failed to call Root Agent: {str(e)}")
        return {"status": "error", "error": str(e)}

def generate_report(user_id: str, report_type: str, time_period: str = "today") -> Dict:
    """Generate a report from backend"""
    logger.info(f"streamlit: Generating report: {report_type}")
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate-report",
            json={"user_id": user_id, "report_type": report_type, "time_period": time_period},
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            logger.info(f"streamlit: Report generated successfully")
            return result
        else:
            logger.error(f"streamlit: Report generation error: {response.status_code}")
            return {"status": "error", "error": response.text}
    except Exception as e:
        logger.error(f"streamlit: Failed to generate report: {str(e)}")
        return {"status": "error", "error": str(e)}

def fetch_plans_data():
    """Fetch plan information from API"""
    print("streamlit: fetchPlansData CALLED")
    try:
        response = requests.get(f"{API_BASE_URL}/plans", timeout=10)
        logger.info(f"streamlit: Fetched plans data, status code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            logger.info(f"streamlit: Fetched plans data: {result}")
            plans = result.get("plans", [])
            plans_data = {}
            for plan in plans:
                raw_name = plan.get("name", "").lower()
                if "bronze" in raw_name:
                    key = "Bronze"
                elif "silver" in raw_name:
                    key = "Silver"
                elif "gold" in raw_name:
                    key = "Gold"
                else:
                    continue
                
                price = plan.get("price", 0)
                price_monthly = f"${price}" if price is not None else "N/A"
                price_annual = f"${plan.get('price_annual', price * 12)}" if plan.get('price_annual') else f"${price * 12}" if price else "N/A"
                transactions = plan.get("transactions", "")
                features = plan.get("features", "")
                if isinstance(features, str):
                    features = [f.strip() for f in features.split(",") if f.strip()]
                reports = plan.get("reports") if "reports" in plan else ""
                support = plan.get("support") if "support" in plan else ""
                history = plan.get("history") if "history" in plan else ""
                
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
        logger.error(f"streamlit: Error fetching plans: {str(e)}")
        return {}

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3 = st.tabs(["💬 Chat Assistant", "📊 Plan Comparison", "📈 Reports"])

# ============================================================================
# TAB 1: CHAT ASSISTANT
# ============================================================================
with tab1:
    st.markdown("### 💬 Chat with Your Financial Assistant")
    st.markdown("Ask me about balances, plans, wire transfers, and more!")
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input (MOVED OUTSIDE TAB)
    user_input = st.chat_input("🔍 Ask about balance, plans, wire transfers...", key="chat_input")
    
    if user_input and user_input != st.session_state.last_processed_input:
        st.session_state.last_processed_input = user_input
        
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Process with Root Agent
        with st.spinner("🔄 Processing your request..."):
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
                    
                    logger.info(f"streamlit: Intent: {intent}, Agent: {agent_used}, Confidence: {confidence}")
                    
                    # HANDLE OUT_OF_SCOPE QUERIES
                    if intent == "OUT_OF_SCOPE":
                        handled = True
                        reasoning_str = str(reasoning) if not isinstance(reasoning, str) else reasoning
                        assistant_response = f"""
                        ### 🤔 Not Sure About That
                        _{reasoning_str}_
                        
                        I can help you with:
                        - **Balance & Transactions** - Showing wire transfers, balance report, intraday balance and ledger
                        - **Plan Management** - Comparing and managing plans
                        - **Feature Info** - Answering questions about features and pricing
                        
                        Feel free to ask me about any of these topics!
                        """
                        
                        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                        with st.chat_message("assistant"):
                            st.markdown(assistant_response)
                    
                    # BALANCE AGENT
                    elif agent_used == "balance_agent":
                        balance = data.get("balance", 0)
                        transactions = data.get("transaction_count", 0)
                        assistant_response = f"""
                        ### 💰 Your Account Balance
                        **Balance:** ${balance:,.2f}
                        
                        **Recent Activity:** {transactions} transactions
                        
                        **Quick Tips:**
                        - Use _show my wire transfers_ for wire details
                        - Use _what's in Gold plan_ for plan comparison
                        - Visit the Reports tab for detailed analytics
                        """
                        
                        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                        with st.chat_message("assistant"):
                            st.markdown(assistant_response)
                    
                    # REPORT AGENT
                    elif agent_used == "report_agent":
                        handled = True
                        reports = data.get("reports", []) if isinstance(data, dict) else data
                        
                        if isinstance(reports, str):
                            try:
                                reports = json.loads(data.get("report_data", "[]"))
                            except:
                                reports = []
                        elif "reports" in data:
                            reports = data.get("reports", [])
                        
                        logger.info(f"streamlit: Report Agent data: {data}")
                        
                        if reports:
                            df = pd.DataFrame(reports)
                            
                            if all(col in df.columns for col in ["balance_date", "opening_balance", "closing_balance"]):
                                assistant_response = f"📊 **Balance Report** - {len(reports)} records"
                                
                                st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                                with st.chat_message("assistant"):
                                    st.markdown(assistant_response)
                                    st.dataframe(df, use_container_width=True)
                                    
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(x=df["balance_date"], y=df["opening_balance"], mode="lines+markers", name="Opening Balance"))
                                    fig.add_trace(go.Scatter(x=df["balance_date"], y=df["closing_balance"], mode="lines+markers", name="Closing Balance"))
                                    fig.update_layout(title="Opening & Closing Balances Over Time", xaxis_title="Date", yaxis_title="Balance", height=500, template="plotly_dark")
                                    st.plotly_chart(fig, use_container_width=True)
                        else:
                            assistant_response = "No report data available."
                            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                            with st.chat_message("assistant"):
                                st.markdown(assistant_response)
                    
                    # PLAN INFO AGENT
                    elif agent_used == "plan_info_agent":
                        if data.get("additional_features"):
                            assistant_response = display_additional_features_in_chat_response(data)
                        else:
                            assistant_response = display_plan_upgrade_in_chat_response(data)
                        
                        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                        with st.chat_message("assistant"):
                            st.markdown(assistant_response)
                    
                    if not handled:
                        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                        with st.chat_message("assistant"):
                            st.markdown(assistant_response)
                
                elif result.get("status") == "low_confidence":
                    error_msg = f"⚠️ I'm not sure about that. ({result.get('confidence', 0) * 100:.0f}% confident) Could you rephrase?"
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
                logger.error(f"streamlit: Exception: {error_msg}")
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.error(error_msg)
            
            st.rerun()

# ============================================================================
# TAB 2: PLAN COMPARISON
# ============================================================================
with tab2:
    st.markdown("### 📊 Plan Comparison & Upgrade Options")
    
    st.markdown("""
    Compare all our plans side-by-side and choose the perfect fit for your financial needs.
    """)
    
    # Fetch plans data
    plans_data = fetch_plans_data()
    
    # Display plans in columns
    col1, col2, col3 = st.columns(3)
    
    def show_payment(plan):
        st.session_state.show_payment_form = True
        st.session_state.selected_upgrade_plan = plan
    
    # Bronze Plan
    with col1:
        bronze = plans_data.get("Bronze", {})
        st.markdown(f"""
        <div class="plan-card">
            <h3>🥉 Bronze</h3>
            <p style="font-size: 24px; color: #00D4FF; font-weight: bold; margin: 15px 0;">
                {bronze.get('price_monthly', 'N/A')}<span style="font-size: 14px; color: #FFD60A;">/mo</span>
            </p>
            <p style="color: #FFD60A; font-size: 14px; margin-bottom: 15px;">
                ✨ Perfect for Getting Started
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if bronze:
            for feature in bronze.get("features", []):
                st.markdown(f"✓ {feature}")
        else:
            st.warning("Bronze plan data not available.")
        
        if st.session_state.current_plan != "BRONZE":
            if st.button("💳 Switch to Bronze", key="bronze_upgrade", use_container_width=True):
                show_payment("Bronze")
        else:
            st.markdown("### ✅ Current Plan")
    
    # Silver Plan (Popular)
    with col2:
        silver = plans_data.get("Silver", {})
        st.markdown(f"""
        <div class="plan-card popular">
            <h3>🥈 Silver <span style="color: #FFD60A; font-size: 14px;">★ POPULAR</span></h3>
            <p style="font-size: 24px; color: #FFD60A; font-weight: bold; margin: 15px 0;">
                {silver.get('price_monthly', 'N/A')}<span style="font-size: 14px; color: #00D4FF;">/mo</span>
            </p>
            <p style="color: #00D4FF; font-size: 14px; margin-bottom: 15px;">
                ⭐ Most Popular Choice
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if silver:
            for feature in silver.get("features", []):
                st.markdown(f"✓ {feature}")
        else:
            st.warning("Silver plan data not available.")
        
        if st.session_state.current_plan != "SILVER":
            if st.button("💳 Switch to Silver", key="silver_upgrade", use_container_width=True):
                show_payment("Silver")
        else:
            st.markdown("### ✅ Current Plan")
    
    # Gold Plan
    with col3:
        gold = plans_data.get("Gold", {})
        st.markdown(f"""
        <div class="plan-card">
            <h3>🥇 Gold</h3>
            <p style="font-size: 24px; color: #00D4FF; font-weight: bold; margin: 15px 0;">
                {gold.get('price_monthly', 'N/A')}<span style="font-size: 14px; color: #FFD60A;">/mo</span>
            </p>
            <p style="color: #FFD60A; font-size: 14px; margin-bottom: 15px;">
                👑 Enterprise Grade
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if gold:
            for feature in gold.get("features", []):
                st.markdown(f"✓ {feature}")
        else:
            st.warning("Gold plan data not available.")
        
        if st.session_state.current_plan != "GOLD":
            if st.button("💳 Switch to Gold", key="gold_upgrade", use_container_width=True):
                show_payment("Gold")
        else:
            st.markdown("### ✅ Current Plan")
    
    # Comparison Table
    st.divider()
    st.markdown("### 📋 Detailed Comparison")
    
    comparison_df = pd.DataFrame({
        "Feature": ["Monthly Price", "Annual Price", "Transactions/Month", "Reports Available", "Support", "History Retention"],
        "Bronze": ["$29", "$290", "500/month", "3 basic", "Email", "7 days"],
        "Silver": ["$99", "$990", "5,000/month", "9 types", "Phone + Email", "90 days"],
        "Gold": ["$299", "$2,990", "Unlimited", "9 + Custom", "24/7 Dedicated", "7 years"]
    })
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# ============================================================================
# TAB 3: REPORTS
# ============================================================================
with tab3:
    st.markdown("### 📈 Financial Reports & Analytics")
    st.markdown("Generate and visualize reports based on your financial activity")
    
    # Report controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        report_type = st.selectbox(
            "📊 Report Type",
            ["Transactions Today", "Daily Transactions", "Wire Transfers", "Account Balance", "Monthly Summary"],
            format_func=lambda x: x,
            key="report_type_selector"
        )
    
    with col2:
        time_period = st.selectbox(
            "📅 Time Period",
            ["Today", "This Week", "This Month", "Last 90 Days"],
            format_func=lambda x: x,
            key="time_period_selector"
        )
    
    with col3:
        chart_type = st.selectbox(
            "📈 Chart Type",
            ["Bar Chart", "Line Chart", "Pie Chart", "Area Chart"],
            format_func=lambda x: x,
            key="chart_type_selector"
        )
    
    if st.button("🔄 Generate Report", use_container_width=True, key="generate_report_btn"):
        with st.spinner("⏳ Generating your report..."):
            report_type_map = {"Transactions Today": "transactions_today", "Daily Transactions": "transactions_daily", "Wire Transfers": "wire_transfers", "Account Balance": "account_balance", "Monthly Summary": "monthly_summary"}
            time_period_map = {"Today": "today", "This Week": "week", "This Month": "month", "Last 90 Days": "90days"}
            
            report_result = generate_report(
                user_id=st.session_state.user_id,
                report_type=report_type_map.get(report_type, "transactions_today"),
                time_period=time_period_map.get(time_period, "today")
            )
            
            if report_result.get("status") == "success":
                st.session_state.report_data = report_result
                st.success("✅ Report generated successfully!")
            else:
                st.error(f"❌ Error: {report_result.get('error', 'Unknown error')}")
    
    # Display report if available
    if st.session_state.report_data:
        report_data = st.session_state.report_data
        
        # Report header
        st.markdown(f"### 📄 {report_data.get('title', 'Report')} Report")
        st.caption(f"Generated: {report_data.get('generated_at', 'N/A')}")
        
        # Metrics
        metrics = report_data.get("data", {})
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            st.metric("📊 Total Transactions", metrics.get("total_transactions", 0))
        with metric_cols[1]:
            st.metric("💰 Total Amount", f"${metrics.get('total_amount', 0):,.2f}")
        with metric_cols[2]:
            st.metric("📈 Average", f"${metrics.get('avg_transaction', 0):,.2f}")
        with metric_cols[3]:
            st.metric("🔝 Highest", f"${metrics.get('max_transaction', 0):,.2f}")
        
        st.divider()
        
        # Charts
        try:
            chart_data = report_data.get("chart_data", [])
            if chart_data:
                df = pd.DataFrame(chart_data)
                
                if chart_type == "Bar Chart":
                    fig = px.bar(df, x=df.columns[0], y=df.columns[1], title=f"{report_type} - Bar Chart", color=df.columns[1], color_continuous_scale="viridis")
                elif chart_type == "Line Chart":
                    fig = px.line(df, x=df.columns[0], y=df.columns[1], title=f"{report_type} - Line Chart", markers=True)
                elif chart_type == "Pie Chart":
                    fig = px.pie(df, names=df.columns[0], values=df.columns[1], title=f"{report_type} - Distribution")
                else:  # Area Chart
                    fig = px.area(df, x=df.columns[0], y=df.columns[1], title=f"{report_type} - Area Chart")
                
                fig.update_layout(height=500, hovermode="x unified", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error rendering chart: {str(e)}")
    else:
        st.info("ℹ️ Select report parameters and click Generate Report to view analytics")

# ============================================================================
# FOOTER
# ============================================================================
st.divider()

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("""
    ### ⚡ Features
    - ✨ Real-time balance tracking
    - 📊 Smart plan comparison
    - 📈 Financial reports
    """)

with footer_col2:
    st.markdown("""
    ### 🔒 Security
    - 🔐 End-to-end encryption
    - 🏦 Bank-level security
    - 🔑 Multi-factor auth
    """)

with footer_col3:
    st.markdown("""
    ### 📞 Support
    - 🕐 24/7 dedicated team
    - 📧 Email support
    - 📱 Phone support
    """)

st.divider()

st.markdown("""
<div style="text-align: center; padding: 20px; color: #00D4FF; font-weight: 600;">
    🚀 Powered by <span style="color: #FFD60A;">AgenticVantage</span> | 
    <span style="color: #FF006E;">Supernova GenAI</span> Hackathon Team
</div>
""", unsafe_allow_html=True)
