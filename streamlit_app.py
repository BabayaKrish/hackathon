# frontend/app.py
import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

# Configuration
API_BASE_URL = "http://localhost:8000"  # Update for production

st.set_page_config(
    page_title="Data Access Plans Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    body, .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%) !important;
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
    }
    .stApp {
        background: linear-gradient(120deg, #fdf6e3 0%, #e0c3fc 100%) !important;
    }
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(120deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .css-18e3th9, .stButton>button, .stDownloadButton>button, .stRadio>div, .stSelectbox>div, .stTextInput>div, .stTextArea>div, .stToggle>div {
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(160, 120, 255, 0.08);
        font-size: 1.08rem;
    }
    .stButton>button, .stDownloadButton>button {
        background: linear-gradient(90deg, #7f53ac 0%, #657ced 100%);
        color: #fff;
        border: none;
        font-weight: 600;
        transition: background 0.2s;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background: linear-gradient(90deg, #ff6a00 0%, #ee0979 100%);
        color: #fff;
    }
    .stTabs [data-baseweb="tab"] {
        background: #fff0f6;
        color: #7f53ac;
        border-radius: 10px 10px 0 0;
        margin-right: 4px;
        font-weight: 600;
        border: 2px solid #e0c3fc;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #7f53ac 0%, #657ced 100%);
        color: #fff;
        border-bottom: 2px solid #fff;
    }
    .stMetric {
        background: #f3e8ff;
        border-radius: 10px;
        padding: 10px 16px;
        margin-bottom: 8px;
        box-shadow: 0 1px 4px rgba(127, 83, 172, 0.08);
    }
    .stContainer {
        background: rgba(255,255,255,0.7);
        border-radius: 14px;
        box-shadow: 0 2px 12px rgba(127, 83, 172, 0.07);
        padding: 18px 20px;
        margin-bottom: 18px;
    }
    .stSidebar {
        background: linear-gradient(120deg, #e0c3fc 0%, #fdf6e3 100%) !important;
        border-radius: 0 20px 20px 0;
        box-shadow: 2px 0 12px rgba(127, 83, 172, 0.07);
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #7f53ac;
        font-weight: 700;
    }
    .stMarkdown p, .stMarkdown ul, .stMarkdown li {
        color: #3d246c;
    }
    .stDataFrame, .stTable {
        background: #fff;
        border-radius: 10px;
        box-shadow: 0 1px 4px rgba(127, 83, 172, 0.08);
    }
    .stJson {
        background: #f3e8ff;
        border-radius: 10px;
        padding: 10px;
    }
    .stSuccess, .stError, .stWarning {
        border-radius: 10px;
        font-weight: 600;
    }
    .stChatInput>div>div>textarea {
        border-radius: 10px !important;
        background: #f3e8ff !important;
        color: #3d246c !important;
    }
    .stHeader {
        background: linear-gradient(90deg, #7f53ac 0%, #657ced 100%);
        color: #fff;
        border-radius: 10px;
        padding: 10px 20px;
        margin-bottom: 18px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_plan' not in st.session_state:
    st.session_state.current_plan = 'BRONZE'
if 'user_id' not in st.session_state:
    st.session_state.user_id = 'demo_user'

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x50/1e3a8a/ffffff?text=Bank+Logo", use_column_width=True)
    st.markdown("## User Profile")
    
    user_id = st.text_input("User ID", value=st.session_state.user_id)
    st.session_state.user_id = user_id
    
    current_plan = st.selectbox(
        "Current Plan",
        ["BRONZE", "SILVER", "GOLD", "PREMIUM"],
        index=["BRONZE", "SILVER", "GOLD", "PREMIUM"].index(st.session_state.current_plan)
    )
    st.session_state.current_plan = current_plan
    
    st.markdown("---")
    st.markdown("### Plan Features")
    
    plan_features = {
        "BRONZE": ["General Balance", "Previous Day Reports", "Transaction Search", "Statements"],
        "SILVER": ["All BRONZE", "Instant Reports", "Expanded Details", "API Access"],
        "GOLD": ["All SILVER", "Wire Tracking", "ACH Details", "Full API", "Product Subscriptions"],
        "PREMIUM": ["All GOLD", "Intraday Activity", "Real-time Monitoring", "SWIFT Reports"]
    }
    
    for feature in plan_features[current_plan]:
        st.markdown(f"✅ {feature}")
    
    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Main content
st.markdown('<h1 class="main-header">📊 Data Access Plans Assistant</h1>', unsafe_allow_html=True)
st.markdown("Ask me anything about reports, account activity, or plan upgrades!")

# Tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat Assistant", "📈 Analytics", "🎯 Plan Comparison"])

with tab1:
    # Chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                    
                    # Display structured response if available
                    if 'structured_data' in message:
                        with st.expander("📋 Detailed Response"):
                            st.json(message['structured_data'])
    

    # Input area
    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_area(
            "Your question:",
            placeholder="E.g., 'Show me my wire transfer details' or 'Can I get intraday balance reports?'",
            height=100,
            key='user_input'
        )
        
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            submit_button = st.form_submit_button("Send 🚀", use_container_width=True)
        with col2:
            example_button = st.form_submit_button("Example", use_container_width=True)
    
    if example_button:
        user_input = "I need to access intraday balance and wire tracking details. What do I need?"
        submit_button = True
    
    if submit_button and user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input
        })
        
        # Show loading
        with st.spinner('Processing your request...'):
            try:
                # Call API
                response = requests.post(
                    f"{API_BASE_URL}/query",
                    json={
                        "query": user_input,
                        "user_id": st.session_state.user_id,
                        "current_plan": st.session_state.current_plan,
                        "context": {}
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Format response
                    formatted_response = format_agent_response(result)
                    
                    # Add assistant message
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': formatted_response,
                        'structured_data': result
                    })
                    
                    # Show metrics
                    st.success(f"✅ Confidence: {result['confidence']:.2%} | Intent: {result['intent']}")
                    
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            
            except Exception as e:
                st.error(f"Failed to connect to API: {str(e)}")
        
        st.rerun()

with tab2:
    st.markdown("### 📊 Usage Analytics")
    
    # Fetch metrics from API
    try:
        metrics_response = requests.get(f"{API_BASE_URL}/metrics")
        if metrics_response.status_code == 200:
            metrics = metrics_response.json()
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Total Queries", metrics.get('total_queries', 0))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Avg Confidence", f"{metrics.get('average_confidence', 0):.2%}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Avg Latency", f"{metrics.get('average_latency_ms', 0):.0f}ms")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Accuracy Rate", f"{metrics.get('accuracy_rate', 0):.2%}")
                st.markdown('</div>', unsafe_allow_html=True)
    
    except Exception as e:
        st.warning("Unable to fetch metrics from API")
    
    # Sample visualizations
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        # Intent distribution
        intent_data = {
            'Intent': ['Report Query', 'Plan Upgrade', 'Data Extraction', 'Plan Info'],
            'Count': [45, 20, 25, 10]
        }
        fig = px.pie(intent_data, values='Count', names='Intent', title='Query Intent Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Confidence over time
        import numpy as np
        dates = [datetime.now().date() for _ in range(30)]
        confidences = np.random.normal(0.85, 0.1, 30)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=confidences,
            mode='lines+markers',
            name='Confidence',
            line=dict(color='#3b82f6', width=2)
        ))
        fig.update_layout(
            title='Confidence Score Trend',
            yaxis_title='Confidence',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### 🎯 Plan Comparison")
    
    # Plan comparison table
    plans_data = {
        'Feature': [
            'General Balance Report',
            'Previous Day Reports',
            'Intraday Balance',
            'Wire Tracking',
            'ACH Details',
            'API Access',
            'Real-time Monitoring',
            'SWIFT Reports'
        ],
        'BRONZE': ['✅', '✅', '❌', '❌', '❌', '❌', '❌', '❌'],
        'SILVER': ['✅', '✅', '❌', '❌', '❌', '⚠️ Limited', '❌', '❌'],
        'GOLD': ['✅', '✅', '❌', '✅', '✅', '✅', '❌', '❌'],
        'PREMIUM': ['✅', '✅', '✅', '✅', '✅', '✅', '✅', '✅']
    }
    
    import pandas as pd
    df = pd.DataFrame(plans_data)
    
    st.dataframe(df, use_container_width=True, height=350)
    
    st.markdown("---")
    
    # Pricing comparison
    col1, col2, col3, col4 = st.columns(4)
    
    plans_info = [
        ('BRONZE', '$50/mo', 'Basic features for small businesses'),
        ('SILVER', '$150/mo', 'Standard features with API access'),
        ('GOLD', '$300/mo', 'Premium features with full integration'),
        ('PREMIUM', '$500/mo', 'Enterprise features with real-time data')
    ]
    
    for col, (plan, price, desc) in zip([col1, col2, col3, col4], plans_info):
        with col:
            is_current = plan == st.session_state.current_plan
            border_color = "#3b82f6" if is_current else "#e5e7eb"
            
            st.markdown(f"""
            <div style="border: 2px solid {border_color}; border-radius: 10px; padding: 1rem; height: 200px;">
                <h3>{plan}</h3>
                <h2 style="color: #3b82f6;">{price}</h2>
                <p style="font-size: 0.9rem;">{desc}</p>
                {"<span style='color: #3b82f6; font-weight: bold;'>Current Plan</span>" if is_current else ""}
            </div>
            """, unsafe_allow_html=True)
            
            if not is_current:
                if st.button(f"Upgrade to {plan}", key=f"upgrade_{plan}"):
                    # Call upgrade API
                    try:
                        upgrade_response = requests.post(
                            f"{API_BASE_URL}/upgrade/execute",
                            params={"user_id": st.session_state.user_id, "new_plan": plan}
                        )
                        if upgrade_response.status_code == 200:
                            st.success(f"Successfully upgraded to {plan}!")
                            st.session_state.current_plan = plan
                            st.rerun()
                    except Exception as e:
                        st.error(f"Upgrade failed: {str(e)}")

# Helper function
def format_agent_response(result):
    """Format agent response for display"""
    intent = result.get('intent', 'UNKNOWN')
    confidence = result.get('confidence', 0)
    agent_result = result.get('result', {})
    
    response_parts = []
    
    # Add confidence indicator
    if confidence > 0.8:
        response_parts.append("I'm confident in this response:")
    elif confidence > 0.6:
        response_parts.append("Based on my analysis:")
    else:
        response_parts.append("Here's what I found (low confidence):")
    
    # Format based on intent
    if intent == 'REPORT_QUERY':
        recommended_reports = agent_result.get('recommended_reports', [])
        if recommended_reports:
            response_parts.append("\n\n**Recommended Reports:**")
            for report in recommended_reports[:3]:
                available = "✅ Available" if report.get('available_in_current_plan') else "⬆️ Upgrade needed"
                response_parts.append(f"\n- **{report.get('report_name')}** ({available})")
                response_parts.append(f"  - {report.get('reasoning')}")
        
        if agent_result.get('upgrade_recommended'):
            response_parts.append(f"\n\n💡 **Recommendation:** Upgrade to {agent_result.get('recommended_plan')} for full access")
    
    elif intent == 'PLAN_UPGRADE':
        if agent_result.get('upgrade_needed'):
            response_parts.append(f"\n\n**Upgrade Recommended:** {agent_result.get('recommended_plan')}")
            response_parts.append(f"\n**Additional Cost:** ${agent_result.get('cost_analysis', {}).get('additional_cost', 0)}/month")
            response_parts.append(f"\n**ROI:** {agent_result.get('roi_justification', 'N/A')}")
    
    return "\n".join(response_parts)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Powered by SuperNova Hackathon team</p>
</div>
""", unsafe_allow_html=True)

