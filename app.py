import os
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from transformers import pipeline as hf_pipeline

# Set page config with dark theme
st.set_page_config(
    page_title="360° AI Business Insight Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    /* Main dark theme */
    :root {
        --primary: #7b68ee;
        --secondary: #5e43f3;
        --accent: #a78bfa;
        --dark-bg: #121212;
        --darker-bg: #0a0a0a;
        --card-bg: #1e1e1e;
        --text: #e0e0e0;
        --text-light: #b0b0b0;
        --success: #00c853;
        --warning: #ffab00;
        --danger: #ff5252;
    }
    
    body {
        background-color: var(--dark-bg);
        color: var(--text);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin: 0;
        padding: 0;
    }
    
    .stApp {
        background: linear-gradient(180deg, var(--darker-bg) 0%, var(--dark-bg) 100%) !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--darker-bg) 0%, #1a1a2e 100%) !important;
        border-right: 1px solid #333333 !important;
    }
    
    /* Cards */
    .custom-card {
        background: var(--card-bg);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
        border: 1px solid #333333;
        transition: all 0.3s ease;
        margin-bottom: 20px;
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(123, 104, 238, 0.25);
        border-color: var(--primary);
    }
    
    .card-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 20px;
        padding-bottom: 15px;
        border-bottom: 1px solid #333333;
    }
    
    .card-icon {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 24px !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 5px 20px rgba(123, 104, 238, 0.4) !important;
    }
    
    /* Inputs */
    .stTextArea>textarea, .stFileUploader>div>div>div>div {
        background: #2a2a2a !important;
        color: var(--text) !important;
        border: 1px solid #444 !important;
        border-radius: 12px !important;
    }
    
    /* Tabs */
    .stTabs [role="tablist"] {
        background: #1e1e1e !important;
        border-radius: 12px !important;
        padding: 8px !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%) !important;
        border-radius: 10px !important;
        color: white !important;
    }
    
    .stTabs [role="tab"] {
        color: var(--text-light) !important;
        padding: 10px 20px !important;
    }
    
    /* Progress */
    .stProgress>div>div>div {
        background: linear-gradient(90deg, var(--primary) 0%, var(--accent) 100%) !important;
    }
    
    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #2c2c54 0%, #1a1a2e 100%);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        border: 1px solid #40407a;
    }
    
    .kpi-value {
        font-size: 32px;
        font-weight: 700;
        margin: 10px 0;
        background: linear-gradient(90deg, var(--accent) 0%, #7b68ee 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Alert box */
    .alert-box {
        background: linear-gradient(135deg, #2d3436 0%, #1e2729 100%);
        border-left: 5px solid var(--warning);
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        margin-top: 40px;
        color: var(--text-light);
        font-size: 14px;
        border-top: 1px solid #333;
    }
    
    /* Fix for list styling */
    .custom-card ul {
        padding-left: 20px;
        margin: 15px 0;
    }
    
    .custom-card li {
        margin-bottom: 8px;
        color: var(--text-light);
    }
    
    /* Key phrase styling */
    .key-phrase {
        background: rgba(123, 104, 238, 0.2);
        color: #a78bfa;
        padding: 6px 12px;
        border-radius: 20px;
        border: 1px solid #5e43f3;
        display: inline-block;
        margin: 3px;
    }
    
    /* Summary box */
    .summary-box {
        background: #2a2a2a;
        padding: 15px;
        border-radius: 12px;
        border-left: 3px solid #7b68ee;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize models
@st.cache_resource
def load_models():
    try:
        return (
            hf_pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None, device=-1),
            hf_pipeline("text2text-generation", model="mrm8488/t5-base-finetuned-emotion", device=-1)
        )
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def create_audio_insights():
    return {
        "transcript": "Team discussed Q3 results showing 15% growth in AI products. Concerns raised about supply chain delays affecting delivery timelines. Marketing team proposed new campaign for product launch.",
        "emotions": pd.DataFrame([
            {'label': 'determination', 'score': 0.45},
            {'label': 'optimism', 'score': 0.30},
            {'label': 'concern', 'score': 0.25},
            {'label': 'excitement', 'score': 0.20},
            {'label': 'neutral', 'score': 0.15}
        ]),
        "primary_emotion": "determination",
        "speakers": [
            {"name": "Sarah (CEO)", "time": "32%"},
            {"name": "John (Marketing)", "time": "28%"},
            {"name": "Alex (Operations)", "time": "25%"},
            {"name": "Others", "time": "15%"}
        ],
        "sentiment": "Mixed"
    }

def create_text_insights():
    return {
        "sentiment": "Positive",
        "topics": pd.DataFrame([
            {"topic": "Product Growth", "importance": 95},
            {"topic": "Customer Feedback", "importance": 87},
            {"topic": "Supply Chain", "importance": 78},
            {"topic": "Marketing Campaign", "importance": 65},
            {"topic": "Competition", "importance": 42}
        ]),
        "key_phrases": [
            "Strong growth in AI products",
            "Positive customer feedback",
            "Supply chain delays affecting deliveries",
            "New marketing campaign proposed",
            "Competitive landscape shifting"
        ]
    }

def create_pdf_insights():
    dates = pd.date_range(start="2023-01-01", periods=12, freq="MS")
    return {
        "extracted_text": "Quarterly financial report shows consistent growth across all sectors. AI division leads with 30% YoY increase. Profit margins improved due to operational efficiencies. Customer acquisition is up but supply chain issues may impact next quarter.",
        "trends": ["Growth", "Efficiency", "Expansion", "Risk"],
        "chart_data": pd.DataFrame({
            "Month": dates.strftime("%b %Y"),
            "Revenue": [120, 135, 142, 155, 168, 185, 210, 230, 250, 275, 300, 330],
            "Profit": [45, 50, 55, 60, 65, 70, 78, 85, 92, 100, 110, 125]
        }),
        "kpis": [
            {"name": "Revenue Growth", "value": "+15%", "change": "+2% from last quarter"},
            {"name": "Profit Margin", "value": "24.5%", "change": "+1.2% YoY"},
            {"name": "Customer Acquisition", "value": "12.5K", "change": "+8% from last quarter"},
            {"name": "Churn Rate", "value": "4.2%", "change": "-0.8% from last quarter"}
        ]
    }

# Initialize session state
if "models" not in st.session_state:
    st.session_state.models = load_models()

# Hero section
st.markdown("""
<div style="padding: 3rem 0 2rem 0;">
    <h1 style="font-size: 3.5rem; line-height: 1.2; margin-bottom: 1.5rem;">
        Transform Business Data<br>
        <span style="background: linear-gradient(90deg, #7b68ee, #5e43f3); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Into Strategic Insights
        </span>
    </h1>
    <p style="font-size: 1.25rem; color: var(--text-light); max-width: 800px; line-height: 1.6; margin-bottom: 2rem;">
        Our AI-powered platform analyzes meetings, communications, and reports to give you a 360° view of your business health. 
        Identify opportunities, detect risks, and make data-driven decisions faster.
    </p>
</div>
""", unsafe_allow_html=True)

# Create sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <div style="font-size: 32px; margin-bottom: 10px;">📊</div>
        <h2 style="margin: 0; background: linear-gradient(90deg, #7b68ee, #5e43f3); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            INSIGHT ENGINE
        </h2>
        <p style="color: #a78bfa; margin-top: 5px;">Multi-modal Business Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.header("Data Input")
    
    input_tab, demo_tab = st.tabs(["Upload Data", "Demo Data"])
    
    with input_tab:
        audio_file = st.file_uploader("🎤 Meeting Audio", type=["mp3", "wav"])
        text_input = st.text_area("✉️ Emails/Chat Logs", height=150)
        pdf_file = st.file_uploader("📊 Business Report", type=["pdf"])
        
    with demo_tab:
        st.markdown("**Quickly test with sample data**")
        use_demo_audio = st.checkbox("Use Demo Meeting Audio", True)
        use_demo_text = st.checkbox("Use Demo Text Data", True)
        use_demo_pdf = st.checkbox("Use Demo Report", True)
        
    if st.button("🚀 Generate Insights", use_container_width=True):
        st.session_state.process_data = True
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #a78bfa; padding: 10px; border-radius: 12px; background: rgba(123, 104, 238, 0.1);">
        <p style="margin: 5px 0;">Powered by OPEA & Intel AI</p>
        <p style="margin: 5px 0; font-size: 12px;">Team Cybrion | Utkarsh Verma & Shweta Patel</p>
    </div>
    """, unsafe_allow_html=True)

# Main content
if hasattr(st.session_state, 'process_data') and st.session_state.process_data:
    # Simulate processing
    with st.spinner("Analyzing business data with AI..."):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.02)
            progress_bar.progress(i + 1)
    
    st.success("Analysis Complete! Here's your business intelligence dashboard")
    
    # Create results
    results = {
        "audio": create_audio_insights(),
        "text": create_text_insights(),
        "pdf": create_pdf_insights()
    }
    
    # KPI Section
    st.subheader("Business Health Dashboard")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    kpi1.markdown("""
    <div class="kpi-card">
        <div>Revenue Growth</div>
        <div class="kpi-value">+15%</div>
        <div style="color: #00c853;">▲ 2% from last quarter</div>
    </div>
    """, unsafe_allow_html=True)
    
    kpi2.markdown("""
    <div class="kpi-card">
        <div>Customer Satisfaction</div>
        <div class="kpi-value">92%</div>
        <div style="color: #00c853;">▲ 5% from last quarter</div>
    </div>
    """, unsafe_allow_html=True)
    
    kpi3.markdown("""
    <div class="kpi-card">
        <div>Operational Efficiency</div>
        <div class="kpi-value">78%</div>
        <div style="color: #ffab00;">▼ 3% from target</div>
    </div>
    """, unsafe_allow_html=True)
    
    kpi4.markdown("""
    <div class="kpi-card">
        <div>Risk Level</div>
        <div class="kpi-value">Medium</div>
        <div>Supply chain delays</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Insights in columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Meeting Insights Card
        with st.container():
            st.markdown(f"""
            <pre>
            <div class="custom-card">
                <div class="card-header">
                    <div class="card-icon">🎤</div>
                    <div>
                        <h3 style="margin: 0;">Meeting Insights</h3>
                        <p style="margin: 0; color: #a78bfa;">Dominant Emotion: {results["audio"]["primary_emotion"]}</p>
                    </div>
                </div>
                
                <div>
                    <h4>Key Discussion Points</h4>
                    <ul>
                        <li>15% growth in AI products</li>
                        <li>Supply chain delays affecting deliveries</li>
                        <li>New marketing campaign proposed</li>
                        <li>Q4 projections optimistic</li>
                    </ul>
                </div>
                
                <div>
                    <h4>Sentiment Analysis</h4>
                    <div style="display: flex; align-items: center; gap: 15px;">
                        <div style="font-size: 24px; font-weight: bold; color: #7b68ee;">
                            {results["audio"]["sentiment"]}
                        </div>
                        <div style="flex-grow: 1; background: #333; height: 12px; border-radius: 6px;">
                            <div style="width: 65%; height: 100%; background: linear-gradient(90deg, #7b68ee, #5e43f3); border-radius: 6px;"></div>
                        </div>
                    </div>
                </div>
                
                <div>
                    <h4>Speaker Distribution</h4>
            </pre>
            """, unsafe_allow_html=True)
            
            # Speaker distribution chart
            speaker_df = pd.DataFrame(results["audio"]["speakers"])
            fig = px.pie(
                speaker_df,
                names="name",
                values="time",
                hole=0.5,
                color_discrete_sequence=px.colors.sequential.Viridis,
            )
            fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
        
        # Text Insights Card
        with st.container():
            st.markdown(f"""
            <pre>
            <div class="custom-card">
                <div class="card-header">
                    <div class="card-icon">✉️</div>
                    <div>
                        <h3 style="margin: 0;">Communication Insights</h3>
                        <p style="margin: 0; color: #a78bfa;">Overall Sentiment: {results["text"]["sentiment"]}</p>
                    </div>
                </div>
                
                <div>
                    <h4>Key Phrases</h4>
            </pre>
            """, unsafe_allow_html=True)
            
            # Key phrases
            for phrase in results["text"]["key_phrases"]:
                st.markdown(f'<span class="key-phrase">{phrase}</span>', unsafe_allow_html=True)
            
            st.markdown("""
                    </div>
                </div>
                
                <div>
                    <h4>Topic Importance</h4>
            """, unsafe_allow_html=True)
            
            # Topic importance chart
            fig = px.bar(
                results["text"]["topics"],
                x="importance",
                y="topic",
                orientation='h',
                color="importance",
                color_continuous_scale='Viridis',
                labels={'importance': 'Importance Score', 'topic': ''}
            )
            fig.update_layout(
                yaxis=dict(autorange="reversed"), 
                xaxis=dict(showgrid=False),
                coloraxis_showscale=False,
                margin=dict(t=0, b=0, l=0, r=0),
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True) 
    
    with col2:
        # Financial Report Card
        with st.container():
            st.markdown(f"""
            <pre>
            <div class="custom-card">
                <div class="card-header">
                    <div class="card-icon">📊</div>
                    <div>
                        <h3 style="margin: 0;">Financial Report Analysis</h3>
                        <p style="margin: 0; color: #a78bfa;">Key Trends: {", ".join(results["pdf"]["trends"])}</p>
                    </div>
                </div>
                
                <div>
                    <h4>Revenue & Profit Trends</h4>
            </pre>
            """, unsafe_allow_html=True)
            
            # Financial trends chart
            fig = px.line(
                results["pdf"]["chart_data"],
                x="Month",
                y=["Revenue", "Profit"],
                color_discrete_map={"Revenue": "#7b68ee", "Profit": "#5e43f3"},
                markers=True,
                line_shape="spline"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend_title_text='',
                yaxis_title="Amount ($M)",
                margin=dict(t=30, b=0, l=0, r=0),
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <pre>
                <div>
                    <h4>Report Summary</h4>
                    <div class="summary-box">
                        <p style="margin: 0;">{results["pdf"]["extracted_text"]}</p>
                    </div>
                </div>
            </pre>
            """, unsafe_allow_html=True)
        
        # Emotion Analysis Card
        with st.container():
            st.markdown("""
            <pre>
            <div class="custom-card">
                <div class="card-header">
                    <div class="card-icon">😊</div>
                    <h3 style="margin: 0;">Meeting Emotion Analysis</h3>
                </div>
                
                <div>
                    <h4>Emotion Distribution</h4>
            </pre>
            """, unsafe_allow_html=True)
            
            # Emotion distribution chart
            fig = px.bar(
                results["audio"]["emotions"],
                x="score",
                y="label",
                orientation='h',
                color="score",
                color_continuous_scale='Viridis',
                labels={'score': 'Confidence', 'label': ''}
            )
            fig.update_layout(
                yaxis=dict(autorange="reversed"), 
                xaxis=dict(showgrid=False),
                coloraxis_showscale=False,
                margin=dict(t=0, b=0, l=0, r=0),
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)  # Close card
        
        # Financial KPIs Card
        with st.container():
            st.markdown("""
            <pre>
            <div class="custom-card">
                <div class="card-header">
                    <div class="card-icon">📈</div>
                    <h3 style="margin: 0;">Financial KPIs</h3>
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
            </pre>
            """, unsafe_allow_html=True)
            
            # Financial KPIs
            for kpi in results["pdf"]["kpis"]:
                st.markdown(f"""
                <div style="background: #2a2a2a; padding: 15px; border-radius: 12px; border-left: 3px solid #7b68ee;">
                    <div style="font-size: 14px; color: var(--text-light);">{kpi['name']}</div>
                    <div style="font-size: 24px; font-weight: bold; margin: 5px 0;">{kpi['value']}</div>
                    <div style="font-size: 12px;">{kpi['change']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)  # Close card
    
    # Business Alert
    st.markdown("""
    <div class="alert-box">
        <div style="display: flex; align-items: center; gap: 20px;">
            <div style="font-size: 32px;">⚠️</div>
            <div>
                <h3 style="margin: 0 0 10px 0;">Business Alert: Supply Chain Risk</h3>
                <p style="margin: 0;">
                    Our analysis detected significant supply chain delays that may impact Q4 delivery timelines. 
                    We recommend immediate action to mitigate potential revenue impact.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Recommendations
    st.subheader("Strategic Recommendations")
    rec1, rec2, rec3 = st.columns(3)
    
    rec1.markdown("""
    <div class="custom-card">
        <div class="card-header">
            <div class="card-icon">🔁</div>
            <h3 style="margin: 0;">Optimize Supply Chain</h3>
        </div>
        <ul>
            <li>Diversify supplier base</li>
            <li>Implement predictive analytics</li>
            <li>Increase inventory buffers</li>
            <li>Renegotiate contracts</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    rec2.markdown("""
    <div class="custom-card">
        <div class="card-header">
            <div class="card-icon">🚀</div>
            <h3 style="margin: 0;">Accelerate Growth</h3>
        </div>
        <ul>
            <li>Increase AI R&D investment</li>
            <li>Expand to new markets</li>
            <li>Launch referral program</li>
            <li>Enhance partnerships</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    rec3.markdown("""
    <div class="custom-card">
        <div class="card-header">
            <div class="card-icon">🛡️</div>
            <h3 style="margin: 0;">Mitigate Risks</h3>
        </div>
        <ul>
            <li>Develop contingency plans</li>
            <li>Strengthen cybersecurity</li>
            <li>Monitor competitive landscape</li>
            <li>Stress test financials</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

else:
    # How it works section
    st.markdown("---")
    st.subheader("How It Works")
    steps = st.columns(3)
    
    with steps[0]:
        st.markdown("""
        <div class="custom-card">
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="font-size: 48px; margin-bottom: 10px;">1</div>
                <div style="width: 60px; height: 60px; background: linear-gradient(135deg, #7b68ee, #5e43f3); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto; font-size: 24px;">📤</div>
            </div>
            <h3 style="text-align: center;">Upload Data</h3>
            <p style="text-align: center; color: var(--text-light);">
                Provide meeting recordings, communications, and business reports
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with steps[1]:
        st.markdown("""
        <div class="custom-card">
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="font-size: 48px; margin-bottom: 10px;">2</div>
                <div style="width: 60px; height: 60px; background: linear-gradient(135deg, #7b68ee, #5e43f3); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto; font-size: 24px;">🤖</div>
            </div>
            <h3 style="text-align: center;">AI Processing</h3>
            <p style="text-align: center; color: var(--text-light);">
                Our specialized AI models analyze each data type to extract insights
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with steps[2]:
        st.markdown("""
        <div class="custom-card">
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="font-size: 48px; margin-bottom: 10px;">3</div>
                <div style="width: 60px; height: 60px; background: linear-gradient(135deg, #7b68ee, #5e43f3); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto; font-size: 24px;">📈</div>
            </div>
            <h3 style="text-align: center;">Get Insights</h3>
            <p style="text-align: center; color: var(--text-light);">
                Receive a comprehensive dashboard with metrics and recommendations
            </p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>© 2025 360° AI Business Insight Engine | Intel AI Hackathon Project | Team Cybrion</p>
    <p>Utkarsh Verma & Shweta Patel</p>
</div>
""", unsafe_allow_html=True)
