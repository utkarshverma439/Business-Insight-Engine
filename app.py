import os
os.environ['USE_TF'] = '0'  # Disable TensorFlow
os.environ['USE_TORCH'] = '1'  # Force PyTorch

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import tempfile
import librosa
from pydub import AudioSegment
from io import BytesIO
import PyPDF2

# Import only what we need from transformers
from transformers import (
    pipeline as hf_pipeline,
    WhisperProcessor, 
    WhisperForConditionalGeneration
)

# Initialize models with error handling
@st.cache_resource
def load_models():
    try:
        # Load emotion classifier
        emotion_classifier = hf_pipeline(
            "text-classification", 
            model="SamLowe/roberta-base-go_emotions", 
            top_k=None,
            device=-1  # Use CPU
        )
        
        # Load sentiment/topic model
        text_analyzer = hf_pipeline(
            "text2text-generation",
            model="mrm8488/t5-base-finetuned-emotion",
            device=-1  # Use CPU
        )
        
        # Load speech recognition
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        
        return emotion_classifier, text_analyzer, processor, asr_model
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

def process_audio(audio_file):
    """Voice microservice: Transcribe audio and analyze emotion"""
    try:
        # Convert to WAV format
        audio = AudioSegment.from_file(audio_file)
        wav_file = BytesIO()
        audio.export(wav_file, format="wav")
        wav_file.seek(0)
        
        # Load audio using librosa
        y, sr = librosa.load(wav_file, sr=16000)
        
        # Get models from session state
        processor, asr_model = st.session_state.models[2], st.session_state.models[3]
        
        # Process audio
        input_features = processor(
            y, 
            sampling_rate=sr, 
            return_tensors="pt"
        ).input_features
        
        # Generate transcription
        predicted_ids = asr_model.generate(input_features)
        transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        # Analyze emotion
        if st.session_state.models[0]:
            emotions = st.session_state.models[0](transcript[:512])[0]
            emotion_df = pd.DataFrame(emotions)
            emotion_df = emotion_df.sort_values("score", ascending=False).head(3)
            primary_emotion = emotion_df.iloc[0]['label']
        else:
            emotion_df = pd.DataFrame()
            primary_emotion = "N/A"
            
        return {
            "transcript": transcript,
            "emotions": emotion_df,
            "primary_emotion": primary_emotion
        }
        
    except Exception as e:
        st.error(f"Audio processing error: {str(e)}")
        return {
            "transcript": "Processing failed",
            "emotions": pd.DataFrame(),
            "primary_emotion": "Error"
        }

def process_text(text):
    """Text microservice: Analyze sentiment and extract topics"""
    try:
        if st.session_state.models[1]:
            # Sentiment analysis
            sentiment_result = st.session_state.models[1](
                f"analyze sentiment: {text[:1000]}", 
                max_length=50
            )[0]['generated_text']
        else:
            sentiment_result = "N/A"
            
        # Simple topic extraction
        words = re.findall(r'\b\w{4,}\b', text.lower())
        word_counts = pd.Series(words).value_counts().head(5)
        topics = list(word_counts.index)
        
        return {
            "sentiment": sentiment_result,
            "topics": topics,
            "processed_text": text[:500] + "..." if len(text) > 500 else text
        }
        
    except Exception as e:
        st.error(f"Text processing error: {str(e)}")
        return {
            "sentiment": "Error",
            "topics": [],
            "processed_text": text[:500] + "..." if len(text) > 500 else text
        }

def process_pdf(pdf_file):
    """Chart microservice: Extract text and detect trends"""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        # Trend detection
        trend_keywords = ["growth", "increase", "decrease", "trend", "forecast", "rise", "fall"]
        trends_found = [kw for kw in trend_keywords if kw in text.lower()]
        
        # Generate mock financial data
        dates = pd.date_range(start="2023-01-01", periods=12, freq="M")
        values = np.random.randint(50, 200, size=12).cumsum()
        
        return {
            "extracted_text": text[:500] + "..." if len(text) > 500 else text,
            "trends": trends_found,
            "chart_data": pd.DataFrame({
                "Month": dates.strftime("%b %Y"),
                "Value": values
            })
        }
        
    except Exception as e:
        st.error(f"PDF processing error: {str(e)}")
        return {
            "extracted_text": "Processing failed",
            "trends": [],
            "chart_data": pd.DataFrame()
        }

# Initialize session state
if "models" not in st.session_state:
    st.session_state.models = load_models()

# Streamlit UI
st.set_page_config(layout="wide")
st.title("🚀 360° AI Business Insight Engine")
st.subheader("Multi-modal Enterprise Insights Dashboard")

# Input Section
st.sidebar.header("Upload Business Data")
audio_file = st.sidebar.file_uploader("Meeting Audio (MP3/WAV)", type=["mp3", "wav"])
text_input = st.sidebar.text_area("Emails/Chat Logs")
pdf_file = st.sidebar.file_uploader("Business Report (PDF)", type=["pdf"])
st.sidebar.caption("Powered by OPEA | Intel AI Hackathon Project")

# Create sample data
demo_text = "The quarterly results show strong growth in our AI division. Customers are responding positively to our new features. However, there are concerns about delivery timelines from the logistics team."

# Processing
if st.sidebar.button("Generate Insights", type="primary"):
    results = {}
    
    with st.spinner("Analyzing business data..."):
        if audio_file:
            results["audio"] = process_audio(audio_file)
        elif st.sidebar.checkbox("Use Demo Audio", True):
            results["audio"] = {
                "transcript": "Demo: Team discussed Q3 results showing 15% growth in AI products. Concerns raised about supply chain delays.",
                "emotions": pd.DataFrame({
                    'label': ['admiration', 'joy', 'neutral'],
                    'score': [0.45, 0.30, 0.25]
                }),
                "primary_emotion": "admiration"
            }
            
        if text_input:
            results["text"] = process_text(text_input)
        else:
            results["text"] = process_text(demo_text)
            
        if pdf_file:
            results["pdf"] = process_pdf(pdf_file)
        elif st.sidebar.checkbox("Use Demo PDF", True):
            # Create demo PDF data
            dates = pd.date_range(start="2023-01-01", periods=12, freq="M")
            values = [100, 120, 130, 145, 160, 190, 210, 230, 250, 270, 300, 330]
            
            results["pdf"] = {
                "extracted_text": "Demo: Quarterly financial report shows consistent growth in all sectors. AI division leads with 30% YoY increase.",
                "trends": ["growth", "increase"],
                "chart_data": pd.DataFrame({
                    "Month": dates.strftime("%b %Y"),
                    "Value": values
                })
            }
    
    # Display Results
    st.success("Analysis Complete!")
    
    # Unified Dashboard
    st.header("Consolidated Business Insights")
    
    col1, col2, col3 = st.columns(3)
    
    # Audio Insights
    if "audio" in results:
        with col1:
            st.subheader("🎤 Meeting Analysis")
            st.write(f"**Transcription:** {results['audio']['transcript']}")
            st.write(f"**Dominant Emotion:** {results['audio']['primary_emotion']}")
            
            if not results['audio']['emotions'].empty:
                fig = px.bar(
                    results['audio']['emotions'],
                    x='label',
                    y='score',
                    title="Emotion Distribution",
                    color='score'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Text Insights
    if "text" in results:
        with col2:
            st.subheader("✉️ Text Analysis")
            st.write(f"**Sentiment:** {results['text']['sentiment']}")
            if results["text"]["topics"]:
                st.write("**Key Topics:**")
                for topic in results["text"]["topics"]:
                    st.markdown(f"- `{topic}`")
    
    # PDF Insights
    if "pdf" in results:
        with col3:
            st.subheader("📊 Business Report")
            if results["pdf"]["trends"]:
                st.write(f"**Detected Trends:** {', '.join(results['pdf']['trends'])}")
            else:
                st.write("**Detected Trends:** None")
                
            if not results['pdf']['chart_data'].empty:
                fig = px.line(
                    results['pdf']['chart_data'],
                    x="Month",
                    y="Value",
                    title="Business Performance",
                    markers=True
                )
                fig.update_layout(yaxis_range=[0, results['pdf']['chart_data']['Value'].max() * 1.1])
                st.plotly_chart(fig, use_container_width=True)
    
    # Show alerts
    alert_triggered = False
    alert_text = []
    
    if "audio" in results and "negative" in results["audio"]["primary_emotion"].lower():
        alert_text.append("Negative emotions in meeting")
        alert_triggered = True
    if "text" in results and "negative" in results["text"]["sentiment"].lower():
        alert_text.append("Negative sentiment in communications")
        alert_triggered = True
    if "pdf" in results and "decrease" in [t.lower() for t in results["pdf"]["trends"]]:
        alert_text.append("Decreasing trends in reports")
        alert_triggered = True
    
    if alert_triggered:
        st.warning(f"⚠️ **Business Alerts:** {' | '.join(alert_text)}")
    else:
        st.info("✅ No critical issues detected")

# Add instructions if no data
if not any([audio_file, text_input, pdf_file]):
    st.info("💡 Upload business data or use the demo options in the sidebar to generate insights")

st.markdown("---")
st.caption("Intel AI Hackathon Project | Team Cybrion | Utkarsh Verma & Shweta Patel")