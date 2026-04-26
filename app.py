import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model and metadata
@st.cache_resource
def load_model():
    return joblib.load('mental_health_model.joblib')

model_data = load_model()
model = model_data['model']
feature_names = model_data['features']
mappings = model_data['mappings']

# Page configuration
st.set_page_config(
    page_title="SentioMetrics | Teen Risk Predictor",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #6366f1;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #4f46e5;
        border: none;
    }
    .risk-high {
        padding: 20px;
        background-color: #fee2e2;
        border-left: 10px solid #ef4444;
        border-radius: 10px;
        color: #991b1b;
    }
    .risk-low {
        padding: 20px;
        background-color: #dcfce7;
        border-left: 10px solid #22c55e;
        border-radius: 10px;
        color: #166534;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.title("🧠 SentioMetrics: Teen Depression Risk Predictor")
st.markdown("### Assessing the impact of digital behavior on adolescent mental health.")
st.divider()

# Sidebar for inputs
with st.sidebar:
    st.header("👤 Demographic Profile")
    age = st.slider("Age", 13, 19, 16)
    gender = st.selectbox("Gender", ["female", "male"])
    
    st.header("📱 Digital Habits")
    daily_hours = st.slider("Daily Social Media (Hours)", 0.0, 12.0, 4.0, 0.5)
    platform = st.selectbox("Primary Platform", ["Both", "Instagram", "TikTok"])
    screen_before = st.slider("Screen Time Before Sleep (Hours)", 0.0, 6.0, 1.5, 0.5)
    
    st.header("💤 Lifestyle Metrics")
    sleep = st.slider("Sleep Duration (Hours)", 2.0, 10.0, 7.0, 0.5)
    physical = st.slider("Physical Activity (Hours/Day)", 0.0, 3.0, 1.0, 0.1)
    academic = st.slider("Academic Performance (GPA/Grade)", 1.0, 4.0, 3.0, 0.1)
    social = st.selectbox("Social Interaction Level", ["high", "medium", "low"])

st.header("📉 Psychological Indicators")
col1, col2, col3 = st.columns(3)
with col1:
    stress = st.select_slider("Stress Level", options=list(range(1, 11)), value=5)
with col2:
    anxiety = st.select_slider("Anxiety Level", options=list(range(1, 11)), value=5)
with col3:
    addiction = st.select_slider("Addiction Level", options=list(range(1, 11)), value=5)

# Prediction Logic
input_dict = {
    'age': age,
    'gender': mappings['gender'][gender],
    'daily_social_media_hours': daily_hours,
    'platform_usage': mappings['platform_usage'][platform],
    'sleep_hours': sleep,
    'screen_time_before_sleep': screen_before,
    'academic_performance': academic,
    'physical_activity': physical,
    'social_interaction_level': mappings['social_interaction_level'][social],
    'stress_level': stress,
    'anxiety_level': anxiety,
    'addiction_level': addiction
}

input_df = pd.DataFrame([input_dict])[feature_names]

st.divider()

if st.button("Generate Risk Assessment"):
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]
    
    col_res1, col_res2 = st.columns([1, 1])
    
    with col_res1:
        st.subheader("Assessment Result")
        if pred == 1:
            st.markdown(f"""<div class='risk-high'>
                <h2>⚠️ HIGH RISK</h2>
                <p>Confidence: {prob:.1%}</p>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class='risk-low'>
                <h2>✅ LOW RISK</h2>
                <p>Confidence: {(1-prob):.1%}</p>
                </div>""", unsafe_allow_html=True)
                
    with col_res2:
        st.subheader("Risk Probability")
        st.progress(prob)
        st.write(f"The model estimates a **{prob:.1%}** likelihood of depression risk based on these behavioral patterns.")

    st.subheader("💡 Recommendations")
    if prob > 0.75:
        st.error("**Clinical Priority:** Behavioral patterns strongly correlate with mental health distress. Professional consultation is highly recommended.")
    elif prob > 0.45:
        st.warning("**Preventative Action:** Moderate risk detected. Reducing social media usage below 4 hours and improving sleep hygiene may significantly reduce stress.")
    else:
        st.success("**Healthy Profile:** Current digital habits appear sustainable. Maintain consistent sleep and physical activity routines.")
