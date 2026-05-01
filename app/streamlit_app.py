import streamlit as st
import pandas as pd
import requests
import random

st.set_page_config(page_title="Fraud Detection System", layout="wide")

# ---------------- STYLING ---------------- #

st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💳 AI Fraud Detection System")
st.caption("Real-time fraud prediction using Machine Learning")

# ---------------- INIT ---------------- #

if "initialized" not in st.session_state:
    for i in range(29):
        st.session_state[f"f{i}"] = 0.0
    st.session_state.initialized = True

@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/creditcard_2023.csv")
    except FileNotFoundError:
        return None

df = load_data()

# ---------------- BUTTON FUNCTIONS ---------------- #

def load_normal():
    if df is not None:
        sample = df[df["Class"] == 0].sample(1).iloc[0]
        values = sample.drop(["Class", "id"]).tolist()
        for i in range(29):
            st.session_state[f"f{i}"] = float(values[i])

def load_fraud():
    if df is not None:
        sample = df[df["Class"] == 1].sample(1).iloc[0]
        values = sample.drop(["Class", "id"]).tolist()
        for i in range(29):
            st.session_state[f"f{i}"] = float(values[i])

def reset_values():
    for i in range(29):
        st.session_state[f"f{i}"] = 0.0

def random_values():
    for i in range(29):
        st.session_state[f"f{i}"] = random.uniform(-5, 5)

# ---------------- BUTTONS ---------------- #

col1, col2, col3, col4 = st.columns(4)

data_available = df is not None

with col1:
    st.button("📥 Normal Sample", on_click=load_normal, disabled=not data_available)

with col2:
    st.button("⚠️ Fraud Sample", on_click=load_fraud, disabled=not data_available)

with col3:
    st.button("🔄 Reset", on_click=reset_values)

with col4:
    st.button("🎲 Random Values", on_click=random_values)

# ---------------- INPUTS ---------------- #

st.subheader("Transaction Details")

cols = st.columns(3)

inputs = []

for i in range(29):
    col = cols[i % 3]
    with col:
        val = st.number_input(
            f"V{i+1}" if i < 28 else "Amount",
            key=f"f{i}"
        )
        inputs.append(val)

# ---------------- PREDICTION ---------------- #

if st.button("🚀 Predict Fraud"):

    response = requests.post(
        "https://fraud-detection-ml-jp1y.onrender.com/predict",
        json={"features": inputs}
    )

    result = response.json()

    prob = result["probability"]

    st.markdown("---")

    if result["fraud"] == 1:
        st.error(f"⚠️ FRAUD DETECTED\n\nConfidence: {prob:.2f}")
    else:
        st.success(f"✅ NORMAL TRANSACTION\n\nConfidence: {prob:.2f}")

    st.progress(prob)