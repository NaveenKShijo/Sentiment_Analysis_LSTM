import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="LSTM Sentiment Analyzer",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------
# Load saved artifacts
# ---------------------------
@st.cache_resource
def load_artifacts():
    tokenizer = joblib.load("tokenizer.pkl")
    # label_encoder = joblib.load("label_encoder.pkl")
    model = load_model("lstm_model.keras")
    return tokenizer, model

tokenizer, lstm_model = load_artifacts()

MAX_LEN = 50
PADDING_TYPE = "post"

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("📌 Model Info")
    st.write("**Architecture:** LSTM")
    st.write("**Max Sequence Length:** 50")
    st.write("**Padding Type:** Post")
    st.success("Model Loaded Successfully ✅")

# ---------------------------
# Main Header
# ---------------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>
        🧠 LSTM Sentiment Intelligence
    </h1>
    <p style='text-align: center; font-size:18px;'>
        Deep Learning Powered Sentiment Detection
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------------------
# Input Section (2 Columns)
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    topic = st.text_input("📂 Topic")

with col2:
    query = st.text_area("✍️ Enter Text", height=150)

st.divider()

# ---------------------------
# Prediction
# ---------------------------
if st.button("🚀 Analyze Sentiment", use_container_width=True):

    if topic and query:

        with st.spinner("Analyzing with LSTM model..."):
            
            input_text = topic + " " + query
            input_seq = tokenizer.texts_to_sequences([input_text])
            input_pad = pad_sequences(
                input_seq,
                maxlen=MAX_LEN,
                padding=PADDING_TYPE
            )

            prediction = lstm_model.predict(input_pad)
            predicted_class = np.argmax(prediction, axis=1)
            confidence = np.max(prediction)

            sentiment = label_encoder.inverse_transform(predicted_class)[0]

            # class_names = ["negative", "neutral", "positive"]  # your original order
            # sentiment = class_names[predicted_class[0]]

        st.divider()

        # ---------------------------
        # Display Result Card
        # ---------------------------
        if sentiment.lower() == "positive":
            emoji = "😊"
            color = "#4CAF50"
        elif sentiment.lower() == "negative":
            emoji = "😡"
            color = "#F44336"
        else:
            emoji = "😐"
            color = "#FFC107"

        st.markdown(
            f"""
            <div style="
                background-color:{color}20;
                padding:30px;
                border-radius:15px;
                text-align:center;
            ">
                <h2>{emoji} Sentiment: {sentiment}</h2>               
            </div>
            """,
            unsafe_allow_html=True
        )

        # st.progress(float(confidence))

    else:
        st.warning("⚠️ Please enter both topic and text.")