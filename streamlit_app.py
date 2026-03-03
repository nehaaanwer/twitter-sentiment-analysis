import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# CONFIG 
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="💬",
    layout="wide"
)
# HEADER 
st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        color: white;
        animation: fadeIn 1.5s ease-in;
    }

    @keyframes fadeIn {
        0% {opacity: 0; transform: translateY(-20px);}
        100% {opacity: 1; transform: translateY(0);}
    }
    </style>

    <div class="hero">
        <h1>💬 Sentiment Analysis Dashboard</h1>
        <p>FastAPI • LSTM • NLP • Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("## 📌 Project Description")

st.write(
"""
This project is an end-to-end **Sentiment Analysis System** that classifies text into:

- 😊 Positive  
- 😐 Neutral  
- 😠 Negative  

### 🔧 Tech Stack
- **Frontend:** Streamlit
- **Backend API:** FastAPI
- **Model:** LSTM (Deep Learning)
- **NLP:** Tokenization & Padding

The system cleans input text, converts it into sequences,
and uses a trained LSTM network to predict sentiment with confidence scores.
"""
)
st.markdown("## ⚙️ How It Works")

col1, col2, col3, col4 = st.columns(4)

col1.metric("1️⃣ Input", "User Text")
col2.metric("2️⃣ NLP", "Tokenization")
col3.metric("3️⃣ Model", "LSTM")
col4.metric("4️⃣ Output", "Sentiment")

from streamlit_lottie import st_lottie
import requests

def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_ai = load_lottie(
    "https://assets2.lottiefiles.com/packages/lf20_kyu7xb1v.json"
)

st_lottie(lottie_ai, height=200)

with st.expander("🧠 Model Architecture"):

    st.write("""
    - Embedding Layer
    - LSTM (128 units)
    - Dense Softmax Output
    - Sparse Categorical Crossentropy Loss
    """)

    st.code("""
    Embedding → LSTM → Dense → Softmax
    """)


st.markdown(
    """
    <hr style="height:3px;border:none;color:#333;
    background:linear-gradient(to right, #ff512f, #dd2476);" />
    """,
    unsafe_allow_html=True
)


API_URL = "http://127.0.0.1:8000/predict"
DATA_PATH = "Twitter_Data.csv"   # change if needed

# SESSION 
if "history" not in st.session_state:
    st.session_state.history = []

# API CHECK 
def check_api():
    try:
        r = requests.get("http://127.0.0.1:8000")
        return r.status_code == 200
    except:
        return False

# SIDEBAR 
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "💬 Sentiment Analyzer",
        "📊 EDA Dashboard",
        "🧠 Model Performance",
        "📜 Prediction History"
    ]
)

st.title("💬 Sentiment Analysis Suite")
st.caption("FastAPI + LSTM + Streamlit")

#  SENTIMENT ANALYZER

if page == "💬 Sentiment Analyzer":

    st.subheader("Analyze Text Sentiment")

    if check_api():
        st.success("🟢 API is running")
    else:
        st.error("🔴 API is not running")
        st.stop()

    text = st.text_area(
        "Enter text",
        height=120,
        placeholder="Type something like: I love this product!"
    )

    col1, col2 = st.columns(2)

    if col1.button("Analyze Sentiment", use_container_width=True):

        if text.strip() == "":
            st.warning("Please enter some text")

        else:
            with st.spinner("Analyzing..."):
                time.sleep(0.5)

                response = requests.post(API_URL, json={"text": text})
                result = response.json()

                sentiment = result["sentiment"]
                confidence = result["confidence"]
                cleaned_text = result["cleaned_text"]

                st.session_state.history.append({
                    "text": text,
                    "sentiment": sentiment,
                    "confidence": confidence
                })

            st.subheader("🔍 Result")

            if sentiment == "positive":
                st.success("😊 Positive")
            elif sentiment == "neutral":
                st.info("😐 Neutral")
            else:
                st.error("😠 Negative")

            st.write("**Confidence**")
            st.progress(min(confidence, 1.0))
            st.caption(f"{confidence:.2f}")

            with st.expander("See cleaned text"):
                st.code(cleaned_text)

    if col2.button("Clear Text", use_container_width=True):
        st.rerun()

# EDA DASHBOARD
elif page == "📊 EDA Dashboard":

    st.subheader("Dataset Exploration")

    df = pd.read_csv(DATA_PATH)

    st.write("### Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Class Distribution
    st.write("### Sentiment Distribution")

    fig1, ax1 = plt.subplots()
    sns.countplot(x="sentiment", data=df, ax=ax1)
    ax1.set_title("Sentiment Class Distribution")
    st.pyplot(fig1)

    # Value Counts
    st.write("### Value Counts")
    st.write(df["sentiment"].value_counts(dropna=False))

    # Text Length Distribution
    st.write("### Text Length Distribution")

    df["text_length"] = df["text"].astype(str).apply(len)

    fig2, ax2 = plt.subplots()
    ax2.hist(df["text_length"], bins=50)
    ax2.set_title("Text Length Histogram")
    st.pyplot(fig2)

# MODEL PERFORMANCE

elif page == "🧠 Model Performance":

    st.subheader("Training Results")

    st.info("Displays model evaluation metrics")

    # Example Confusion Matrix (replace with real saved data)
    cm = np.array([
        [120, 10, 5],
        [8, 95, 12],
        [4, 9, 140]
    ])

    fig3, ax3 = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Neutral", "Positive"],
        yticklabels=["Negative", "Neutral", "Positive"],
        ax=ax3
    )

    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    ax3.set_title("Confusion Matrix")

    st.pyplot(fig3)

    # Loss Curve
    st.write("### Training vs Validation Loss")

    train_loss = [0.9, 0.6, 0.4, 0.3]
    val_loss   = [1.0, 0.7, 0.5, 0.45]

    fig4, ax4 = plt.subplots()

    ax4.plot(train_loss, label="Training Loss")
    ax4.plot(val_loss, label="Validation Loss")

    ax4.set_xlabel("Epochs")
    ax4.set_ylabel("Loss")
    ax4.legend()

    st.pyplot(fig4)

# HISTORY

elif page == "📜 Prediction History":

    st.subheader("Past Predictions")

    if st.session_state.history:

        df_hist = pd.DataFrame(st.session_state.history)

        st.dataframe(df_hist, use_container_width=True)

        st.write("### Sentiment Distribution")
        st.bar_chart(df_hist["sentiment"].value_counts())

    else:
        st.info("No predictions yet.")