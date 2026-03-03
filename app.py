
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from joblib import load
import re

# FastAPI app
app = FastAPI(title="Sentiment Analysis API")

model = None
tokenizer = None

MAX_SEQUENCE_LENGTH = 40

ID_TO_LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# Load model and tokenizer ONCE at startup
@app.on_event("startup")
def load_assets():
    global model, tokenizer
    model = tf.keras.models.load_model("sentiment_model.keras")
    tokenizer = load("tokenizer.joblib")

# Text cleaning (SAME as training)

def clean_text(text: str):
    text = str(text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"[^a-zA-Z\s!?]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Request schema

class TextInput(BaseModel):
    text: str

# Routes

@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running"}

@app.post("/predict")
def predict_sentiment(data: TextInput):
    cleaned = clean_text(data.text)

    seq = tokenizer.texts_to_sequences([cleaned])

    pad = tf.keras.preprocessing.sequence.pad_sequences(
        seq,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post"
    )

    preds = model.predict(pad)
    class_id = int(np.argmax(preds, axis=1)[0])

    return {
        "text": data.text,
        "cleaned_text": cleaned,
        "sentiment": ID_TO_LABEL[class_id],
        "confidence": float(np.max(preds))
    }
