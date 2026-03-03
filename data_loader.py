import pandas as pd
from preprocessing import clean_text
from config import LABEL_TO_ID

def load_and_clean_data(path):
    df = pd.read_csv(path)

    df = df.dropna(subset=["text", "sentiment"]).copy()
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 2]

    df["label_id"] = df["sentiment"].map(LABEL_TO_ID)
    df = df.dropna(subset=["label_id"])
    df["label_id"] = df["label_id"].astype(int)

    return df