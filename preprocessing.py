import re

def clean_text(text: str):
    text = str(text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"[^a-zA-Z\s!?]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text
