from sklearn.model_selection import train_test_split

from config import *
from data_loader import load_and_clean_data
from tokenizer import build_tokenizer, tokenize_and_pad
from model import build_model
from train import train_model
from save_load import save_all

# 1. Load data
df = load_and_clean_data(DATA_PATH)

X = df["clean_text"].values
y = df["label_id"].values

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# 3. Tokenizer
tokenizer = build_tokenizer(X_train)
X_train_pad = tokenize_and_pad(tokenizer, X_train)
X_test_pad  = tokenize_and_pad(tokenizer, X_test)

# 4. Model
model = build_model(
    MAX_NUM_WORDS,
    EMBEDDING_DIM,
    MAX_SEQUENCE_LENGTH
)

# 5. Train
history = train_model(
    model,
    X_train_pad,
    y_train,
    X_test_pad,
    y_test,
    EPOCHS,
    BATCH_SIZE
)

save_all(model, tokenizer, MODEL_PATH, TOKENIZER_PATH)

print("✅ Training complete. Model & tokenizer saved.")
