DATA_PATH = "Twitter_Data.csv"
MODEL_PATH = "sentiment_model.keras"
TOKENIZER_PATH = "tokenizer.joblib"

RANDOM_STATE = 42

MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 40
EMBEDDING_DIM = 100
EPOCHS = 15
BATCH_SIZE = 32

LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {0: "negative", 1: "neutral", 2: "positive"}
NUM_CLASSES = len(LABEL_TO_ID)
