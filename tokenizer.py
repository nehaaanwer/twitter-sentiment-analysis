from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import MAX_NUM_WORDS, MAX_SEQUENCE_LENGTH

def build_tokenizer(X_train):
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    return tokenizer

def tokenize_and_pad(tokenizer, X):
    seq = tokenizer.texts_to_sequences(X)
    pad = pad_sequences(
        seq,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post"
    )
    return pad
