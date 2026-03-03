from joblib import dump, load
import tensorflow as tf

def save_all(model, tokenizer, model_path, tokenizer_path):
    model.save(model_path)
    dump(tokenizer, tokenizer_path)

def load_all(model_path, tokenizer_path):
    model = tf.keras.models.load_model(model_path)
    tokenizer = load(tokenizer_path)
    return model, tokenizer
