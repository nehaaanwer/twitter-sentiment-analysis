from tensorflow.keras import layers, models
from tensorflow.keras.layers import Bidirectional, LSTM

def build_model(max_num_words, embedding_dim, max_sequence_length):
    model = models.Sequential([
        layers.Embedding(
            input_dim=max_num_words,
            output_dim=embedding_dim,
            input_length=max_sequence_length
        ),
        layers.SpatialDropout1D(0.1),
        Bidirectional(LSTM(128)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(3, activation="softmax")
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model
