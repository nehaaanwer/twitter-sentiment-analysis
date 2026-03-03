import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

def train_model(model, X_train_pad, y_train, X_test_pad, y_test, EPOCHS, BATCH_SIZE):

    X_train_pad = np.array(X_train_pad, dtype="int32")
    X_test_pad  = np.array(X_test_pad, dtype="int32")
    y_train     = np.array(y_train, dtype="int32")
    y_test      = np.array(y_test, dtype="int32")

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train_pad,
        y_train,
        validation_data=(X_test_pad, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=1
    )

    return history
