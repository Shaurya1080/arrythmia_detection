import os
import random

import numpy as np
import tensorflow as tf

SEED = 42
EPOCHS = int(os.environ.get("CNN_EPOCHS", "20"))
BATCH_SIZE = int(os.environ.get("CNN_BATCH_SIZE", "128"))
USE_CLASS_WEIGHTS = os.environ.get("USE_CLASS_WEIGHTS", "0") == "1"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def conv_block(filters: int, kernel_size: int) -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv1D(filters, kernel_size, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling1D(2),
        ]
    )


def build_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(360, 1))
    x = conv_block(32, 7)(inputs)
    x = conv_block(64, 5)(x)
    x = conv_block(128, 3)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def get_class_weight(y_train):
    if not USE_CLASS_WEIGHTS:
        return None

    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    return dict(zip(classes, weights))


def main() -> None:
    set_seed(SEED)

    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")
    X_val = np.load("X_val.npy")
    y_val = np.load("y_val.npy")

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    model.summary()

    os.makedirs("models", exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="models/ecg_cnn_best.keras",
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
    ]

    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        class_weight=get_class_weight(y_train),
        callbacks=callbacks,
    )

    model.save("models/ecg_cnn.keras")
    print("Model saved")


if __name__ == "__main__":
    main()
