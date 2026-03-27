import numpy as np
import tensorflow as tf
import os
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load pre-split data (patient-wise split)
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

# Class weights to handle imbalance (more weight for minority class: abnormal)
from sklearn.utils.class_weight import compute_class_weight
classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)
class_weight = dict(zip(classes, weights))

model = tf.keras.Sequential([

    tf.keras.layers.Conv1D(32,5,activation='relu',input_shape=(360,1)),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv1D(64,5,activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(1,activation='sigmoid')

])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(

    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
    ]

)

model.summary()

os.makedirs("models", exist_ok=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath="models/ecg_cnn_best.keras",
        monitor="val_loss",
        save_best_only=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
    ),
]

model.fit(
    X_train,
    y_train,
    epochs=8,
    batch_size=32,
    validation_data=(X_val, y_val),
    class_weight=class_weight,
    callbacks=callbacks
)

model.save("models/ecg_cnn.keras")

print("Model saved")
