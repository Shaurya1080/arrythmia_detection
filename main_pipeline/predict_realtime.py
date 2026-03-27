import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("models/ecg_cnn.keras")
stats = np.load("norm_stats.npz")
GLOBAL_MEAN = float(stats["mean"])
GLOBAL_STD = float(stats["std"])

def predict_ecg(ecg_signal):
    ecg_signal = np.array(ecg_signal)
    if ecg_signal.shape[0] != 360:
        raise ValueError(f"Expected 360 samples, got {ecg_signal.shape[0]}")
    if not np.all(np.isfinite(ecg_signal)):
        raise ValueError("Input contains NaN or Inf values")
    if GLOBAL_STD == 0 or not np.isfinite(GLOBAL_STD):
        raise ValueError("Invalid normalization stats (std)")

    ecg_signal = (ecg_signal - GLOBAL_MEAN) / GLOBAL_STD

    ecg_signal = ecg_signal.reshape(1,360,1)

    prediction = model.predict(ecg_signal, verbose=0)

    return prediction[0][0]
