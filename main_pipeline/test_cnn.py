import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

model = tf.keras.models.load_model("models/ecg_cnn.keras")

# Use held-out test set only (no data leakage)
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

metrics = model.evaluate(X_test, y_test, verbose=0)
y_prob = model.predict(X_test, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)

acc = np.mean(y_pred == y_test)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
try:
    auc = roc_auc_score(y_test, y_prob)
except ValueError:
    auc = float("nan")
cm = confusion_matrix(y_test, y_pred)

print("Loss:", float(metrics[0]))
print("Accuracy:", float(acc))
print("Precision:", float(precision))
print("Recall:", float(recall))
print("F1-score:", float(f1))
print("ROC-AUC:", float(auc))
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Abnormal"],
        zero_division=0,
    )
)

false_negatives = int(cm[1, 0])
print("False Negatives (abnormal predicted normal):", false_negatives)
