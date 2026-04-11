# Final Model Comparison

| Model | Accuracy | Precision | Recall | F1 | Notes |
|---|---:|---:|---:|---:|---|
| CNN | 73.41% | 27.37% | 74.30% | 40.00% | Best performance, automatic feature learning |
| LogReg_no_pca | 60.11% | 22.08% | 92.66% | 35.66% | Interpretable and fast |
| LogReg_with_pca | 60.11% | 22.08% | 92.66% | 35.66% | Interpretable and fast |
| RandomForest_no_pca | 72.43% | 28.98% | 90.38% | 43.89% | Stable baseline |
| RandomForest_with_pca | 74.00% | 29.46% | 84.57% | 43.70% | Stable baseline |
| SVM_no_pca | 58.31% | 21.23% | 92.02% | 34.50% | Strong baseline |
| SVM_with_pca | 64.64% | 24.21% | 92.14% | 38.34% | Strong baseline |

## Key Insight
- CNN: best performance due to automatic feature learning.
- Classical ML: more interpretable and often faster.
- Practical trade-off: **performance vs interpretability**.
