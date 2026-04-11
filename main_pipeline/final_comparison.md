# Final Model Comparison

| Model | Accuracy | Precision | Recall | F1 | Notes |
|---|---:|---:|---:|---:|---|
| CNN | 99.44% | 90.91% | 100.00% | 95.24% | Best performance, automatic feature learning |
| LogReg_no_pca | 95.56% | 55.56% | 100.00% | 71.43% | Interpretable and fast |
| LogReg_with_pca | 95.56% | 55.56% | 100.00% | 71.43% | Interpretable and fast |
| RandomForest_no_pca | 97.78% | 100.00% | 60.00% | 75.00% | Stable baseline |
| RandomForest_with_pca | 97.78% | 87.50% | 70.00% | 77.78% | Stable baseline |
| SVM_no_pca | 97.78% | 80.00% | 80.00% | 80.00% | Strong baseline |
| SVM_with_pca | 97.78% | 80.00% | 80.00% | 80.00% | Strong baseline |

## Key Insight
- CNN: best performance due to automatic feature learning.
- Classical ML: more interpretable and often faster.
- Practical trade-off: **performance vs interpretability**.
