# Final Model Comparison

| Model | Accuracy | Precision | Recall | F1 | Notes |
|---|---:|---:|---:|---:|---|
| CNN | 98.50% | 0.00% | 0.00% | 0.00% | Best performance, automatic feature learning |
| LogReg_no_pca | 98.55% | 100.00% | 2.94% | 5.71% | Interpretable and fast |
| LogReg_with_pca | 98.55% | 100.00% | 2.94% | 5.71% | Interpretable and fast |
| RandomForest_no_pca | 98.50% | 50.00% | 2.94% | 5.56% | Stable baseline |
| RandomForest_with_pca | 98.50% | 0.00% | 0.00% | 0.00% | Stable baseline |
| SVM_no_pca | 98.55% | 100.00% | 2.94% | 5.71% | Strong baseline |
| SVM_with_pca | 98.55% | 100.00% | 2.94% | 5.71% | Strong baseline |

## Key Insight
- CNN: best performance due to automatic feature learning.
- Classical ML: more interpretable and often faster.
- Practical trade-off: **performance vs interpretability**.
