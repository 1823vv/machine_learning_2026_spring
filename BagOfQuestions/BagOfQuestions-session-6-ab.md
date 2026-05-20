## Question: Metrics Under Class Imbalance and Task-Dependent Trade-offs

A binary medical screening classifier produces the confusion matrix below:

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | TP = 40 | FN = 10 |
| **Actual Negative** | FP = 160 | TN = 790 |

Use:
- Accuracy $= \frac{TP+TN}{TP+FP+FN+TN}$
- Precision $= \frac{TP}{TP+FP}$
- Recall $= \frac{TP}{TP+FN}$
- F1 $= 2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}$

1. Compute accuracy, precision, recall, and F1. Show each calculation and final numeric value.
2. Using your results, explain concretely why accuracy alone can be misleading on imbalanced data in this medical setting.
3. In this screening context, argue why recall is often prioritized. Interpret the real-world consequence of the 10 false negatives.
4. Switch to a spam-filter scenario and explain why precision may become the primary objective. Interpret the real-world cost of many false positives.
5. Propose one practical threshold-tuning strategy (or evaluation workflow) to rebalance precision vs recall for each of the two domains above.
