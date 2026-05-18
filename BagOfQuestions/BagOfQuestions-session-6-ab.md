## Question: Classification Metrics for an Imbalanced Medical Screening Task

Consider a binary classifier for a **medical screening task** where the positive class (disease present) is rare. The confusion matrix counts are:

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | True Positive (TP) = 40 | False Negative (FN) = 10 |
| **Actual Negative** | False Positive (FP) = 160 | True Negative (TN) = 790 |

**Definitions for reference:**
- **Accuracy**: $\frac{TP + TN}{TP + FP + FN + TN}$
- **Precision**: $\frac{TP}{TP + FP}$ (of all predicted positive, how many are actually positive?)
- **Recall**: $\frac{TP}{TP + FN}$ (of all actual positive, how many did we catch?)
- **F1 Score**: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$

1. Compute **accuracy** for this classifier. Show your calculation.

2. Compute **precision**. Compute **recall**. Show your calculations.

3. Compute the **F1 score**. Explain why **accuracy alone can be misleading** for imbalanced datasets like this one, using the numbers above to illustrate your point.

4. In this **medical screening setting**, why might **recall be especially important**? What is the real-world cost of the 10 false negatives?

5. Now consider a **spam detection** task instead. In that setting, why might **precision be more important than recall**? What is the real-world cost of too many false positives (legitimate emails marked as spam)?


