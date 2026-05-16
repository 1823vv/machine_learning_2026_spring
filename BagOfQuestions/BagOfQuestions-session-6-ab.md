## Question: Classification Metrics for an Imbalanced Dataset

Consider a binary classifier for a medical screening task. The positive class is rare. The confusion matrix counts are:

```text
True Positive  (TP) = 40
False Positive (FP) = 160
False Negative (FN) = 10
True Negative  (TN) = 790
```

1. Compute accuracy.
2. Compute precision. Compute recall.
3. Compute the F1 score. Explain why accuracy alone can be misleading for imbalanced datasets.
4. In this medical screening setting, why might recall be especially important?

## Question: Choosing the Right Metric

Different machine-learning applications require different evaluation metrics. A metric should match the cost of different mistakes.

1. For spam detection, when might precision be more important than recall?
2. For disease screening, when might recall be more important than precision? For a balanced classification problem, when might accuracy be acceptable?
3. Explain why the “best” model can change when the evaluation metric changes.
4. Draw a small decision diagram for selecting accuracy, precision, recall, or F1.
