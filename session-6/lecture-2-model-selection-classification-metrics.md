# Confusion Matrix and Classification Metrics

## 1. Purpose and Intuition

![](./img/confusionmatrixgreenblackboard.jpg)

In classification problems, we need more than just a single number to understand model performance.

The **confusion matrix** provides a structured breakdown of prediction outcomes, showing not only what is correct, but also *how* predictions fail.

It separates errors into meaningful categories, which is essential for deeper evaluation beyond simple accuracy.

---

## 2. Binary Classification Setup

We consider a binary classification task with two classes:

* Positive
* Negative

Each prediction can be categorized based on comparison with the ground truth.

---

## 3. Confusion Matrix Structure

|                 | Predicted Positive | Predicted Negative |
| --------------- | ------------------ | ------------------ |
| Actual Positive | TP                 | FN                 |
| Actual Negative | FP                 | TN                 |

### 3.1 Definitions

* $TP$: True Positive, correctly predicted positive
* $TN$: True Negative, correctly predicted negative
* $FP$: False Positive, predicted positive but actually negative
* $FN$: False Negative, predicted negative but actually positive

> [!INFO]
> The confusion matrix is the fundamental structure behind most classification metrics.

---

## 4. Core Classification Metrics


![](./img/classificationmatricschalkboard.jpg)


## 4.1 Accuracy


Accuracy measures the overall correctness of predictions:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

### Interpretation

> Among all samples, how many are correctly classified?

### Limitation

> [!WARNING]
> Accuracy can be misleading when classes are imbalanced.

For example:

* 99% Negative, 1% Positive
* Always predicting Negative → 99% accuracy
* But the model learns nothing meaningful about the Positive class

---

## 4.2 Precision

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

### Interpretation

> Among all predicted positives, how many are actually correct?

### Insight

* High precision means few false alarms
* Useful when false positives are costly (e.g., spam filtering)

---

## 4.3 Recall

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

### Interpretation

> Among all actual positives, how many are successfully detected?

### Insight

* High recall means few missed positives
* Important when missing a positive is costly (e.g., medical diagnosis)

---

## 4.4 F1 Score

$$
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

### Interpretation

F1 score balances precision and recall into a single metric.

### When to use

* Imbalanced datasets
* When both false positives and false negatives matter

---

## 5. Multi-class Extension (Optional)

For a classification problem with $K$ classes, the confusion matrix generalizes to a $K \times K$ structure:

### Structure

* Rows represent actual classes
* Columns represent predicted classes
* Each entry $(i, j)$ counts samples from class $i$ predicted as class $j$

> [!INFO]
> Multi-class confusion matrices preserve the same idea: comparing predictions against ground truth, but across multiple categories instead of two.
