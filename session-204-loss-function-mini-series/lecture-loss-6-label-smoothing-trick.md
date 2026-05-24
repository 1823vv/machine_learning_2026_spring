# Label Smoothing — A Soft Target for Stable Training

---

## 1. Standard Target (One-Hot)

For vocabulary size $V$, the target is:

$$
y_c =
\begin{cases}
1 & \text{if } c = \text{correct token} \\
0 & \text{otherwise}
\end{cases}
$$

---

### Cross-Entropy (Token-Level)

$$
\mathcal{L}_t = -\log P_{\text{correct}}
$$

---

### Issue

> forces $P_{\text{correct}} \rightarrow 1$ and all others to 0

This is often too strict for natural language.

---

## 2. Simple Example (Shanghai Sentence)

Suppose the model sees:

```text
The Bund in Shanghai is famous for its
```

Correct token:

```text
skyline
```

---

### One-Hot Target

| Token     | Target |
| --------- | ------ |
| skyline   | 1      |
| river     | 0      |
| buildings | 0      |
| food      | 0      |

---

### Problem

Even though:

* “river” is also plausible
* “buildings” is reasonable

the model is forced to say:

> only “skyline” is 100% correct

---

## 3. Label Smoothing Idea

Instead of hard labels, we soften them:

$$
y^{LS}_{\text{correct}} = 1 - \varepsilon
$$

$$
y^{LS}_{i \neq \text{correct}} = \frac{\varepsilon}{V - 1}
$$

---

### Intuition

> the model is allowed to be “slightly unsure”

---

## 4. Example (Same Sentence)

Using label smoothing:

| Token     | Target Probability |
| --------- | ------------------ |
| skyline   | 0.9                |
| river     | 0.033              |
| buildings | 0.033              |
| food      | 0.033              |

---

### Interpretation

Now the model learns:

> skyline is best, but others are not impossible

---

## 5. Loss with Label Smoothing

For a single position:

$$
\mathcal{L}_t = - \sum_{c=1}^{V} y^{LS}_c \log P_c
$$

---

### Key Difference

* one-hot: focuses only on correct token
* smoothed: spreads attention to alternatives

---

## 6. Why This Helps 

### Without smoothing

Model becomes:

* overly confident
* easily overfits
* poor at uncertainty

---

### With smoothing

Model becomes:

* less “certain but wrong”
* more robust
* better calibrated probabilities

---

## 7. Key Takeaway

Label smoothing replaces:

$$
\text{hard one-hot target}
$$

with:

$$
\text{soft probability distribution}
$$

It prevents the model from becoming overconfident and improves generalization.
