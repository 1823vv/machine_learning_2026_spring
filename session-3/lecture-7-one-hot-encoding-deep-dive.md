# One-Hot Encoding Deep Dive

## 1. Why One-Hot Encoding Matters

Machine learning models require numbers, but many variables are categorical:

* city
* product type
* language
* class label (cat, dog, bird)

One-hot encoding converts each category into a vector of length $K$ (for $K$ categories), with:

* one entry = $1$
* all others = $0$

> [!INFO]
> One category = one active position.

---

## 2. One-Hot Encoding for Input Features

### 2.1 Categorical Inputs

Suppose `State` has three values:

* New York
* California
* Florida

| Category   | NY | CA | FL |
| ---------- | -: | -: | -: |
| New York   |  1 |  0 |  0 |
| California |  0 |  1 |  0 |
| Florida    |  0 |  0 |  1 |

This lets the model treat categories numerically without implying false relationships.

---

### 2.2 Why Not Integer Encoding?

Bad encoding:

| Category   | Integer |
| ---------- | ------: |
| New York   |       0 |
| California |       1 |
| Florida    |       2 |

This incorrectly suggests:

$$
0 < 1 < 2
$$

and artificial distances.

> [!WARNING]
> Integer labels can mislead models when categories have no true order.

---

### 2.3 Input Dimension Expansion

If there are:

* $d_{num}$ numerical features
* $K$ categories

Then:

$$
d = d_{num} + K
$$

One categorical column becomes multiple binary columns.

---

### 2.4 Dummy Variable Trap

For linear models:

$$
State_{NY}+State_{CA}+State_{FL}=1
$$

With an intercept term, one column may be dropped, so often $K-1$ columns are used.

> [!INFO]
> Neural networks often keep all $K$ columns.

---

## 3. One-Hot Encoding for Output Labels

For multiclass classification with $K$ classes:

$$
y \in \mathbb{R}^{1 \times K}
$$

Example: digit classification ($K=10$).

If the true class is $3$:

$$
y = [0,0,0,1,0,\dots,0]
$$

The model outputs logits:

$$
z^{(L)} \in \mathbb{R}^{1 \times K}
$$

Softmax converts logits to probabilities:

$$
\hat{y}_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$

---

### Batch Form

$$
Y \in \mathbb{R}^{n \times K}
$$

$$
\hat{Y} \in \mathbb{R}^{n \times K}
$$

Each row of $Y$ has one correct class.

---

## 4. One-Hot Labels and Cross-Entropy

For one example:

$$
\ell = -\sum_{k=1}^{K} y_k \log \hat{y}_k
$$

Since one-hot labels contain only one $1$:

$$
\ell = -\log \hat{y}_c
$$

where $c$ is the correct class.

For a batch:

$$
\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K} y_k^{(i)} \log \hat{y}_k^{(i)}
$$

> [!INFO]
> One-hot labels make cross-entropy naturally select the correct class probability.

---

## 5. Sparse Labels vs One-Hot Labels

Instead of storing full vectors, many systems store only the class index:

$$
y^{(i)} = c
$$

Example:

$$
3
$$

This is called sparse labeling.

### Shapes

One-hot:

$$
Y \in \mathbb{R}^{n \times K}
$$

Sparse:

$$
Y_{sparse} \in \mathbb{R}^{n}
$$

> [!WARNING]
> Sparse labels are storage shortcuts—not ordinal values.

---

## 6. Output Layer Dimension

For $K$ classes:

* Output neurons: $K$
* One-hot label size: $K$

Final layer:

$$
W^{(L)} \in \mathbb{R}^{n_{L-1} \times K}
$$

$$
z^{(L)} \in \mathbb{R}^{1 \times K}
$$

Prediction:

$$
\arg\max_k \hat{y}_k
$$

---

## 7. Gradients with Softmax + Cross-Entropy

A major advantage:

$$
\frac{\partial \ell}{\partial z_k} = \hat{y}_k - y_k
$$

Batch form:

$$
\frac{\partial \mathcal{L}}{\partial Z} = \frac{1}{n}(\hat{Y}-Y)
$$

This elegant result is one reason one-hot labels are standard.

---

## 8. Input One-Hot vs Output One-Hot

| Aspect     | Input               | Output               |
| ---------- | ------------------- | -------------------- |
| Represents | feature category    | target class         |
| Used in    | preprocessing       | loss                 |
| Affects    | input dimension $d$ | output dimension $K$ |

---

## 9. Common Mistakes

##### Mistake 1: Treating categories as ordered numbers

Class IDs are labels, not magnitudes.

##### Mistake 2: Wrong output size

$K$ classes usually require $K$ outputs.

##### Mistake 3: One-hot encoding continuous data

Use one-hot only for categories.

##### Mistake 4: Ignoring unseen categories

Use unknown tokens or embeddings.

---

## 10. Embeddings in LLMs (Preview, Optional)

For very large $K$, one-hot vectors become inefficient.

Instead:

$$
e = xE
$$

where:

$$
x \in \mathbb{R}^{1 \times K}
$$

$$
E \in \mathbb{R}^{K \times d_{emb}}
$$

Embeddings learn dense category representations.
