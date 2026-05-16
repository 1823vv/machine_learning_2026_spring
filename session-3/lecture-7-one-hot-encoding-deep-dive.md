# One-Hot Encoding Deep Dive

## 1. Why One-Hot Encoding Matters

Machine learning models work with numbers. However, many important variables are categories:

- a city name
- a company sector
- a product type
- a digit class
- a language label
- a class name such as cat, dog, or bird

One-hot encoding is a simple way to represent a category as a vector.

If there are $K$ possible categories, one-hot encoding represents each category by a vector in $\mathbb{R}^{1 \times K}$ with exactly one entry equal to $1$ and all other entries equal to $0$.

For example, if the categories are three locations, then each location can be represented by one vector of length $3$.

> [!INFO]
> The main idea is simple: one category becomes one active position in a vector.

## 2. Row-Vector Convention

We follow the row-vector convention used throughout the main machine-learning material.

A single input example is a row vector:

$$
x \in \mathbb{R}^{1 \times d}
$$

A batch of $n$ examples is a matrix whose rows are examples:

$$
X \in \mathbb{R}^{n \times d}
$$

For a neural-network layer, the row-vector form is:

$$
z^{(l)} = a^{(l-1)} W^{(l)} + b^{(l)}
$$

where

$$
a^{(l-1)} \in \mathbb{R}^{1 \times n_{l-1}}
$$

$$
W^{(l)} \in \mathbb{R}^{n_{l-1} \times n_l}
$$

$$
b^{(l)} \in \mathbb{R}^{1 \times n_l}
$$

$$
z^{(l)} \in \mathbb{R}^{1 \times n_l}
$$

One-hot vectors also follow this row-vector convention.

## 3. One-Hot Encoding for Input Features

### 3.1 Categorical Input Features

Suppose an input dataset contains a categorical feature called `State`, with possible values:

- `New York`
- `California`
- `Florida`

A model cannot directly multiply the text value `New York` by a weight matrix. We first convert the category into numerical features.

One possible one-hot encoding is:

| Category | State_New_York | State_California | State_Florida |
|---|---:|---:|---:|
| New York | 1 | 0 | 0 |
| California | 0 | 1 | 0 |
| Florida | 0 | 0 | 1 |

After this conversion, the category becomes part of the input vector $x$.

### 3.2 Example Input Vector

Suppose the original numerical features are:

- research and development spending
- administration spending
- marketing spending

and the categorical feature is `State`.

After one-hot encoding, one row may look like:

| R_and_D | Administration | Marketing | State_New_York | State_California | State_Florida |
|---:|---:|---:|---:|---:|---:|
| 165349.20 | 136897.80 | 471784.10 | 1 | 0 | 0 |

The input dimension increases because one categorical column becomes several indicator columns.

If there are $d_{num}$ numerical features and $K$ state categories, then the new input dimension is often

$$
d = d_{num} + K
$$

### 3.3 Why One-Hot Instead of Integer Labels for Inputs

A bad idea is to encode the states as integers such as:

| Category | Integer Encoding |
|---|---:|
| New York | 0 |
| California | 1 |
| Florida | 2 |

This can mislead a model because it suggests an artificial order:

$$
0 < 1 < 2
$$

It can also suggest artificial distances:

$$
2 - 0 = 2
$$

For a category like state or color, these distances usually do not mean anything.

> [!WARNING]
> Integer encoding is dangerous for nominal categories when the numbers do not represent a real order or magnitude.

### 3.4 The Dummy Variable Trap

For linear models, one-hot encoding all $K$ categories together with an intercept can create redundant columns.

For the three state columns above, every row satisfies:

$$
State\_New\_York + State\_California + State\_Florida = 1
$$

If the model also has a bias term, one of the one-hot columns can often be dropped without losing information.

So a linear model may use $K-1$ columns instead of $K$ columns.

> [!INFO]
> In many machine-learning libraries, this is handled by an option such as dropping one category. In neural networks, keeping all $K$ columns is also common, especially when the model has no need for strict linear-model interpretability.

## 4. One-Hot Encoding for Output Labels

### 4.1 Multiclass Classification Labels

For multiclass classification, the target label is one of $K$ possible classes.

For digit recognition, there are $K=10$ classes:

$$
0,1,2,3,4,5,6,7,8,9
$$

The output layer often produces $K$ logits:

$$
z^{(L)} \in \mathbb{R}^{1 \times K}
$$

Softmax converts logits into probabilities:

$$
\hat{y}_k = \frac{e^{z_k}}{\sum_{j=1}^{K}e^{z_j}}
$$

The target class can be represented as a one-hot row vector:

$$
y \in \mathbb{R}^{1 \times K}
$$

If the true digit is $3$, the one-hot target has a $1$ in the position for digit $3$ and $0$ everywhere else.

### 4.2 Batch Output Labels

For a batch of $n$ examples, the one-hot label matrix is:

$$
Y \in \mathbb{R}^{n \times K}
$$

The predicted probability matrix is:

$$
\hat{Y} \in \mathbb{R}^{n \times K}
$$

Each row of $Y$ has exactly one $1$ for a standard single-label multiclass problem.

Each row of $\hat{Y}$ sums to $1$ after softmax:

$$
\sum_{k=1}^{K}\hat{y}^{(i)}_k = 1
$$

## 5. One-Hot Labels and Cross-Entropy

For one example, multiclass cross-entropy with one-hot labels is:

$$
\ell = -\sum_{k=1}^{K} y_k \log \hat{y}_k
$$

Because one-hot $y$ has exactly one $1$, only the correct class contributes to the sum.

If the correct class is $c$, then:

$$
y_c = 1
$$

and

$$
y_k = 0 \quad \text{for} \quad k \neq c
$$

Therefore:

$$
\ell = -\log \hat{y}_c
$$

For a batch of $n$ examples:

$$
\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K} y^{(i)}_k \log \hat{y}^{(i)}_k
$$

> [!INFO]
> One-hot labels make the cross-entropy formula look like a sum over all classes, but mathematically it selects the probability assigned to the correct class.

## 6. Sparse Labels versus One-Hot Labels

### 6.1 Sparse Integer Labels

Instead of storing the full one-hot vector, many implementations store only the correct class index.

For example, a digit label may be stored as:

$$
y^{(i)} = 3
$$

This is called a sparse label representation.

### 6.2 Same Meaning, Different Storage

Sparse labels and one-hot labels can represent the same target.

If the correct class is $c$, then the one-hot version satisfies:

$$
y_c = 1
$$

and

$$
y_k = 0 \quad \text{for} \quad k \neq c
$$

The sparse version stores only $c$.

The cross-entropy can be computed as:

$$
\ell = -\log \hat{y}_c
$$

This avoids explicitly building a full one-hot vector.

### 6.3 Practical Difference

One-hot labels have shape:

$$
Y \in \mathbb{R}^{n \times K}
$$

Sparse labels have shape:

$$
Y_{sparse} \in \mathbb{R}^{n \times 1}
$$

or sometimes simply a one-dimensional array of length $n$ in code.

> [!WARNING]
> Sparse labels are not the same as treating class numbers as ordered numerical values. Sparse labels are just a compact way to identify the correct class for a classification loss.

## 7. One-Hot Encoding and the Output Layer Dimension

For multiclass classification with $K$ classes, the output layer usually has $K$ neurons.

The final affine transformation is:

$$
z^{(L)} = a^{(L-1)} W^{(L)} + b^{(L)}
$$

where

$$
W^{(L)} \in \mathbb{R}^{n_{L-1} \times K}
$$

$$
b^{(L)} \in \mathbb{R}^{1 \times K}
$$

$$
z^{(L)} \in \mathbb{R}^{1 \times K}
$$

The one-hot target also has $K$ entries:

$$
y \in \mathbb{R}^{1 \times K}
$$

So the output dimension and the one-hot label dimension match.

### 7.1 MNIST Example

For MNIST digit recognition:

$$
K = 10
$$

The logits have shape:

$$
z^{(L)} \in \mathbb{R}^{1 \times 10}
$$

The softmax probabilities have shape:

$$
\hat{y} \in \mathbb{R}^{1 \times 10}
$$

The one-hot label has shape:

$$
y \in \mathbb{R}^{1 \times 10}
$$

The predicted class is often obtained by:

$$
\arg\max_k \hat{y}_k
$$

## 8. One-Hot Encoding and Gradients

For softmax output with cross-entropy loss, the gradient with respect to logits has a simple form:

$$
\frac{\partial \ell}{\partial z_k} = \hat{y}_k - y_k
$$

For a batch:

$$
\frac{\partial \mathcal{L}}{\partial Z} = \frac{1}{n}\left(\hat{Y}-Y\right)
$$

This formula is one reason one-hot labels are so natural for multiclass classification.

For the correct class $c$, the one-hot entry is $1$, so the gradient is:

$$
\hat{y}_c - 1
$$

For an incorrect class $k$, the one-hot entry is $0$, so the gradient is:

$$
\hat{y}_k
$$

This pushes the correct logit up and the incorrect logits down during learning.

## 9. Input One-Hot versus Output One-Hot

Input one-hot encoding and output one-hot labels look similar, but they play different roles.

| Aspect | Input One-Hot | Output One-Hot |
|---|---|---|
| Represents | categorical feature | target class |
| Location | input vector $x$ | label vector $y$ |
| Used by | first layer or preprocessing | loss function |
| Example | state category | digit class |
| Learned? | no | no |
| Affects | input dimension $d$ | output dimension $K$ |

### 9.1 Input One-Hot

Input one-hot encoding becomes part of the feature vector:

$$
x \in \mathbb{R}^{1 \times d}
$$

It affects the first layer weight shape:

$$
W^{(1)} \in \mathbb{R}^{d \times n_1}
$$

### 9.2 Output One-Hot

Output one-hot labels define the target distribution:

$$
y \in \mathbb{R}^{1 \times K}
$$

They affect the output layer shape:

$$
W^{(L)} \in \mathbb{R}^{n_{L-1} \times K}
$$

and the cross-entropy loss.

## 10. Common Mistakes

### 10.1 Mistake: Treating Class IDs as Continuous Numbers

For multiclass classification, class IDs such as $0,1,2,3$ are names, not continuous values.

It is wrong to train a standard multiclass classifier as if digit $9$ is numerically closer to digit $8$ than to digit $0$.

### 10.2 Mistake: Mismatched Output Dimension

If there are $K$ classes, the softmax output should have $K$ probabilities.

A model with $K$ classes should not output only one number unless the task is binary classification with a sigmoid output.

### 10.3 Mistake: Applying One-Hot Encoding to Continuous Variables

One-hot encoding is for categories. A continuous feature such as salary, temperature, or pixel intensity should usually not be one-hot encoded directly.

### 10.4 Mistake: Forgetting Unknown Categories

A model may see a category at prediction time that did not appear during training.

Possible strategies include:

- use an `unknown` category
- reject the example and ask for data cleaning
- use a different representation such as embeddings

## 11. Optional: One-Hot Encoding and Embeddings

For high-cardinality categorical variables, one-hot vectors can become very large.

If there are $K=100000$ categories, then one one-hot vector has $100000$ entries.

An embedding layer learns a dense vector representation for each category.

One-hot representation can be seen as selecting one row from an embedding matrix.

If the one-hot vector is $x$ and the embedding matrix is $E$, then:

$$
e = xE
$$

where

$$
x \in \mathbb{R}^{1 \times K}
$$

$$
E \in \mathbb{R}^{K \times d_{emb}}
$$

$$
e \in \mathbb{R}^{1 \times d_{emb}}
$$

This is optional for now, but it is an important idea in deep learning for text and recommendation systems.

## 12. Label Smoothing (Optional)

Standard one-hot labels put probability $1$ on the correct class and $0$ on every incorrect class.

Label smoothing changes the target distribution so that the correct class is slightly less than $1$, and incorrect classes receive small positive values.

For $K$ classes and smoothing strength $\epsilon$, one common form is:

$$
y^{smooth}_c = 1 - \epsilon
$$

and for incorrect classes:

$$
y^{smooth}_k = \frac{\epsilon}{K-1}
$$

Label smoothing can reduce overconfidence, but it changes the target distribution and should be used carefully.

## 13. Summary

One-hot encoding is used in two major places:

1. For input categorical features, it converts categories into numerical feature columns.
2. For output labels, it represents the correct class as a target distribution for softmax cross-entropy.

Important shapes:

$$
x \in \mathbb{R}^{1 \times d}
$$

$$
X \in \mathbb{R}^{n \times d}
$$

$$
y \in \mathbb{R}^{1 \times K}
$$

$$
Y \in \mathbb{R}^{n \times K}
$$

$$
\hat{Y} \in \mathbb{R}^{n \times K}
$$

Important formula:

$$
\ell = -\sum_{k=1}^{K} y_k\log\hat{y}_k
$$

For one-hot labels, this becomes:

$$
\ell = -\log\hat{y}_c
$$
