# Dropout

---

## 1. Motivation: Why Regularization is Needed

> One good example of NN overfitting: [TensorFlow Playground](https://playground.tensorflow.org/#activation=relu&batchSize=10&dataset=gauss&regDataset=reg-plane&learningRate=0.03&regularizationRate=0.001&noise=50&networkShape=8,8,8,8,8,8&seed=0.67177&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=true&xSquared=true&ySquared=true&cosX=false&sinX=true&cosY=false&sinY=true&collectStats=false&problem=classification&initZero=false&hideText=false&activation_hide=false&percTrainData_hide=true&batchSize_hide=true&noise_hide=false&numHiddenLayers_hide=false&problem_hide=true&dataset_hide=false)

Neural networks often contain a large number of parameters. This makes them highly expressive but also prone to overfitting.

A model with high capacity can:

* memorize training samples
* fit noise in the data
* fail to generalize to unseen inputs

Dropout is a regularization technique designed to reduce this effect by preventing co-adaptation of neurons.

---

## 2. Core Idea of Dropout

![](img/dropoutgif.gif)

During training, we randomly deactivate a subset of activations.

For each activation coordinate $a_i$, sample a binary keep mask:

$$
m_i \sim \operatorname{Bernoulli}(1-p),
$$

where:

* $p$ is the drop probability
* $1-p$ is the keep probability
* $m_i = 0$ means activation $a_i$ is dropped
* $m_i = 1$ means activation $a_i$ is kept
* $\odot$ means element-wise multiplication

Without scaling, the masked activation would be:

$$
\tilde a = a \odot m.
$$

This simple version lowers the expected activation magnitude during training, because only a fraction $1-p$ of units are kept on average.

---

## 3. Inverted Dropout

In practice, we use **inverted dropout** to maintain a consistent expected activation scale.

Use the same binary keep mask:

$$
m_i \sim \operatorname{Bernoulli}(1-p).
$$

Then scale the kept activations by $1/(1-p)$ during training:

$$
\tilde a = a \odot \frac{m}{1-p}.
$$

For one coordinate:

$$
\tilde a_i = a_i\frac{m_i}{1-p}.
$$

### Key property

Because $\mathbb{E}[m_i]=1-p$,

$$
\mathbb{E}[\tilde a_i]
= \mathbb{E}\left[a_i\frac{m_i}{1-p}\right]
= a_i\frac{\mathbb{E}[m_i]}{1-p}
= a_i\frac{1-p}{1-p}
= a_i.
$$

This ensures that:

* training and inference operate on the same expected activation scale
* no rescaling is needed during inference

---

## 4. Intuition: Implicit Ensemble Learning

Dropout can be interpreted as training a large ensemble of sub-networks.

Each forward pass samples a different sub-network:

* different subsets of activations are active
* parameters are shared across all sub-networks

This leads to:

* model averaging effect
* improved robustness
* reduced overfitting

---

## 5. Effect on Representation Learning

Without dropout, neurons can develop strong dependencies.

With dropout:

* neurons must work independently
* redundant representations are learned
* features become more robust

This reduces co-adaptation and improves generalization.

---

## 6. Dropout in Neural Networks

A typical layer activation is:

$$
z = xW + b,
\qquad
a = \sigma(z).
$$

During training, with $m_i \sim \operatorname{Bernoulli}(1-p)$:

$$
\tilde a = a \odot \frac{m}{1-p}.
$$

During inference:

$$
\tilde a = a.
$$

No dropout mask and no extra scaling are applied at inference time.

---

## 7. PyTorch Implementation

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(256, 10)
)
```

Here:

* $p = 0.5$ means 50% drop probability
* $1-p = 0.5$ means 50% keep probability
* dropout is active only during training mode

---

## 8. Geometric Interpretation

Dropout injects noise into hidden representations:

* training becomes stochastic
* decision boundaries are smoothed
* sharp fitting is discouraged

From a function perspective:

* the model learns a family of functions
* robustness is enforced across perturbations of hidden units
