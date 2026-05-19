# Dropout

---

## 1. Why We Need It

> [!INFO]
> One good example of NN overfitting: [TensorFlow Playground](https://playground.tensorflow.org/#activation=relu&batchSize=10&dataset=gauss&regDataset=reg-plane&learningRate=0.03&regularizationRate=0.001&noise=50&networkShape=8,8,8,8,8,8&seed=0.67177&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=true&xSquared=true&ySquared=true&cosX=false&sinX=true&cosY=false&sinY=true&collectStats=false&problem=classification&initZero=false&hideText=false&activation_hide=false&percTrainData_hide=true&batchSize_hide=true&noise_hide=false&numHiddenLayers_hide=false&problem_hide=true&dataset_hide=false)



Neural networks have high capacity and easily overfit by memorizing training data or noise.

This leads to:

* weak generalization
* overly fragile feature dependencies
* co-adaptation between neurons

Dropout is a simple way to regularize training by injecting randomness into hidden activations.

---

## 2. Core Mechanism

![](img/dropoutgif.gif)


During training, each neuron is randomly kept or removed.

$$
m_i \sim \operatorname{Bernoulli}(1 - p)
$$

$$
\tilde a = a \odot m
$$

Each forward pass effectively uses a different sub-network.

---

## 3. Scaling Issue

If we directly use:

$$
\tilde a = a \odot m
$$

then the expected activation becomes smaller:

only $(1 - p)$ of units are active on average.

This creates a mismatch between training and inference.

---

## 4. Inverted Dropout (Standard Practice)

To fix the scale mismatch, we rescale during training:

$$
\tilde a = a \odot \frac{m}{1 - p}
$$

so that:

$$
\mathbb{E}[\tilde a_i] = a_i
$$

This keeps activation scale consistent, and inference becomes identity:

$$
\tilde a = a
$$

---

## 5. Where It Happens

Given a standard layer:

$$
z = xW + b, \quad a = \sigma(z)
$$

Dropout is applied on activations:

$$
\tilde a = a \odot \frac{m}{1 - p}
$$

At inference time:

$$
\tilde a = a
$$

No masking is used.

---

## 6. What It Actually Does

Each training step samples a different sub-network.

So dropout behaves like:

* training an implicit ensemble of networks
* sharing parameters across all sub-networks

This improves robustness and reduces overfitting.

---

## 7. Effect on Features

Dropout forces representations to be independent:

* neurons cannot rely on fixed partners
* redundant features are encouraged
* co-adaptation is reduced

---

## 8. Implementation

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(256, 10)
)
```

Dropout is active only during training mode.
