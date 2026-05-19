# Why Inverted Dropout Divides by $1-p$

![](./img/whyinverteddropout.png)

---

## 1. Question

In our neural-network-from-scratch implementation, as well as in PyTorch and TensorFlow-style APIs, why does dropout use

```python
self.mask = np.random.binomial(1, 1 - self.p, size=input.shape) / (1 - self.p)
```

?

Here $p$ is the drop probability, so $1-p$ is the keep probability. We assume $0 \le p < 1$.

---

## 2. Answer: Keep the Expected Activation Magnitude Constant

Let $a$ be an activation before dropout. During training, dropout samples a binary keep mask

$$
m_i \sim \operatorname{Bernoulli}(1-p),
$$

so $m_i=1$ means activation $a_i$ is kept and $m_i=0$ means it is dropped.

If we only did

```python
mask = np.random.binomial(1, 1 - p, size=input.shape)
output = input * mask
```

then on average only a fraction $1-p$ of activations would be alive. The expected activation would shrink:

$$
\mathbb{E}[a_i m_i] = a_i\mathbb{E}[m_i] = a_i(1-p).
$$

To compensate, inverted dropout scales the kept activations by $1/(1-p)$ during training:

$$
\tilde a_i = a_i\frac{m_i}{1-p}.
$$

Then

$$
\mathbb{E}[\tilde a_i]
= \mathbb{E}\left[a_i\frac{m_i}{1-p}\right]
= a_i\frac{\mathbb{E}[m_i]}{1-p}
= a_i\frac{1-p}{1-p}
= a_i.
$$

That is why the implementation stores the scaled mask:

```python
self.mask = np.random.binomial(1, 1 - self.p, size=input.shape) / (1 - self.p)
output = input * self.mask
```

In math notation, the code variable `self.mask` is the scaled mask $m/(1-p)$, and the training-time output is

$$
\tilde a = a \odot \frac{m}{1-p}.
$$

---

## 3. Compared to the Original Dropout Implementation

With inverted dropout, scaling happens during training, so inference is a plain feed-forward pass with no masking and no scaling.

By contrast, the original or naïve dropout implementation works like this:

1. **Training**

   ```python
   mask = np.random.binomial(1, 1 - p, size=input.shape)
   output = input * mask
   ```

   We drop activations but do **not** rescale the surviving activations.

2. **Inference**

   ```python
   output = input * (1 - p)
   ```

   We multiply every activation by the keep probability $1-p$ to match the expected magnitude during training.

### 3.1 Key Differences

|                           | Original Dropout                         | Inverted Dropout                          |
|:-------------------------:|:----------------------------------------:|:-----------------------------------------:|
| **Train-time scaling**    | None                                     | Scale kept activations by $1/(1-p)$      |
| **Test-time scaling**     | Multiply activations by $1-p$            | None: pass activations through           |
| **Code simplicity at test** | Requires an extra scaling step         | Forward is a no-op in the dropout layer  |
| **Expected activation**   | Corrected at test time                   | Corrected during training                |

### 3.2 Why Inverted Dropout is Preferred

- **No special inference logic.** Once we scale by $1/(1-p)$ during training, we simply skip the mask in evaluation mode.
- **Consistent expected magnitudes.** Training activations have the same expectation as inference activations.
- **Cleaner code.** Only the dropout layer needs to branch on `training`; the rest of the model can remain unchanged.

In short, inverted dropout trades a little extra work during training for a simpler, faster, and more robust inference pass.

See the original article proposing Dropout:
- [https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)


---

## 4. Something wrong?

![](./img/dp.jpg)


<details>
<summary>
Details
</summary>

The important detail is that <b>p</b> in the original paper denotes the probability of a unit being <b>present</b>, while in our implementation, TensorFlow, and PyTorch, <b>p</b> denotes the probability of a unit being <b>dropped</b>.

![](./img/pd-1.jpg)
</details>
