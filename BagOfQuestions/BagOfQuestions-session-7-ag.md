## Question: Dropout Fill-in-the-Blank

Fill in the `_____YOUR_TEXT_HERE_N_____` blanks in the following text.

In our own NumPy neural-network implementation, dropout uses inverted dropout during training:

```python
self.mask = np.random.binomial(1, 1 - p, size=input.shape) / _____YOUR_TEXT_HERE_1_____
output = input * self.mask
```

Let $p$ be the drop probability, so $1-p$ is the keep probability. Mathematically, for an activation coordinate $a_i$ and binary keep mask $m_i$,

$$
m_i \sim \operatorname{Bernoulli}(1-p),
\qquad
\tilde a_i = a_i\frac{m_i}{1-p}.
$$

If we only did

```python
mask = np.random.binomial(1, 1 - p, size=input.shape)
output = input * mask
```

then on average only a fraction `_____YOUR_TEXT_HERE_2_____` of activations would be alive, so the expected activation magnitude would shrink by a factor of `_____YOUR_TEXT_HERE_3_____`. To compensate, inverted dropout scales up the surviving activations by `_____YOUR_TEXT_HERE_4_____` during training.

Because the scaling is already done during training, we do not need special scaling during inference. When we set `self.training = _____YOUR_TEXT_HERE_5_____`, the dropout layer simply returns the raw input activations.


Your answer goes here, after the `:`:

- `_____YOUR_TEXT_HERE_1_____`:
- `_____YOUR_TEXT_HERE_2_____`:
- `_____YOUR_TEXT_HERE_3_____`:
- `_____YOUR_TEXT_HERE_4_____`:
- `_____YOUR_TEXT_HERE_5_____`:
