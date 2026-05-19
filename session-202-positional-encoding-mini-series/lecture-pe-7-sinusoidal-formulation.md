# Positional Encoding: Standard Sinusoidal Formula

![](./img/positionencodingfigure.jpg)

---
## 1. Definition with 0-Based Indexing

For position index $i\in\{0,1,\dots,n-1\}$ and channel pair index $k\in\{0,1,\dots,d_{\text{model}}/2-1\}$,

$$
PE_{i,2k}=\sin\left(\frac{i}{10000^{2k/d_{\text{model}}}}\right)
$$

$$
PE_{i,2k+1}=\cos\left(\frac{i}{10000^{2k/d_{\text{model}}}}\right)
$$

---
## 2. Frequency Schedule Intuition

The denominator creates geometrically spaced frequencies across channels.

### Consequence

- some channels vary quickly
- some channels vary slowly
- together they encode both local and global position structure

---
## 3. Matrix Form

$$
PE\in\mathbb{R}^{n\times d_{\text{model}}}, \quad \tilde{X}=X+PE
$$

Rows correspond to positions and columns correspond to frequency channels.

---
## 4. Why This Specific Design Works Well

- smooth positional variation
- no extra learned parameters for position
- extrapolation to sequence lengths beyond training range
