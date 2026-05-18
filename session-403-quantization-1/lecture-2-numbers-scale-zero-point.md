# Numbers, Scale, and Zero Point

Previously, we saw why quantization matters. Now we ask the central mechanism question:

> How can real-valued tensors be represented using integers?

---

## 1. Floating Point vs Integer

A trained neural network usually contains values like:

```text
-0.137, 0.004, 1.276, 3.91
```

An INT8 tensor can only store integer values in a limited range. For signed INT8:

$$
q \in [-128, 127]
$$

For unsigned UINT8:

$$
q \in [0, 255]
$$

So quantization needs a mapping between real numbers and integers.

---

## 2. Affine Quantization

The most common formula is affine quantization:

$$
\boxed{x \approx s(q - z)}
$$

where:

- $x$ is the real value;
- $q$ is the stored integer value;
- $s$ is the **scale**;
- $z$ is the **zero point**.

To quantize a real number:

$$
\boxed{q = \text{round}\left(\frac{x}{s}\right) + z}
$$

Then we clip $q$ so it stays inside the integer range.

---

## 3. What the Scale Means

The scale says how much real-world value one integer step represents.

If:

$$
s = 0.1
$$

then adjacent integer values differ by 0.1 in real space.

Small scale:

- more precision;
- smaller representable range.

Large scale:

- larger representable range;
- less precision.

This is one of the core trade-offs in quantization.

---

## 4. What the Zero Point Means

The zero point is the integer value that represents real zero.

We want:

$$
x = 0 \Rightarrow q = z
$$

This matters because many neural-network operations produce or depend on exact zeros, especially padding and ReLU outputs.

If zero is exactly representable, the quantized model is easier to implement correctly.

---

## 5. Symmetric vs Asymmetric Quantization

### Symmetric Quantization

Symmetric quantization centers the integer range around zero:

$$
z = 0
$$

This is common for weights.

Advantages:

- simpler math;
- efficient matrix multiplication;
- good when values are roughly centered around zero.

### Asymmetric Quantization

Asymmetric quantization allows nonzero zero points:

$$
z \neq 0
$$

This is common for activations, especially when values are not centered around zero.

Advantages:

- better use of integer range for skewed distributions;
- useful after ReLU, where many values are nonnegative.

---

## 6. Per-Tensor vs Per-Channel Quantization

### Per-Tensor

One scale and zero point for the whole tensor:

```text
entire weight matrix → one scale
```

Simple, but a single outlier can force a large scale.

### Per-Channel

Different channels get different scales:

```text
each output channel → its own scale
```

This is often better for neural-network weights because different channels may have different value ranges.

---

## 7. Worked Example

Suppose values are in the range:

$$
x \in [-1.0, 1.0]
$$

Using signed INT8 gives 256 possible integer values. A simple symmetric scale is approximately:

$$
s = \frac{1.0}{127}
$$

A value $x = 0.5$ becomes:

$$
q = \text{round}\left(\frac{0.5}{1/127}\right) = 64
$$

Dequantizing:

$$
\tilde{x} = \frac{64}{127} \approx 0.504
$$

The representation is not exact, but it is close.

---

## 8. Key Takeaways

- Quantization maps real values to integers using scale and zero point.
- Scale controls the precision/range trade-off.
- Zero point lets real zero be represented exactly.
- Symmetric quantization is common for weights.
- Asymmetric quantization is common for activations.
- Per-channel quantization often preserves quality better than per-tensor quantization.
