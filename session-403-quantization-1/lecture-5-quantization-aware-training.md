# Quantization-Aware Training

---

## 1. The Problem with Naive Quantization

Quantization introduces approximation error:

$$
x \rightarrow \tilde{x}
$$

If the model was trained only in FP32, it may rely on small numerical differences that disappear after quantization.

This can cause:

- lower accuracy;
- unstable outputs;
- sensitivity to outliers;
- large errors in some layers.

---

## 2. Core Idea of QAT

Quantization-aware training, or QAT, simulates quantization during training.

```text
forward pass: pretend quantization happens
backward pass: update FP32 shadow weights
```

The model experiences quantization noise during training and can adapt to it.

---

## 3. Fake Quantization

During QAT, frameworks often use fake-quantization operations.

A fake-quant operation does:

```text
FP32 value → quantize to integer grid → dequantize back to FP32
```

So the tensor is still stored as FP32 during training, but its values are restricted to quantized levels.

Mathematically:

$$
\tilde{x} = s\left(\text{round}\left(\frac{x}{s}\right) + z - z\right)
$$

This lets the model learn under quantization-like noise.

---

## 4. Why Backpropagation Needs a Trick

The rounding function is not differentiable in the usual useful way.

QAT commonly uses a straight-through estimator:

```text
forward: use rounded value
backward: pretend rounding was identity in a reasonable range
```

This is not exact calculus. It is a practical engineering approximation.

---

## 5. PTQ vs QAT

| Method | Needs Training Data? | Needs Retraining? | Typical Use |
|---|---:|---:|---|
| PTQ | maybe calibration only | no | first deployment attempt |
| QAT | yes | yes | when PTQ accuracy loss is too high |

QAT can preserve accuracy better, but it costs more engineering effort.

---

## 6. When to Use QAT

Consider QAT when:

- PTQ causes unacceptable quality loss;
- the model is very sensitive to numerical precision;
- deployment constraints require aggressive quantization;
- calibration data is not enough;
- you control the training pipeline.

Avoid QAT as the first step if simple PTQ already meets the target.

---

## 7. Key Takeaways

- QAT trains the model while simulating quantization.
- Fake quantization makes quantization error visible during training.
- QAT can improve low-precision quality.
- QAT is more expensive than PTQ and should be justified by measurements.
