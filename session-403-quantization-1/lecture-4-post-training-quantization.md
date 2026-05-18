# Post-Training Quantization

Previously, we inspected an ONNX model graph. Now we ask:

> Can we quantize a model after it has already been trained?

The answer is yes. This is called post-training quantization.

---

## 1. Core Idea

Post-training quantization, often abbreviated PTQ, starts with a trained model and converts some tensors or operators to lower precision.

```text
trained FP32 model → quantization procedure → lower-precision model
```

The model weights are already learned. We do not redo full training.

---

## 2. Why PTQ Is Popular

PTQ is attractive because it is practical:

- no training pipeline needed;
- fast to try;
- useful when only a trained model is available;
- often good enough for deployment;
- easy to compare against the original model.

PTQ is usually the first quantization method to try.

---

## 3. Dynamic Quantization

Dynamic quantization commonly quantizes weights ahead of time and computes activation scales dynamically during inference.

Conceptually:

```text
weights: quantized before inference
activations: quantized based on runtime values
```

This is often simple and effective for models with large matrix multiplications.

Advantages:

- easy to apply;
- no calibration dataset required;
- can reduce model size;
- often works well for linear layers.

Limitations:

- activations may still require runtime overhead;
- speedup depends on hardware and runtime support;
- not every operator benefits equally.

---

## 4. Static Quantization

Static quantization determines activation ranges before deployment using calibration data.

Workflow:

```text
FP32 model
→ run representative calibration samples
→ collect activation ranges
→ choose scales and zero points
→ produce quantized model
```

Calibration data should resemble real inference data. Bad calibration can cause bad quantization.

---

## 5. Calibration

Calibration estimates ranges such as:

$$
x_{min}, x_{max}
$$

Then those ranges determine scale and zero point.

A simple asymmetric scale might be:

$$
s = \frac{x_{max} - x_{min}}{q_{max} - q_{min}}
$$

Calibration is a measurement step:

> What numerical ranges does this model actually use on real inputs?

---

## 6. Accuracy Trade-Off

After PTQ, always evaluate the model.

Compare:

- file size;
- inference latency;
- memory usage;
- accuracy or task metric;
- per-class errors;
- robustness on unusual inputs.

A useful table:

| Model | Size | Accuracy | Notes |
|---|---:|---:|---|
| FP32 | large | baseline | original model |
| INT8 PTQ | smaller | maybe slightly lower | deployment candidate |

---

## 7. Key Takeaways

- PTQ quantizes a model after training.
- Dynamic quantization is easy and often a good first attempt.
- Static quantization uses calibration data to choose activation ranges.
- Quantization must be measured, not assumed safe.
