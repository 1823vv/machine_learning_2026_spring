# Advanced Quantization Topics and Deployment Trade-Offs

Previously, we connected quantization to LLMs. Now we organize advanced topics into practical decision rules.

---

## 1. Quantization Is a System Design Choice

A quantized model is useful only if the full system benefits.

Measure:

- model size;
- peak memory;
- latency for one request;
- throughput under load;
- warm-up time;
- quality metric;
- hardware compatibility;
- engineering complexity.

A smaller model is not automatically faster. Runtime kernels and hardware support matter.

---

## 2. Operator Support

Quantization works best when the inference runtime has optimized kernels for the quantized operators.

If a graph repeatedly does:

```text
quantize → dequantize → float operation
```

then it may not get much faster.

Good deployment asks:

- Does the runtime support INT8 matrix multiplication?
- Does the hardware accelerate this precision?
- Are operators fused?
- Are layout conversions expensive?
- Does quantization reduce memory bandwidth enough to matter?

---

## 3. Layer Sensitivity

Not all layers tolerate quantization equally.

Sensitive components may include:

- first and last layers;
- normalization layers;
- attention score computation;
- layers with large outliers;
- small matrices where overhead dominates benefits.

A common strategy is mixed precision:

```text
most layers: low precision
sensitive layers: higher precision
```

---

## 4. Quantization Granularity

| Granularity | Description | Trade-Off |
|---|---|---|
| Per-tensor | one scale for entire tensor | simple but less precise |
| Per-channel | one scale per channel | better quality, more metadata |
| Per-group | one scale per group | strong balance for large matrices |
| Per-token | activation scale changes by token | adaptive but more runtime work |

Granularity is one of the most important knobs in practical quantization.

---

## 5. Weight, Activation, and Cache Choices

Different deployment goals suggest different strategies:

| Goal | Common First Attempt |
|---|---|
| Reduce model file size | weight-only quantization |
| Reduce CPU inference cost | INT8 dynamic or static quantization |
| Fit LLM on limited GPU memory | 8-bit or 4-bit weight quantization |
| Serve long contexts | KV cache quantization or attention memory optimization |
| Preserve maximum quality | mixed precision with sensitive layers kept high precision |

---

## 6. A Practical Decision Workflow

Use this workflow:

1. Establish FP32/FP16 baseline quality and latency.
2. Apply the simplest quantization method available.
3. Measure quality, size, and speed.
4. Inspect the graph or runtime profile.
5. Identify layers or operators that fail to quantize well.
6. Increase granularity or use mixed precision.
7. Consider QAT only if PTQ cannot meet the target.
8. Re-evaluate on real deployment data.

---

## 7. Common Failure Modes

- Calibration data is not representative.
- Outliers dominate the scale.
- Accuracy is measured on too small a test set.
- Runtime silently falls back to floating-point kernels.
- Model gets smaller but latency does not improve.
- Only average quality is checked, hiding rare severe failures.
- Quantized model is tested on easy examples but deployed on harder inputs.

---

## 8. Final Synthesis

Quantization is a compression and deployment technique, but it is also a measurement discipline.

The central question is not:

> Can this model be quantized?

The better question is:

> Which parts can be quantized, to which precision, on which hardware, with what quality loss, and under what workload?

---

## 9. Key Takeaways

- Quantization quality depends on model, data, runtime, and hardware.
- Mixed precision is often more practical than forcing every operation into the same format.
- Graph inspection and benchmarking are essential.
- The best quantization strategy is the simplest one that meets the deployment target.
