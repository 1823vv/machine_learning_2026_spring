# Quantization for Large Language Models

Previously, we studied quantization with a small MNIST classifier. Now we connect the same principle to large language models.

---

## 1. Why LLM Quantization Matters

LLMs can contain billions of parameters. Weight memory becomes enormous.

Approximate memory for parameters:

$$
\text{memory} = \text{parameters} \times \text{bytes per parameter}
$$

For a 7-billion-parameter model:

| Format | Approximate Weight Memory |
|---|---:|
| FP32 | 28 GB |
| FP16/BF16（Brain Floating Point） | 14 GB |
| INT8 | 7 GB |
| INT4 | 3.5 GB |

This simple calculation explains why low-bit LLM quantization is so important.

---

## 2. What Is Different from MNIST?

The basic idea is the same:

```text
represent values with fewer bits
```

But LLMs are harder because:

- they are much larger;
- quality is harder to measure than classification accuracy;
- activations can contain extreme outliers;
- attention uses a growing KV cache during generation;
- small numerical errors can affect long generated sequences;
- different layers may have different sensitivity.

---

## 3. Weight-Only Quantization

Many LLM deployment methods begin with weight-only quantization.

```text
weights: low precision
activations: higher precision during computation
```

This reduces model memory while avoiding some of the difficulty of fully quantizing activations.

Common idea:

- store weights in INT8, INT4, or another compact format;
- dequantize or use specialized kernels during matrix multiplication;
- keep sensitive operations in higher precision.

---

## 4. Activation Outliers

LLM activations may have outlier channels with much larger magnitudes than the rest.

A single large outlier can force a large scale:

```text
one huge value → large quantization range → poor precision for normal values
```

This motivates techniques such as:

- per-channel scaling;
- group-wise quantization;
- keeping some channels in higher precision;
- smoothing or redistributing activation ranges;
- special handling for sensitive layers.

---

## 5. Group-Wise Quantization

Instead of one scale for a whole tensor, group-wise quantization uses one scale per group of values.

```text
weight matrix → groups → one scale per group
```

Smaller groups usually mean:

- better accuracy;
- more scale metadata;
- more complex kernels.

This is a recurring deployment trade-off.

---

## 6. KV Cache Quantization

During autoregressive generation, transformers store keys and values for previous tokens.

This storage is called the KV cache.

For long contexts, KV cache memory can become a major bottleneck:

```text
more layers × more heads × longer sequence × hidden dimension
```

Quantizing the KV cache can reduce memory, but it may affect generation quality because future tokens attend to these stored values.

---

## 7. Quality Evaluation Is Harder

For MNIST, evaluation is simple:

```text
accuracy on test set
```

For LLMs, evaluation may include:

- perplexity;
- benchmark scores;
- instruction-following quality;
- coding ability;
- reasoning tasks;
- safety behavior;
- long-context performance;
- human preference evaluation.

A quantized LLM should be tested on the tasks that matter for its deployment.

---

## 8. Key Takeaways

- LLM quantization is motivated mainly by memory and bandwidth.
- Weight-only quantization is often a practical first step.
- Activation outliers make LLM quantization difficult.
- Group-wise and per-channel methods improve quality at the cost of metadata and complexity.
- KV cache quantization matters for long-context generation.
