# Why Quantization?

> [!INFO]
> Motivation:
> - Check out https://ollama.com/library/gemma3  

Quantization is the practice of representing model computations with fewer bits. The goal is simple: make models cheaper to store, move, and run while preserving as much quality as possible.

---

## 1. The Core Problem

Neural networks are usually trained with floating-point numbers such as FP32.

A model with many parameters consumes memory roughly proportional to:

$$
\text{model size} \approx \text{number of parameters} \times \text{bytes per parameter}
$$

Common storage costs:

| Format | Bits | Bytes per value |
|---|---:|---:|
| FP32 | 32 | 4 |
| FP16/BF16 | 16 | 2 |
| INT8 | 8 | 1 |
| INT4 | 4 | 0.5 |

So if a model has $N$ parameters:

$$
\text{FP32 size} = 4N \text{ bytes}
$$

$$
\text{INT8 size} = 1N \text{ bytes}
$$

This gives a theoretical 4× reduction in weight storage.

---

## 2. Why This Matters

Quantization matters because modern models are constrained by:

- **memory capacity**: can the model fit on the device?
- **memory bandwidth**: how fast can weights be moved to the compute units?
- **latency**: how quickly can one request be answered?
- **throughput**: how many requests can be served per second?
- **energy**: how much power is used per inference?
- **cost**: how much hardware is needed?

For many deployments, moving data is more expensive than doing arithmetic. Smaller weights can mean faster inference even when the mathematical operation is similar.

---

## 3. A First Intuition

Imagine measuring temperature.

A precise thermometer may report:

```text
21.382716 °C
```

But for many decisions, this is enough:

```text
21 °C
```

Quantization asks a similar question:

> How much numerical precision does the model actually need to make a good prediction?

If the model can tolerate approximate numbers, we can save memory and computation.

---

## 4. Quantization as Approximation

A real number $x$ is replaced by a nearby representable value $\tilde{x}$:

$$
x \approx \tilde{x}
$$

The difference is quantization error:

$$
\epsilon = x - \tilde{x}
$$

The central tension is:

```text
fewer bits → smaller/faster model → more approximation error
```

Good quantization methods reduce memory and latency while keeping $\epsilon$ small enough that predictions remain useful.

---

## 5. What Can Be Quantized?

Different parts of a model can be quantized:

- **weights**: learned parameters;
- **activations**: intermediate tensor values;
- **inputs/outputs**: data entering or leaving operators;
- **KV cache** in transformers: stored attention keys and values during generation;
- **optimizer states** during training, in some memory-saving training methods.

In this mini-series, the first concrete workflow focuses on weights and activations in an ONNX MNIST classifier. Later, we connect the same ideas to LLMs.

---

## 6. Key Takeaways

- Quantization reduces the number of bits used to represent values.
- The main benefits are smaller memory footprint, lower bandwidth pressure, and potentially faster inference.
- The main risk is quality loss from approximation error.
- Quantization is a deployment trade-off, not only a mathematical trick.
