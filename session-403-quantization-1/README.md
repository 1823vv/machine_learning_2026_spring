# Quantization Mini-Series: From MNIST ONNX to LLM Compression

This mini-series builds quantization from first principles to modern large-language-model deployment. The running artifact is a small MNIST classifier exported to ONNX, inspected in Netron, and then quantized so learners can see what changes in the graph, weights, activations, file size, and accuracy.

---

## Learning Promise

By the end of this mini-series, you should be able to:

1. explain why quantization is needed for efficient inference;
2. describe the difference between floating-point and integer representations;
3. read an ONNX model graph in Netron;
4. compare FP32 and INT8 models in terms of size, accuracy, and operators;
5. explain post-training quantization and quantization-aware training at a high level;
6. discuss why LLM quantization is harder than small CNN/MLP quantization;
7. reason about practical trade-offs such as latency, memory, calibration, outliers, and quality loss.

---

## Suggested Order

| Order | File | Main Question |
|---:|---|---|
| 1 | `lecture-1-why-quantization.md` | Why does quantization exist? |
| 2 | `lecture-2-numbers-scale-zero-point.md` | How do real numbers become integers? |
| 3 | `lecture-3-onnx-and-netron-mnist.md` | How do we inspect a real model graph? |
| 4 | `lecture-4-post-training-quantization.md` | How can we quantize after training? |
| 5 | `lecture-5-quantization-aware-training.md` | How can training prepare the model for quantization? |
| 6 | `lecture-6-llm-quantization.md` | What changes when the model is an LLM? |
| 7 | `lecture-7-advanced-quantization-topics.md` | What are the advanced deployment trade-offs? |
| 8 | `practice-1-netron-onnx-mnist.md` | How do we generate and inspect the artifact? |
| 9 | `practice-2-quantization-analysis.md` | How do we compare FP32 vs quantized models? |
| 10 | `code-mnist-onnx-quantization.py` | Runnable artifact for export, quantization, and evaluation. |

---

## Running Example

We use MNIST because it is small, visual, and easy to evaluate. The model is intentionally simple:

```text
784 input pixels → hidden layer → ReLU → 10 logits
```

This is not the best possible MNIST model. It is a teaching model. Its purpose is to make the quantization mechanics easy to inspect.

---

## Tooling Focus

The central tooling workflow is:

```text
train small model
→ export ONNX
→ open in Netron
→ quantize ONNX
→ open quantized model in Netron
→ compare size, operators, and accuracy
```

Netron is especially useful because quantization is not only a number-format idea. It changes the concrete graph that an inference runtime executes.

---

## Final Discussion Goal

The final synthesis question is:

> If you had to deploy a model under strict memory and latency limits, which quantization method would you try first, and what evidence would convince you that it is safe?
