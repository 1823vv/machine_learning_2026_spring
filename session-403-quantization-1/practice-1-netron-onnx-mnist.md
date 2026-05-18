# Practice 1: Export an MNIST ONNX Model and Inspect It in Netron

This practice turns the quantization mini-series into a concrete artifact.

---

## Goal

Create a small MNIST classifier, export it to ONNX, and inspect the graph in Netron.

---

## Setup

Recommended packages:

```bash
pip install -r requirements.txt
```

If you only want to inspect prebuilt ONNX files, you need Netron. If you want to run the full script, install all packages above.

---

## Step 1: Generate the ONNX Model

From this folder, run:

```bash
python code-mnist-onnx-quantization.py
```

Expected generated files:

```text
artifacts/mnist_mlp_fp32.onnx
artifacts/mnist_mlp_dynamic_int8.onnx
artifacts/metrics.json
```

---

## Step 2: Open the FP32 Model in Netron

Run:

```bash
netron artifacts/mnist_mlp_fp32.onnx
```

Or open the file manually in the Netron desktop/browser interface.

Inspect:

- model input shape;
- model output shape;
- linear/matrix multiplication operators;
- ReLU operator;
- weight initializers.

---

## Step 3: Answer These Questions

1. What is the input tensor shape?
2. What is the output tensor shape?
3. Which node represents the hidden layer computation?
4. Where do you see the learned parameters?
5. How many output logits are produced?

---

## Step 4: Open the Quantized Model

Run:

```bash
netron artifacts/mnist_mlp_dynamic_int8.onnx
```

Inspect:

- new quantization-related nodes;
- integer weight tensors;
- scale tensors;
- zero-point tensors;
- any operators that remain floating point.

---

## Deliverable

Create a short note with:

- one screenshot or description of the FP32 graph;
- one screenshot or description of the quantized graph;
- three differences you observed;
- one question you still have about quantized inference.
