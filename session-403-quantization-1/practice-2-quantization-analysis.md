# Practice 2: Analyze FP32 vs Quantized MNIST Inference

This practice focuses on measurement and interpretation.

---

## Goal

Compare the FP32 and quantized ONNX models using file size and accuracy.

---

## Step 1: Run the Script

```bash
python code-mnist-onnx-quantization.py
```

Open:

```text
artifacts/metrics.json
```

---

## Step 2: Build a Comparison Table

Fill in:

| Model | File Size | Test Accuracy | Notes |
|---|---:|---:|---|
| FP32 ONNX | | | |
| Dynamic INT8 ONNX | | | |

---

## Step 3: Interpret the Result

Answer:

1. How much smaller is the quantized model?
2. Did accuracy change?
3. Is the quantized model automatically better? Why or why not?
4. Which extra benchmark would you run before deploying it?

---

## Step 4: Modify the Experiment

Change one of these settings in the script:

- hidden layer size;
- number of training samples;
- maximum training iterations;
- random seed.

Then rerun the experiment.

Answer:

1. Did FP32 accuracy change?
2. Did quantized accuracy change?
3. Did the size ratio change?
4. What does this tell you about architecture and quantization?

---

## Step 5: Connect to LLMs

Write a short paragraph comparing this MNIST experiment to LLM quantization.

Use these prompts:

- What is similar?
- What is much harder for LLMs?
- Why might weight-only quantization be attractive for LLMs?
- Why is one accuracy number not enough for LLM evaluation?

---

## Optional Challenge

Try static quantization with calibration data using ONNX Runtime's calibration workflow. Compare the resulting graph to dynamic quantization in Netron.
