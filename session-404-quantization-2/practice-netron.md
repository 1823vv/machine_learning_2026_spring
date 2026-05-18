# Inspecting ONNX Models with Netron

ONNX models can be **inspected visually** using [Netron](https://netron.app/), a powerful tool to explore:

* Layer structure
* Input/output shapes
* Weight types (precision)
* Activation flows

---

## 1 — Why Visualize Models?

* **Understand model architecture**

  * See how layers are connected
  * Check the sequence of operations (Linear → ReLU → Linear, etc.)
* **Verify quantization effects**

  * Compare FP32, FP16, and INT8 versions
  * Inspect weight precision and layer parameters
* **Debug and validate models**

  * Ensure that exported ONNX models match your intended architecture
  * Spot potential issues before deployment

---

## 2 — Install and Run Netron

### 2.1 Install

* **Option 0: Web Version**

  Use https://netron.app/ directly. Go to netron.app and drag your `.onnx` file into the browser.

* **Option 1: Desktop app**
  Download from [https://netron.app](https://netron.app) and install for your OS (Windows, macOS, Linux).

* **Option 2: Python / pip**

```bash
pip install netron
```

---

### 2.2 Launch Netron

* **Desktop:** Open the app, then open a model from `models/` folder.
* **Command line:**

```bash
netron models/mnist_model.onnx
```

* The command opens a browser window with a **graphical interface** for the model.

---

## 3 — Visualizing FP32, FP16, and INT8 Models

1. Open `mnist_model.onnx` → notice **FP32 weights** and layer shapes.
2. Open `mnist_model_quantized_fp16.onnx` → compare **weight precision** (16-bit) and check that the **network structure is preserved**.
3. Open `mnist_model_quantized_int8.onnx` → see **8-bit integer weights**. Some layers may show scaling parameters due to quantization.

---

## 4 — Key Things to Observe

### 4.1 Layer Connections

* Verify that **Linear → ReLU → Linear → ReLU → Linear** is maintained.
* Flatten layer should be **at the beginning**.

### 4.2 Input and Output Shapes

* Input shape: `(1,1,28,28)`
* Output shape: `(1,10)` (10 classes)
* Dynamic axes (batch dimension) may appear in ONNX export.

### 4.3 Weight Precision

* FP32 → 32-bit floating point
* FP16 → 16-bit floating point
* INT8 → 8-bit integer
* Check the nodes; in INT8 some nodes may include **quantize/dequantize** operations.

### 4.4 Node Types

* Linear / Gemm (Fully Connected)
* ReLU (Activation)
* Flatten / Reshape
* QuantizeLinear / DequantizeLinear (in INT8 model)

---

## 5 — Practice Ideas

1. Open all three models in **Netron simultaneously**.
2. Check **which layers changed in precision** after quantization.
3. Identify **QuantizeLinear / DequantizeLinear** nodes in INT8 model.
4. Explore **weight distributions**: some nodes show min/max ranges in INT8.
5. Try **hovering on nodes** to see their attributes: shapes, scales, bitwidth.

---

### Summary

Using **Netron**:

* You get a **visual overview** of your model architecture.
* You can **inspect quantized models** for FP16 and INT8 to see the effects on layers, weights, and activations.
* It’s an **essential practice tool** for debugging, validating, and understanding ONNX exports.
