# ONNX and Netron with an MNIST Model


> [!INFO]
> - https://netron.app/
 

---

## 1. Why Use ONNX (Open Neural Network Exchange) ?

ONNX is a portable model representation. It stores a model as a computation graph:

```text
inputs → operators → outputs
```

For this mini-series, ONNX is useful because it lets us separate three concerns:

- training the model;
- saving the model graph;
- inspecting and running the model in inference tools.

---

## 2. Why Use Netron?

Netron is a model graph viewer. It helps answer questions such as:

- What are the input and output tensor shapes?
- Which operators does the model use?
- Where are the weight tensors?
- Does the quantized model contain quantization operators?
- Did the graph become simpler or more complex after quantization?

Quantization can feel abstract. Netron makes it visible.

---

## 3. The Running MNIST Model

The teaching model is intentionally small:

$$
x \in \mathbb{R}^{784}
$$

$$
h = \text{ReLU}(xW_1 + b_1)
$$

$$
\text{logits} = hW_2 + b_2
$$

The output has 10 logits, one for each digit class.

```text
image pixels → flatten → linear → ReLU → linear → logits
```

This small model is enough to show:

- weights;
- activations;
- matrix multiplication;
- graph nodes;
- quantized operators or quantize/dequantize patterns.

---

## 4. What to Look For in the FP32 Graph

Open the floating-point model in Netron and inspect:

1. **Input shape**

   The model should accept one flattened image:

   ```text
   [batch, 784]
   ```

2. **Linear layers**

   Depending on export details, linear layers may appear as `Gemm`, `MatMul`, or related operators.

3. **Activation**

   The hidden layer should use `Relu`.

4. **Output shape**

   The model should output:

   ```text
   [batch, 10]
   ```

5. **Initializers**

   These are stored tensors such as weights and biases.

---

## 5. What to Look For in the Quantized Graph

After quantization, inspect the quantized ONNX file.

Depending on the quantization format, you may see operators such as:

- `QuantizeLinear`
- `DequantizeLinear`
- quantized matrix multiplication operators;
- integer weight tensors;
- scale tensors;
- zero-point tensors.


> [!INFO]
> - FP32 graph: values are directly consumed as floating point
> - quantized graph: values may be quantized, computed, and dequantized around operators

---

## 6. Inspection Questions

When comparing the two models, answer:

1. Is the quantized model file smaller?
2. Which new nodes appear?
3. Where are scale and zero-point tensors stored?
4. Are all operators quantized, or only some?
5. Does the output accuracy change?
6. Is the graph easier or harder to read?
