# Why Quantization?

> [!INFO]
> Motivation:
> - Check out https://ollama.com/library/gemma3  


## 1 — Motivation: Why We Do Quantization

### 1.1 Reduce Model Memory Footprint

* Most neural networks store weights in **FP32 (32-bit floating-point)**.
* Large models like CNNs or LLMs can consume **hundreds of megabytes or even tens of gigabytes**.
* Quantizing to **FP16 (16-bit)** or **INT8 (8-bit)** reduces memory usage dramatically:

  * FP16 → ~50% memory reduction
  * INT8 → ~75% memory reduction
* Smaller models mean:

  * Easier deployment on **memory-constrained devices**
  * Ability to fit **more models in GPU vRAM** for batch inference

---

### 1.2 Speed Up Inference

* Lower-precision arithmetic is **faster** because:

  * FP16 operations leverage **Tensor Cores** on modern GPUs (NVIDIA, AMD).
  * INT8 operations often have **hardware acceleration** on CPU, GPU, or specialized AI chips (TPU, EdgeTPU).
* Faster inference is critical for:

  * **Real-time computer vision** (object detection, video analysis)
  * **Interactive NLP applications** (chatbots, LLMs)

---

### 1.3 Reduce Energy Consumption

* Lower precision reduces:

  * **Memory bandwidth** usage
  * **Number of FLOPs** (floating-point operations)
* Critical for **battery-powered or embedded devices**

---

## 2 — Why Quantization Works: Theory

Neural networks are **robust to small numerical perturbations**:

1. **Redundant Representations**

   * Many weights in trained networks are **not sensitive to small changes**.
   * Quantizing 32-bit weights to 16-bit or 8-bit introduces small rounding errors, often negligible for model output.

2. **Activation and Weight Distributions**

   * Most activations are within a limited range.
   * Quantization maps these to **lower-precision numeric ranges** with minimal information loss.

3. **Post-Training vs Quantization-Aware Training**

   * **Post-Training Quantization (PTQ)** works for many networks without retraining.
   * **Quantization-Aware Training (QAT)** allows the network to adjust to low-precision weights during training, reducing accuracy loss.

---

## 3 — Quantization in Practice

### 3.1 Precision Types

| Precision | Bits | Notes                                                |
| --------- | ---- | ---------------------------------------------------- |
| FP32      | 32   | Standard training precision                          |
| FP16      | 16   | Minimal accuracy drop, widely supported on GPUs      |
| INT8      | 8    | Aggressive compression, may slightly reduce accuracy |
| INT4      | 4    | Rare, extreme compression, experimental              |

---

### 3.2 Practical Deployment Considerations

1. **GPU vRAM Optimization**

   * FP16 models use **half the memory of FP32**
   * INT8 models use **quarter of FP32 memory**
   * Enables:

     * Larger batch sizes
     * Running multiple models simultaneously

2. **LLMs and Large Models**

   * Large Language Models (LLMs) like GPT, LLaMA, or BLOOM have **billions of parameters**.
   * Storing them in FP32 would require **tens of GB of VRAM**.
   * Quantization allows inference on **smaller GPUs** without significant accuracy loss.

3. **Inference Latency**

   * Lower precision reduces data movement between memory and GPU cores
   * Reduces **compute latency** for each forward pass

4. **Edge and Mobile AI**

   * INT8 quantization is particularly valuable on **mobile AI chips** and **embedded devices**.
   * Memory and power constraints are strict; quantization is often required.

---

## 4 — Observed Benefits

* **Memory Reduction:** FP16 ~50%, INT8 ~75% compared to FP32
* **Faster Inference:** Leveraging hardware-specific low-precision units
* **Deployability:** Models can run on **smaller GPUs, CPUs, or edge devices**
* **Sustainable AI:** Lower power consumption and heat generation

---

## 5 — Trade-offs and Considerations

1. **Accuracy Loss**

   * FP16 usually maintains accuracy almost identical to FP32
   * INT8 may drop accuracy slightly, especially for sensitive models

2. **Hardware Support**

   * FP16 is widely supported on modern GPUs (Tensor Cores)
   * INT8 requires hardware support or may rely on software emulation (slower)

3. **Model Type Sensitivity**

   * CNNs, RNNs, and transformers tolerate quantization differently
   * Large models like LLMs often use **mixed precision**: FP16 for weights, FP32 for sensitive layers

4. **Quantization Strategy**

   * **Static Quantization:** Pre-computed scaling factors
   * **Dynamic Quantization:** Scales activations on-the-fly
   * **Quantization-Aware Training (QAT):** Adjusts weights during training to improve INT8 accuracy

---

## 6 — Summary

Quantization works because **neural networks tolerate small numerical errors**, allowing weights and activations to be stored at lower precision.

Key takeaways:

* Reduce **model size** → deploy larger models or multiple models in limited VRAM
* Increase **inference speed** → lower latency for real-time applications
* Reduce **energy consumption** → essential for edge AI and sustainable AI
* Enable **LLM deployment on practical hardware** without losing significant accuracy

> By practicing FP32 → FP16 → INT8 conversion, you can **directly observe trade-offs in accuracy, memory usage, and model efficiency**.
