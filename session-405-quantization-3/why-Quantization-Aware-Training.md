# Why Quantization Aware Training (QAT)?

> [!INFO]
> Motivation:
> - Check out https://ollama.com/library/gemma3  



## The Core Problem

Standard neural networks are usually trained in FP32 because high precision makes optimization easier.

### FP32 Advantages

* High numerical precision
* Stable training
* Better gradient flow

### FP32 Disadvantages

* Large model size
* Slower inference
* Higher power consumption
* Poor deployment on edge devices

> [!INFO]
> INT8 weights typically reduce memory usage by about 4× compared to FP32.

---

## The Limitation of Post-Training Quantization (PTQ)

PTQ converts a trained FP32 model directly into INT8 after training.

### Main Issue

The model was never trained to handle:

* Weight rounding
* Activation clipping
* Reduced precision

### Result

Accuracy often drops, especially for smaller or sensitive models.

> [!WARNING]
> PTQ is fast, but accuracy degradation can be significant.

---

## How QAT Solves This

QAT simulates quantization during training.

### Key Idea

During training:

* Weights behave like INT8
* Activations behave like INT8
* Forward pass includes quantization noise

### Benefit

The model learns how to adapt before deployment.

---

## Main Advantages of QAT

### Better Accuracy

QAT usually preserves accuracy much better than PTQ.

### Smaller Models

INT8 deployment reduces memory footprint.

### Faster Inference

Useful for:

* Mobile devices
* Edge AI
* Robotics
* Embedded systems

### Lower Energy Usage

Important for battery-powered hardware.

---

## Trade-Offs

### Downsides

* More training complexity
* Longer training time
* Harder implementation
* Export pipelines can be more difficult

---

## PTQ vs QAT

| Feature             | PTQ    | QAT       |
| ------------------- | ------ | --------- |
| Training Complexity | Low    | Higher    |
| Accuracy            | Medium | Near-FP32 |
| Deployment Speed    | Fast   | Fast      |
| Edge Performance    | Good   | Best      |

---

## In Your MNIST Case

For a network like:

$784 \rightarrow Hidden \rightarrow Hidden \rightarrow 10$

### Without QAT

INT8 may lose noticeable accuracy.

### With QAT

INT8 often stays close to FP32 while remaining smaller and faster.

---

## Final Takeaway

QAT is the best choice when you need:

* Small model size
* Fast inference
* Low power usage
* Minimal accuracy loss

> [!INFO]
> QAT trains the model for the hardware limitations it will face later, making deployment much more reliable.
