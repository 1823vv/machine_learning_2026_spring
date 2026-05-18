# 16-Bit Floating-Point Numbers

## Introduction: The Illusion of Continuity

When we write programs, we often treat numbers as though they exist on a continuous line. We write $\pi$, and we expect the machine to understand the infinite, non-repeating majesty of $3.1415926535...$

But computers are finite machines. They do not possess infinite memory, nor do they natively understand continuous mathematics. Instead, they work with discrete intervals—fractions built entirely out of powers of two.

Today, we are going to look under the hood of **FP16 (Half-Precision Floating-Point)** to understand exactly why it cannot store $\pi$ accurately, why it rounds specifically to **$3.140625$**, and how we can use Python to expose these architectural constraints.

---

## 1. The Anatomy of an FP16 Number

To understand why $\pi$ changes, we must first look at how a 16-bit floating-point container is built. Under the IEEE 754 standard, those 16 bits of memory are strictly rationed into three distinct components:

* **Sign (1 bit):** Determines if the number is positive ($0$) or negative ($1$).
* **Exponent (5 bits):** Determines the dynamic range or "scale" of the number (where it sits between powers of two).
* **Mantissa / Significand (10 bits):** Determines the actual precision or "digits" of the number.

Because the mantissa only has 10 bits allocated to it, it can only hold $2^{10}$ or **1,024 distinct values** between any two given powers of two. This finite bucket of values introduces what we call *quantization error*.

---

## 2. The Mathematics of Scale and Step Size

Every floating-point number is computed using a formula that includes a hidden, implicit leading $1$, followed by the fractional binary bits of the mantissa, all scaled by the exponent:

$$\text{Value} = (-1)^{\text{sign}} \times \left(1 + \sum_{i=1}^{10} \text{bit}_i \times 2^{-i}\right) \times 2^{\text{exponent} - \text{bias}}$$

For FP16, the exponent bias is $15$.

Because $\pi$ ($3.14159...$) lives between $2^1$ ($2$) and $2^2$ ($4$), the exponent must scale our numbers by exactly $2^1$. When we multiply that $2^1$ scale across our 10 available mantissa bits, each bit takes on a rigid, specific decimal value:

| Bit Position | Binary Weight | Multiplied by $2^1$ (Decimal Step Size) |
| --- | --- | --- |
| Bit 1 | $2^{-1} = 1/2$ | $1.0$ |
| Bit 2 | $2^{-2} = 1/4$ | $0.5$ |
| Bit 3 | $2^{-3} = 1/8$ | $0.25$ |
| Bit 4 | $2^{-4} = 1/16$ | $0.125$ |
| Bit 5 | $2^{-5} = 1/32$ | $0.0625$ |
| Bit 6 | $2^{-6} = 1/64$ | $0.03125$ |
| Bit 7 | $2^{-7} = 1/128$ | $0.015625$ |
| Bit 8 | $2^{-8} = 1/256$ | $0.0078125$ |
| Bit 9 | $2^{-9} = 1/512$ | $0.00390625$ |
| Bit 10 | $2^{-10} = 1/1024$ | **$0.001953125$** |

### The Precision Gap

Look closely at Bit 10. In this specific numerical range (between 2 and 4), the absolute finest resolution an FP16 number can achieve is **$0.001953125$**. It is completely impossible for the hardware to represent a number that falls *between* these steps.

---

## 3. The Nearest Neighbor Problem

Because of this rigid grid, $\pi$ finds itself trapped between two representable numbers. It must round to the closest available option:

* **Option A (Below $\pi$):** $3.140625$
* **Option B (Above $\pi$):** $3.142578125$ (which is exactly $3.140625 + 0.001953125$)

If we calculate the absolute distance from actual $\pi$ to both options, we see:

* $\text{Distance to Option A}: |3.14159265 - 3.140625| = \mathbf{0.00096765}$
* $\text{Distance to Option B}: |3.14159265 - 3.142578125| = 0.000985475$

Following standard IEEE 754 round-to-nearest rules, the processor selects Option A because it is mathematically closer. Thus, $\pi$ becomes $3.140625$. This is the price we pay for the massive speed and low memory footprints required by modern graphics and deep learning workloads.

---

## 4. Empirical Verification via Python

We do not have to take this theory on faith. We can use Python and the `numpy` library to force the hardware to reveal this quantization process.

### Experiment 1: The Quick Cast

First, let's observe the rounding directly by casting Python's high-precision float (which is a 64-bit float) into a 16-bit float.

```python
import numpy as np

# Convert standard 64-bit pi to a 16-bit float
pi_fp16 = np.float16(np.pi)

print(f"Standard 64-bit Pi: {np.pi}")
print(f"Truncated FP16 Pi:   {pi_fp16}")

```

When run, the output confirms our math:

```text
Standard 64-bit Pi: 3.141592653589793
FP16 Pi:             3.140625

```

### Experiment 2: Binary Bit Extraction

To prove that the binary components match our formulas perfectly, we can instruct Python to view the raw 16-bit integer memory layout of that float, unpack the bits, and map them out.

```python
import numpy as np

# 1. Store pi as an FP16
pi_fp16 = np.float16(np.pi)

# 2. View the raw bits as an unsigned 16-bit integer
raw_bits = pi_fp16.view(np.uint16)

# 3. Format the integer into a 16-character binary string
binary_str = f"{raw_bits:016b}"

# 4. Slice the string into its structural components
sign = binary_str[0]
exponent = binary_str[1:6]
mantissa = binary_str[6:]

print(f"FP16 Value: {pi_fp16}")
print(f"Binary Breakdown: {sign} {exponent} {mantissa}")
print(f"                  └─Sign └─Exponent └─Mantissa")

```

Execution yields the following bit pattern:

```text
FP16 Value: 3.140625
Binary Breakdown: 0 10000 1001001000
                  └─Sign └─Exponent └─Mantissa

```

---

## Conclusion: Reconstructing the Bitstring

Let's reverse-engineer this binary output to finish the lecture:

1. **Sign bit is `0**`, meaning the value is positive.
2. **Exponent bits are `10000**`. In decimal, this is 16. Subtracting our bias ($16 - 15$), we get an exponent multiplier of $2^1$.
3. **Mantissa bits are `1001001000**`. Counting from left to right, Bit 1, Bit 4, and Bit 7 are active (`1`).

If we re-apply our step-size table to those active bits and include the implicit leading $1$:

$$\text{Value} = \left(1 + \frac{1}{2} + \frac{1}{16} + \frac{1}{128}\right) \times 2^1$$

$$\text{Value} = (1 + 0.5 + 0.0625 + 0.0078125) \times 2$$

$$\text{Value} = 1.5703125 \times 2 = \mathbf{3.140625}$$

The machine has done no wrong; it has simply run out of resolution. When writing high-performance computing or machine learning code, always remember that every drop in precision changes the foundational constants of your mathematics.