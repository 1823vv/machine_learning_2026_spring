# Accurate Numerical Examples: FP32, FP16, INT8, INT4

---

## **1 — FP32 (32-bit floating-point)**

*   **Memory:** 32 bits (4 bytes) per number.
*   **Format:** 1 sign bit, 8 exponent bits, 23 mantissa bits (IEEE 754 standard).
*   **Range:** ≈ ±3.4028235 × 10³⁸
*   **Precision:** ~7 significant decimal digits.
*   **Example:**
    *   **Number:** 1.5
        *   **Binary (Exact):** `0 01111111 10000000000000000000000`
        *   **Hex:** 0x3FC00000
    *   **Number:** -2.75
        *   **Binary (Exact):** `1 10000000 01100000000000000000000`
        *   **Hex:** 0xC0300000
    *   **Number:** 0.1
        *   **Binary (Rounded):** `0 01111011 10011001100110011001101`
        *   **Hex:** 0x3DCCCCCD
        *   **Note:** Cannot be represented exactly in binary; this is the closest single-precision value.
*   **Memory for 10,000 numbers:** 10,000 * 4 bytes = 40,000 bytes = **~39.06 KiB**.

---

### 🔍 Why is 0.1 an Infinite Repeating Decimal in Binary?

This is a classic computer science problem! Let me explain in detail:

---

#### 1️⃣ Core Reason: Essential Difference in Number Bases

**Decimal 0.1 = 1/10**

The problem: **Prime factorization of 10 = 2 × 5**

**Binary can only exactly represent fractions with denominators that are powers of 2**, such as:
- 1/2 = 0.5 ✅
- 1/4 = 0.25 ✅
- 1/8 = 0.125 ✅
- 3/8 = 0.375 ✅

But **1/10's denominator contains the prime factor 5**, which binary cannot represent exactly!

---

#### 2️⃣ Converting Decimal 0.1 to Binary

Using the **multiply-by-2 method**:

```
0.1 × 2 = 0.2  → take integer part 0
0.2 × 2 = 0.4  → take integer part 0
0.4 × 2 = 0.8  → take integer part 0
0.8 × 2 = 1.6  → take integer part 1
0.6 × 2 = 1.2  → take integer part 1
0.2 × 2 = 0.4  → take integer part 0  ← Cycle starts here!
0.4 × 2 = 0.8  → take integer part 0
0.8 × 2 = 1.6  → take integer part 1
0.6 × 2 = 1.2  → take integer part 1
...
```

**Result: 0.1(decimal) = 0.0001100110011001100... (binary)**

You can see **`1100` repeats infinitely**! Just like 1/3 = 0.333... in decimal.

---

#### 3️⃣ How Does FP32 Store 0.1?

Since there are only 23 mantissa bits, it must be **truncated**:

```
Exact value:    0.000110011001100110011001100110011001100...
FP32 storage:   0.00011001100110011001101  (23-bit mantissa, last bit rounded)
```

**Actual stored value: 0.100000001490116119384765625**

Error from 0.1: approximately **1.49 × 10⁻⁹**

---

#### 4️⃣ Problems in Actual Programming

```python
# Python Example
>>> 0.1 + 0.2
0.30000000000000004  # Not 0.3!

>>> 0.1 + 0.2 == 0.3
False  # Surprising!

>>> 0.3 - 0.2
0.09999999999999998  # Not 0.1!
```

```javascript
// JavaScript Example
> 0.1 + 0.2
0.30000000000000004

> 0.1 * 3
0.30000000000000004
```

---

#### 5️⃣ Which Numbers Can Be Represented Exactly?

| Decimal | Binary | Can Be Exact |
|--------|--------|-------------|
| 0.5 | 0.1 | ✅ Yes (1/2) |
| 0.25 | 0.01 | ✅ Yes (1/4) |
| 0.75 | 0.11 | ✅ Yes (3/4) |
| 0.1 | 0.00011001100... | ❌ No |
| 0.2 | 0.00110011001... | ❌ No |
| 0.3 | 0.01001100110... | ❌ No |

**Rule:** Only fractions with denominators of 2ⁿ can be represented exactly.

---

#### 6️⃣ Solutions

| Scenario | Recommended Solution |
|------|---------|
| Financial calculations | Use **fixed-point** or **Decimal type** (e.g., Python's `decimal` module) |
| Comparing floats | Use **epsilon tolerance**: `abs(a - b) < epsilon` |
| High precision needs | Use **arbitrary precision libraries** (e.g., Python's `decimal`, `fractions`) |
| Display purposes | Format output: `"{:.2f}".format(value)` |

```python
if abs(0.1 + 0.2 - 0.3) == 0:
    print("Equal")
else:
    print("Not Equal")

# Correct float comparison
epsilon = 1e-9
if abs(0.1 + 0.2 - 0.3) < epsilon:
    print("Equal")
else:
    print("Not Equal")

# Financial calculations with Decimal
from decimal import Decimal
print(Decimal('0.1') + Decimal('0.2') == Decimal('0.3'))  # True
```

---

### 📌 Summary

| Problem | Reason |
|------|------|
| 0.1 cannot be represented exactly | 1/10's denominator contains prime factor 5, binary cannot represent it |
| Produces error | 23-bit mantissa must truncate infinite repeating binary decimal |
| Impact | Floating-point operations may produce tiny errors, be careful with comparisons |
| Solution | Use Decimal for finance, epsilon for comparisons, never use `==` directly |

**This is why you should NEVER compare floating-point numbers with `==` directly!** 🎯

---

### 🐍 FP32 Python Deep Dive

```python
import struct
import numpy as np

def float_to_bits(f):
    """Convert float to binary string representation"""
    return ''.join(f'{b:08b}' for b in struct.pack('>f', f))

def float_to_hex(f):
    """Convert float to hex representation"""
    return hex(struct.unpack('>I', struct.pack('>f', f))[0])

# Test values
test_values = [1.5, -2.75, 0.1, 0.2, 0.5]

print("=" * 60)
print("FP32 Deep Analysis")
print("=" * 60)

for val in test_values:
    bits = float_to_bits(val)
    hex_val = float_to_hex(val)
    sign = bits[0]
    exponent = bits[1:9]
    mantissa = bits[9:]
    
    print(f"\nValue: {val}")
    print(f"  Binary : {sign} {exponent} {mantissa}")
    print(f"  Hex    : {hex_val}")
    print(f"  Sign   : {'+' if sign == '0' else '-'}")
    print(f"  Exp    : {int(exponent, 2)} (biased), {int(exponent, 2) - 127} (actual)")

# Demonstrate precision loss
print("\n" + "=" * 60)
print("Precision Loss Demonstration")
print("=" * 60)

x = 0.1
y = 0.2
z = 0.3

print(f"0.1 + 0.2 = {x + y}")
print(f"Expected  : {z}")
print(f"Difference: {abs(x + y - z):.20f}")
print(f"Equal?    : {x + y == z}")

# Show cumulative error
print("\nCumulative Error (adding 0.1 ten times):")
total = 0.0
for i in range(10):
    total += 0.1
    print(f"  Step {i+1}: {total:.20f}")
print(f"  Expected: 1.0")
print(f"  Error   : {abs(total - 1.0):.20f}")
```

---

## **2 — FP16 (16-bit floating-point)**

*   **Memory:** 16 bits (2 bytes) per number.
*   **Format:** 1 sign bit, 5 exponent bits, 10 mantissa bits (IEEE 754 half-precision).
*   **Range:** ≈ ±6.55 × 10⁴
*   **Precision:** ~3-4 significant decimal digits.
*   **Example:**
    *   **Number:** 1.5
        *   **Binary (Exact):** `0 01111 1000000000`
        *   **Hex:** 0x3E00
    *   **Number:** 0.1
        *   **Binary (Rounded):** `0 01011 1001100110`
        *   **Hex:** 0x2E66
        *   **Decimal Value:** ~0.0999755859375
        *   **Note:** Not exact; this is the closest half-precision value.
*   **Memory for 10,000 numbers:** 10,000 * 2 bytes = 20,000 bytes = **~19.53 KiB** (50% of FP32).

---

### 🔬 FP16 Deep Analysis

#### Key Characteristics

| Aspect | Detail |
|--------|--------|
| **Exponent Bias** | 15 (vs 127 for FP32) |
| **Smallest Positive** | 5.96 × 10⁻⁸ (normal), 5.96 × 10⁻⁸ (subnormal) |
| **Largest Value** | 65,504 |
| **Machine Epsilon** | ~9.77 × 10⁻⁴ |
| **Decimal Precision** | ~3.31 digits (log₁₀(2¹⁰)) |

#### ⚠️ FP16 Limitations

```
┌─────────────────────────────────────────────────────────┐
│  FP16 Danger Zones                                      │
├─────────────────────────────────────────────────────────┤
│  • Overflow: values > 65,504 → ∞                        │
│  • Underflow: values < 5.96e-8 → 0                      │
│  • Precision: only 1024 distinct values per exponent    │
│  • Gradient vanishing in deep neural networks           │
└─────────────────────────────────────────────────────────┘
```

#### 🎯 FP16 Sweet Spots

| Use Case | Why FP16 Works |
|----------|---------------|
| GPU Training | NVIDIA Tensor Cores optimized for FP16 |
| Inference | 2x memory bandwidth, 2-4x speedup |
| Graphics | HDR rendering, sufficient precision |
| Mobile AI | Reduced power consumption |

---

### 🐍 FP16 Python Code

```python
import numpy as np

def analyze_fp16():
    """Deep analysis of FP16 precision and range"""
    
    print("=" * 60)
    print("FP16 Deep Analysis")
    print("=" * 60)
    
    # FP16 limits
    fp16_max = np.finfo(np.float16).max
    fp16_min = np.finfo(np.float16).min
    fp16_eps = np.finfo(np.float16).eps
    
    print(f"\nFP16 Limits:")
    print(f"  Max value     : {fp16_max}")
    print(f"  Min value     : {fp16_min}")
    print(f"  Epsilon       : {fp16_eps}")
    print(f"  Memory saving : 50% vs FP32")
    
    # Precision demonstration
    print("\n" + "-" * 60)
    print("Precision Comparison (FP32 vs FP16)")
    print("-" * 60)
    
    test_values = [0.1, 0.01, 0.001, 1.0, 10.0, 100.0]
    
    for val in test_values:
        fp32_val = np.float32(val)
        fp16_val = np.float16(val)
        error = abs(float(fp32_val) - float(fp16_val))
        rel_error = error / abs(float(fp32_val)) * 100
        
        print(f"  {val:6.3f} → FP16: {fp16_val:8.5f} | Error: {rel_error:6.3f}%")
    
    # Overflow demonstration
    print("\n" + "-" * 60)
    print("Overflow Demonstration")
    print("-" * 60)
    
    overflow_vals = [10000, 50000, 65504, 65505, 100000]
    
    for val in overflow_vals:
        fp16_result = np.float16(val)
        status = "✓" if np.isfinite(fp16_result) else "✗ OVERFLOW"
        print(f"  {val:6d} → FP16: {fp16_result:8} | {status}")
    
    # Accumulation error
    print("\n" + "-" * 60)
    print("Accumulation Error (sum 0.1 × 100)")
    print("-" * 60)
    
    fp32_sum = np.sum([np.float32(0.1) for _ in range(100)])
    fp16_sum = np.sum([np.float16(0.1) for _ in range(100)])
    
    print(f"  FP32 sum: {fp32_sum:.6f}")
    print(f"  FP16 sum: {fp16_sum:.6f}")
    print(f"  Expected: 10.0")
    print(f"  FP32 err: {abs(fp32_sum - 10.0):.6f}")
    print(f"  FP16 err: {abs(fp16_sum - 10.0):.6f}")

analyze_fp16()
```

#### Fun FP16 Numbers

```python
# Interesting FP16 boundary values
interesting_fp16 = {
    "Smallest normal": np.float16(2**-14),      # 6.1035e-05
    "Smallest subnormal": np.float16(2**-24),   # 5.9605e-08
    "Largest finite": np.float16(65504),        # 65504
    "One + epsilon": np.float16(1.0) + np.finfo(np.float16).eps,
    "Pi approximation": np.float16(np.pi),      # 3.1406
    "E approximation": np.float16(np.e),        # 2.7188
}

print("\nInteresting FP16 Values:")
for name, val in interesting_fp16.items():
    print(f"  {name:20s}: {val}")
```

---

## **3 — INT8 (8-bit signed integer)**

*   **Memory:** 8 bits (1 byte) per number.
*   **Range:** -128 to 127.
*   **Precision:** Integer only. Floating-point values require quantization (scale & zero point).
*   **Example with Symmetric Quantization (FP32 range ≈ [-1.0, 1.0]):**
    *   **Scale Factor:** 127
    *   **Quantization Formula:** `INT8 = round(FP32 * 127)`
    *   **FP32 value:** 1.0 → **INT8 value:** 127
    *   **FP32 value:** 0.56 → **INT8 value:** round(71.12) = 71
    *   **FP32 value:** -0.5 → **INT8 value:** -64
*   **Memory for 10,000 numbers:** 10,000 * 1 byte = 10,000 bytes = **~9.77 KiB** (25% of FP32).

---

### 🔬 INT8 Deep Analysis

#### Quantization Mathematics

```
┌─────────────────────────────────────────────────────────┐
│  Quantization Pipeline                                  │
├─────────────────────────────────────────────────────────┤
│  FP32 → [Scale] → [Round] → [Clip] → INT8               │
│  INT8 → [Dequantize] → FP32                             │
├─────────────────────────────────────────────────────────┤
│  Scale = (max_range) / 127                              │
│  Zero_Point = round(-min_range / scale)                 │
│  Quantized = clip(round(value / scale) + zero_point)    │
└─────────────────────────────────────────────────────────┘
```

#### Quantization Error Analysis

| FP32 Value | INT8 Value | Dequantized | Absolute Error | Relative Error |
|------------|------------|-------------|----------------|----------------|
| 1.0 | 127 | 1.0 | 0.0 | 0.0% |
| 0.56 | 71 | 0.5591 | 0.0009 | 0.16% |
| 0.1 | 13 | 0.1024 | 0.0024 | 2.4% |
| 0.01 | 1 | 0.0079 | 0.0021 | 21% |
| 0.001 | 0 | 0.0 | 0.001 | 100% |

**Key Insight:** Small values suffer disproportionately from quantization error!

---

### 🐍 INT8 Python Code

```python
import numpy as np

class INT8Quantizer:
    """INT8 Symmetric Quantization"""
    
    def __init__(self, scale=127):
        self.scale = scale
    
    def quantize(self, fp32_array):
        """FP32 → INT8"""
        int8_array = np.round(fp32_array * self.scale)
        int8_array = np.clip(int8_array, -128, 127)
        return int8_array.astype(np.int8)
    
    def dequantize(self, int8_array):
        """INT8 → FP32"""
        return int8_array.astype(np.float32) / self.scale
    
    def analyze_error(self, fp32_array):
        """Analyze quantization error"""
        int8_array = self.quantize(fp32_array)
        recovered = self.dequantize(int8_array)
        
        abs_error = np.abs(fp32_array - recovered)
        rel_error = abs_error / (np.abs(fp32_array) + 1e-10) * 100
        
        return {
            'max_abs_error': np.max(abs_error),
            'mean_abs_error': np.mean(abs_error),
            'max_rel_error': np.max(rel_error),
            'mean_rel_error': np.mean(rel_error),
        }

# Demonstration
print("=" * 60)
print("INT8 Quantization Analysis")
print("=" * 60)

quantizer = INT8Quantizer(scale=127)

# Test various values
test_values = np.array([1.0, 0.56, 0.5, 0.1, 0.01, 0.001, -0.5, -1.0])

print("\nQuantization Table:")
print(f"{'FP32':>8} | {'INT8':>6} | {'Recovered':>10} | {'Abs Err':>10} | {'Rel Err':>8}")
print("-" * 60)

for val in test_values:
    int8_val = quantizer.quantize(np.array([val]))[0]
    recovered = quantizer.dequantize(np.array([int8_val]))[0]
    abs_err = abs(val - recovered)
    rel_err = abs_err / (abs(val) + 1e-10) * 100
    print(f"{val:8.4f} | {int8_val:6d} | {recovered:10.6f} | {abs_err:10.6f} | {rel_err:7.2f}%")

# Error distribution analysis
print("\n" + "-" * 60)
print("Error Distribution (1000 random values)")
print("-" * 60)

np.random.seed(42)
random_values = np.random.uniform(-1, 1, 1000)
error_stats = quantizer.analyze_error(random_values)

print(f"  Max Abs Error  : {error_stats['max_abs_error']:.6f}")
print(f"  Mean Abs Error : {error_stats['mean_abs_error']:.6f}")
print(f"  Max Rel Error  : {error_stats['max_rel_error']:.2f}%")
print(f"  Mean Rel Error : {error_stats['mean_rel_error']:.2f}%")
```

#### Fun INT8 Facts

```python
# INT8 boundary behaviors
int8_facts = {
    "Max positive": 127,
    "Min negative": -128,
    "Total values": 256,
    "Zero representation": 0,
    "Overflow example": "127 + 1 = -128 (wraparound)",
    "Symmetric range": "-127 to 127 (often used for quantization)",
}

print("\nInteresting INT8 Facts:")
for fact, value in int8_facts.items():
    print(f"  {fact:20s}: {value}")

# Demonstrate wraparound
print("\nWraparound Demonstration:")
a = np.int8(127)
b = np.int8(1)
print(f"  127 + 1 = {np.int8(a + b)}  (overflow!)")
print(f"  -128 - 1 = {np.int8(-128 - 1)}  (underflow!)")
```

---

## **4 — INT4 (4-bit signed integer)**

*   **Memory:** 4 bits per number (2 numbers packed into 1 byte).
*   **Range (for symmetric quantization):** -7 to 7 (using all 16 values for -8 to 7 is asymmetric).
*   **Precision:** Very coarse integer quantization.
*   **Example with Symmetric Quantization (FP32 range ≈ [-1.0, 1.0]):**
    *   **Scale Factor:** 7
    *   **Quantization Formula:** `INT4 = round(FP32 * 7)`
    *   **FP32 value:** 1.0 → **INT4 value:** 7
    *   **FP32 value:** 0.54 → **INT4 value:** round(3.78) = 4
    *   **FP32 value:** -0.5 → **INT4 value:** -4
*   **Memory for 10,000 numbers:** (10,000 numbers * 4 bits) / 8 bits/byte = 5,000 bytes = **~4.88 KiB** (12.5% of FP32).

---

### 🔬 INT4 Deep Analysis

#### The 16-Value Problem

```
┌─────────────────────────────────────────────────────────┐
│  INT4 Only Has 16 Distinct Values!                      │
├─────────────────────────────────────────────────────────┤
│  Symmetric: -7, -6, -5, -4, -3, -2, -1, 0,              │
│              1,  2,  3,  4,  5,  6,  7  (15 values)     │
│  Full range: -8 to 7 (16 values, asymmetric)            │
├─────────────────────────────────────────────────────────┤
│  Quantization step: 1/7 ≈ 0.143                         │
│  Max quantization error: ±0.0715 (7.15%)                │
└─────────────────────────────────────────────────────────┘
```

#### Error Comparison Across Formats

| Format | Distinct Values | Min Step | Max Error | Typical Use |
|--------|----------------|----------|-----------|-------------|
| FP32 | ~4 billion | ~10⁻⁷ | Negligible | Training |
| FP16 | 65,536 | ~10⁻⁴ | Small | Inference |
| INT8 | 256 | 0.0079 | Medium | Deployment |
| **INT4** | **16** | **0.143** | **Large** | **Research** |

---

### 🐍 INT4 Python Code

```python
import numpy as np

class INT4Quantizer:
    """INT4 Symmetric Quantization with bit packing"""
    
    def __init__(self, scale=7):
        self.scale = scale
    
    def quantize(self, fp32_array):
        """FP32 → INT4"""
        int4_array = np.round(fp32_array * self.scale)
        int4_array = np.clip(int4_array, -7, 7)
        return int4_array.astype(np.int8)  # Store as int8, pack later
    
    def dequantize(self, int4_array):
        """INT4 → FP32"""
        return int4_array.astype(np.float32) / self.scale
    
    def pack_two_per_byte(self, int4_array):
        """Pack two INT4 values into one byte"""
        assert len(int4_array) % 2 == 0, "Array length must be even"
        
        # Shift first value to upper 4 bits, second stays in lower 4 bits
        # Add 8 to make values 0-15 for packing
        upper = (int4_array[0::2] + 8).astype(np.uint8) << 4
        lower = (int4_array[1::2] + 8).astype(np.uint8)
        
        return upper | lower
    
    def unpack_from_bytes(self, packed_array):
        """Unpack bytes back to INT4 values"""
        upper = (packed_array >> 4).astype(np.int8) - 8
        lower = (packed_array & 0x0F).astype(np.int8) - 8
        
        result = np.empty(len(packed_array) * 2, dtype=np.int8)
        result[0::2] = upper
        result[1::2] = lower
        
        return result

# Demonstration
print("=" * 60)
print("INT4 Quantization Analysis")
print("=" * 60)

quantizer = INT4Quantizer(scale=7)

# Test various values
test_values = np.array([1.0, 0.54, 0.5, 0.25, 0.14, 0.1, -0.5, -1.0])

print("\nQuantization Table:")
print(f"{'FP32':>8} | {'INT4':>6} | {'Recovered':>10} | {'Abs Err':>10} | {'Rel Err':>8}")
print("-" * 60)

for val in test_values:
    int4_val = quantizer.quantize(np.array([val]))[0]
    recovered = quantizer.dequantize(np.array([int4_val]))[0]
    abs_err = abs(val - recovered)
    rel_err = abs_err / (abs(val) + 1e-10) * 100
    print(f"{val:8.4f} | {int4_val:6d} | {recovered:10.6f} | {abs_err:10.6f} | {rel_err:7.2f}%")

# Bit packing demonstration
print("\n" + "-" * 60)
print("Bit Packing Demonstration")
print("-" * 60)

sample_int4 = np.array([7, -4, 3, 0, -7, 5, 2, -1])
packed = quantizer.pack_two_per_byte(sample_int4)

print(f"  Original INT4 values: {sample_int4.tolist()}")
print(f"  Packed bytes (hex)  : {[hex(b) for b in packed]}")
print(f"  Memory: {len(sample_int4)} INT4 → {len(packed)} bytes")
print(f"  Compression: {len(packed) / len(sample_int4) * 100:.0f}% of unpacked")

# Unpack and verify
unpacked = quantizer.unpack_from_bytes(packed)
print(f"  Unpacked matches: {np.array_equal(sample_int4, unpacked)}")

# Error distribution
print("\n" + "-" * 60)
print("Error Distribution (1000 random values)")
print("-" * 60)

np.random.seed(42)
random_values = np.random.uniform(-1, 1, 1000)
int4_values = quantizer.quantize(random_values)
recovered = quantizer.dequantize(int4_values)

abs_errors = np.abs(random_values - recovered)
rel_errors = abs_errors / (np.abs(random_values) + 1e-10) * 100

print(f"  Max Abs Error  : {np.max(abs_errors):.6f}")
print(f"  Mean Abs Error : {np.mean(abs_errors):.6f}")
print(f"  Max Rel Error  : {np.max(rel_errors):.2f}%")
print(f"  Mean Rel Error : {np.mean(rel_errors):.2f}%")

# Compare all formats
print("\n" + "=" * 60)
print("Format Comparison (same 1000 random values)")
print("=" * 60)

formats = {
    'FP32': (4, lambda x: x),
    'FP16': (2, lambda x: x.astype(np.float16).astype(np.float32)),
    'INT8': (1, lambda x: quantizer.INT8Quantizer().dequantize(
             quantizer.INT8Quantizer().quantize(x))),
    'INT4': (0.5, lambda x: quantizer.dequantize(quantizer.quantize(x))),
}

for name, (bytes_per_num, transform) in formats.items():
    if name == 'FP32':
        recovered = random_values
    elif name == 'FP16':
        recovered = random_values.astype(np.float16).astype(np.float32)
    elif name == 'INT8':
        q = INT8Quantizer(127)
        recovered = q.dequantize(q.quantize(random_values))
    else:  # INT4
        recovered = quantizer.dequantize(quantizer.quantize(random_values))
    
    mse = np.mean((random_values - recovered) ** 2)
    print(f"  {name:6s}: MSE = {mse:.8f}, Memory = {bytes_per_num} bytes/num")
```

---

## **Summary Table**

| Type | Bits | Bytes/Num | ~Range | Example (1.0 → Quantized) | Memory for 10k nums |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FP32** | 32 | 4 | ±3.4e38 | 1.0 → 0x3F800000 | ~39.1 KiB |
| **FP16** | 16 | 2 | ±6.6e4 | 1.0 → 0x3C00 | ~19.5 KiB |
| **INT8** | 8 | 1 | -128 … 127 | 1.0 → 127 (scaled) | ~9.8 KiB |
| **INT4** | 4 | 0.5 | -7 … 7 (sym) | 1.0 → 7 (scaled) | ~4.9 KiB |

---

## **Visual Memory Comparison**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Memory Efficiency Comparison                 │
├─────────────────────────────────────────────────────────────────┤
│  FP32  ████████████████████████████████████████  100%           │
│  FP16  ████████████████████                      50%            │
│  INT8  ██████████                                25%            │
│  INT4  ██████                                    12.5%          │
└─────────────────────────────────────────────────────────────────┘
```

---

## **Observations:**

*   **FP16** offers a direct 50% memory saving over FP32 while maintaining a floating-point format, suitable for many inference tasks.
*   **INT8** requires quantization (scale/zero point) to map floats to integers, achieving 75% memory reduction. It is the most common format for efficient deployment.
*   **INT4** is an aggressive 87.5% compression, often leading to noticeable accuracy degradation. It is used in research or for highly compressed, non-critical models.

---

## **🎯 Format Selection Guide**

| Scenario | Recommended Format | Why |
|----------|-------------------|-----|
| **Model Training** | FP32 | Maximum precision, stable gradients |
| **Mixed Precision Training** | FP16 + FP32 | Speed + stability (master weights in FP32) |
| **GPU Inference** | FP16 | Native hardware support, good accuracy |
| **Edge Deployment** | INT8 | Wide hardware support, 75% memory saving |
| **Mobile/Embedded** | INT8 | Power efficiency, small model size |
| **Extreme Compression** | INT4 | Research, very large models, accuracy trade-off |
| **Financial Calculations** | Decimal/Fixed | Exact decimal representation required |
| **Scientific Computing** | FP64 | Double precision for numerical stability |

