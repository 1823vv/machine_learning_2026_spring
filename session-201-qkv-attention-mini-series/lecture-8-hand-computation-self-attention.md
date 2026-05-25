# Self Attention - Hand Computation Example

---

## Setup

For clarity, we use extremely small dimensions:

- Sequence length: $n = 3$ tokens
- Model dimension: $d_{\text{model}} = 4$
- Key/Query dimension: $d_k = 2$
- Value dimension: $d_v = 2$

#### Input Embeddings

Let the input token embeddings be:

| Token | Embedding (4D) |
|-------|----------------|
| I     | $x_0 = [1.0, 0.0, 0.0, 0.0]$ |
| love  | $x_1 = [0.0, 1.0, 0.0, 0.0]$ |
| Shanghai    | $x_2 = [0.0, 0.0, 1.0, 0.0]$ |

In matrix form:

$$
X = \begin{bmatrix}
1.0 & 0.0 & 0.0 & 0.0 \\
0.0 & 1.0 & 0.0 & 0.0 \\
0.0 & 0.0 & 1.0 & 0.0
\end{bmatrix}
$$

#### Learned Weight Matrices

For this example, we define simplified weight matrices:

**Query weights** $W_Q \in \mathbb{R}^{4 \times 2}$:

$$
W_Q = \begin{bmatrix}
0.5 & 0.1 \\
0.2 & 0.6 \\
0.3 & 0.4 \\
0.1 & 0.2
\end{bmatrix}
$$

**Key weights** $W_K \in \mathbb{R}^{4 \times 2}$:

$$
W_K = \begin{bmatrix}
0.4 & 0.2 \\
0.1 & 0.5 \\
0.6 & 0.3 \\
0.2 & 0.1
\end{bmatrix}
$$

**Value weights** $W_V \in \mathbb{R}^{4 \times 2}$:

$$
W_V = \begin{bmatrix}
0.3 & 0.4 \\
0.5 & 0.2 \\
0.2 & 0.6 \\
0.1 & 0.3
\end{bmatrix}
$$

---

## Step 1: Compute Q, K, V

#### Computing Queries

$$Q = X W_Q$$

For each token:

**Token "I" (row 0 of X):**

$$
q_0 = [1.0, 0.0, 0.0, 0.0] \quad W_Q = [0.5, 0.1]
$$

**Token "love" (row 1 of X):**

$$
q_1 = [0.0, 1.0, 0.0, 0.0]  \quad W_Q = [0.2, 0.6]
$$

**Token "Shanghai" (row 2 of X):**

$$
q_2 = [0.0, 0.0, 1.0, 0.0]  \quad W_Q = [0.3, 0.4]
$$

So:

$$
Q = \begin{bmatrix}
0.5 & 0.1 \\
0.2 & 0.6 \\
0.3 & 0.4
\end{bmatrix}
$$

#### Computing Keys

$$K = X W_K$$

$$
k_0 = [0.4, 0.2] \\
k_1 = [0.1, 0.5] \\
k_2 = [0.6, 0.3]
$$

$$
K = \begin{bmatrix}
0.4 & 0.2 \\
0.1 & 0.5 \\
0.6 & 0.3
\end{bmatrix}
$$

#### Computing Values

$$V = X W_V$$

$$
v_0 = [0.3, 0.4] \\
v_1 = [0.5, 0.2] \\
v_2 = [0.2, 0.6]
$$

$$
V = \begin{bmatrix}
0.3 & 0.4 \\
0.5 & 0.2 \\
0.2 & 0.6
\end{bmatrix}
$$

---

## Step 2: Compute Attention Scores

#### The Score Matrix

$$\text{Scores} = Q K^T$$

This gives us a $3 \times 3$ matrix where entry $(i, j)$ is $q_i \cdot k_j$.

#### Computing Each Entry

**Row 0 (Query for "I"):**

$$
\begin{aligned}
s_{00} &= q_0 \cdot k_0 = [0.5, 0.1] \cdot [0.4, 0.2] = 0.5 \times 0.4 + 0.1 \times 0.2 = 0.20 + 0.02 = 0.22 \\
s_{01} &= q_0 \cdot k_1 = [0.5, 0.1] \cdot [0.1, 0.5] = 0.5 \times 0.1 + 0.1 \times 0.5 = 0.05 + 0.05 = 0.10 \\
s_{02} &= q_0 \cdot k_2 = [0.5, 0.1] \cdot [0.6, 0.3] = 0.5 \times 0.6 + 0.1 \times 0.3 = 0.30 + 0.03 = 0.33
\end{aligned}
$$

**Row 1 (Query for "love"):**

$$
\begin{aligned}
s_{10} &= q_1 \cdot k_0 = [0.2, 0.6] \cdot [0.4, 0.2] = 0.08 + 0.12 = 0.20 \\
s_{11} &= q_1 \cdot k_1 = [0.2, 0.6] \cdot [0.1, 0.5] = 0.02 + 0.30 = 0.32 \\
s_{12} &= q_1 \cdot k_2 = [0.2, 0.6] \cdot [0.6, 0.3] = 0.12 + 0.18 = 0.30
\end{aligned}
$$

**Row 2 (Query for "Shanghai"):**

$$
\begin{aligned}
s_{20} &= q_2 \cdot k_0 = [0.3, 0.4] \cdot [0.4, 0.2] = 0.12 + 0.08 = 0.20 \\
s_{21} &= q_2 \cdot k_1 = [0.3, 0.4] \cdot [0.1, 0.5] = 0.03 + 0.20 = 0.23 \\
s_{22} &= q_2 \cdot k_2 = [0.3, 0.4] \cdot [0.6, 0.3] = 0.18 + 0.12 = 0.30
\end{aligned}
$$

#### Complete Score Matrix

$$
\text{Scores} = \begin{bmatrix}
0.22 & 0.10 & 0.33 \\
0.20 & 0.32 & 0.30 \\
0.20 & 0.23 & 0.30
\end{bmatrix}
$$

---

##  Step 3: Scale the Scores

With $d_k = 2$, we have $\sqrt{d_k} = \sqrt{2} \approx 1.414$.

$$
\text{Scaled Scores} = \frac{\text{Scores}}{\sqrt{d_k}} = \frac{1}{1.414} \cdot \text{Scores}
$$

$$
\text{Scaled Scores} \approx \begin{bmatrix}
0.156 & 0.071 & 0.233 \\
0.141 & 0.226 & 0.212 \\
0.141 & 0.163 & 0.212
\end{bmatrix}
$$

---

## Step 4: Apply Softmax

For each row, compute (using scaled scores $s_{ij}/\sqrt{d_k}$):

$$
\alpha_{ij} = \frac{\exp(s_{ij}/\sqrt{d_k})}{\sum_{j'=0}^{n-1} \exp(s_{ij'}/\sqrt{d_k})}
$$

#### Row 0 (for token "I"):

$$
\begin{aligned}
\exp(0.156) &\approx 1.169 \\
\exp(0.071) &\approx 1.074 \\
\exp(0.233) &\approx 1.262 \\
\text{Sum} &= 1.169 + 1.074 + 1.262 = 3.505
\end{aligned}
$$

$$
\alpha_{00} = \frac{1.169}{3.505} \approx 0.334 
$$

$$
\alpha_{01} = \frac{1.074}{3.505} \approx 0.306
$$

$$
\alpha_{02} = \frac{1.262}{3.505} \approx 0.360
$$

#### Row 1 (for token "love"):

$$
\begin{aligned}
\exp(0.141) &\approx 1.151 \\
\exp(0.226) &\approx 1.254 \\
\exp(0.212) &\approx 1.236 \\
\text{Sum} &= 3.641
\end{aligned}
$$

$$
\alpha_{10} \approx 0.316, \quad \alpha_{11} \approx 0.344, \quad \alpha_{12} \approx 0.340
$$

#### Row 2 (for token "Shanghai"):

$$
\begin{aligned}
\exp(0.141) &\approx 1.151 \\
\exp(0.163) &\approx 1.177 \\
\exp(0.212) &\approx 1.236 \\
\text{Sum} &= 3.564
\end{aligned}
$$

$$
\alpha_{20} \approx 0.323, \quad \alpha_{21} \approx 0.330, \quad \alpha_{22} \approx 0.347
$$

#### Complete Attention Weight Matrix

$$
A = \begin{bmatrix}
0.334 & 0.306 & 0.360 \\
0.316 & 0.344 & 0.340 \\
0.323 & 0.330 & 0.347
\end{bmatrix}
$$

**Verify**: Each row sums to 1.0 

---

## Step 5: Compute Output

$$
\text{Output} = A V
$$

#### For Token "I" (Row 0):

$$
\begin{aligned}
\text{output}_0 &= 0.334 \cdot [0.3, 0.4] + 0.306 \cdot [0.5, 0.2] + 0.360 \cdot [0.2, 0.6] \\
&= [0.100, 0.134] + [0.153, 0.061] + [0.072, 0.216] \\
&= [0.325, 0.411]
\end{aligned}
$$

#### For Token "love" (Row 1):

$$
\begin{aligned}
\text{output}_1 &= 0.316 \cdot [0.3, 0.4] + 0.344 \cdot [0.5, 0.2] + 0.340 \cdot [0.2, 0.6] \\
&= [0.095, 0.126] + [0.172, 0.069] + [0.068, 0.204] \\
&= [0.335, 0.399]
\end{aligned}
$$

#### For Token "Shanghai" (Row 2):

$$
\begin{aligned}
\text{output}_2 &= 0.323 \cdot [0.3, 0.4] + 0.330 \cdot [0.5, 0.2] + 0.347 \cdot [0.2, 0.6] \\
&= [0.097, 0.129] + [0.165, 0.066] + [0.069, 0.208] \\
&= [0.331, 0.403]
\end{aligned}
$$

#### Final Output Matrix

$$
\text{Output} = \begin{bmatrix}
0.325 & 0.411 \\
0.335 & 0.399 \\
0.331 & 0.403
\end{bmatrix}
$$

---

## Interpretation


Each output is a **weighted blend** of all value vectors:

- **Output for "I" (row 0)**: 33.4% from "I", 30.6% from "love", 36.0% from "Shanghai"
- **Output for "love" (row 1)**: 31.6% from "I", 34.4% from "love", 34.0% from "Shanghai"
- **Output for "Shanghai" (row 2)**: 32.3% from "I", 33.0% from "love", 34.7% from "Shanghai"

