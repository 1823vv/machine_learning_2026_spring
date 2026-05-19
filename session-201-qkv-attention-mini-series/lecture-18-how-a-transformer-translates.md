# How a Transformer Translates

![](./img-4/transformer_decoding_2.gif)

We examine how a Transformer generates a translation **one token at a time**.

> English: “I visited Shanghai today”
> French: “J’ai visité Shanghai aujourd’hui”

The focus is not on linguistic rules, but on the **mechanical process**:

> At each step, the decoder produces one token by querying a fixed encoder memory.

---

## 1. Encoder: Build Once, Use Repeatedly

Input sequence:

> I / visited / Shanghai / today

After embedding:

$$
X = (x_1, x_2, x_3, x_4)
$$

The encoder produces contextual representations:

$$
H_{\text{enc}} = (h_1, h_2, h_3, h_4)
$$

These are used as:

$$
K = H_{\text{enc}}, \quad V = H_{\text{enc}}
$$

Interpretation:

* $K$ defines how each position can be matched
* $V$ defines what information each position contains

This memory is fixed for the entire decoding process.

---

## 2. Decoder: Generate One Token at a Time

The decoder produces the output sequence incrementally:

$$
y_1, y_2, y_3, \dots
$$

At step $t$, it has already generated:

$$
(y_1, \dots, y_{t-1})
$$

These tokens are embedded and transformed into:

$$
X^{(t)}_{\text{dec}}
$$

From this, the query is formed:

$$
Q^{(t)} = X^{(t)}_{\text{dec}} W_Q
$$

Then cross-attention retrieves from encoder memory:

$$
Z^{(t)} = \text{softmax}\left(\frac{Q^{(t)} K^T}{\sqrt{d_k}}\right) V
$$

Finally, the model outputs the next token $y_t$.

---

## 3. Step-by-Step Generation

We now trace the decoding process token by token.

---

#### Step 1: Generate the First Token

**Input to decoder:**

$$
\langle SOS \rangle
$$

Query:

$$
Q^{(1)} = X^{(1)}_{\text{dec}} W_Q
$$

Cross-attention:

$$
Z^{(1)} = \text{softmax}\left(\frac{Q^{(1)} K^T}{\sqrt{d_k}}\right) V
$$

Output:

$$
y_1 = \text{“J’”}
$$

Interpretation:

* The first query attends broadly to the encoder memory
* It extracts a global signal to initiate the target sentence

---

#### Step 2: Generate the Second Token

**Input to decoder:**

$$
\langle SOS \rangle, \; \text{“J’”}
$$

Query:

$$
Q^{(2)} = X^{(2)}_{\text{dec}} W_Q
$$

Cross-attention:

$$
Z^{(2)} = \text{softmax}\left(\frac{Q^{(2)} K^T}{\sqrt{d_k}}\right) V
$$

Output:

$$
y_2 = \text{“ai”}
$$

Interpretation:

* The query now depends on the previous output
* Attention shifts toward representations associated with the main action

---

#### Step 3: Generate the Third Token

**Input to decoder:**

$$
\langle SOS \rangle, \; \text{“J’”}, \; \text{“ai”}
$$

Query:

$$
Q^{(3)} = X^{(3)}_{\text{dec}} W_Q
$$

Cross-attention:

$$
Z^{(3)} = \text{softmax}\left(\frac{Q^{(3)} K^T}{\sqrt{d_k}}\right) V
$$

Output:

$$
y_3 = \text{“visité”}
$$

Interpretation:

* The query becomes more focused
* It retrieves the core semantic content of the action

---

#### Step 4: Generate the Fourth Token

**Input to decoder:**

$$
\langle SOS \rangle, \; \text{“J’”}, \; \text{“ai”}, \; \text{“visité”}
$$

Query:

$$
Q^{(4)} = X^{(4)}_{\text{dec}} W_Q
$$

Cross-attention:

$$
Z^{(4)} = \text{softmax}\left(\frac{Q^{(4)} K^T}{\sqrt{d_k}}\right) V
$$

Output:

$$
y_4 = \text{“Shanghai”}
$$

Interpretation:

* The query now targets a specific entity
* Attention concentrates on the corresponding position in the encoder

---

#### Step 5: Generate the Final Token

**Input to decoder:**

$$
\langle SOS \rangle, \; \text{“J’”}, \; \text{“ai”}, \; \text{“visité”}, \; \text{“Shanghai”}
$$

Query:

$$
Q^{(5)} = X^{(5)}_{\text{dec}} W_Q
$$

Cross-attention:

$$
Z^{(5)} = \text{softmax}\left(\frac{Q^{(5)} K^T}{\sqrt{d_k}}\right) V
$$

Output:

$$
y_5 = \text{“aujourd’hui”}
$$

Interpretation:

* The query shifts to remaining contextual information
* It retrieves the temporal component

---

## 4. What Remains Constant

Across all steps:

* Encoder memory $(K, V)$ does not change
* The attention formula does not change
* Model parameters do not change

Only the query evolves:

$$
Q^{(1)}, Q^{(2)}, Q^{(3)}, \dots
$$

---

## 5. Final Interpretation

The translation process can be summarized as:

$$
Q^{(t)} \rightarrow \text{retrieve from } (K, V) \rightarrow y_t
$$

repeated for each token.

Thus:

> A Transformer translates by generating one token at a time, where each token is produced by querying a fixed memory of the source sentence.

The sequence emerges not from a single transformation, but from a chain of retrieval steps.
