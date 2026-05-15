## Question: Batch Normalization Formulas

Batch normalization standardizes intermediate activations within a mini-batch.

For a mini-batch of pre-activation values $z_1,z_2,\dots,z_m$:

1. Write the batch mean formula $\mu_B$.
2. Write the batch variance formula $\sigma_B^2$.
3. Write the normalized activation formula $\hat{z}_i$ using $\epsilon$.
4. Write the learnable scale-and-shift formula using $\gamma$ and $\beta$.
5. Explain why $\epsilon$ is included.
6. Explain why batch normalization includes learnable parameters $\gamma$ and $\beta$ after normalization.
7. Draw a layer block diagram:

   ```text
   Linear -> BatchNorm -> Activation
   ```

   and label where $z$, $\hat{z}$, and the final transformed output appear.

## Question: BatchNorm and Standardization

1. Write the standardization formula for input features.
2. Explain how batch normalization is similar to standardization.
3. Explain how batch normalization is different from standardization.
4. Why does batch normalization use mini-batch statistics during training?
