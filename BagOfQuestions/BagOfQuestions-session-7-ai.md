## Question: Batch Normalization versus Z-Score Normalization

In machine learning, we use normalization tricks to make training faster and more stable. We are familiar with standard **Z-score normalization** used on input features, as well as **Batch Normalization** used inside neural networks.

1. Write the standard Z-score normalization formula used to scale an input feature $x$ using its dataset mean $\mu$ and standard deviation $\sigma$.
2. In a neural network, *where* exactly is Batch Normalization applied compared to standard Z-score feature normalization?
3. Batch Normalization standardizes intermediate activations $z_1, z_2, \dots, z_B$ across a mini-batch to get $\hat{z}_i$. Write the formula for $\hat{z}_i$ using the mini-batch mean $\mu_B$, mini-batch variance $\sigma_B^2$, and a small safety constant $\epsilon$.
4. After standardizing the values to get $\hat{z}_i$, Batch Normalization applies two learnable parameters ($\gamma$ and $\beta$) to calculate the final output: $\tilde{z}_i = \gamma \hat{z}_i + \beta$. What happens to this output if the network learns $\gamma = \sqrt{\sigma_B^2 + \epsilon}$ and $\beta = \mu_B$?
5. Why does Batch Normalization need these extra learnable parameters ($\gamma$ and $\beta$), whereas standard input Z-score normalization does not?