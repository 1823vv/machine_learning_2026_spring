## Question: Data Augmentation as Label-Preserving Transformation

Data augmentation creates new training examples by transforming existing data.

1. Write the general data-augmentation idea as a formula:

   $$
   x' = T(x)
   $$

2. What important condition must be true about the label after transformation?
3. For an image-classification task, list four possible transformations that often preserve the label.
4. For each transformation, give one example where it is safe and one example where it might not be safe.
5. Draw a small diagram showing one original image going through several transformations to create several training examples.
6. Explain how data augmentation can be understood as regularization.
7. Explain the phrase: “augmentation teaches invariance.”

## Question: Augmentation for Text

For a sentiment-classification task, consider these augmentations:

- synonym replacement,
- back translation,
- random deletion.

1. Explain why text augmentation is more delicate than image augmentation.
2. Give one synonym-replacement example that preserves meaning.
3. Give one synonym-replacement example that changes meaning or label.
4. Explain why back translation can preserve semantics while changing surface form.
