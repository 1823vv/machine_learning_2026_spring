## Question: Data Augmentation as Label-Preserving Transformation

Data augmentation creates new training examples by applying transformations that should not change the label. For example, a small shift of an MNIST digit usually keeps the same digit label.

1. Define data augmentation in your own words.
2. Give three label-preserving transformations for image classification.
3. Give one transformation that might not be label-preserving for digit recognition.
4. Explain why augmentation can improve generalization.
5. Draw a simple diagram showing one original image becoming several augmented training examples.
6. Why should augmentation usually be applied to training data, not to the final test labels or test-set definition?

## Question: Augmentation for Text

For text classification, augmentation is more delicate than for images because small word changes can change meaning.

1. Give one possible label-preserving text augmentation.
2. Give one text transformation that could accidentally change the label.
3. Why is augmentation domain-dependent?
4. Explain why human judgment may be needed to decide whether an augmentation is valid.
5. Compare image augmentation and text augmentation in one short paragraph.
