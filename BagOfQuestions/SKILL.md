# BagOfQuestions Skill: Generating Independent Exam-Ready Question Files

## Purpose of This Skill

This skill explains how to create, revise, and maintain the files named:

```text
BagOfQuestions/BagOfQuestions-session-N-XX.md
```

where:

- `N` is a session number. Sessions `1` to `7` are the deeper main sessions; `N >= 8` files can also be generated with easier extra-topic questions.
- `XX` is a two-letter suffix such as `aa`, `ab`, `ac`, ..., possibly up to `zz`.
- Each file contains one or more Markdown sections beginning with `## Question`.

The purpose of these files is to build a large bank of final-exam-style questions for the course **ML01 Machine Learning**. The instructor may later choose only one file, or may randomly select individual `## Question` blocks from many files, and place those questions into an exam paper. Therefore, every question must be written so that it is clear, independent, and useful even when removed from its original file.

This skill is intentionally long and explicit. Future AI agents should follow it carefully instead of guessing the instructor's preferences from only one or two examples. The instructor is expected to keep dynamically improving this folder by editing, removing, renaming, or replacing Markdown files. Therefore, treat the current contents of `BagOfQuestions/` as the instructor's latest preference unless the user explicitly says otherwise.

---

## Instructor Edits Are Authoritative

The instructor may dynamically improve the `BagOfQuestions/` folder at any time. He may remove files, rewrite files, rename files, shorten questions, expand questions, or change the style of generated question sets. Future agents must respect those edits.

Operational rules:

1. Treat every existing `BagOfQuestions/BagOfQuestions-session-N.md` and `BagOfQuestions/BagOfQuestions-session-N-XX.md` file as already reflecting the instructor's good current preference, unless the user explicitly identifies a problem in a specific file.
2. Do not "normalize" or rewrite existing files just because they differ from older examples in this skill.
3. When generating new files, imitate the strongest current examples in the relevant part of `BagOfQuestions/`, not stale wording from old prompts.
4. If the instructor removed a file or removed a question style, do not recreate it unless explicitly asked.
5. If there is a conflict between this skill and the current contents of a file the instructor has clearly edited, prefer the user's latest instruction and the current file style.
6. Keep student-facing generated questions free of internal planning details such as score proportions, file-management notes, or statements about how the instructor might later adjust the question bank.

---

## Core Principle: Every `## Question` Must Stand Alone

The most important rule is:

> Every `## Question` section must be independently understandable when copied into a final exam paper.

A question should not depend on a previous question in the same file unless the dependency is fully restated inside the same `## Question` block. The instructor may randomly choose a single `## Question` section from a file and use it alone.

### What Standalone Means

A standalone question must include enough background for students to know:

1. The topic being tested.
2. The setting or implementation being referenced.
3. The notation or variables used in formulas.
4. The expected type of answer: explanation, drawing, formula, code, debugging, parameter counting, etc.
5. Any necessary shapes, class names, function names, or conventions.

### Good Standalone Openings

Use openings like these:

```markdown
## Question: Linear Regression from Scratch

We are implementing linear regression `class MyOwnLinearRegression` from scratch with NumPy. The model receives a feature matrix `X` with shape `(n_samples, n_features)` and a target vector `y` with shape `(n_samples,)`.
```

```markdown
## Question: One Gradient Descent Step by Hand in Logistic Regression

Consider binary logistic regression with two features. For one training example, let `x = [2, -1]`, `y = 1`, weights `w = [0.1, -0.2]`, and bias `b = 0`.
```

```markdown
## Question: Dropout and Expected Activation Magnitude

In our neural network from scratch implementation, dropout randomly drops activations during training. We use inverted dropout, which scales the surviving activations during training so that no special scaling is needed during inference.
```

```markdown
## Question: MNIST Network Parameter Counting

In our own Neural Network From Scratch implementation with NumPy for MNIST digit recognition, suppose the architecture is `Dense(784, 64) -> ReLU -> Dense(64, 32) -> ReLU -> Dense(32, 10)`. Under the course row-vector convention, `Dense(d_in, d_out)` has $W \in \mathbb{R}^{d_{in} \times d_{out}}$ and $b \in \mathbb{R}^{1 \times d_{out}}$.
```

```markdown
## Question: GitHub CI/CD/Actions/Release Project

In our GitHub CI/CD/Actions/Release project, a trained model file such as `linear_model.txt` is generated by code and then handled by the workflow.
```

### Bad Non-Standalone Openings

Avoid these because they require missing context:

```markdown
## Question
Explain why it works.
```

```markdown
## Question
Continue from the previous question.
```

```markdown
## Question
What is the formula?
```

```markdown
## Question
Fix the code below.
```

If code is shown, say what the code is supposed to implement. If a formula is requested, define the symbols. If a diagram is requested, state what should appear in the diagram.


---

## Mandatory Notation Source

Before writing or revising any `BagOfQuestions` exam question for Sessions 1--7, explicitly consult:

```text
session-0/lecture-0-notation-for-session-1-to-session-7.md
```

Follow that notation in both formulas and examples. In particular:

1. Use the **row-vector convention** used by the course and by modern ML frameworks. For one example, write linear layers as

   $$
   y = xW + b,
   $$

   where $x \in \mathbb{R}^{1 \times d_{in}}$, $W \in \mathbb{R}^{d_{in} \times d_{out}}$, $b \in \mathbb{R}^{1 \times d_{out}}$, and $y \in \mathbb{R}^{1 \times d_{out}}$.

2. For neural-network layer $l$, use

   $$
   z^{(l)} = a^{(l-1)} W^{(l)} + b^{(l)},
   $$

   $$
   a^{(l)} = f^{(l)}(z^{(l)}).
   $$

3. For a batch of examples stacked as rows, use

   $$
   Z^{(l)} = A^{(l-1)} W^{(l)} + \mathbf{1} b^{(l)}.
   $$

4. For regression and binary classification batches, prefer $X \in \mathbb{R}^{n \times d}$, $Y \in \mathbb{R}^{n \times 1}$, and $\hat{Y} \in \mathbb{R}^{n \times 1}$. If a code skeleton uses a one-dimensional NumPy array such as `y.shape == (n_samples,)`, say explicitly that this is the implementation shape, while the mathematical convention is $Y \in \mathbb{R}^{n \times 1}$.

5. For gradient-descent updates in math, use the course's left-arrow notation:

   $$
   g = \frac{\partial \mathcal{L}}{\partial W}, \quad W \leftarrow W - \eta g.
   $$

   Do not use `:=` or `\gets` in newly generated math. In Python code blocks, normal assignment with `=` is still correct.

6. For absolute values in formulas, follow the course guideline and use double pipes, for example $\|x\|$, not single-pipe notation.

7. Do not write final-exam question backgrounds that depend on repository file paths or session numbers. File paths are useful for the AI agent while preparing the question, but the exam prompt should usually say something like "In our own Neural Network From Scratch implementation with NumPy for MNIST digit recognition".

For `N >= 8` files, still use this notation when the topic overlaps with Sessions 1--7. If the extra topic introduces its own notation, define that notation clearly inside each standalone question.

---

## Expected File Structure

A typical generated file should look like this:

```markdown
## Question: Descriptive Title

Standalone background paragraph.

1. Sub-question.
2. Sub-question.
3. Sub-question.

## Question: Another Descriptive Title

Standalone background paragraph.

1. Sub-question.
2. Sub-question.
```

### Titles

Prefer descriptive titles after `## Question:`. Examples:

- `## Question: Linear Regression from Scratch`
- `## Question: Debug Linear Regression from Scratch`
- `## Question: Binary Cross-Entropy for Logistic Regression`
- `## Question: Softmax by Hand`
- `## Question: MNIST Network Architecture and Parameter Counting`
- `## Question: Output Layers Depend on the Task`
- `## Question: Training, Validation, and Test Sets`

A plain `## Question` without a title is allowed only when the context is extremely clear, but descriptive titles are strongly preferred.

### Numbered Sub-Questions

Use numbered lists for sub-questions. They are easier to grade and easier to copy into an exam.

Good:

```markdown
1. Write the sigmoid formula.
2. Compute the predicted probability.
3. Write the binary cross-entropy loss for this one example.
4. Explain why this loss becomes large when the model is confidently wrong.
```

Avoid overly long lists. A question can have many parts, but if it reaches 12 or more sub-questions, consider splitting it into two independent `## Question` blocks. The instructor likes coherent development, but not a giant unstructured checklist.

---


### Public-Facing Wording Ban for Generated Question Files

In generated files such as `BagOfQuestions-session-N-XX.md`, especially for `N=5`, `N=6`, and `N=7`, do **not** use misleading meta-words inside the student-facing question text. Avoid words that reveal internal organization or sound like authoring notes rather than exam prompts.

Banned or strongly discouraged in generated question files:

- `story`
- `session`
- `course`
- `lecture`
- `chapter`
- `exam`
- `question bank`
- repository file paths such as `session-4/code-my_nn.py`

Use neutral replacements instead:

- Instead of "story", write "scenario", "setting", or simply describe the task.
- Instead of "in this course", write the mathematical or implementation convention directly.
- Instead of "exam answer", write "written answer".
- Instead of naming a session or file path, describe the concept, model, or implementation students need to reason about.

A useful validation command for upper-session generated files is:

```bash
rg -n -i "\b(story|session|course|exam|lecture|chapter|repository|file path|code-my)\b" BagOfQuestions/BagOfQuestions-session-{5,6,7}-*.md
```

This command should return no matches unless the user explicitly asks for one of those terms. Also avoid administrative wording such as score percentages, assessment policies, no-materials rules, or comments that questions may be adjusted later; generated question files should contain only the student-facing problem statement.

---

## Course Scope and Session Mapping

The main course sessions are:

1. **Session 1:** Linear Regression
2. **Session 2:** Logistic Regression for classification
3. **Session 3:** Neural networks, basic ideas, activations, output layers, logits/probabilities, parameter counting
4. **Session 4:** Neural network from scratch, especially the course's own NumPy neural-network implementation. Use source files to understand the implementation, but avoid exposing file paths in final-exam question wording unless the instructor explicitly asks for them.
5. **Session 5:** Neural networks, regularization, dropout, optimization, deeper neural-network concepts depending on existing course files
6. **Session 6:** Model selection, validation, cross-validation, train/validation/test split, overfitting/underfitting
7. **Session 7:** Model selection continuation, metrics, practical evaluation, final model choice depending on existing course files

Before generating a new question file, inspect existing files for that session, especially:

- `BagOfQuestions/BagOfQuestions-session-N.md`
- Existing `BagOfQuestions/BagOfQuestions-session-N-XX.md`
- Relevant lecture files under `session-N/`
- Relevant code files under `session-N/`
- `Readme.md` for exam expectations

Do not assume a topic belongs to a session if the course files clearly place it somewhere else.

### Files for `N >= 8`: Extra-Topic Questions Are Easier

`Readme.md` states that the main deeper material is in Sessions 1--7, while later sessions are extra topics. Therefore, this skill also applies to generated files such as `BagOfQuestions-session-8-XX.md` or `BagOfQuestions-session-N-XX.md` with `N >= 8`, but the expected difficulty should be different:

1. Make `N >= 8` questions easier and more conceptual than the deeper Session 1--7 questions.
2. Prefer clear definitions, intuition, simple diagrams, and small calculations over long derivations.
3. Keep every `## Question` standalone, just as for Sessions 1--7.
4. Still inspect current files and relevant source material before writing new questions.
5. Do **not** write grading proportions, final-assessment rules, no-materials rules, or other administrative details inside generated student-facing question files. Those details may change later and belong in planning documents, not in the question prompt.
6. Do **not** say "this is an easy extra-session question" inside the generated question. Just write an appropriately easy question.


---

## What the Instructor Likes

The instructor's preferences are strong and should guide generation.

### 1. Main Ideas with Drawings or Schemas

The instructor likes questions where students explain an important idea and draw a figure or schema. These should not be too trivial. Avoid asking students to draw something so simple that it becomes meaningless.

Good drawing/schema prompts:

- Draw a neural network architecture for MNIST, showing input dimension, hidden layers, activation layers, logits, and probabilities.
- Draw a decision boundary for logistic regression with two features.
- Draw how training error and validation error change when a model overfits.
- Draw a schema of train/validation/test split and explain what each split is used for.
- Draw dropout during training and inference, showing dropped units and scaling.
- Draw a CI/CD pipeline that generates and releases `linear_model.txt`.

Bad drawing prompts:

- Draw a straight line for linear regression with no further reasoning.
- Draw a dot.
- Draw a sigmoid without asking about saturation, probability interpretation, or classification threshold.

### 2. Important Math Formulas

The instructor likes questions that test essential formulas. These formulas should be central, not obscure.

Important formulas include:

- Linear regression prediction:
  $$\hat{Y} = XW + \mathbf{1}b$$
- Mean squared error:
  $$\mathcal{L}_{\mathrm{MSE}} = \frac{1}{n}\sum_{i=1}^n(\hat y^{(i)}-y^{(i)})^2$$
- Mean absolute error:
  $$\mathcal{L}_{\mathrm{MAE}} = \frac{1}{n}\sum_{i=1}^n\|\hat y^{(i)}-y^{(i)}\|$$
- Sigmoid:
  $$\sigma(z)=\frac{1}{1+e^{-z}}$$
- Binary cross-entropy per example:
  $$\ell^{(i)}= -\left[y^{(i)}\log(\hat y^{(i)})+(1-y^{(i)})\log(1-\hat y^{(i)})\right]$$
- Multiclass softmax:
  $$p_k=\frac{e^{z_k}}{\sum_j e^{z_j}}$$
- Numerically stable softmax:
  $$p_k=\frac{e^{z_k-m}}{\sum_j e^{z_j-m}},\quad m=\max_j z_j$$
- Multiclass cross-entropy per example for correct class $c$:
  $$\ell^{(i)}=-\log(p_c^{(i)})$$
- L1 regularization:
  $$\lambda\sum_j \|W_j\|$$
- L2 regularization:
  $$\lambda\sum_j W_j^2$$
  or sometimes
  $$\frac{\lambda}{2}\sum_j W_j^2$$
  depending on the convention used in the course material.
- Momentum, using the course's update notation:
  $$v \leftarrow \beta v+(1-\beta)g,\quad W \leftarrow W-\eta v$$
  or the equivalent course convention if lecture files use a different one.
- Adam:
  $$m \leftarrow \beta_1m+(1-\beta_1)g$$
  $$v \leftarrow \beta_2v+(1-\beta_2)g^2$$
  $$\hat m=\frac{m}{1-\beta_1^t},\quad \hat v=\frac{v}{1-\beta_2^t}$$
  $$W \leftarrow W-\eta\frac{\hat m}{\sqrt{\hat v}+\epsilon}$$

When asking formulas, define variables if the question is standalone. For example, say that `z_k` is the logit for class `k`, `p_k` is the softmax probability for class `k`, and `c` is the correct class.

### 3. Parameter Counting in Neural Networks

The instructor likes parameter-counting questions. These are especially good for neural-network sessions.

A good parameter-counting question should include:

1. The input dimension.
2. The hidden layer sizes.
3. The output layer size.
4. Whether biases are included.
5. A request for total trainable parameters.
6. A reminder that activation functions such as ReLU usually do not add trainable parameters.

Example:

```markdown
## Question: MNIST Parameter Counting

Consider an MNIST network with input dimension `28 * 28 = 784`, two hidden layers with 10 neurons each, and 10 output neurons. Each dense layer has weights and biases.

1. How many weights are in the first dense layer?
2. How many biases are in the first dense layer?
3. Repeat for the second dense layer.
4. Repeat for the output dense layer.
5. How many trainable parameters are there in total?
6. Does ReLU add trainable parameters? Explain.
```

### 4. Coherent Development Across Sub-Questions

The instructor likes a sequence of related sub-questions that develops a coherent scenario. A good question might move from concept to formula to small calculation to explanation to drawing.

Example structure:

1. Define the setting.
2. Ask for the main formula.
3. Ask for a small numerical computation.
4. Ask for an interpretation.
5. Ask for a diagram or debugging insight.

Do not make the scenario depend on a previous `## Question`. The scenario should be contained within one `## Question` block.

### 5. Code Fill-in-the-Blank Questions

The instructor likes fill-in-the-blank code questions with placeholders such as:

```text
____YOUR_CODE_HERE__1_____
____YOUR_CODE_HERE__2_____
```

Use exactly this style when possible. It is clear and consistent with existing files.

Good prompt phrasing:

```markdown
Fill in the `____YOUR_CODE_HERE__N_____` blanks in the code skeleton below.

Your answer goes here, after the `:`:

- `____YOUR_CODE_HERE__1_____`:
- `____YOUR_CODE_HERE__2_____`:
```

Use code questions mainly for the allowed areas listed below.

---

## Strong Restrictions: What Not to Ask

These restrictions are important.

### Neural-Network Math Is Allowed in Session 3 and Session 4 Question Bags

There is **no blanket restriction** against mathematical questions on neural networks in the Session 3 and Session 4 exam-question bags. It is acceptable to ask students to write, explain, or compute important neural-network math when the question is aligned with lecture/code material and uses the notation guideline from `session-0/lecture-0-notation-for-session-1-to-session-7.md`.

Allowed neural-network math includes, but is not limited to:

- Forward-pass formulas using the course row-vector convention, such as $z^{(l)} = a^{(l-1)}W^{(l)} + b^{(l)}$ and $a^{(l)} = f^{(l)}(z^{(l)})$.
- Shape checks for $A^{(l-1)}$, $W^{(l)}$, $b^{(l)}$, $Z^{(l)}$, and $A^{(l)}$.
- Logits versus probabilities.
- Softmax, the stable softmax trick, and cross-entropy.
- Activation formulas and derivatives if they are central to the course material.
- Parameter counting and architecture diagrams.
- Conceptual or formula-level backpropagation questions if the question is exam-appropriate and does not become a long unsupported derivation.
- Code fill-in-the-blank questions based on the course's own NumPy neural-network implementation. When writing the exam prompt, describe it as the course implementation rather than by file path.
- Code-debugging questions for layer operations, shapes, and updates if this matches existing course files.

Even though math is allowed, keep each `## Question` standalone and fair: define variables, state shapes, use the row-vector convention, and avoid making a single question so long that it becomes impossible to grade.

### No Code Questions on Optimization or Model Selection

Do **not** create code-writing questions for optimization algorithms or model selection.

Avoid code questions like:

- Implement Adam from scratch.
- Implement momentum from scratch.
- Implement cross-validation from scratch.
- Implement grid search from scratch.
- Write code for early stopping.
- Write code to split data into train/validation/test sets.

Allowed non-code questions:

- Write the formula for momentum or Adam.
- Explain the intuition of momentum or Adam.
- Explain training/validation/test split.
- Explain cross-validation conceptually.
- Draw overfitting and underfitting curves.
- Compare model-selection strategies.

### Code Questions Are Limited to Specific Areas

Code questions should be limited to:

1. **Linear Regression from scratch**
   - Especially `fit`
   - Maybe `predict`
   - Matrix multiplication with `np.dot`
   - Gradient update signs
   - Shape debugging

2. **Logistic Regression from scratch**
   - Especially `fit`
   - Maybe `predict`
   - Sigmoid
   - Binary probability threshold
   - Gradient form using `(y_predicted - y)`

3. **Neural Network from scratch**
   - Only based on the course's own NumPy neural-network implementation. Use the source file internally, but do not make the final-exam question depend on a file path.
   - Fill in code around existing architecture, training loop, layers, ReLU, Dense, softmax-cross-entropy, or similar material already present in the course
   - Avoid creating totally new neural-network code that students have not seen

---

## Style Guidelines

### Language

Use English. The course README says English should be used for everything.

### Tone

Use a clear exam-question tone. Be direct but not overly terse. Students should understand what is being asked without needing to guess.

Good:

```markdown
Explain why the sigmoid output can be interpreted as a probability for class 1.
```

Less good:

```markdown
What's going on with sigmoid?
```

### Difficulty

The desired difficulty is final-exam appropriate for an introductory machine-learning course with coding practice. Questions should be deeper than simple memorization but not research-level.

A good question often mixes:

- Conceptual explanation
- Formula recall
- Small numerical calculation
- Code reading or debugging
- Drawing/schema
- Interpretation of model output

### Mathematical Formatting

Use Markdown and LaTeX. Prefer display math for important formulas:

```markdown
$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$
```

Use inline math for short symbols:

```markdown
The target is $y \in \{0,1\}$.
```

### Code Formatting

Use fenced code blocks with language tags when useful:

```markdown
```python
linear_model = np.dot(X, self.weights) + self.bias
```
```

For shape descriptions, use backticks:

```markdown
`X.shape == (50, 4)`
```

### Typos and Grammar

Fix obvious typos when editing. Common typos in older files include:

- `mathmatical` -> `mathematical`
- `recogonization` -> `recognition`
- `represention` -> `representation`
- `Fill the the` -> `Fill in the`
- `From Sratch` -> `From Scratch`

Do not rewrite every old file just for style unless asked. But for files you touch, remove draft markers and make the wording clear.

---

## Question Types to Generate

The bank should contain a mix of question types.

### Type A: Formula Recall with Interpretation

Example:

```markdown
## Question: Binary Cross-Entropy for Logistic Regression

In binary logistic regression, the model outputs a probability $\hat y \in (0,1)$ for class 1, and the true label is $y \in \{0,1\}$.

1. Write the binary cross-entropy loss for one example.
2. What does the loss become when `y = 1`?
3. What does the loss become when `y = 0`?
4. Explain why the loss is large when the model is confidently wrong.
```

### Type B: Small Numerical Calculation

Example:

```markdown
## Question: Softmax by Hand

Consider logits $z=[2,1,0]$ for a three-class classifier.

1. Write the softmax formula.
2. Using $e^2\approx 7.39$, $e^1\approx 2.72$, and $e^0=1$, compute the denominator.
3. Compute the three softmax probabilities approximately.
4. Which class is predicted by argmax?
5. Draw a bar chart of the logits and a bar chart of the probabilities.
```

### Type C: Code Fill-in-the-Blank

Example:

```markdown
## Question: Logistic Regression from Scratch

We are implementing binary logistic regression `class MyOwnLogisticRegression` from scratch with NumPy.

Fill in the `____YOUR_CODE_HERE__N_____` blanks.

```python
linear_model = np.dot(____YOUR_CODE_HERE__1_____, ____YOUR_CODE_HERE__2_____) + self.bias
y_predicted = self.sigmoid(____YOUR_CODE_HERE__3_____)

dw = (1 / n_samples) * np.dot(____YOUR_CODE_HERE__4_____, (y_predicted - y))
db = (1 / n_samples) * np.sum(____YOUR_CODE_HERE__5_____)
```

Your answer goes here, after the `:`:

- `____YOUR_CODE_HERE__1_____`:
- `____YOUR_CODE_HERE__2_____`:
- `____YOUR_CODE_HERE__3_____`:
- `____YOUR_CODE_HERE__4_____`:
- `____YOUR_CODE_HERE__5_____`:
```

### Type D: Debugging Code

Example:

```markdown
## Question: Debug Linear Regression from Scratch

We are implementing `class MyOwnLinearRegression` from scratch with NumPy. A student wrote the following incorrect code:

```python
y_predicted = X * self.weights + self.bias
dw = (2 / n_samples) * X.T * (y_predicted - y)
self.weights = self.weights + self.lr * dw
```

1. Explain why `X * self.weights` is wrong for multiple linear regression.
2. Write the correct prediction line.
3. Explain why `X.T * (y_predicted - y)` is wrong for `dw`.
4. Write the correct `dw` line.
5. Explain why the update should subtract the gradient.
```

### Type E: Drawing or Schema

Example:

```markdown
## Question: Neural Network Schema for MNIST

In our own Neural Network From Scratch implementation with NumPy for MNIST digit recognition, consider the architecture `Dense(784, 64) -> ReLU -> Dense(64, 32) -> ReLU -> Dense(32, 10)`. Under the course row-vector convention, each image is flattened as $x \in \mathbb{R}^{1 \times 784}$.

1. Draw the network architecture.
2. Label the input dimension, hidden dimensions, output logits, and softmax probabilities.
3. Explain why the input dimension is 784.
4. Explain why the output dimension is 10.
5. In your drawing, clearly separate logits from probabilities.
```

### Type F: Compare and Contrast

Example:

```markdown
## Question: Linear Regression versus Logistic Regression

Both linear regression and logistic regression compute a linear score from input features. However, they solve different tasks.

1. Write the prediction formula for linear regression.
2. Write the linear score and sigmoid probability formula for logistic regression.
3. Compare the typical target values for the two models.
4. Compare the typical loss functions.
5. Explain why logistic regression is a classification model even though it contains the word "regression".
```

### Type G: Explain a Trick

Example:

```markdown
## Question: Why Subtract the Maximum in Softmax?

For logits $z_1,\ldots,z_K$, the softmax probability for class $k$ is usually written as

$$
p_k=\frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}.
$$

In code, we often compute `z_shifted = z - np.max(z)` before exponentiation.

1. Write the numerically stable softmax formula.
2. Show algebraically why subtracting the same constant from all logits does not change the softmax probabilities.
3. Explain what numerical problem this trick helps avoid.
4. Does subtracting the maximum change the predicted class from `argmax`? Explain.
```

---

## Session-Specific Guidance

## Session 1: Linear Regression

Topics to emphasize:

- Linear prediction with the course row-vector convention: per example $\hat{y}=xW+b$ and in batch form $\hat{Y}=XW+\mathbf{1}b$
- Multiple linear regression shapes
- MSE and MAE
- Gradient descent intuition
- Linear regression from scratch with NumPy
- `fit` and `predict`
- Dot product versus elementwise multiplication
- Debugging shapes
- CI/CD project references if present in session material

Good question ideas:

1. Fill in `MyOwnLinearRegression.fit`.
2. Debug wrong use of `X * self.weights`.
3. Compare MAE and MSE.
4. Draw residuals and explain error.
5. Explain why gradient descent updates parameters by subtracting gradients.
6. Given a tiny dataset, compute one prediction and one squared error.
7. Explain why `dw` has shape `(n_features,)`.

Avoid:

- Overly advanced linear algebra derivations.
- Closed-form normal equation unless clearly part of the session material.
- Too many trivial line-fitting drawing questions.

### Session 1 Code Skeleton Pattern

Use skeletons like:

```python
class MyOwnLinearRegression:
    def __init__(self, learning_rate=0.0001, n_iters=30000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(____YOUR_CODE_HERE__1_____)
        self.bias = ____YOUR_CODE_HERE__2_____

        for _ in range(self.n_iters):
            y_predicted = np.dot(____YOUR_CODE_HERE__3_____, ____YOUR_CODE_HERE__4_____) + self.bias

            dw = (2 / n_samples) * np.dot(____YOUR_CODE_HERE__5_____, (y_predicted - y))
            db = (2 / n_samples) * np.sum(____YOUR_CODE_HERE__6_____)

            self.weights = self.weights - self.lr * ____YOUR_CODE_HERE__7_____
            self.bias = self.bias - self.lr * ____YOUR_CODE_HERE__8_____
```

Remember that for MSE written without `1/2`, the gradient has `2 / n_samples`. If the course uses a different convention, follow the course file.

---

## Session 2: Logistic Regression

Topics to emphasize:

- Binary classification
- Linear score with the course row-vector convention: per example $z=xW+b$ and in batch form $Z=XW+\mathbf{1}b$
- Sigmoid probability
- Decision threshold, usually `0.5`
- Binary cross-entropy
- Decision boundary for two features
- Logistic regression from scratch with NumPy
- Similarities and differences with linear regression code

Good question ideas:

1. Write sigmoid and binary cross-entropy.
2. Compute sigmoid for a small logit.
3. Interpret probability and threshold.
4. Draw a linear decision boundary in 2D.
5. Fill in logistic regression `fit` and `predict` code.
6. Compare logistic regression and linear regression.
7. Explain why BCE is preferred over MSE for logistic regression.

Avoid:

- Multiclass logistic regression unless the session explicitly includes it.
- Heavy maximum-likelihood derivations unless lecture files support it.
- Code questions beyond the scratch implementation.

### Session 2 Code Skeleton Pattern

Use skeletons like:

```python
class MyOwnLogisticRegression:
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
```

Turn exact lines into fill-in-the-blank questions when useful.

---

## Session 3: Neural Network Concepts

Topics to emphasize:

- What a neural network is conceptually
- Activation functions: ReLU, sigmoid, tanh
- Why nonlinear activation functions matter
- Logits versus probabilities
- Output layer choices by task
- Regression output, binary classification output, multiclass output
- Softmax and cross-entropy
- Parameter counting
- Drawing architectures

Good question ideas:

1. Explain why stacking only linear layers is still linear.
2. Draw ReLU, sigmoid, and tanh; write formulas where appropriate.
3. Match output layer and loss to task.
4. Softmax calculation by hand.
5. Explain negative logits.
6. Parameter counting for a small network.
7. Distinguish logits and probabilities.

Avoid:

- Very long unsupported derivations that are too large to grade fairly.
- Multi-layer calculations with so many numbers that the arithmetic hides the main idea.
- Implementation-heavy questions unless they belong to Session 4.

### Output Layer Question Pattern

A good standalone output-layer question says:

```markdown
A neural network's final layer must match the prediction task. Under the course row-vector convention, assume the last hidden representation of one example is $a^{(L-1)}$, and the final affine transformation is $z^{(L)} = a^{(L-1)} W^{(L)} + b^{(L)}$.
```

Then ask about:

- Regression: output dimension 1, linear output, MSE.
- Binary classification: one logit or sometimes two logits; sigmoid and BCE for one-logit case.
- Multiclass classification: `K` logits, softmax, cross-entropy.

Avoid making this into a table if the instruction asks for more math. Tables are okay sometimes, but formula-driven prompts are often better.

---

## Session 4: Neural Network from Scratch

Topics to emphasize:

- The course's own NumPy neural-network implementation. Use the source file internally, but phrase exam questions without relying on its file path.
- Dense layer code
- ReLU layer code
- Softmax cross-entropy code
- Training loop code
- MNIST architecture
- Logits and softmax probabilities
- He initialization if present in code
- Parameter counting
- Debugging matrix multiplication in Dense

Good question ideas:

1. Fill in the training loop blanks.
2. Debug Dense layer code using `np.dot` instead of elementwise multiplication.
3. Draw the MNIST network in the code.
4. Count parameters in the MNIST network.
5. Explain logits versus probabilities in the implementation.
6. Explain why `np.max(logits, axis=1, keepdims=True)` is subtracted.
7. Treat linear regression and binary logistic regression as one-layer neural networks and ask detailed backpropagation math using simple shapes and numbers.
8. Ask students to derive or compute error signals such as $\delta=\partial \ell/\partial z$ for identity and sigmoid outputs.
9. Write `Sigmoid` or `LeakyReLU` class only if consistent with source materials.

Avoid:

- Very long unsupported backpropagation derivations for deep networks with many layers and many numbers.
- New architectures not connected to the course's own NumPy neural-network implementation unless the question is conceptual.

### Session 4 Placeholder Style

Follow existing style:

```markdown
Fill in the `____YOUR_CODE_HERE__N_____`.

```python
def train(network, X, y):
    activations = ____YOUR_CODE_HERE__1_____(network, X)
    logits = activations____YOUR_CODE_HERE__2_____
```

Your answer goes here, after the `:`:

- `____YOUR_CODE_HERE__1_____`:
- `____YOUR_CODE_HERE__2_____`:
```

---

## Session 5: Regularization, Dropout, and Optimization Concepts

The exact scope depends on course files. Inspect existing session 5 materials before generating.

Likely topics:

- Overfitting in neural networks
- L1 and L2 regularization
- Dropout
- Inverted dropout
- Training mode versus inference mode
- Momentum and Adam formulas if covered
- Learning rate intuition

Good question ideas:

1. Explain L1 versus L2 regularization and write formulas.
2. Explain dropout with a drawing.
3. Fill in inverted dropout blanks similar to `BagOfQuestions-Dropout.md`.
4. Explain why dropout is disabled at inference time.
5. Write momentum or Adam formulas and explain the terms.
6. Compare training loss and validation loss under overfitting.

Avoid:

- Coding Adam or momentum.
- Coding model selection.
- Heavy backpropagation math.

---

## Session 6 and Session 7: Model Selection

Topics to emphasize:

- Train/validation/test split
- Overfitting and underfitting
- Bias-variance intuition if covered
- Hyperparameters versus parameters
- Cross-validation conceptually
- Choosing final model
- Data leakage
- Evaluation metrics if covered
- Confusion matrix, accuracy, precision, recall, F1 if in session files

Good question ideas:

1. Draw train/validation/test workflow.
2. Explain why test set should be used only at the end.
3. Explain overfitting from training and validation curves.
4. Compare parameters and hyperparameters.
5. Explain k-fold cross-validation conceptually.
6. Identify data leakage in a scenario.
7. Choose between models based on validation performance and explain.

Avoid:

- Code implementation of cross-validation or grid search.
- Questions requiring libraries not used in class.
- Advanced statistical learning theory unless present in lecture files.

---

## Handling Existing Files

When asked to fix draft markers:

1. Search only the requested scope if the user specifies a scope.
2. Remove the draft marker text entirely.
3. Replace it with finished question content.
4. Preserve the file's intent if it is clear.
5. Do not introduce answer keys unless explicitly asked.
6. Do not modify session base files such as `BagOfQuestions-session-N.md` unless explicitly asked.

When asked to generate new `BagOfQuestions-session-N-XX.md` files:

1. Do not overwrite existing `-XX` files unless asked.
2. Pick the next available suffix in alphabetical order if the user does not specify one.
3. Keep file names exactly in the pattern.
4. Include one or more independent `## Question` blocks.
5. Make each block standalone.

When asked to revise existing generated files:

1. Respect the user's requested file range.
2. Do not edit unrelated sessions.
3. Do not edit deleted/renamed files unless necessary.
4. Keep the style consistent with neighboring files.

---

## Draft-Marker Removal Standards

A draft marker is not fixed by simply deleting the marker text. It is fixed only when the surrounding content becomes a complete usable question.

### Example: Bad Fix

Original:

```markdown
TODO: make this better.
```

Bad fix:

```markdown
Make this better.
```

### Example: Good Fix

Original:

```markdown
TODO: make it more like the fill-in-the-blank code style.
```

Good fix:

```markdown
We are implementing binary logistic regression `class MyOwnLogisticRegression` from scratch with NumPy. Fill in the `____YOUR_CODE_HERE__N_____` blanks in the code skeleton below.
```

Then provide an actual skeleton and answer slots.

---

## Answer Keys

By default, do **not** include answer keys in `BagOfQuestions-session-N-XX.md` files unless the existing file style or the user explicitly asks for answers.

The bag is intended for exam-question selection. Including answers may make it less directly usable as exam material.

If an answer key is requested, put it clearly under a separate heading such as:

```markdown
## Answer Key
```

But again: do not add this unless requested.

---

## Formatting Checklist Before Finishing

Before finalizing edits, verify:

- [ ] Every question begins with `## Question`.
- [ ] Preferably every question has a descriptive title.
- [ ] Every `## Question` is standalone.
- [ ] No `draft marker`, `placeholder marker`, or `fix marker` remains in the requested scope.
- [ ] Code blocks are fenced correctly.
- [ ] LaTeX formulas use correct Markdown syntax.
- [ ] Placeholder names are consistent and numbered in order.
- [ ] Every placeholder in code has a corresponding answer slot.
- [ ] Every answer slot corresponds to a placeholder in code.
- [ ] Any neural-network backpropagation math is standalone, notation-consistent, and fair to grade.
- [ ] The question does not ask forbidden optimization/model-selection coding.
- [ ] The topic fits the session.
- [ ] There are no obvious spelling mistakes in titles.

Suggested command for draft marker check in the BagOfQuestions folder:

```bash
rg -n "TODO|TBD|FIXME" BagOfQuestions
```

If the user requested only sessions 1 to 3 generated files, use a narrower command:

```bash
rg -n "TODO|TBD|FIXME" BagOfQuestions/BagOfQuestions-session-[123]-*.md
```

---

## Examples of Strong Questions

### Example 1: Linear Regression Fill-in-the-Blank

```markdown
## Question: Linear Regression from Scratch

We are implementing linear regression `class MyOwnLinearRegression` from scratch with NumPy. In the mathematical notation from the course, $X \in \mathbb{R}^{n \times d}$ and $Y \in \mathbb{R}^{n \times 1}$. In the NumPy implementation, the target may be stored as `y.shape == (n_samples,)`. The batch prediction rule is

$$
\hat{Y} = XW + \mathbf{1}b.
$$

Fill in the `____YOUR_CODE_HERE__N_____` blanks.

```python
y_predicted = np.dot(____YOUR_CODE_HERE__1_____, ____YOUR_CODE_HERE__2_____) + self.bias
dw = (2 / n_samples) * np.dot(____YOUR_CODE_HERE__3_____, (y_predicted - y))
db = (2 / n_samples) * np.sum(____YOUR_CODE_HERE__4_____)
self.weights = self.weights - self.lr * ____YOUR_CODE_HERE__5_____
```

Your answer goes here, after the `:`:

- `____YOUR_CODE_HERE__1_____`:
- `____YOUR_CODE_HERE__2_____`:
- `____YOUR_CODE_HERE__3_____`:
- `____YOUR_CODE_HERE__4_____`:
- `____YOUR_CODE_HERE__5_____`:

1. Why do we use `X.T` in the gradient for `dw`?
2. What shape should `dw` have?
```

### Example 2: Logistic Regression Formula and Interpretation

```markdown
## Question: Binary Cross-Entropy and Probability

In binary logistic regression, under the course row-vector convention, the model computes a linear score $z = xW + b$ for one example and a probability $\hat y = \sigma(z)$ for class 1.

1. Write the sigmoid formula.
2. Write the binary cross-entropy loss for one example.
3. If $y=1$, simplify the loss.
4. If $y=0$, simplify the loss.
5. Explain why a probability threshold such as 0.5 can convert $\hat y$ into a class label.
```

### Example 3: Output Layers

```markdown
## Question: Output Layers Depend on the Task

A neural network's final layer must match the prediction task. Under the course row-vector convention, assume the final affine transformation is $z^{(L)} = a^{(L-1)} W^{(L)} + b^{(L)}$.

1. For one-output regression, what output dimension and activation should be used? Write the MSE loss for one example.
2. For binary classification with one logit, what output dimension and activation should be used? Write the sigmoid and BCE formulas.
3. For 10-class classification, what output dimension and activation should be used? Write the softmax and cross-entropy formulas.
4. Explain why the output activation and the loss function should be designed together.
```

### Example 4: Model Selection Conceptual Schema

```markdown
## Question: Train, Validation, and Test Sets

In model selection, we often split data into training, validation, and test sets. The training set is used to fit model parameters, the validation set is used to choose hyperparameters or compare models, and the test set is used only at the end.

1. Draw a schema showing the three splits and their roles.
2. Explain what can go wrong if the test set is used repeatedly during model selection.
3. Give two examples of hyperparameters.
4. Give two examples of learned parameters.
5. Explain why the final test score should be reported only after the model-selection process is finished.
```

---

## Examples of Weak Questions and How to Improve Them

### Weak Question

```markdown
## Question
What is softmax?
```

### Improved Question

```markdown
## Question: Softmax Formula and Interpretation

In multiclass classification, a neural network often outputs logits $z_1,\ldots,z_K$ before converting them to probabilities.

1. Write the softmax formula for class $k$.
2. Explain why the softmax outputs sum to 1.
3. Explain why the largest logit also gives the largest softmax probability.
4. What is the difference between logits and probabilities?
```

### Weak Question

```markdown
## Question
Draw neural network.
```

### Improved Question

```markdown
## Question: Draw an MNIST Neural Network

Consider an MNIST classifier with input dimension 784, two hidden layers of 10 neurons each, and an output layer of 10 neurons.

1. Draw the network architecture.
2. Label the input, hidden layers, output logits, and softmax probabilities.
3. Explain why the input dimension is 784.
4. Explain why the output dimension is 10.
5. Count the total number of trainable parameters, including biases.
```

### Weak Question

```markdown
## Question
Fix code.
```

### Improved Question

```markdown
## Question: Debug Matrix Multiplication in Linear Regression

We are implementing multiple linear regression from scratch with NumPy. The feature matrix has shape `(50, 4)`, and the weight vector has shape `(4,)`. A student wrote:

```python
y_predicted = X * self.weights + self.bias
```

1. What shape does this elementwise multiplication tend to produce?
2. Why is that not the desired prediction shape?
3. Write the correct prediction line using `np.dot`.
4. What should the prediction shape be?
```

---

## Working Procedure for Future AI Agents

When a user asks you to work in `BagOfQuestions`, follow this procedure.

### Step 1: Inspect Instructions and Recent Intent

Run commands such as:

```bash
pwd
git status --short
git log -1 --stat --oneline
find .. -name AGENTS.md -print
```

Read any relevant `AGENTS.md` files if present. Inspect the most recent commit if the user asks you to infer style or direction from it.

### Step 2: Inspect Relevant Files

For a task involving sessions 1 to 3, inspect:

```bash
find BagOfQuestions -maxdepth 1 -type f | sort
sed -n '1,220p' BagOfQuestions/BagOfQuestions-session-1-aa.md
sed -n '1,220p' BagOfQuestions/BagOfQuestions-session-2-aa.md
sed -n '1,220p' BagOfQuestions/BagOfQuestions-session-3-aa.md
```

Adjust filenames depending on which files exist.

### Step 3: Search for draft markers in Scope

Use ripgrep, not recursive grep:

```bash
rg -n "TODO|TBD|FIXME" BagOfQuestions/BagOfQuestions-session-[123]-*.md
```

If the user asked all of `BagOfQuestions`, use:

```bash
rg -n "TODO|TBD|FIXME" BagOfQuestions
```

### Step 4: Edit Only Requested Files

If the user says only sessions 1, 2, and 3 generated files, edit only files matching:

```text
BagOfQuestions/BagOfQuestions-session-1-XX.md
BagOfQuestions/BagOfQuestions-session-2-XX.md
BagOfQuestions/BagOfQuestions-session-3-XX.md
```

Do not edit:

```text
BagOfQuestions/BagOfQuestions-session-1.md
BagOfQuestions/BagOfQuestions-session-2.md
BagOfQuestions/BagOfQuestions-session-3.md
```

unless explicitly requested.

### Step 5: Validate

At minimum, run:

```bash
rg -n "TODO|TBD|FIXME" BagOfQuestions/BagOfQuestions-session-[123]-*.md
```

For Markdown-only changes, there may be no test suite. A TODO/draft-marker search and `git diff --check` are still useful:

```bash
git diff --check
```

### Step 6: Commit and Pull Request if Required by Environment

If the environment instructions require committing changes and creating a PR, do so after validation. Use a clear commit message such as:

```bash
git add BagOfQuestions
git commit -m "Fix BagOfQuestions draft markers and add generation skill"
```

Then create a PR with a concise title and body.

---

## Final Quality Standard

A finished `BagOfQuestions-session-N-XX.md` file should feel like something a teacher can copy directly into a final exam. It should not feel like notes to an AI, a draft, or a draft marker list.

The best questions are:

- Clear
- Independent
- Exam-ready
- Connected to course code or lecture material
- Mathematically important
- Not too trivial
- Not too advanced
- Easy to grade
- Written in English
- Consistent with the instructor's style

When in doubt, write a little more context at the beginning of each `## Question`. A short standalone background paragraph is usually better than a vague question.
