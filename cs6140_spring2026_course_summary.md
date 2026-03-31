# CS 6140 Machine Learning
## Comprehensive Course Summary: Weeks 1–11

**Course:** CS6140.39092 Machine Learning, Section 06, Spring 2026
**Instructor:** Prof. Aanchan Mohan | **TA:** Chencheng (Jane) Feng
**Format:** Hybrid — Vancouver Campus | **Mid-term:** April 6, in-person, paper-based

---

## Table of Contents

1. [Course Structure and Grading](#course-structure)
2. [Week 1: Introduction to Machine Learning](#week-1)
3. [Week 2: Linear Algebra, Eigendecomposition, PCA Setup](#week-2)
4. [Week 3: K-Means and PCA](#week-3)
5. [Week 4: Probabilistic View of Data](#week-4)
6. [Week 5: Logistic Regression and Maximum Likelihood](#week-5)
7. [Weeks 6–7: Metrics, Overfitting, and Regularization](#week-6-7)
8. [Week 8: MAP Estimation and Neural Networks Introduction](#week-8)
9. [Weeks 9–10: Backpropagation and PyTorch Autograd](#week-9-10)
10. [Week 11: Midterm Review](#week-11)
11. [Project: Voice Activity Detection](#project)
12. [Algorithm and Formula Reference](#formula-reference)
13. [Key Tools and Resources](#tools-reference)

---

## <a name="course-structure"></a>Course Structure and Grading

### Weekly Activity Format

Every week has three tiers of activity:

| Tier | Due | Format | Points |
|---|---|---|---|
| **Pre-class** | Monday 9 AM | Reading + video questions; screenshot evidence required | 5 pts/week |
| **In-class** | Same day (programming due Tuesday 9 AM) | Paper worksheet or Colab notebook | 5 pts/week |
| **Post-class** | Thursday 9 PM | Larger coding or analysis problem | Varies (5–124 pts) |

Attendance is recorded via Qwickly PIN check-in on the Canvas Student App.

### Grade Weights

| Component | Weight | Notes |
|---|---|---|
| Pre-class activities | 15% | Drop lowest 1 |
| In-class activities | 15% | Drop lowest 1 |
| Post-class activities | 15% | Drop lowest 1 |
| Project proposal | 5% | Team submission |
| Mid-term project video | 10% | Individual code walkthrough |
| In-class mid-term exam | 20% | April 6; paper-based |
| End-of-semester presentation & Q&A | 20% | Team |

### Rubric Standard (pre/post-class written answers)
- **0** — AI-generated sounding, no demonstrated understanding
- **1** — Brief, minimal engagement
- **2** — Satisfactory, shows understanding
- **3** — Detailed, demonstrates depth

### Exam Format
- 14 questions, 10 pts each, best 8 count (64 pts maximum contributing)
- Written justification required — a correct answer with no justification earns at most 3 points
- **The justification is what is being tested**

---

## <a name="week-1"></a>Week 1 — Introduction to Machine Learning

### Main Topics
- What is machine learning? Classification and regression as supervised tasks
- Representing data as vectors: word embeddings, one-hot vectors
- Introduction to cost functions
- Linear algebra foundations: dot product, norms, cosine similarity, matrix multiplication

### Key Concepts

**Types of ML problems:**
- **Classification** — discrete output labels (e.g., spam/not spam, speech/silence)
- **Regression** — continuous output (e.g., predicting a price)
- **Clustering** — no labels; find structure in data (introduced later)

**Vector representations:**
- **One-hot vector** — binary, exactly one 1, all others 0; orthogonal to all other one-hot vectors
- **Word embeddings** — dense, low-dimensional; dot product measures semantic similarity
- **Dot product** — `a·b = ∑ᵢ aᵢbᵢ`; zero if and only if vectors are orthogonal
- **Cosine similarity** — `cos(θ) = (a·b) / (‖a‖ · ‖b‖)`; scale-invariant similarity measure
- **p-Norm** — `‖x‖ₚ = (∑ᵢ |xᵢ|ᵖ)^(1/p)`; L1 (p=1), L2 (p=2, Euclidean) most common

**Cost function for linear regression:**
- Model a straight line: ŷ = wx + b
- Parameters w, b initialized and iteratively adjusted to minimize squared error
- `C = ∑ᵢ (yᵢ − ŷᵢ)²`

### Activities
- **Pre-class (5 pts):** Week 1 survey
- **In-class (5 pts):** Run SpeechBrain VAD notebook on Northeastern Open On Demand; upload launch screenshot and VAD output
- **Post-class (21 pts):** Read Murphy Ch. 1; answer 7 questions on Figs 1.3/1.4/1.7, Eq 1.1/1.4, misclassification rate, overfitting, Section 1.5 datasets

### Resources
- Murphy, *Probabilistic Machine Learning*, Chapter 1
- CMU word embedding tutorial
- Stanford ML cheat sheet

---

## <a name="week-2"></a>Week 2 — Linear Algebra, Eigendecomposition, PCA Setup

### Main Topics
- Training pipeline vs inference pipeline
- Eigenvalues and eigenvectors
- Eigendecomposition
- Mean vectors and covariance matrices
- Motivation for GPU parallelism

### Key Concepts

**Training vs inference pipelines:**
- **Training:** data → preprocessing → compute loss through model → update parameters (iteratively)
- **Inference:** new data → same preprocessing → trained model → predictions (no parameter updates, no gradient storage)

**Eigenvalues and eigenvectors:**
- Definition: `Av = λv` — a matrix A acting on vector v only scales it, not rotates it
- v is the **eigenvector**, λ is the **eigenvalue**
- Eigenvectors of the covariance matrix point in directions of maximum variance

**Eigendecomposition:**
```
A = Q Λ Qᵀ
```
where Q is the matrix of eigenvectors (as columns) and Λ is diagonal with eigenvalues.

**Covariance matrix:**
- `Σ = (1/N) Xᵀ X` (after mean-centering X)
- Diagonal entries = variance of each feature
- Off-diagonal entries = covariance between features
- Always symmetric; eigenvectors are orthogonal

**Gram matrix:**
- `G = X Xᵀ` — shape N×N instead of D×D
- Used when D >> N (more features than samples); eigenvectors of G recover PCA directions

### Activities
- **Pre-class (25 pts):** 25-question linear algebra activity using 3B1B playlist + Murphy Ch. 7.1–7.4. Topics: span, basis vectors, linear transformations, determinants, matrix inverse, rank, eigenvectors, trace
- **In-class (5 pts):** Paper worksheet — matrix multiplications as composed transformations; verify eigendecomposition; calculate mean and covariance by hand
- **Post-class (100 pts):** Coding + hand-calculation: Part A0 — hand calculation of mean/covariance/centering; Part A — data loading/preprocessing/visualization; Part B — covariance computation, textbook connections, dual (Gram) formulation

---

## <a name="week-3"></a>Week 3 — K-Means and PCA

### Main Topics
- PCA: encode/decode formulation
- K-Means clustering
- The signal compression connection between PCA and K-Means
- Neural networks as nonlinear autoencoders

### Key Concepts

**PCA as encode-decode:**
```
encode(x) = Wᵀ(x − μ)       [project to low-dim code]
decode(z) = Wz + μ            [reconstruct from code]
```
PCA minimises: `‖x − decode(encode(x))‖²`

- Optimal W = matrix of top-k eigenvectors of the covariance matrix
- PCA is the **globally optimal linear autoencoder**
- **Neural autoencoders** do the same thing but with nonlinear encode/decode functions = nonlinear PCA

**PCA limitations:**
- Maximises **variance**, not **class separability**
- Can silently discard a direction that perfectly separates two classes if that direction has low variance
- To detect: compare classification accuracy before and after applying PCA
- Assumes data follows a **multivariate Gaussian distribution**

**K-Means:**
- Minimises **global distortion** (within-cluster sum of squared distances):
  `D = ∑ₙ ‖xₙ − μ_{c(n)}‖²`
- Algorithm: (1) assign each point to nearest centroid, (2) recompute centroids, repeat until convergence
- K-Means++ — smarter initialisation to avoid poor local minima
- Related to **Vector Quantization (VQ)** / Linde-Buzo-Gray (LBG) algorithm from signal compression

**PCA vs K-Means:**
| | PCA | K-Means |
|---|---|---|
| Output | Continuous low-dim code | Discrete cluster assignment |
| Analogy | Continuous compression | Codebook/VQ compression |
| When to use | Continuous latent structure | Discrete groupings |
| Supervised? | No | No |

### Activities
- **Pre-class (5 pts):** Murphy Eqs 20.1–20.13 (PCA), Eqs 21.13–21.15 (K-Means), formal algorithm, k-means++ variant
- **In-class (5 pts):** Hand-written worksheet on PCA
- **Post-class (100 pts):** PCA implementation and analysis activity

---

## <a name="week-4"></a>Week 4 — Probabilistic View of Data

### Main Topics
- Data and labels as samples from probability distributions
- Multivariate Gaussian distribution
- Bayesian reasoning fundamentals
- Voice Activity Detection (VAD) as the course running example

### Key Concepts

**Core assumption of ML:**
Every training example (xᵢ, yᵢ) is treated as a sample drawn from an underlying joint probability distribution p(x, y). The goal of learning is to approximate this distribution.

**Probability distributions used in this course:**
- **Gaussian / Normal** — for continuous real-valued data (images, audio features, fMRI)
- **Bernoulli** — for binary labels (speech/silence, spam/not-spam)
- **Categorical** — for multi-class labels (digit 0–9)

**Bayes' Theorem:**
```
P(H | D) = P(D | H) · P(H) / P(D)
```
- P(H) = **prior** — belief before seeing data
- P(D|H) = **likelihood** — how probable is the data given hypothesis H?
- P(H|D) = **posterior** — updated belief after seeing data

**Voice Activity Detection (VAD):**
- Binary classification: is there speech in this audio frame? (1 = speech, 0 = silence)
- Running example throughout the course; project uses SpeechBrain framework

### Activities
- **Pre-class (5 pts):** StatQuest probability playlist (11 videos); 6 questions on distributions, Bayes' theorem, parameter estimation, modelling of text/image/audio
- **In-class (5 pts):** Multivariate Gaussian worksheet; K-Means vs LBG demo
- **Post-class (40 pts):** Probability and distribution activity

### Resources
- StatQuest with Josh Starmer — probability playlist
- Seeing Theory (interactive probability visualisations)
- Observable Multivariate Gaussian and Distribution Explorer

---

## <a name="week-5"></a>Week 5 — Logistic Regression and Maximum Likelihood

### Main Topics
- Logistic regression as a discriminative model
- Sigmoid function and the Bernoulli likelihood connection
- Maximum Likelihood Estimation (MLE)
- Generative vs discriminative models
- Naive Bayes introduction

### Key Concepts

**Logistic regression forward pass:**
```
z = wᵀx + b
ŷ = σ(z) = 1 / (1 + e^{−z})
```
- Output ŷ ∈ (0,1) — interpreted as P(y=1 | x)

**MLE for logistic regression:**
- Model label y as a Bernoulli random variable with parameter ŷ
- Per-sample likelihood: `p(yᵢ | xᵢ) = ŷᵢ^{yᵢ} · (1 − ŷᵢ)^{1−yᵢ}`
- Joint likelihood (i.i.d. assumption): `L(w) = ∏ᵢ p(yᵢ | xᵢ)`
- Maximise log-likelihood → minimise **negative log-likelihood (NLL)** = **cross-entropy loss**:
  `C = −∑ᵢ [yᵢ log ŷᵢ + (1−yᵢ) log(1−ŷᵢ)]`

**Generative vs discriminative:**
| | Discriminative | Generative |
|---|---|---|
| Models | p(y\|x) directly | p(x\|y) and prior p(y) |
| Examples | Logistic regression | Naive Bayes, GDA |
| Sampling | Cannot generate x | Can generate new x |
| Typically | More accurate in large-data regime | Works with less data |

**Gradient descent:**
- Minimise cost C by moving parameters in the negative gradient direction:
  `w ← w − η · ∂C/∂w`
- η = learning rate (hyperparameter); too large → divergence; too small → slow convergence

### Activities
- **Pre-class (5 pts):** StatQuest MLE video; Murphy Sec 4.2 Eqs 4.2–4.8; Kamper logistic regression video; Murphy Sec 10.2; gradient descent video
- **In-class (5 pts):** Forward pass by hand
- **Post-class (124 pts):** Full logistic regression implementation; gradient descent; learning curves

---

## <a name="week-6-7"></a>Weeks 6–7 — Metrics, Overfitting, and Regularization

### Main Topics
- Evaluation metrics beyond accuracy
- Bias-variance tradeoff
- Overfitting and underfitting
- L1 and L2 regularization
- MAP estimation as a Bayesian view of regularization
- Curse of dimensionality

### Key Concepts

**Evaluation metrics:**
- **Accuracy** = correct predictions / total; misleading on imbalanced datasets
- **Precision** = TP / (TP + FP) — of all predicted positives, how many are correct?
- **Recall (TPR)** = TP / (TP + FN) — of all true positives, how many did we catch?
- **F1** = 2 · Precision · Recall / (Precision + Recall) — harmonic mean

**Bias-variance tradeoff:**
- **Bias** — error from wrong assumptions; high bias = underfitting
- **Variance** — error from sensitivity to training data fluctuations; high variance = overfitting
- Adding model capacity reduces bias but increases variance

**Regularization:**

| | L2 (Ridge) | L1 (LASSO) |
|---|---|---|
| Added term | `+ λ‖w‖²` | `+ λ‖w‖₁` |
| Effect on weights | Shrink toward zero uniformly | Drive some weights to exactly zero |
| Weight distribution | Gaussian (bell curve) | Laplace (spiky/peaked) |
| Key property | Reduces magnitude | **Feature selection** (sparsity) |
| MAP prior | Gaussian prior on w | Laplace prior on w |

**Why L1 produces sparsity:**
The subdifferential of |w| at w=0 includes 0. Once a weight reaches zero, there is no restoring force — it stays at zero. L2 always has a restoring force proportional to w, so weights shrink toward but never reach zero.

**MAP estimation:**
```
θ_MAP = argmax  p(θ | D)
       = argmax  p(D | θ) · p(θ)         [Bayes' rule]
       = argmax  log p(D | θ) + log p(θ) [log transform]
```
- **Gaussian prior** on θ → log p(θ) = const − λ‖θ‖² → **L2 regularization**
- **Laplace prior** on θ → log p(θ) = const − λ‖θ‖₁ → **L1 regularization**
- λ = 1/(2σ₀²) for Gaussian prior with variance σ₀²
- As N → ∞, MAP → MLE (data overwhelms prior)

**MLE vs MAP:**
- **MLE** = maximum of likelihood only; no prior; can overfit with small data
- **MAP** = maximum of likelihood × prior; regularises with a probabilistic prior

**Curse of dimensionality:**
As the number of features D grows, the amount of data required to maintain the same density grows exponentially. Distances become less meaningful in high dimensions. Motivation for dimensionality reduction (PCA) before applying distance-based methods (K-Means, KNN).

### Post-class Activity — Fake News Classification
Using logistic regression with sklearn on a fake news binary classification dataset:
1. 70/15/15 train/validation/test split; histogram of class counts
2. Choose appropriate metric given class imbalance
3. `CountVectorizer` feature construction; justify matrix dimensions
4. Train with `SGDClassifier.partial_fit`; plot learning curves; tune learning rate
5. Sweep L2 regularization strength; plot 15 coefficient values vs λ (Ridge)
6. Sweep L1 regularization strength; plot coefficient values vs λ (LASSO)
7. Compare L1 vs L2 weight distribution histograms
8. Compare three strategies: PCA only vs regularization only vs PCA + regularization

---

## <a name="week-8"></a>Week 8 — MAP Estimation and Neural Networks Introduction

### Main Topics
- Bayesian derivation of regularization (Bishop PRML connection)
- Composition of linear and nonlinear transformations
- Neural network architecture: layers, activations, depth vs width
- Hierarchical feature learning
- TensorFlow Playground exploration

### Key Concepts

**From logistic regression to neural networks:**
- Logistic regression = single linear layer + sigmoid activation
- Neural network = multiple compositions of (linear layer + nonlinear activation)
- Adding layers allows learning nonlinear decision boundaries (XOR problem)

**Why nonlinearity is essential:**
A composition of linear transformations is itself linear: `W₂(W₁x) = (W₂W₁)x`. Without nonlinear activations between layers, a deep network collapses to a single linear layer.

**Activation functions:**
| Activation | Formula | Gradient | Use case |
|---|---|---|---|
| Sigmoid | `σ(z) = 1/(1+e^{−z})` | `σ'(z) = σ(z)(1−σ(z)) ≤ 0.25` | Binary output, logistic regression |
| Softmax | `σₖ(z) = e^{zₖ}/∑ⱼe^{zⱼ}` | Jacobian (row-wise) | Multi-class output |
| ReLU | `max(0, z)` | 0 or 1 | Hidden layers; avoids vanishing gradients |
| Linear | `z` | 1 | Regression output |

**Hierarchical feature learning:**
Deep networks learn increasingly abstract features in later layers. Early layers detect edges; later layers detect shapes; final layers detect semantic content.

**TensorFlow Playground observations:**
- Linear activation → cannot learn nonlinear decision boundaries
- Sigmoid activation → can, but slow on deep networks (vanishing gradients)
- Circle dataset requires ≥ 1 hidden layer with nonlinearity
- Spiral dataset requires deeper network or more neurons

### Activities
- **Pre-class (5 pts):** Bishop PRML Eq 1.43 worksheet (Bayesian derivation of regularization); 3B1B neural networks video (5 questions: output layer, layers of abstraction, sigmoid, nonlinearity necessity, ReLU)
- **In-class (5 pts):** Guest lecture notes — 2–3 themes, 75–100 words each
- **Post-class (18 pts):** TensorFlow Playground (7 pts); classification heads with BERT video (3 pts); BERT vs GPT from CoNLL 2024 paper (2 pts); language models blog post + Murphy Ch. 15 (2 pts)

---

## <a name="week-9-10"></a>Weeks 9–10 — Backpropagation and PyTorch Autograd

### Main Topics
- The four backpropagation equations (BP1–BP4)
- Computation graphs and automatic differentiation
- PyTorch `.backward()`, `.grad`, `requires_grad`
- The `zero_grad()` bug
- BCE+sigmoid gradient cancellation
- Vanishing gradients

### Key Concepts

**The four backprop equations (Nielsen notation):**

| Equation | Formula | Meaning |
|---|---|---|
| **BP1** | `δᴸ = ∂C/∂aᴸ ⊙ σ'(zᴸ)` | Error at output layer |
| **BP2** | `δˡ = ((wˡ⁺¹)ᵀ δˡ⁺¹) ⊙ σ'(zˡ)` | Error at hidden layer l |
| **BP3** | `∂C/∂bˡ = δˡ` | Bias gradient |
| **BP4** | `∂C/∂wˡ = δˡ (aˡ⁻¹)ᵀ` | Weight gradient |

**How `.backward()` works:**
1. During the forward pass, PyTorch builds a computation graph recording every operation
2. `.backward()` traverses this graph in **reverse topological order**
3. At each node, it calls the stored backward function (chain rule)
4. It does **not** recompute — it uses **cached activations** from the forward pass
5. This is why forward-pass activations must stay in GPU memory during training

**Memory comparison — training vs inference:**

| | Training | Inference |
|---|---|---|
| Activations | Kept in memory (needed for BP2/BP4) | Discarded immediately |
| Gradients | Computed and stored | Not computed |
| Optimizer state (Adam) | 2 extra tensors per parameter (m, v) | None |
| Memory multiplier (Adam) | ~4× weights | ~1× weights |

For a 7B-parameter FP32 model: ~28 GB inference, ~112+ GB training with Adam.

**The BCE+sigmoid cancellation:**
```
∂C/∂aᴸ = −y/a + (1−y)/(1−a)
σ'(z) = a(1−a)

BP1: δᴸ = ∂C/∂aᴸ · σ'(z)
        = [−y/a + (1−y)/(1−a)] · a(1−a)
        = a − y    ← σ' cancels completely
```
This means the output-layer error signal is simply `a − y` — the prediction error. No sigmoid derivative appears, **removing a source of vanishing gradient at the output layer**.

**MSE+sigmoid vs BCE+sigmoid vs CE+softmax:**

| | MSE + sigmoid | BCE + sigmoid | CE + softmax |
|---|---|---|---|
| **Dimensionality** | `a ∈ ℝᴷ`, vectors — one sigmoid per class | `aₖ ∈ ℝ` scalar — single neuron | `a ∈ ℝᴷ`, vectors — one-hot y |
| **Cost** | `C = ½‖a−y‖²` | `C = −[yₖ log aₖ + (1−yₖ) log(1−aₖ)]` | `C = −∑ₖ yₖ log aₖ` |
| **Output δ** | `δₖ = (aₖ−yₖ)·aₖ(1−aₖ)` — σ' does NOT cancel | `δₖ = aₖ−yₖ` — σ' cancels | `δₖ = aₖ−yₖ` — Jacobian cancels |
| **Vanishing gradient?** | Yes — when aₖ ≈ 0 or 1 | No | No |
| **Used in** | Nielsen, 3B1B (pedagogical) | Binary classification, VAD project | Multi-class; practical MNIST |
| **MLE-principled?** | No (Gaussian likelihood) | Yes (Bernoulli NLL) | Yes (categorical NLL) |

**Why Nielsen uses MSE+sigmoid:** Deliberate pedagogical choice — keeps the backprop derivation clean. BCE+sigmoid is better in practice because σ' cancels, and it connects cleanly to MLE.

**Vanishing gradient problem:**

| Activation | σ'(z) | 10-layer worst case |
|---|---|---|
| Sigmoid | ≤ 0.25 | 0.25¹⁰ ≈ 10⁻⁶ |
| ReLU | 0 or 1 | 1¹⁰ = 1 (active region) |

Early layers in a deep sigmoid network receive effectively zero gradient — they cannot learn.

**The universal training loop (PyTorch):**
```python
predictions = model(batch)        # forward pass — build computation graph
loss = criterion(predictions, y)  # compute scalar loss
optimizer.zero_grad()             # clear gradients from previous step
loss.backward()                   # backprop — fill .grad attributes
optimizer.step()                  # update weights: w ← w − η·∂L/∂w
```
**The `zero_grad()` bug:** Forgetting `optimizer.zero_grad()` causes gradients to accumulate across steps. This causes divergence, not just slow convergence — effective learning rate grows unboundedly.

**Gradient checking (numerical verification):**
```
∂C/∂θ ≈ (C(θ+ε) − C(θ−ε)) / (2ε)
```
Compare to backprop gradient; relative error < 10⁻⁵ is acceptable.

**Transfer learning / frozen encoders:**
- `requires_grad=False` on a module stops gradient flow at that boundary
- Weights in that module do not update — the encoder is "frozen"
- Used in SpeechBrain VAD: frozen speech encoder + trainable classification head

### Activities
- **Week 9 Pre-class (5 pts):** 3B1B gradient descent + backprop intuition videos; submit learning notes as PDF slides
- **Week 9 In-class (5 pts):** Colab neural network notebook — output layer, cost function, weight update, gradient computation, misclassification screenshots
- **Week 9 Post-class (5 pts):** Backpropagation by hand — chain rule derivatives, MNIST weight matrix dimensions, BP1–BP4 in vectorized form, gradient checking
- **Week 10 Pre-class (5 pts):** PyTorch autograd video; run Colab autograd notebook; submit study notes

---

## <a name="week-11"></a>Week 11 — Midterm Review

### Main Topics
- Consolidation of Weeks 1–10
- Clarifying persistent confusions from Piazza

### What the Piazza questions reveal

The most-asked clusters on Piazza going into the midterm:

| Topic cluster | # threads |
|---|---|
| Backpropagation (chain rule, PyTorch graph, efficiency, vanishing gradients) | ~10 |
| MLE / likelihood / loss function connection | ~4 |
| PCA limitations (discriminative signal loss, Gram matrix) | ~4 |
| L1 vs L2 (mechanism, MAP connection) | ~3 |
| Activation functions (sigmoid vs ReLU choice) | ~3 |
| Neural network geometry / depth vs width | ~3 |
| Generative vs discriminative | ~2 |
| When/how to stop training | ~2 |

### Activities
- **Pre-class 1 (5 pts, due Mar 28):** Post one Piazza question reviewing past assignments; submit screenshot
- **Pre-class 2 (5 pts, due Mar 30):** Leave two comments on other students' Piazza questions

---

## <a name="project"></a>Project — Voice Activity Detection (VAD)

### Overview
Teams build a VAD classifier that distinguishes speech from non-speech audio frames. The baseline approach uses the SpeechBrain framework with a frozen speech encoder and a trained classification head. Teams then extend to a sub-problem variation.

### Milestones

| Milestone | Points | Notes |
|---|---|---|
| Team member names | 1 pt | Individual; due Week 4 |
| Project proposal slides | 20 pts | Team; 5 slides: problem, 2 block diagrams, metrics, resources |
| Project progress slides | 10 pts | 1 slide per team member minimum; due Week 10 |
| Individual code walkthrough video 1 | 30 pts | 4–6 min MP4; individual |
| Final presentation | 50 pts | Team in-class; Q&A worth 25 of the 50 pts |

### Key Technical Concepts for the Project
- **Transfer learning** — use a pre-trained speech encoder; train only the classification head
- **Frozen encoder** — `requires_grad=False` stops gradient at encoder boundary; reduces training memory
- **Binary cross-entropy** — BCE+sigmoid output; δ = a−y at output layer
- **ROC curves** — threshold-independent evaluation; AUC measures ranking ability
- **Threshold tuning** — set operating point based on cost of false alarms vs missed detections

---

## <a name="formula-reference"></a>Algorithm and Formula Reference

### Linear Algebra
| Formula | Description |
|---|---|
| `a·b = ∑ᵢ aᵢbᵢ` | Dot product |
| `cos(θ) = (a·b)/(‖a‖·‖b‖)` | Cosine similarity |
| `‖x‖ₚ = (∑ᵢ\|xᵢ\|ᵖ)^{1/p}` | p-norm (L1: p=1, L2: p=2) |
| `A = QΛQᵀ` | Eigendecomposition |
| `Σ = (1/N) Xᵀ X` | Covariance matrix (mean-centered X) |

### Unsupervised Learning
| Formula | Description |
|---|---|
| `encode(x) = Wᵀ(x−μ)` | PCA encoder |
| `decode(z) = Wz + μ` | PCA decoder |
| `D = ∑ₙ ‖xₙ − μ_{c(n)}‖²` | K-Means global distortion |

### Supervised Learning
| Formula | Description |
|---|---|
| `ŷ = σ(wᵀx + b)` | Logistic regression forward pass |
| `σ(z) = 1/(1+e^{−z})` | Sigmoid; σ'(z) = σ(z)(1−σ(z)) ≤ 0.25 |
| `C = −∑ᵢ[yᵢ log ŷᵢ + (1−yᵢ) log(1−ŷᵢ)]` | Binary cross-entropy (BCE) |
| `C = −∑ₖ yₖ log aₖ` | Cross-entropy with softmax |
| `C = ½‖a−y‖²` | MSE loss |
| `w ← w − η·∂C/∂w` | Gradient descent update |

### Regularization and MAP
| Formula | Description |
|---|---|
| `C_L2 = C + λ‖w‖²` | L2 regularized cost |
| `C_L1 = C + λ‖w‖₁` | L1 regularized cost |
| `θ_MAP = argmax p(D\|θ)·p(θ)` | MAP estimation |
| Gaussian prior → `−log p(θ) ∝ ‖θ‖²` | Gives L2 penalty |
| Laplace prior → `−log p(θ) ∝ ‖θ‖₁` | Gives L1 penalty |

### Neural Networks / Backpropagation
| Formula | Description |
|---|---|
| `zˡ = wˡ aˡ⁻¹ + bˡ` | Linear combination at layer l |
| `aˡ = σ(zˡ)` | Activation at layer l |
| `δᴸ = ∂C/∂aᴸ ⊙ σ'(zᴸ)` | BP1: output error |
| `δˡ = ((wˡ⁺¹)ᵀ δˡ⁺¹) ⊙ σ'(zˡ)` | BP2: hidden layer error |
| `∂C/∂bˡ = δˡ` | BP3: bias gradient |
| `∂C/∂wˡ = δˡ (aˡ⁻¹)ᵀ` | BP4: weight gradient |
| `δᴸ = a − y` (BCE+sigmoid) | Output δ when σ' cancels |
| `(C(θ+ε)−C(θ−ε))/(2ε)` | Numerical gradient check |

### ROC Curves
| Concept | Formula / Description |
|---|---|
| TPR (recall) | `TP / (TP + FN)` |
| FPR | `FP / (FP + TN)` |
| AUC | P(rank positive above random negative) |
| ROC invariance | TPR and FPR both normalise by own class total → invariant to class prevalence |

---

## <a name="tools-reference"></a>Key Tools and Resources

### Software and Frameworks
| Tool | Purpose |
|---|---|
| **PyTorch** | Neural network training; autograd; `.backward()`, `requires_grad`, `.grad` |
| **SpeechBrain** | Speech processing framework; used for VAD project with frozen encoder |
| **scikit-learn** | `SGDClassifier`, `LogisticRegression`, `CountVectorizer`, `Ridge`, `Lasso` |
| **NumPy** | Matrix operations; `np.dot`, `np.cov`, `np.linalg.eig` |
| **torchviz** | Visualize PyTorch computation graphs with `make_dot()` |
| **TensorFlow Playground** | Interactive neural network visualisation |

### Key External References
| Resource | What it covers |
|---|---|
| Michael Nielsen — *Neural Networks and Deep Learning* | BP1–BP4 equations; MSE+sigmoid MNIST (pedagogical) |
| 3Blue1Brown — Neural Networks playlist | Intuitive backpropagation; gradient descent visualisation |
| Murphy — *Probabilistic Machine Learning* | Textbook; PCA Ch. 20, K-Means Ch. 21, logistic regression Ch. 10 |
| Jurafsky & Martin — *Speech and Language Processing* | Logistic regression Ch. 4/6; NLP applications |
| Herman Kamper — lecture videos | Logistic regression; regularization |
| Andrej Karpathy — Micrograd | Scalar autograd from scratch |
| StatQuest — Josh Starmer | MLE, bias-variance, probability distributions |
| Stanford CS294A — Sparse Autoencoder notes | Gradient checking (Section 2.3) |
| ConvnetJS (Karpathy) | 2D classification demo |
| CNN Explainer (Polo Club) | CNN visualisation |
| Seeing Theory | Interactive probability |

### Northeastern Computing Resources
- **Open On Demand** — browser-based HPC access; used to run SpeechBrain VAD notebook
- **GitHub** — all project code submitted via GitHub; reproducibility required
- **Canvas** — all assignment submission; Qwickly for attendance
- **Piazza** — Q&A; component 2 requires posting questions and leaving comments
- **Office hours** — Room 1507, Mon 1–2 PM in-person; Mon 4–5 PM online on Teams
- **Learn Data Science meetup** — bi-weekly on campus (organized by TA Jane Feng); 2% extra credit for attending
