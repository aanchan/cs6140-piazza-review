# CS 6140 Midterm Review Lecture Plan
**Duration:** 2 hours | **Exam date:** April 6 | **Format:** In-person, paper-based

---

## What the Piazza data tells you

Mapping the 64 followup threads to clusters reveals where student understanding has real gaps vs. where peers gave each other good answers:

| Cluster | # threads |
|---|---|
| Backpropagation (chain rule, PyTorch graph, efficiency, vanishing gradients) | ~10 |
| MLE / likelihood / loss function connection | ~4 |
| PCA limitations (discriminative signal loss, Gram matrix) | ~4 |
| L1 vs L2 (mechanism, MAP connection) | ~3 |
| Activation functions (sigmoid vs ReLU choice) | ~3 |
| Neural network geometry / depth vs width | ~3 |
| Generative vs discriminative | ~2 |
| When/how to stop training | ~2 |

---

## Guiding principle

Use the Piazza questions as the curriculum — not your slides. Students have already seen the slides. What they need is *resolution* of confusion, not re-exposition.

---

## Block 0 — Setup (5 min)

Tell students exactly what the exam looks like: 14 questions, 10 pts each, best 8 count, written justification required. Show the rubric. Emphasise:

> **A correct answer with no justification earns at most 3 points. The justification is what they are being tested on.**

---

## Block 1 — The probabilistic backbone (20 min)
*Resolves: f16 (likelihood confusion), f26 (why NLL?), f27 (MSE vs BCE for sigmoid), f11 (L1/L2 MAP connection)*

This is the highest-leverage block. Students are confused about a chain of ideas that underlies questions across every major topic on the exam:

```
Data → likelihood → log-likelihood → NLL → loss function
                                      ↓
                                 + log prior → MAP → regularisation
                                                      L2 = Gaussian prior
                                                      L1 = Laplace prior
```

Draw this on the board once and walk through it. Then show the algebraic payoffs:
- NLL under Bernoulli model → **cross-entropy loss**
- NLL under Gaussian model → **MSE loss**

### Addressing f27 directly — MSE vs BCE is a context problem, not a wrong/right problem

The student question in f27 is well-founded, not confused. Michael Nielsen's book (neuralnetworksanddeeplearning.com) and the 3B1B MNIST videos both use **MSE + sigmoid** for the output layer across 10 classes. This is a deliberate pedagogical choice: the backprop math is simpler to follow, the δ equations come out cleanly, and it works well enough for MNIST. Students who did those pre-class resources have seen MSE+sigmoid used in practice and are right to ask why.

The actual answer to tell them:

| | MSE + sigmoid output | BCE + sigmoid output | CE + softmax output |
|---|---|---|---|
| **Dimensionality** | `a ∈ ℝᴷ`, `y ∈ ℝᴷ` — **vectors**: one sigmoid neuron per class, e.g. K=10 for MNIST | `aₖ ∈ ℝ`, `yₖ ∈ {0,1}` — **single neuron** k: one sigmoid output for the binary decision | `a ∈ ℝᴷ`, `y ∈ ℝᴷ` — **vectors**: one probability per class, y is one-hot |
| **Cost function** | `C = ½ ‖a − y‖² = ½ ∑ₖ(aₖ − yₖ)²` where each `aₖ = σ(zₖ)` independently | `C = −[yₖ log aₖ + (1−yₖ) log(1−aₖ)]` where `aₖ = σ(zₖ)` | `C = −∑ₖ yₖ log aₖ` where `aₖ = e^zₖ / ∑ⱼ e^zⱼ` |
| **Output δ** | `δₖ = (aₖ − yₖ) · σ'(zₖ)` = `(aₖ − yₖ) · aₖ(1−aₖ)` per neuron — σ' does not cancel | `δₖ = aₖ − yₖ` — σ' cancels via `aₖ(1−aₖ) · [−yₖ/aₖ + (1−yₖ)/(1−aₖ)]` | `δₖ = aₖ − yₖ` — softmax Jacobian simplifies the same way |
| **Vanishing gradient at output?** | Yes — when aₖ ≈ 0 or 1, aₖ(1−aₖ) ≈ 0 and that neuron's learning stalls | No — aₖ(1−aₖ) cancels out of δₖ entirely | No — softmax Jacobian cancels similarly |
| **Used in** | Nielsen's book, 3B1B MNIST — pedagogical, keeps backprop derivation clean | Binary classification, this course's VAD project | Multi-class classification; what you'd actually use for MNIST in practice |
| **Principled via MLE?** | No — MSE corresponds to a Gaussian output likelihood, not Bernoulli or categorical | Yes — NLL under Bernoulli model | Yes — NLL under categorical model |

The key sentence: **MSE+sigmoid works and Nielsen uses it to keep the backprop derivation clean. BCE+sigmoid is better in practice because the σ' term cancels, removing one source of vanishing gradient at the output layer. Both are valid; this course uses BCE because it connects cleanly to MLE and trains faster.**

The BCE+sigmoid cancellation is a specific algebraic fact, not a general property of "good loss functions."


---

## Block 2 — PCA: what it does, what it can't do (20 min)
*Resolves: f1 (edge cases), f3 (PCA vs autoencoders), f8 (PCA vs K-Means choice), f18 (discriminative info loss)*

Open with Jiaming Pei's question (f18) — the best question in the entire thread:

> *"PCA can silently discard a direction that perfectly separates two classes because that direction has low variance. How would you know this happened?"*

Walk through:
1. PCA maximises variance, not class separability
2. How to detect it: compare classification accuracy before and after PCA
3. Gram matrix: a computational shortcut when D >> N, not a different algorithm
4. PCA as linear autoencoder vs. neural autoencoders (f3) — one sentence: *PCA is the globally optimal linear autoencoder; neural autoencoders do non-linear PCA*
5. PCA vs K-Means (f8): continuous codes vs. discrete codes — both are signal compression

Use the scree plot slide to anchor the "how many components" question.

---

## Block 3 — Backpropagation: the one true explanation (30 min)
*Resolves: f4, f5, f15, f17, f20, f23, f24 — the largest cluster*

The peer answers in this cluster are partially correct but miss the mechanistic precision the exam requires. Structure as three precise statements, each with a worked number.

### Statement 1 — What `.backward()` does

`.backward()` traverses the computation graph built *during the forward pass* in reverse topological order, calling each node's stored backward function. It does not recompute — it reuses cached activations. This is why forward-pass activations must stay in GPU memory during training.

*Resolves f5, f15, f20 in one statement.*

### Statement 2 — The delta equations (write on board)

```
δ^(L)     = ∂L/∂a · σ'(z)          ← BP1: output error
∂L/∂b     = δ                       ← BP3: bias gradient
∂L/∂w     = δ · a_prev              ← BP4: weight gradient
∂L/∂a_prev = Σ_j  δ_j · w_jk       ← aggregation across downstream neurons
```

Walk through the BCE + sigmoid cancellation live:
- `∂L/∂a = -y/a + (1-y)/(1-a)`
- multiply by `σ'(z) = a(1-a)`
- result: `δ = a - y`

Then ask: *"Does this hold for ReLU?"* — No. The cancellation requires `σ'= a(1−a)` specifically.

### Statement 3 — The vanishing gradient problem

| Activation | Derivative | 10-layer worst case |
|---|---|---|
| Sigmoid | ≤ 0.25 | 0.25^10 ≈ **10⁻⁶** |
| ReLU | 0 or 1 | 1^10 = **1** (active region) |

Write both numbers. Early layers in a deep sigmoid network receive effectively zero gradient. This resolves f2, f10, f27 simultaneously.


---

## Block 4 — Training pipeline, inference, and deployment (15 min)

Use a single diagram with two columns:

| | Training | Inference |
|---|---|---|
| **Forward pass** | Yes — store activations | Yes — discard activations immediately |
| **Backward pass** | Yes | No |
| **What's in memory** | Weights + gradients + activations + optimizer state | Weights only |
| **Adam overhead** | 3× weight memory (m, v, gradient) | 0× |
| **Typical hardware** | Cloud GPU, large batch | Edge device or cloud, single request |

**Key number to remember:** For a 7B-parameter FP32 model:
- Inference: ~28 GB (weights only)
- Training with Adam: ~112+ GB (weights × 4 tensors)

**On-device vs cloud decision rule:** Ask one question — *"What is the latency budget?"*
- Hard real-time requirement (< 100 ms) → edge device, even if model is smaller
- Throughput matters more than latency → cloud, batch many requests together
- Privacy-sensitive data → edge, data never leaves the device

---

## Block 5 — ROC curves: 3 facts (10 min)
*Resolves: f2, f9*

### Fact 1 — What AUC actually means
AUC = 0.85 means **85% probability that the model ranks a random positive above a random negative**. It does not mean 85% accuracy. The all-negative naive classifier on a 99/1 imbalanced dataset: accuracy = 99%, AUC = 0.5.

### Fact 2 — Why ROC is invariant to class prevalence
TPR = TP / (TP + FN) and FPR = FP / (FP + TN) both normalise by their own class totals. Doubling the number of positives doubles both TP and the denominator — TPR is unchanged. Accuracy, precision, and F1 do not have this property.

### Fact 3 — Operating point vs model quality
- AUC measures the **classifier** (its ranking ability across all thresholds)
- The **threshold** is a deployment decision driven by cost structure
- Fraud detection → high recall, tolerate false alarms → low threshold
- Spam filter → high precision, tolerate missed spam → high threshold

---

## Block 6 — Exam simulation: 3 worked questions (15 min)

Walk through three questions exactly as students will face them. Spend time on the *justification*, not just the answer letter.

### L1 sparsity mechanism

Most students know L1 produces sparse weights. The exam tests whether they know *why*. The justification must mention: subdifferential of |w| at w=0 includes 0, so a weight that reaches zero has no restoring force. "L1 produces sparsity" alone earns 3/10.

### PCA discards the discriminative direction

This is the most commonly misunderstood question type. Students will read option C ("PCA automatically detects class structure") and be tempted. Walk through why PCA is unsupervised — it never sees class labels — and why maximum-variance direction and maximum-discriminative direction are independent concepts.

### Vanishing gradient: compute 0.25^10

Walk through the arithmetic explicitly. The justification must show: σ'(z) ≤ 0.25 at each of 10 layers, product = 0.25^10 ≈ 10^{-6}, conclusion: early layers receive negligible gradient signal. This is the level of precision the rubric rewards.

---

## Closing (5 min)

Two things to say explicitly:

1. **The justification is the exam.** Write the mechanism, not the conclusion.
   - Weak: "L1 produces sparsity" → 3/10
   - Strong: "the subdifferential of |w| at w=0 includes 0, so once a weight reaches zero no restoring force acts on it" → 9/10

2. **If you can't remember a formula, reason from the definition.** The exam rewards understanding over memorisation. Deriving the BCE+sigmoid gradient from scratch during the exam is worth full marks.

---

## What NOT to cover

- **CNN and RNN backprop** (f13, f14, f21) — out of scope for the midterm, say so explicitly
- **PCA from scratch** — students have seen the derivation; go straight to implications

