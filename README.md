# ClimbingBoardGPT

[![Live Demo](https://img.shields.io/badge/demo-webapp-teal)](https://cbgpt.pawelsarkowicz.xyz)

**ClimbingBoardGPT uses AI to generate new climbing routes and predict their difficulty — for Tension Board 2 and Kilter Board.**

The route generator follows the small causal-transformer recipe from Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT): train a model to predict the next token, then sample from it autoregressively. The twist here is that the "language" is made of board, angle, grade, and hold-role tokens instead of text.

You give it a board, a wall angle, and a target grade. It gives you a route. You can also paste in a route you already know, and it will guess the grade.

---

## Table of contents

- [What is this, exactly?](#what-is-this-exactly)
- [What are Tension Board 2 and Kilter Board?](#what-are-tension-board-2-and-kilter-board)
- [What can it do?](#what-can-it-do)
- [Try it](#try-it)
- [How it works](#how-it-works)
- [Climbing-board ML $\leftrightarrow$ Linear Algebra Dictionary](#climbing-board-ml--linear-algebra-dictionary)
- [Quantitative results](#quantitative-results)
- [Setup](#setup)
- [Run the web demo locally](#run-the-web-demo-locally)
- [CLI demos](#cli-demos)
- [Full training pipeline](#full-training-pipeline)
- [API endpoints (for the webapp)](#api-endpoints-for-the-webapp)
- [Repository layout](#repository-layout)
- [Important caveats](#important-caveats)
- [Background and related work](#background-and-related-work)
- [License](#license)

---

## What is this, exactly?

If you've climbed on a **Tension Board 2 (TB2)** or a **Kilter Board**, you know these are standardized training boards with a fixed set of holds. Routes on these boards are described as a list of holds and their roles (start, foot, hand, finish). The holds are identified by placement ID numbers and the route is stored as a short string like `p652r5p631r6p322r6p326r7`.

This project trains two small AI models on hundreds of thousands of real community-set routes from both boards:

- **A route generator** — you ask for a V6 at 40° on the Kilter, and the model samples a novel sequence of holds that should produce something around that difficulty.
- **A grade predictor** — you give it any route (board + angle + holds), and the model estimates the difficulty.

Both models are transformer-based neural networks, the same family of architecture behind large language models. Here the "language" is not English words but climbing-hold tokens: each hold-role combination gets its own symbol, and a route is a short sentence in that language.

The whole thing is small by modern standards (~1.2–1.4M parameters each) and runs on a CPU.

---

## What are Tension Board 2 and Kilter Board?

**Tension Board 2 (TB2)** is an adjustable training wall made by Tension Climbing. It has a fixed set of holds placed in a regular grid. Climbers set and share routes through a companion app; the community has set tens of thousands of problems. We work with the 12x12 ft mirror in this project.

**Kilter Board** is a similar product from Kilter (Setter Closet). It also has a large library of community-set problems. We work with the 16x12 ft Kilter original board in this project.

Both boards store routes as placement-ID strings. That is what this project trains on.

---

## What can it do?

| Feature | How to use it |
|---|---|
| Generate new routes | Web app or CLI script |
| Predict grade from holds | Web app or CLI script |
| Visualize routes on a board image | CLI script, saved as PNG/SVG |
| Run a local web demo | Docker or `uvicorn` |
| Retrain from scratch | Four numbered scripts |

---

## Try it

The live demo is at **[cbgpt.pawelsarkowicz.xyz](https://cbgpt.pawelsarkowicz.xyz)**. You can:

- Pick a board (TB2 or Kilter), a wall angle, and a target grade, and click **Generate** to get a new route drawn on the board image.
- Paste a frames string — the compact route code used by the board apps — into the **Predict** tab to estimate the grade of any route you already know.

---

## How it works

### Turning a route into text

Every route is converted into a short sequence of symbols, one per hold:

```
<BOS> <BOARD_TB2> <ANGLE_40> <GRADE_V6>
<TB2_p344_start> <TB2_p369_middle> <TB2_p603_finish>
<EOS>
```

`<BOS>` and `<EOS>` mark the start and end. The board, angle, and grade tokens say: "this is a TB2 problem, set at 40 degrees, graded V6." The rest are hold tokens — each one encodes a specific hold and whether it is a start, middle, finish, or foot hold.

Both boards share one vocabulary, so the model can learn patterns from TB2 and Kilter routes together without confusing hold positions from one board with the other.

### Grade predictor

The grade predictor reads the sequence above (minus the grade token, which it has to guess) and outputs a single number. It is a **transformer encoder** — roughly the same kind of model used for text classification, just applied to climbing holds instead of words.

It also gets the physical (x, y) board coordinates of each hold as extra input, so it can reason about route geometry: how high the holds are, how far apart they are, whether the route traverses sideways, etc.

Accuracy on the held-out test set: **79% within ±1 V-grade**.

### Route generator

The route generator is a small **GPT-style model** — the same general idea as ChatGPT, but tiny and trained only on climbing routes. You give it the prompt:

```
<BOS> <BOARD_KILTER> <ANGLE_40> <GRADE_V6>
```

and it predicts what hold token should come next, then the next, then the next, until it produces an `<EOS>` token. The result is a novel sequence of holds that the model thinks is a plausible V6 on the Kilter at 40°.

**~89% of generated routes pass basic structural checks** (has a start hold, has a finish hold, holds exist on the right board, no duplicates).

---

## Climbing-board ML $\leftrightarrow$ Linear Algebra Dictionary

The single most useful thing to keep in mind: **a route is a short sequence of discrete symbols**, and almost every operation in this project is a standard linear-algebra operation on the matrices and vectors built from those symbols.

Each pipeline stage (tokenization, encoder training, generator training, evaluation) has its own table below. The project-wide reference, with the notebook(s) where each term appears, follows.

Notebook numbering: **01** Tokenization · **02** Grade prediction (transformer encoder) · **03** Route generation (GPT-style decoder) · **04** Generated-route evaluation.

### 1. Routes, tokens, and the vocabulary

| Term | Linear-algebra meaning | Notebooks |
|------|------------------------|:---------:|
| **Hold** | A discrete symbol drawn from a finite placement set — one element of the board's hold-index set | 01 |
| **Role** | A label (start, middle, finish, foot) attached to a hold; part of the token's identity, not a separate axis | 01 |
| **Frames string** | Compact serialization $p\,i_1\,r\,j_1\,p\,i_2\,r\,j_2\cdots$ of a route — the source object before tokenization | 01 |
| **Route** | An ordered list of $(\text{placement}, \text{role})$ pairs; later mapped to a sequence in $\mathbb{R}^{L\times d}$ | 01 |
| **Vocabulary** $V$ | The finite symbol set, $\|V\| = 4438$, identified with the standard basis $\{e_1,\dots,e_{\|V\|}\}$ of $\mathbb{R}^{\|V\|}$ | 01 |
| **`stoi` / `itos`** | The bijection between tokens and basis-vector indices $\{1,\dots,\|V\|\}$ | 01 |
| **Special tokens** | Distinguished basis vectors ($e_{\text{PAD}}, e_{\text{BOS}}, e_{\text{EOS}}, \dots$) reserved for structural roles | 01 |
| **Board namespacing** | A direct-sum partition $V = V_{\text{TB2}} \oplus V_{\text{Kilter}} \oplus V_{\text{special}} \oplus \cdots$ so placement IDs cannot collide | 01 |
| **Encode / decode** | Convert tokens to basis indices and back: $\text{encode}(t) = i$ where $t \equiv e_i$ | 01 |
| **Canonical ordering** | A fixed sort key on holds (role, $y$, $x$, placement_id) selecting one representative per orbit of the permutation group $S_L$ | 01 |
| **Grade tokens** | Discrete grade bins $V_0,\dots,V_{16}$ — a 17-symbol sub-vocabulary indexed by `GRADE_TO_V` | 01 |
| **Train / val / test split** | A partition of the row index set into three disjoint subsets, stratified by board $\times$ grade | 01 |

### 2. Embeddings: discrete tokens → $\mathbb{R}^d$

| Term | Linear-algebra meaning | Notebooks |
|------|------------------------|:---------:|
| **One-hot map** $\phi$ | $\phi: V \to \mathbb{R}^{\|V\|}$, $t \mapsto e_t$ — the canonical embedding of a discrete symbol into the standard basis | 01, 02, 03 |
| **Token embedding matrix** $E$ | Learned linear map $E \in \mathbb{R}^{d\times \|V\|}$; column $i$ is the vector assigned to token $i$ in $\mathbb{R}^d$ (with $d = 128$) | 02, 03 |
| **Embedding lookup** | The matrix-vector product $E e_t$ — selects column $t$ of $E$ | 02, 03 |
| **`padding_idx=0`** | Pins column $0$ of $E$ to the zero vector; never updated by gradient descent | 02, 03 |
| **Position embedding** | A second embedding matrix $E_{\text{pos}} \in \mathbb{R}^{d \times L_{\max}}$, indexed by position $n \in \{0,\dots,L-1\}$ rather than token | 02, 03 |
| **Coordinate features** $C$ | A fixed matrix $C \in \mathbb{R}^{\|V\| \times 3}$ whose rows are $(x_{\text{norm}}, y_{\text{norm}}, \mathbf{1}_{\text{hold}})$ — one geometric vector per token ID | 02 |
| **Coordinate normalisation** | Affine rescaling $x \mapsto 2(x - x_{\min})/(x_{\max} - x_{\min}) - 1$ mapping board coordinates into $[-1, 1]$ | 01, 02 |
| **Coordinate projection** $W_{\text{coord}}$ | Learned linear map $W_{\text{coord}} \in \mathbb{R}^{d \times 3}$ that lifts physical coordinates into $\mathbb{R}^d$ | 02 |
| **Token + position + coord** | Three vectors added componentwise in $\mathbb{R}^d$: $X_{n,:} = E e_{t_n} + E_{\text{pos}} e_n + W_{\text{coord}}\, C_{t_n}$ | 02 |
| **Sequence matrix** $X$ | Stacked row embeddings: $X \in \mathbb{R}^{L \times d}$, one row per token | 02, 03 |

### 3. The transformer encoder (grade predictor)

| Term | Linear-algebra meaning | Notebooks |
|------|------------------------|:---------:|
| **Self-attention** | A learned bilinear form followed by a convex combination: $X \mapsto \mathrm{softmax}(QK^\top/\sqrt{d_h})\,V$ | 02 |
| **Query / Key / Value** | Three learned linear images of $X$: $Q = X W_Q$, $K = X W_K$, $V = X W_V$ with $W_\bullet \in \mathbb{R}^{d \times d}$ | 02 |
| **Multi-head attention** | Block-diagonal decomposition of $\mathbb{R}^d$ into 4 orthogonal 32-dim subspaces that attend independently | 02 |
| **Gram matrix** $QK^\top$ | The $L \times L$ matrix of learned inner products — entry $(i,j)$ is the query-key affinity $\langle Q_i, K_j \rangle$ | 02 |
| **$1/\sqrt{d_h}$ scaling** | Variance normalisation so the inner products have $O(1)$ variance as $d_h$ grows | 02 |
| **Row-wise softmax** | Each row of $QK^\top/\sqrt{d_h}$ is mapped onto the probability simplex $\Delta^{L-1}$ | 02 |
| **Convex combination** | $\mathrm{softmax}(\cdot)\,V$ — each output row is a learned weighted average of value rows | 02 |
| **`key_padding_mask`** | Zeros out rows of the attention matrix that correspond to pad tokens | 02 |
| **Pre-norm residual** | $X \leftarrow X + \mathrm{Sublayer}(\mathrm{LN}(X))$ — additive update in $\mathbb{R}^{L \times d}$ | 02 |
| **LayerNorm** | Affine map subtracting the mean and dividing by the std of each row, then scaling/shifting — projection onto the unit sphere followed by a learned affine transform | 02 |
| **Feed-forward block** | Two linear maps $\mathbb{R}^d \xrightarrow{W_1} \mathbb{R}^{4d} \xrightarrow{\text{GELU}} \mathbb{R}^{4d} \xrightarrow{W_2} \mathbb{R}^d$ | 02 |
| **`<CLS>` pooling** | Selecting row $0$ of the encoder output as the route-level summary vector in $\mathbb{R}^d$ | 02 |
| **Regression head** | A learned nonlinear functional $\mathbb{R}^d \xrightarrow{\text{Linear}} \mathbb{R}^d \xrightarrow{\text{GELU}} \mathbb{R}^d \xrightarrow{\text{Linear}} \mathbb{R}$ returning a difficulty scalar | 02 |

### 4. The causal transformer (route generator)

| Term | Linear-algebra meaning | Notebooks |
|------|------------------------|:---------:|
| **Causal mask** | Strictly upper-triangular boolean matrix $M \in \{0,1\}^{L \times L}$, $M_{ij} = \mathbf{1}_{j > i}$ — token $i$ cannot attend to token $j > i$ | 03 |
| **Masked attention** | Attention restricted to the lower-triangular part of the Gram matrix | 03 |
| **Teacher forcing** | Training pairs $(X, Y)$ with $Y$ equal to $X$ shifted one position right — the model learns the conditional $p(x_{n+1} \mid x_{1:n})$ | 03 |
| **Logits** $\ell$ | A vector $\ell \in \mathbb{R}^{\|V\|}$ whose $i$-th entry is the inner product $\langle E_i, h \rangle$ between the current hidden state and the embedding of token $i$ | 03 |
| **Weight tying** | `lm_head.weight = token_emb.weight`; the output projection reuses $E^\top$, so logit $i = \langle E_i, h \rangle$ — read and score in the same geometry | 03 |
| **Cross-entropy loss** | $\mathcal{L} = -\log \dfrac{\exp(\langle E_y, h\rangle)}{\sum_j \exp(\langle E_j, h\rangle)}$ — negative log of the softmax-inner-product with the target basis vector | 03 |
| **Perplexity** | $\exp(\mathcal{L}_{\text{CE}})$ — effective average number of tokens the model was choosing between at each step | 03 |
| **Block size** | The context window $L_{\max}$ — maximum number of tokens the model can attend over | 03 |

### 5. Sampling and decoding

| Term | Linear-algebra meaning | Notebooks |
|------|------------------------|:---------:|
| **Logits vector** $\ell$ | An arbitrary vector in $\mathbb{R}^{\|V\|}$ about to be mapped to a probability distribution | 03, 04 |
| **Temperature** $T$ | Divides $\ell$ by $T$; rescales the logit vector's spread. $T \to 0^+$ concentrates on the argmax basis vector; $T \to \infty$ flattens to the barycentre of the simplex | 03, 04 |
| **Top-$k$ filter** | Masks every logit outside the top $k$ to $-\infty$, projecting the output distribution onto a $k$-element face of the simplex $\Delta^{\|V\|-1}$ | 03, 04 |
| **`forbidden_ids`** | Sets specified logits to $-\infty$ — hard projection onto a face of the simplex that excludes those basis vectors | 03 |
| **Softmax** | Map $\mathbb{R}^{\|V\|} \to \mathrm{int}(\Delta^{\|V\|-1})$, $\ell_i \mapsto \exp(\ell_i)/\sum_j \exp(\ell_j)$ | 03, 04 |
| **Multinomial sample** | Draw one basis vector $e_i$ with probability $p_i$ | 03, 04 |
| **Autoregressive loop** | Ancestral sampling: concatenate $e_i$ to the sequence, re-run the model, repeat until `<EOS>` | 03, 04 |

### 6. Training and optimisation

| Term | Linear-algebra meaning | Notebooks |
|------|------------------------|:---------:|
| **MSE loss** | Squared Euclidean distance $\|y - \hat y\|^2 / N$ — used for grade regression | 02 |
| **AdamW** | Adaptive-gradient SGD using a diagonal preconditioner (running means of $\nabla^2$) with decoupled L2 pull toward $\mathbf{0}$ | 02, 03 |
| **Weight decay** | Decoupled L2 regularisation that shrinks every parameter vector toward $\mathbf{0}$ each step | 02, 03 |
| **Gradient clipping** | Rescale the gradient: $g \leftarrow \min(1, 1/\|g\|_2)\, g$ to cap its spectral norm at $1$ | 02, 03 |
| **Learning rate** $\eta$ | Step size along $-\nabla \mathcal{L}$ in parameter space | 02, 03 |
| **Batch** | A uniform sample of training rows; loss is the mean over the batch | 02, 03 |
| **Early stopping** | Halt when the validation metric stops improving; restore the parameters with best validation score | 02, 03 |
| **Train / val / test** | Three disjoint samples of rows used for fitting, hyperparameter selection, and final evaluation | 02, 03 |

### 7. Evaluation metrics

| Term | Linear-algebra meaning | Notebooks |
|------|------------------------|:---------:|
| **Jaccard similarity** | $\|A \cap B\| / \|A \cup B\| = \langle \mathbf{1}_A, \mathbf{1}_B\rangle \,/\, (\|\mathbf{1}_A\|_1 + \|\mathbf{1}_B\|_1 - \langle \mathbf{1}_A, \mathbf{1}_B\rangle)$ — a normalised inner product of indicator vectors on $\{0,1\}^N$ | 04 |
| **Novelty distance** | $1 - \mathrm{Jaccard}(A, B)$ — a metric on the Boolean hypercube | 04 |
| **Nearest real route** | Argmax of Jaccard over the training set — an exhaustive nearest-neighbour query under the Jaccard metric | 04 |
| **MAE** | $\ell_1$ distance $\|y - \hat y\|_1 / N$ between prediction and target vectors | 04 |
| **RMSE** | $\ell_2$ distance $\sqrt{\|y - \hat y\|_2^2 / N}$ | 04 |
| **$R^2$** | Fraction of variance explained: $1 - \|y - \hat y\|^2 / \|y - \bar y\, \mathbf{1}\|^2$ | 04 |
| **Within ±$k$ V-grades** | $\mathbf{1}[\|y - \hat y\|_1 \le k]$ after grade binning | 04 |
| **Hand-reach distances** | `scipy.spatial.distance.pdist(holds[["x","y"]].values)` — pairwise Euclidean distances in $\mathbb{R}^2$ | 04 |
| **Composite ranking score** | Learned linear combination of validity, novelty, and grade-consistency features used to rank generated routes | 04 |

---

## Quantitative results

These numbers are from the full training run documented by this repository. The notebooks in `notebooks/` are self-contained walkthroughs of the pipeline stages. The reported pipeline run was executed on Kaggle; notebooks 01-04 took about 8h 1m 59s total using GPU T4 x2.

In practice: the grade model is usually within one V-grade, and the generator usually makes structurally valid routes, but exact grade control is still imperfect.

### Dataset size

| Board | Training routes | Validation | Test |
|---|---:|---:|---:|
| TB2 | 33,719 | 4,430 | 4,447 |
| Kilter | 223,112 | 27,555 | 27,822 |
| **Total** | **256,831** | **31,985** | **32,269** |

Shared vocabulary: **$\|V\| = 4438$ tokens** (6 special + 2 board + 12 angle + 16 grade + 4,402 hold-role tokens).

### Grade prediction accuracy

The model has ~1.17M parameters (scalar entries across all parameter tensors). Early stopping selected epoch 11 (validation MAE $\approx 1.488$).

| Metric | Overall | TB2 | Kilter |
|---|---:|---:|---:|
| Exact V-grade | 35.8% | 35.8% | 35.8% |
| Within ±1 V-grade | 79.2% | 79.4% | 79.1% |
| Within ±2 V-grades | 94.9% | 95.5% | 94.8% |
| $R^2$  | 0.763 | 0.793 | 0.758 |

### Route generation

The generator has ~1.41M parameters. Best validation perplexity: 24.3.

| Metric | TB2 | Kilter |
|---|---:|---:|
| Routes evaluated | 200 | 200 |
| Structurally valid | 91.5% | 86.0% |
| Exact requested grade (critic) | 34.5% | 37.0% |
| Within ±1 V-grade (critic) | 73.0% | 79.5% |
| Within ±2 V-grades (critic) | 91.0% | 96.5% |
| Mean novelty (Jaccard distance) | 0.656 | 0.643 |

---

## Setup

Requires Python 3.12+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## Run the web demo locally

### Without Docker

```bash
uvicorn webapp.app:app --host 127.0.0.1 --port 8055
```

Then open `http://127.0.0.1:8055`.

The webapp needs these files (generated by training, or copied from a previous run):

```
models/joint_route_gpt_generator.pth
models/joint_transformer_grade_predictor.pth
data/processed/tokenized/token_metadata.csv
data/processed/tokenized/token_vocab.json
data/processed/tokenized/route_sequences.csv
configs/
images/
```

### With Docker

```bash
docker compose -f docker-compose.webapp.yml up -d --build
```

Binds to `127.0.0.1:8055`.

### With Docker behind a reverse proxy

Use the reverse-proxy compose file when another container, such as Caddy,
owns the public ports and shares an external Docker network with the webapp.
The file defaults that external network to `caddy`:

```bash
docker network create caddy
docker compose -f docker-compose.webapp.proxy.yml up -d --build
```

This does not publish a host port. It exposes port `8055` only on the proxy
network. For Caddy, a site block can reverse proxy to the webapp container:

```caddyfile
cbgpt.example.com {
    reverse_proxy climbingboardgpt-webapp:8055
}
```

---

## CLI demos

Once the trained model checkpoints are in `models/`, you can run demos from the terminal.

### Generate routes

```bash
# TB2, 40 degrees, V6, 4 routes
python scripts/demo_generate_tb2.py --angle 40 --grade 6 --n 4

# Kilter
python scripts/demo_generate_kilter.py --angle 40 --grade 6 --n 4

# With custom temperature and top-k sampling
python scripts/demo_generate_and_visualize.py \
  --board tb2 --angle 40 --grade 6 --n 4 \
  --temperature 0.9 --top-k 50
```

Generated routes are saved to:

```
outputs/demo_routes/<board>/angle_<angle>/V<grade>/
├── generated_routes.csv
├── generated_route_001.png
├── generated_route_001.svg
...
```

**What does temperature do?**

| Temperature | Effect |
|---:|---|
| `0.3`–`0.6` | Conservative — picks safer, more common moves |
| `0.9` | Balanced default |
| `1.0` | Samples directly from learned probabilities |
| `1.1`–`1.3` | More creative — can produce weirder routes |

**What does top-k do?**

| Top-k | Effect |
|---:|---|
| `1` | Greedy — always picks the single most likely token |
| `5`–`15` | Very focused — limited vocabulary per step |
| `50` | Balanced default |
| `200`–`500` | Less restricted — more variety per step |

The two parameters work at different stages of the same step. At each position in the sequence, top-k first discards every token outside the k most likely candidates. Temperature then rescales the probabilities of those remaining candidates before drawing from them. Setting `top_k=1` makes temperature irrelevant — there is only one candidate left. With a larger top-k, temperature controls how evenly spread the draw is: low temperature concentrates weight on the top few candidates, high temperature flattens the distribution toward equal probability across all k.

In short: **top-k controls the breadth of what is considered; temperature controls how adventurously the model picks within that breadth.** The defaults (`top_k=50`, `temperature=0.9`) aim for a balance between variety and structural coherence.


### Predict grade

```bash
# TB2
python scripts/demo_predict_tb2.py \
  --angle 40 --frames 'p652r5p631r6p322r6p326r7'

# Kilter
python scripts/demo_predict_kilter.py \
  --angle 40 --frames 'p1127r12p1196r13p1216r13p1388r14'
```

Example output:

```
Board:      Tension Board 2 Mirror (tb2)
Angle:      40°
Frames:     p652r5p631r6p322r6p326r7
Predicted:  V6
```

Additional flags: `--json` for machine-readable output, `--visualize` to save a board image, `--show-tokens` to inspect the token sequence.

---

## Full training pipeline

To train from scratch you need the raw board databases at:

```
data/raw/tb2.db
data/raw/kilter.db
```

These can be downloaded with the [`BoardLib`](https://github.com/lemeryfertitta/BoardLib) CLI — the commands are recorded in `configs/tb2.json` and `configs/kilter.json`.

Then run the four scripts in order:

```bash
python scripts/01_tokenize_routes.py --boards tb2,kilter
python scripts/02_train_grade_predictor.py
python scripts/03_train_route_generator.py
python scripts/04_evaluate_generated_routes.py
```

This produces trained checkpoints in `models/` and evaluation outputs in `data/processed/`.

### Fast smoke test (no GPU needed)

To verify the pipeline runs end-to-end without retraining the real models, once the raw board databases are in `data/raw/`:

```bash
python scripts/01_tokenize_routes.py \
  --out-dir /tmp/cbgpt_smoke/tokenized \
  --max-routes-per-board 20

python scripts/02_train_grade_predictor.py \
  --tokenized-dir /tmp/cbgpt_smoke/tokenized \
  --out-dir /tmp/cbgpt_smoke/grade_prediction \
  --model-dir /tmp/cbgpt_smoke/models \
  --smoke-test

python scripts/03_train_route_generator.py \
  --tokenized-dir /tmp/cbgpt_smoke/tokenized \
  --out-dir /tmp/cbgpt_smoke/generation \
  --model-dir /tmp/cbgpt_smoke/models \
  --smoke-test \
  --generate-angles 40 \
  --generate-grades 6

python scripts/04_evaluate_generated_routes.py \
  --tokenized-dir /tmp/cbgpt_smoke/tokenized \
  --generated-dir /tmp/cbgpt_smoke/generation \
  --out-dir /tmp/cbgpt_smoke/evaluation \
  --grade-model-path /tmp/cbgpt_smoke/models/joint_transformer_grade_predictor.pth \
  --device cpu
```

The numbers from this run are meaningless — it only checks that the code runs.

---

## API endpoints (for the webapp)

```
GET  /api/health
GET  /api/boards
POST /api/generate
POST /api/predict
```

Example generation payload:

```json
{
  "board": "tb2",
  "angle": 40,
  "grade": 6,
  "temperature": 0.9,
  "top_k": 50,
  "max_new_tokens": 40
}
```

Example prediction payload:

```json
{
  "board": "kilter",
  "angle": 40,
  "frames": "p1127r12p1196r13p1216r13p1388r14"
}
```

---

## Repository layout

```
ClimbingBoardGPT/
├── configs/          Board-specific config files (role IDs, angle ranges, etc.)
├── data/
│   ├── raw/          Raw SQLite databases (not in git)
│   └── processed/    Tokenized data and training outputs (not in git)
├── images/           Board background images
├── models/           Trained model checkpoints (not in git)
├── notebooks/        Executed Jupyter notebooks documenting each pipeline step
├── scripts/          Training scripts (01–04) and CLI demo scripts
├── src/climbingboardgpt/   Importable package — models, tokenization, inference, etc.
├── tests/            Unit tests
├── webapp/           FastAPI server + browser-side SVG route builder
├── docker-compose.webapp.yml
├── docker-compose.webapp.proxy.yml
├── requirements.txt
└── pyproject.toml
```

The main package modules:

| Module | What it does |
|---|---|
| `config.py` | Loads board JSON configs and role mappings |
| `data.py` | Reads from the SQLite board databases |
| `tokenization.py` | Converts frames strings to/from token sequences |
| `datasets.py` | PyTorch dataset adapters for training |
| `models.py` | Transformer encoder (grade predictor) and GPT (generator) |
| `generation.py` | Sampling, validity checks, frames reconstruction |
| `inference.py` | Model loading and inference helpers used by the webapp and demos |
| `evaluation.py` | Validity, novelty, and grade-consistency metrics |
| `visualization.py` | Board image overlays |
| `grades.py`, `metrics.py`, `utils.py` | Grade mapping, reporting helpers |

---

## Important caveats

- **Generated routes are machine-made candidates.** Use at your own risk. They are not guaranteed to be safe, fun, or even physically possible.
- **Grade predictions are estimates, not ground truth.** Climbing grades are subjective, board-dependent, and noisy even in the training data.
- **The hold sequence is a canonical ordering, not intended beta.** The model sorts holds by role and position; this is not necessarily the order you would climb them.
- **This is a research/hobby project**, not affiliated with Tension Climbing or Kilter/Setter Closet.

---

## Background and related work

This repo is the transformer/GPT follow-up to two earlier analysis projects:
- [Tension-Board-2-Analysis](https://github.com/psark007/Tension-Board-2-Analysis)
- [Kilter-Board-Analysis](https://github.com/psark007/Kilter-Board-Analysis)

Acknowledgement: the route generator draws on ideas from Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).

Board layouts, hold metadata, and route data are from the [Tension Board 2](https://tensionclimbing.com/products/tension-board-2) and [Kilter Board](https://settercloset.com/collections/kilter-board) apps, loaded via [`BoardLib`](https://github.com/lemeryfertitta/BoardLib). This project is unaffiliated with Tension Climbing or Kilter.

---

## License

MIT — see [LICENSE](LICENSE). Educational use.
