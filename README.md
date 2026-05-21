# ClimbingBoardGPT

**Applying LLM-style transformer techniques to climbing board route generation and grade prediction.**

This project treats climbing routes as language and trains transformer models on data from the **Tension Board 2 Mirror** and **Kilter Board Original** — learning to predict grades and generate entirely new routes.

---

## The Core Idea

Large language models process text as sequences of tokens and learn statistical patterns from billions of examples. Climbing routes have the same structure:

| NLP Concept | Climbing Analog |
|---|---|
| Word / Subword | Hold token (`TB2_p344_start`) |
| Sentence | Route (sequence of holds) |
| Document language | Board type (TB2 vs Kilter) |
| POS tag | Semantic role (start / middle / finish / foot) |
| Genre / Domain | Angle + Grade conditioning |
| Special tokens | `<BOS>`, `<EOS>`, `<PAD>`, `<CLS>`, `<MASK>`, `<UNK>` |

A route becomes a symbolic sequence:

```text
<BOS> <BOARD_TB2> <ANGLE_40> <GRADE_V6>
<TB2_p344_start> <TB2_p369_middle> <TB2_p603_finish>
<EOS>
```

The same transformer architectures that power GPT and BERT can learn "climb grammar" — which holds tend to follow which, how start holds differ from finish holds, and how difficulty emerges from spatial relationships.

---

## What This Repo Does

### 1. Tokenization (`01_tokenize_routes`)

Converts raw SQLite data into tokenized sequences:

- Parses `frames` strings (e.g., `p344r5p369r6p603r7`) into structured hold records
- Maps board-specific role IDs to shared semantic roles (TB2: 5/6/7/8 → Kilter: 12/13/14/15 → `start`/`middle`/`finish`/`foot`)
- Sorts holds canonically by (role priority, y-position, x-position)
- Generates two sequence versions:
  - **With grade** — for GPT generation training
  - **Without grade** — for BERT-style grade prediction
- Builds a shared vocabulary (~4,400 tokens), stratified train/val/test splits, and coordinate metadata

### 2. Grade Prediction (`02_train_grade_predictor`)

Trains a **transformer encoder** (BERT-style) to predict climb difficulty:

- Input: `<CLS> <BOARD_TB2> <ANGLE_40> <TB2_p344_start> ...` (grade excluded)
- Output: Single difficulty score (regression)
- Coordinate features (x, y, is_hold) are projected and added to token embeddings
- Joint training across both boards with board-conditioning tokens

**Results (joint model, test set):**

| Metric | Overall | TB2 | Kilter |
|---|---|---|---|
| MAE | 1.47 | 1.42 | 1.48 |
| R² | 0.787 | 0.816 | 0.782 |
| Within ±1 V-grade | 80.1% | 81.3% | 80.0% |
| Within ±2 V-grades | 95.3% | 96.1% | 95.2% |

### 3. Route Generation (`03_train_route_generator`)

Trains a **GPT-style causal transformer** to generate new routes:

- Input prompt: `<BOS> <BOARD_TB2> <ANGLE_40> <GRADE_V6>`
- Output: Sequence of hold tokens ending with `<EOS>`
- Uses causal masking (each position attends only to previous positions)
- Generation uses temperature sampling and top-k filtering

**Training results:**
- Best validation perplexity: ~24.6
- 88.8% basic validity rate for generated routes

### 4. Evaluation (`04_evaluate_generated_routes`)

Evaluates generated routes on four dimensions:

- **Validity**: Structural correctness (start/finish holds, no duplicates, single board)
- **Novelty**: Jaccard distance from nearest real route
- **Geometric plausibility**: Height, width, reach distances
- **Grade consistency**: Uses the trained grade predictor as a critic

**Evaluation results:**

| Metric | TB2 | Kilter |
|---|---|---|
| Basic validity | 87.0% | 90.5% |
| Mean novelty distance | 0.661 | 0.642 |
| Exact V-grade match | 27.5% | 33.5% |
| Within ±1 V-grade | 66.0% | 67.5% |
| Within ±2 V-grades | 91.0% | 90.0% |

---

## Key Design Decisions

### Board Namespacing

Hold tokens include the board prefix (`TB2_p344_start` vs `KILTER_p1084_start`). Placement 344 on TB2 is a completely different physical hold than placement 344 on Kilter — the prefix prevents ID collisions.

### Semantic Role Mapping

Different boards use different numeric role IDs, but they all map to the same semantic roles:

| Role | TB2 | Kilter |
|---|---|---|
| Start | 5 | 12 |
| Middle | 6 | 13 |
| Finish | 7 | 14 |
| Foot | 8 | 15 |

This shared vocabulary lets the model learn transferable patterns across boards.

### Coordinate Features

Each hold token carries physical (x, y) position information that gets projected and added to token embeddings. This gives the model direct spatial knowledge — similar to how some vision-language models inject spatial features.

### Conditioning Tokens

Routes are prefixed with board, angle, and grade tokens. This is analogous to how modern LLMs use system prompts to condition generation.

---

## Repository Structure

```text
ClimbingBoardGPT/
├── configs/
│   ├── tb2.json                    # Tension Board 2 configuration
│   └── kilter.json                 # Kilter Board configuration
├── data/
│   ├── raw/                        # SQLite databases (not in repo)
│   └── processed/
│       ├── tokenized/               # Tokenized route data
│       ├── grade_prediction/        # Grade predictor outputs
│       ├── generation/              # Generated route data
│       └── evaluation/              # Evaluation results
├── models/                          # Saved model checkpoints
├── notebooks/
│   ├── 01_unified_route_tokenization.ipynb
│   ├── 02_joint_transformer_grade_prediction.ipynb
│   ├── 03_joint_nanogpt_route_generator.ipynb
│   └── 04_generated_route_evaluation.ipynb
├── scripts/
│   ├── 01_tokenize_routes.py
│   ├── 02_train_grade_predictor.py
│   ├── 03_train_route_generator.py
│   └── 04_evaluate_generated_routes.py
├── src/climbingboardgpt/
│   ├── __init__.py
│   ├── config.py                   # Board configuration loading
│   ├── data.py                     # SQLite data loading
│   ├── datasets.py                 # PyTorch Dataset classes
│   ├── evaluation.py               # Route evaluation functions
│   ├── generation.py               # Route generation logic
│   ├── grades.py                   # Grade-to-V mapping
│   ├── metrics.py                  # Evaluation metrics
│   ├── models.py                   # Transformer architectures
│   ├── paths.py                    # Project root detection
│   ├── tokenization.py             # Core tokenization logic
│   └── utils.py                    # Utility functions
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/ClimbingBoardGPT.git
cd ClimbingBoardGPT

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Retrieving Raw Databases

The project expects SQLite databases at `data/raw/tb2.db` and `data/raw/kilter.db`.

Using [BoardLib](https://github.com/lemeryf/BoardLib):

```bash
pip install boardlib
boardlib database tension data/raw/tb2.db
boardlib database kilter data/raw/kilter.db
```

---

## Running the Pipeline

### 1. Tokenize both boards

```bash
python scripts/01_tokenize_routes.py --boards tb2,kilter
```

Creates `data/processed/tokenized/` with vocabulary, route sequences, and metadata.

### 2. Train the grade predictor

```bash
python scripts/02_train_grade_predictor.py
```

Trains a BERT-style transformer encoder and saves to `models/joint_transformer_grade_predictor.pth`.

### 3. Train the route generator

```bash
python scripts/03_train_route_generator.py
```

Trains a GPT-style causal transformer and saves to `models/joint_route_gpt_generator.pth`.

### 4. Evaluate generated routes

```bash
python scripts/04_evaluate_generated_routes.py
```

Evaluates validity, novelty, geometry, and grade consistency. Saves results to `data/processed/evaluation/`.

---

## Model Architectures

### JointRouteTransformerRegressor (Grade Prediction)

```
Input: [CLS] BOARD ANGLE HOLDS...
         ↓
Token Embedding + Position Embedding + Coordinate Features
         ↓
Transformer Encoder (4 layers, 4 heads, d_model=128)
         ↓
[CLS] token output → Regression Head → difficulty score
```

- ~1.17M parameters
- MSE loss, AdamW optimizer
- Early stopping on validation MAE

### JointRouteGPT (Route Generation)

```
Input: BOS BOARD ANGLE GRADE HOLDS...
         ↓
Token Embedding + Position Embedding
         ↓
Causal Transformer (4 layers, 4 heads, d_embd=128)
         ↓
Language Modeling Head → next token logits
```

- ~1.41M parameters
- Cross-entropy loss, AdamW optimizer
- Weight tying between embedding and output layers

---

## Board Configuration

| Setting | TB2 Mirror | Kilter Original |
|---|---:|---|
| `layout_id` | 10 | 1 |
| `token_prefix` | TB2 | KILTER |
| `max_angle` | 50 | 55 |
| `role_definitions` | start=5, middle=6, finish=7, foot=8 | start=12, middle=13, finish=14, foot=15 |
| `include_mirror_placement_id` | true | false |
| `min_fa_date` | null | 2016-01-01 |

To add a new board, create a JSON config in `configs/` following the same format.

---

## Comparison with Classical Approach

The earlier TB2 project used hand-engineered features with Random Forest and neural networks. This project replaces feature engineering with transformer attention:

| Aspect | Classical (TB2 Notebooks 01-06) | Transformer (This Project) |
|---|---|---|
| Input | 30+ engineered features | Raw token sequences |
| Feature engineering | Manual (spatial, geometric) | Learned via attention |
| Board handling | Single board (TB2) | Joint model with board token |
| Grade prediction | Random Forest / MLP | Transformer encoder |
| Route generation | Not supported | GPT-style decoder |
| Interpretability | Feature importance | Attention weights |

---

## Future Extensions

- **Masked hold prediction**: Mask holds and predict them (like BERT's MLM)
- **Stronger legality constraints**: Enforce valid start/finish positions in generation
- **Board transfer experiments**: Train on TB2, evaluate on Kilter (zero-shot)
- **GUI for route generation**: Interactive tool to generate and visualize climbs
- **Integration with classical features**: Combine transformer embeddings with engineered features

---

## Acknowledgments

- Board data from [Tension Climbing](https://tensionclimbing.com/) and [Kilter Board](https://kilterboard.com/)
- Database access via [BoardLib](https://github.com/lemeryf/BoardLib)
- Original TB2 analysis notebooks for foundational data exploration