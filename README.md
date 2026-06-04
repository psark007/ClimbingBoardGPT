# ClimbingBoardGPT

[![Live Demo](https://img.shields.io/badge/demo-webapp-teal)](https://cbgpt.pawelsarkowicz.xyz)

**ClimbingBoardGPT uses AI to generate new climbing routes and predict their difficulty — for Tension Board 2 and Kilter Board.**

You give it a board, a wall angle, and a target grade. It gives you a route. You can also paste in a route you already know, and it will guess the grade.

**[Try it live →](https://cbgpt.pawelsarkowicz.xyz)**

---

## What is this, exactly?

If you've climbed on a **Tension Board 2 (TB2)** or a **Kilter Board**, you know these are standardised training boards with a fixed set of holds. Routes on these boards are described as a list of holds and their roles (start, foot, hand, finish). The holds are identified by placement ID numbers and the route is stored as a short string like `p652r5p631r6p322r6p326r7`.

This project trains two small AI models on hundreds of thousands of real community-set routes from both boards:

- **A route generator** — you ask for a V6 at 40° on the Kilter, and the model samples a novel sequence of holds that should produce something around that difficulty.
- **A grade predictor** — you give it any route (board + angle + holds), and the model estimates the difficulty.

Both models are transformer-based neural networks, the same family of architecture behind large language models. Here the "language" is not English words but climbing-hold tokens: each hold-role combination gets its own symbol, and a route is a short sentence in that language.

The whole thing is small by modern standards (~1.2–1.4M parameters each) and runs on a CPU.

---

## What are Tension Board 2 and Kilter Board?

**Tension Board 2 (TB2)** is an adjustable training wall made by Tension Climbing. It has a fixed set of holds placed in a regular grid. Climbers set and share routes through a companion app; the community has set tens of thousands of problems. We work with the 12x12ft mirror in this project. 

**Kilter Board** is a similar product from Kilter (Setter Closet). It also has a large library of community-set problems. We work with the 16ftx12ft Kilter original board in this project.

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

## Try it — no setup needed

The live demo is at **[cbgpt.pawelsarkowicz.xyz](https://cbgpt.pawelsarkowicz.xyz)**. You can:

- Pick a board (TB2 or Kilter), a wall angle, and a target grade, and click **Generate** to get a new route drawn on the board image.
- Paste a frames string — the compact route code used by the board apps — into the **Predict** tab to estimate the grade of any route you already know.

---

## How it works (plain-English version)

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

**~91% of generated routes pass basic structural checks** (has a start hold, has a finish hold, holds exist on the right board, no duplicates).

---

## Quantitative results

These numbers are from the full training run documented by this repository.

In practice: the grade model is usually within one V-grade, and the generator usually makes structurally valid routes, but exact grade control is still imperfect.

### Dataset size

| Board | Training routes | Validation | Test |
|---|---:|---:|---:|
| TB2 | 33,719 | 4,430 | 4,447 |
| Kilter | 223,112 | 27,555 | 27,822 |
| **Total** | **256,831** | **31,985** | **32,269** |

Shared vocabulary: **4,438 tokens** (6 special + 2 board + 12 angle + 16 grade + 4,402 hold-role tokens).

### Grade prediction accuracy

The model has ~1.17M parameters. Early stopping selected epoch 8 (validation MAE ≈ 1.480).

| Metric | Overall | TB2 | Kilter |
|---|---:|---:|---:|
| Exact V-grade | 36.0% | 37.3% | 35.8% |
| Within ±1 V-grade | 79.3% | 80.0% | 79.2% |
| Within ±2 V-grades | 94.8% | 95.5% | 94.7% |
| R² | 0.768 | 0.800 | 0.763 |

### Route generation

The generator has ~1.41M parameters. Best validation perplexity: 24.2.

| Metric | TB2 | Kilter |
|---|---:|---:|
| Routes evaluated | 200 | 200 |
| Structurally valid | 89.0% | 94.0% |
| Exact requested grade (critic) | 29.5% | 27.0% |
| Within ±1 V-grade (critic) | 68.5% | 73.0% |
| Within ±2 V-grades (critic) | 90.5% | 93.5% |
| Mean novelty (Jaccard distance) | 0.656 | 0.634 |

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

**What does temperature do?**

| Temperature | Effect |
|---:|---|
| `0.3`–`0.6` | Conservative — picks safer, more common moves |
| `0.9` | Balanced default |
| `1.0` | Samples directly from learned probabilities |
| `1.1`–`1.3` | More creative — can produce weirder routes |

Generated routes are saved to:

```
outputs/demo_routes/<board>/angle_<angle>/V<grade>/
├── generated_routes.csv
├── generated_route_001.png
├── generated_route_001.svg
...
```

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

- **Generated routes are machine-made candidates.** Always inspect them before climbing. They are not guaranteed to be safe, fun, or even physically possible.
- **Grade predictions are estimates, not ground truth.** Climbing grades are subjective, board-style-dependent, and noisy even in the training data.
- **The hold sequence is a canonical ordering, not intended beta.** The model sorts holds by role and position; this is not necessarily the order you would climb them.
- **This is a research/hobby project**, not affiliated with Tension Climbing or Kilter/Setter Closet.

---

## Background and related work

This repo is the transformer/GPT follow-up to two earlier analysis projects:
- [Tension-Board-2-Analysis](https://github.com/psark007/Tension-Board-2-Analysis)
- [Kilter-Board-Analysis](https://github.com/psark007/Kilter-Board-Analysis)

The route generator architecture is inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).

Board layouts, hold metadata, and route data are from the [Tension Board 2](https://tensionclimbing.com/products/tension-board-2) and [Kilter Board](https://settercloset.com/collections/kilter-board) apps, loaded via [`BoardLib`](https://github.com/lemeryfertitta/BoardLib). This project is unaffiliated with Tension Climbing or Kilter.

---

## License

MIT — see [LICENSE](LICENSE). Educational use.
