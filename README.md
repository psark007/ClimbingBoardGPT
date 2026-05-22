# ClimbingBoardGPT

[![Live Demo](https://img.shields.io/badge/demo-webapp-teal)](https://cbgpt.pawelsarkowicz.xyz)

**ClimbingBoardGPT** is a unified transformer-style modeling project for climbing-board routes on:

- **Tension Board 2 Mirror** (12ftx12ft)
- **Kilter Board Original** (16ftx12ft)

The project treats climbing-board problems as symbolic sequences of board-aware hold-role tokens. It supports:

1. joint route tokenization for TB2 and Kilter,
2. transformer-based grade prediction,
3. GPT-style route generation conditioned on board, wall angle, and target grade,
4. calibrated board-background visualization,
5. command-line demo scripts for generation and grade prediction,
6. interactive FastAPI webapp with board-image overlay and click-to-build route prediction.

This repo is the transformer/GPT follow-up project to [Tension-Board-2-Analysis](https://github.com/psark007/Tension-Board-2-Analysis) and [Kilter-Board-Analysis](https://github.com/psark007/Kilter-Board-Analysis). 

---

## Core idea

A route is represented as a sequence like:

```text
<BOS> <BOARD_TB2> <ANGLE_40> <GRADE_V6>
<TB2_p344_start> <TB2_p369_middle> <TB2_p603_finish>
<EOS>
```

or:

```text
<BOS> <BOARD_KILTER> <ANGLE_40> <GRADE_V6>
<KILTER_p1084_start> <KILTER_p1231_middle> <KILTER_p1395_finish>
<EOS>
```

Hold tokens are **board-namespaced**, so a TB2 placement ID and a Kilter placement ID never collide.

For grade prediction, the grade token is removed:

```text
<CLS> <BOARD_TB2> <ANGLE_40>
<TB2_p344_start> <TB2_p369_middle> <TB2_p603_finish>
<EOS>
```

The model then predicts the climb difficulty from the board, angle, and hold-role tokens.


---

## Quantitative results from the executed notebooks

These numbers come from the executed four-notebook run included with the project. They should be treated as the current benchmark for this checkpoint/data snapshot; rerun the pipeline if the raw databases, tokenization, model sizes, or train/validation/test split change.

### Dataset and tokenization scale

The unified tokenizer builds one shared corpus across TB2 and Kilter.

| Quantity | Value |
|---|---:|
| Total route/angle entries | 321,085 |
| TB2 entries | 42,596 |
| Kilter entries | 278,489 |
| Placement metadata rows | 1,139 |
| Shared vocabulary size | 4,438 tokens |
| Special tokens | 6 |
| Board tokens | 2 |
| Angle tokens | 12 |
| Grade tokens | 16 |
| Hold-role tokens | 4,402 |
| Grade-predictor max sequence length | 398 |
| GPT-generator max sequence length | 399 |

The train/validation/test split used in the executed notebooks was:

| Board | Train | Validation | Test |
|---|---:|---:|---:|
| TB2 | 33,719 | 4,430 | 4,447 |
| Kilter | 223,112 | 27,555 | 27,822 |
| **Total** | **256,831** | **31,985** | **32,269** |

### Grade prediction performance

The grade predictor is a transformer encoder trained jointly on both boards. It receives board, angle, hold-role tokens, and coordinate features, but **does not receive the grade token**.

| Metric | Overall | TB2 | Kilter |
|---|---:|---:|---:|
| MAE | 1.481 | 1.420 | 1.490 |
| RMSE | 1.941 | 1.845 | 1.956 |
| R² | 0.768 | 0.800 | 0.763 |
| Exact grouped V-grade | 36.0% | 37.3% | 35.8% |
| Within ±1 V-grade | 79.3% | 80.0% | 79.2% |
| Within ±2 V-grades | 94.8% | 95.5% | 94.7% |

The model has about **1.17M parameters**. In the executed run, early stopping selected epoch 8 with validation MAE ≈ **1.480**.

### Route generator training

The route generator is a GPT-style causal transformer trained on grade-conditioned route sequences.

| Quantity | Value |
|---|---:|
| Model size | ~1.41M parameters |
| Best validation loss | 3.187 |
| Best validation perplexity | 24.2 |
| Evaluation sample size | 400 generated routes |
| Overall basic validity | 91.5% |
| Overall strict validity | 91.5% |

During the generator evaluation run, routes were sampled across both boards, common angles, and target grades V1–V8.

### Generated-route evaluation

Generated routes are evaluated by structural validity, novelty against real climbs, geometric features, and grade consistency using the trained grade predictor as a critic.

| Metric | TB2 | Kilter |
|---|---:|---:|
| Generated routes evaluated | 200 | 200 |
| Basic validity | 89.0% | 94.0% |
| Strict validity | 89.0% | 94.0% |
| Mean novelty distance | 0.656 | 0.634 |
| Median novelty distance | 0.667 | 0.652 |
| Mean generated hold count | 11.11 | 12.90 |
| Mean route height | 130.76 | 142.32 |
| Mean route width | 61.66 | 74.94 |
| Mean hand-reach distance | 50.41 | 57.53 |

Grade consistency of generated climbs, measured by the trained grade predictor:

| Metric | Overall | TB2 | Kilter |
|---|---:|---:|---:|
| Exact requested V-grade | 28.2% | 29.5% | 27.0% |
| Within ±1 V-grade | 70.8% | 68.5% | 73.0% |
| Within ±2 V-grades | 92.0% | 90.5% | 93.5% |
| Mean V-grade error | — | -0.18 | -0.30 |

Interpretation: the generator is usually structurally valid and usually close to the requested grade according to the critic, but exact grade control remains imperfect. That is expected: this is a small GPT-style model trained on symbolic route data, not a production setter.


---

## Repository layout

```text
ClimbingBoardGPT/
├── configs/
│   ├── tb2.json
│   └── kilter.json
├── data/
│   ├── raw/
│   │   ├── tb2.db
│   │   └── kilter.db
│   └── processed/
├── images/
│   ├── tb2_board_12x12_composite.png
│   └── kilter-original-16x12_compose.png
├── models/
│   ├── joint_transformer_grade_predictor.pth
│   └── joint_route_gpt_generator.pth
├── notebooks/
│   ├── 01_unified_route_tokenization.ipynb
│   ├── 02_joint_transformer_grade_prediction.ipynb
│   ├── 03_joint_route_generator.ipynb
│   └── 04_generated_route_evaluation.ipynb
├── scripts/
│   ├── 01_tokenize_routes.py
│   ├── 02_train_grade_predictor.py
│   ├── 03_train_route_generator.py
│   ├── 04_evaluate_generated_routes.py
│   ├── demo_generate_and_visualize.py
│   ├── demo_generate_tb2.py
│   ├── demo_generate_kilter.py
│   ├── demo_predict_grade.py
│   ├── demo_predict_tb2.py
│   └── demo_predict_kilter.py
├── src/climbingboardgpt/
├── webapp/
│   ├── app.py
│   ├── app.css
│   ├── app.js
│   ├── index.html
│   └── Dockerfile
├── docker-compose.webapp.yml
├── LICENSE
├── README.md
├── requirements.txt
└── pyproject.toml
```

---

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the package:

```bash
pip install -r requirements.txt
pip install -e .
```

For CPU-only demo use on a small VPS, the scripts support:

```bash
--torch-threads 1
```

This caps PyTorch CPU thread usage.

---

## Data expected by the full training pipeline

The full tokenization/training pipeline expects raw BoardLib databases at:

```text
data/raw/tb2.db
data/raw/kilter.db
```

The project configs are:

```text
configs/tb2.json
configs/kilter.json
```

They define board-specific details such as:

- database path,
- layout ID,
- role IDs,
- token prefix,
- angle cutoff,
- optional date / placement filters.

The demo scripts do **not** need the raw databases if the processed tokenization artifacts and trained model checkpoints already exist.

---

## Full training pipeline

From the repository root:

```bash
python scripts/01_tokenize_routes.py --boards tb2,kilter
python scripts/02_train_grade_predictor.py
python scripts/03_train_route_generator.py
python scripts/04_evaluate_generated_routes.py
```

This produces the main processed artifacts and trained checkpoints.

### Tokenization outputs

```text
data/processed/tokenized/
├── route_sequences.csv
├── routes_tokenized.jsonl
├── token_vocab.json
├── token_metadata.csv
├── placement_metadata.csv
└── board_summary.csv
```

### Grade-prediction outputs

```text
data/processed/grade_prediction/
├── training_history.csv
├── test_predictions.csv
├── board_metrics.csv
└── overall_metrics.json

models/
└── joint_transformer_grade_predictor.pth
```

### Route-generation outputs

```text
data/processed/generation/
├── training_history.csv
└── generated_routes.csv

models/
└── joint_route_gpt_generator.pth
```

### Generated-route evaluation outputs

```text
data/processed/evaluation/
├── generated_route_evaluation.csv
└── top_generated_candidates.csv
```

---

## Generate routes and visualize them

After training the route generator, or after placing a trained checkpoint at:

```text
models/joint_route_gpt_generator.pth
```

you can generate and visualize climbs.

### TB2

```bash
python scripts/demo_generate_tb2.py --angle 40 --grade 6 --n 4
```

### Kilter

```bash
python scripts/demo_generate_kilter.py --angle 40 --grade 6 --n 4
```

### Generic version

```bash
python scripts/demo_generate_and_visualize.py \
  --board tb2 \
  --angle 40 \
  --grade 6 \
  --n 4 \
  --temperature 0.9 \
  --top-k 50
```

Outputs are written to:

```text
outputs/demo_routes/<board>/angle_<angle>/V<grade>/
├── generated_routes.csv
├── generated_route_001.png
├── generated_route_001.svg
├── generated_route_002.png
├── generated_route_002.svg
└── ...
```

### Generated-route visualization

The visualization uses calibrated board backgrounds:

```text
images/tb2_board_12x12_composite.png
images/kilter-original-16x12_compose.png
```

These are overlaid using product-size coordinate windows:

```text
TB2:    x = [-68, 68],  y = [0, 144]
Kilter: x = [-24, 168], y = [0, 156]
```

These extents match the old visualization notebooks better than simply using the min/max of observed hold coordinates, because the hold coordinates are inset from the product boundary.

The role markers are:

| Role | Marker |
|---|---|
| start | green circle |
| middle | blue circle |
| finish | red star |
| foot | small yellow square |


### Annotate holds

To label route holds by placement ID:

```bash
python scripts/demo_generate_tb2.py \
  --angle 40 \
  --grade 6 \
  --n 2 \
  --annotate
```

### CPU-  friendly run

```bash
python scripts/demo_generate_tb2.py \
  --angle 40 \
  --grade 6 \
  --n 2 \
  --torch-threads 1
```

---

## Temperature and sampling

The `--temperature` argument controls generation randomness.

The model predicts probabilities for the next token. Temperature rescales those probabilities before sampling.

| Temperature | Effect |
|---:|---|
| `0.3`–`0.6` | conservative; picks safer/common tokens |
| `0.9` | balanced default |
| `1.0` | samples directly from the learned probabilities |
| `1.1`–`1.3` | more exploratory; can produce weirder climbs |

Example:

```bash
python scripts/demo_generate_kilter.py \
  --angle 40 \
  --grade 6 \
  --n 4 \
  --temperature 0.6
```

---

## Predict grade from board, angle, and frames string

After training the grade predictor, or after placing a trained checkpoint at:

```text
models/joint_transformer_grade_predictor.pth
```

you can predict a grade directly from a BoardLib-style frames string.

### Generic

```bash
python scripts/demo_predict_grade.py \
  --board tb2 \
  --angle 40 \
  --frames 'p652r5p631r6p322r6p326r7'
```

### TB2 wrapper

```bash
python scripts/demo_predict_tb2.py \
  --angle 40 \
  --frames 'p652r5p631r6p322r6p326r7'
```

### Kilter wrapper

```bash
python scripts/demo_predict_kilter.py \
  --angle 40 \
  --frames 'p1127r12p1196r13p1216r13p1388r14'
```

Example output:

```text
Board:        Tension Board 2 Mirror (tb2)
Angle:        40°
Frames:       p652r5p631r6p322r6p326r7
Predicted:    V6
Difficulty:   22.400
```

The `Predicted` line is the grouped V-grade. The `Difficulty` line is the model's continuous prediction in the underlying BoardLib difficulty scale.

### JSON output

```bash
python scripts/demo_predict_grade.py \
  --board kilter \
  --angle 40 \
  --frames 'p1127r12p1196r13p1216r13p1388r14' \
  --json
```

### Show model tokens

```bash
python scripts/demo_predict_tb2.py \
  --angle 40 \
  --frames 'p652r5p631r6p322r6p326r7' \
  --show-tokens
```

### Save a visualization of the input climb

```bash
python scripts/demo_predict_tb2.py \
  --angle 40 \
  --frames 'p652r5p631r6p322r6p326r7' \
  --visualize
```

This writes:

```text
outputs/grade_predictions/<board>/angle_<angle>/
├── <name>.png
├── <name>.svg
└── <name>.json
```

Example with custom output name:

```bash
python scripts/demo_predict_kilter.py \
  --angle 40 \
  --frames 'p1127r12p1196r13p1216r13p1388r14' \
  --visualize \
  --output-name my_kilter_climb
```

---

## Grade prediction in generated-route visualizations

If both checkpoints exist:

```text
models/joint_route_gpt_generator.pth
models/joint_transformer_grade_predictor.pth
```

then the generation demo automatically scores each generated climb with the grade predictor.

Example:

```bash
python scripts/demo_generate_tb2.py --angle 40 --grade 6 --n 4
```

The terminal output includes something like:

```text
predicted=V5 (difficulty=20.81, error=-1 V)
```

The visualization subtitle also includes:

```text
predicted V5 (20.81) | error -1V
```

To disable this scoring:

```bash
python scripts/demo_generate_tb2.py \
  --angle 40 \
  --grade 6 \
  --n 4 \
  --no-grade-prediction
```

To use a non-default grade predictor:

```bash
python scripts/demo_generate_and_visualize.py \
  --board kilter \
  --angle 40 \
  --grade 6 \
  --grade-model-path models/joint_transformer_grade_predictor.pth
```

---

## Important caveats

Generated climbs are **machine-generated candidates**, not guaranteed to be safe, good, or fun.

The grade predictor is a model-based estimate, not ground truth. Climbing grades are noisy and subjective, and board climbs can be highly style-dependent.

The route sequence is a canonical ordering of holds, not necessarily actual beta order. This is fine for symbolic modeling, but it should not be interpreted as the intended movement sequence.

The visualizations are calibrated to match the existing board images, but any change in image file, crop, or coordinate convention may require adjusting board extents in:

```text
src/climbingboardgpt/visualization.py
```


---

## Webapp demo

The repository includes a lightweight FastAPI webapp. It is inference-only:

- loads the generator and grade predictor once at startup,
- serves the TB2/Kilter board images as static assets,
- returns hold coordinates and roles as JSON,
- draws the climb overlay in the browser as SVG.

### Run locally

From the repository root:

```bash
pip install fastapi "uvicorn[standard]" pydantic
uvicorn webapp.app:app --host 127.0.0.1 --port 8055
```

Then open:

```text
http://127.0.0.1:8055
```

### Run with Docker

```bash
docker compose -f docker-compose.webapp.yml up -d --build
```

The service binds to localhost only:

```text
127.0.0.1:8055
```

### Required files for the webapp

The webapp does not need raw SQLite databases. It needs:

```text
models/joint_route_gpt_generator.pth
models/joint_transformer_grade_predictor.pth
data/processed/tokenized/token_metadata.csv
data/processed/tokenized/token_vocab.json
data/processed/tokenized/route_sequences.csv
configs/
images/
src/climbingboardgpt/
webapp/
```

### API endpoints

```text
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

# Future Work
- Board-size-specific generation is a planned future extension. For now, the demo uses the full TB2 12x12 and Kilter 16x12-style background images and placement sets.
- "No Match" token and "No Match" options in the demo. 



# License
This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.

The project is for educational purposes. Climb data belongs to Tension Climbing and Kilter respectively. 