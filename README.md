# SAFE: Semantic-Aware Flexible Evaluation for Quad-ABSA

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A semantic embedding-based evaluation metric for Quadruple Aspect-Based Sentiment Analysis (Quad-ABSA) that addresses the systematic undervaluation of LLM outputs by exact-match F1.

## Overview

**The Problem**: Traditional exact-match F1 scores fail to recognize semantically equivalent predictions:

```python
Gold:       ("service", "quality", "bad smell", "negative")
Prediction: ("service", "quality", "very bad smell", "negative")

Exact F1:   0.00  ❌  (treated as total failure)
SAFE:       99.2  ✅  (recognizes semantic equivalence)
```

**The Solution**: SAFE uses sentence embeddings to compute semantic similarity between quadruple elements, replacing brittle exact string matching with robust semantic comparison.

## Key Features

- 🎯 **Human-aligned**: Correlates more strongly with human similarity judgments than exact-match F1
- 🔧 **Configurable threshold**: τ parameter balances precision/recall (default: 0.70)
- 📊 **Three variants**: Standalone, Hard-SAFE, Soft-SAFE for different use cases
- 🚀 **Fast**: Lightweight sentence-transformers model (all-MiniLM-L6-v2)
- 📦 **Easy to use**: Simple Python API + command-line interface

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 2GB free disk space (for embedding model)

### Step-by-Step Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/safe-quad-absa.git
cd safe-quad-absa
```

#### 2. Create Virtual Environment (Recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt.

#### 3. Install SAFE Package

```bash
# Upgrade pip first
pip install --upgrade pip

# Install in development mode (recommended for reproducibility)
pip install -e .
```

This installs:
- `sentence-transformers==2.2.2` (embedding model)
- `pandas>=1.5.0` (data processing)
- `numpy>=1.23.0` (numerical operations)
- `scikit-learn>=1.2.0` (F1 calculation)
- `torch>=2.0.0` (deep learning backend)

**Installation time:** ~2-5 minutes (depending on internet speed)

#### 4. Verify Installation

```bash
# Test import
python -c "from safe_metric import SAFEMetric; print('✅ SAFE installed successfully')"

# Run unit tests (optional but recommended)
pip install pytest
pytest tests/ -v
```

**Expected output:** `24 passed` (or 22 passed with 2 known tolerance issues)

## Quick Start

### Reproduce Paper Results

#### Option 1: Using Sample Data (30 seconds)

```bash
# Process sample dataset
python scripts/evaluate_dataset.py \
    examples/sample_input.csv \
    output_sample.csv \
    --tau 70 \
    --verbose

# View results
head -20 output_sample.csv
```

**Expected output:**
- 9 rows processed
- SAFE scores ranging 0-100
- Classifications: TP/FP/FN
- Review-level F1 scores

#### Option 2: Using Paper Dataset (2-3 minutes)

```bash
# Download dataset from Dataverse (see Dataset section below)
# Then run:
python scripts/evaluate_dataset.py \
    examples/real_data_sample.csv \
    output_paper.csv \
    --tau 70 \
    --append-metrics \
    --verbose
```

**Expected output:**
- 109 quadruple pairs processed
- Global metrics: Precision ~76%, Recall ~92%, F1 ~83%
- 78 TP, 24 FP, 7 FN classifications

### Python API Usage

```python
from safe_metric import SAFEMetric, QuadEvaluator
import pandas as pd

# Initialize SAFE metric with threshold
safe = SAFEMetric(tau=70.0)

# Example 1: Evaluate single quad pair
gold = ("service", "quality", "excellent", "positive")
pred = ("service", "quality", "very good", "positive")
score = safe.calculate_quad_score(gold, pred)
print(f"SAFE score: {score:.2f}")  # Output: ~98.50

# Example 2: Canonical divergence case (Paper Section 4.1)
gold = ("service", "quality", "bad smell", "negative")
pred = ("service", "quality", "very bad smell", "negative")
score = safe.calculate_quad_score(gold, pred)
print(f"SAFE score: {score:.2f}")  # Output: ~99.20
print(f"Exact F1 would be: 0.00")  # Demonstrates the problem

# Example 3: Evaluate entire dataset
df = pd.read_csv("examples/sample_input.csv")
evaluator = QuadEvaluator(safe)
results_df, metrics = evaluator.evaluate_dataset(df)

print(f"Global SAFE F1: {metrics['f1']:.2f}%")
print(f"Precision: {metrics['precision']:.2f}%")
print(f"Recall: {metrics['recall']:.2f}%")
print(f"TP: {metrics['total_tp']}, FP: {metrics['total_fp']}, FN: {metrics['total_fn']}")
```

### Command-Line Interface

```bash
# Basic usage
python scripts/evaluate_dataset.py INPUT.csv OUTPUT.csv

# Full options (for reproducibility)
python scripts/evaluate_dataset.py INPUT.csv OUTPUT.csv \
    --tau 70.0 \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --gold-cols Aspect Category Opinion Sentiment \
    --pred-cols Aspect.1 Category.1 Opinion.1 Sentiment.1 \
    --review-id-col Review_ID \
    --append-metrics \
    --verbose

# Threshold sensitivity analysis (Paper Section 5.3)
for tau in 60 65 70 75 80; do
    python scripts/evaluate_dataset.py \
        examples/real_data_sample.csv \
        results_tau${tau}.csv \
        --tau $tau \
        --verbose
done
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `input_file` | Path to input CSV with gold and predicted quads | Required |
| `output_file` | Path to save results CSV | Required |
| `--tau` | Similarity threshold (0-100) for TP classification | 70.0 |
| `--model` | SentenceTransformer model name | all-MiniLM-L6-v2 |
| `--gold-cols` | Gold standard column names (4 required) | Aspect Category Opinion Sentiment |
| `--pred-cols` | Prediction column names (4 required) | Aspect.1 Category.1 Opinion.1 Sentiment.1 |
| `--review-id-col` | Review ID column name | Review_ID |
| `--append-metrics` | Append global metrics to output CSV | False |
| `--verbose` | Print detailed progress | False |

## Input Data Format

Your CSV must contain these columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| Review_ID | int/str | Groups quads from same review | 1, 2, 3 |
| Aspect | str | Gold aspect term | "service", "room" |
| Category | str | Gold aspect category | "quality", "cleanliness" |
| Opinion | str | Gold opinion expression | "excellent", "dirty" |
| Sentiment | str | Gold sentiment polarity | "positive", "negative" |
| Aspect.1 | str | Predicted aspect term | "service", "room" |
| Category.1 | str | Predicted aspect category | "quality", "cleanliness" |
| Opinion.1 | str | Predicted opinion expression | "very good", "not clean" |
| Sentiment.1 | str | Predicted sentiment polarity | "positive", "negative" |

**Notes:**
- Column names are configurable via CLI arguments
- Empty cells (NaN) indicate missing quads (FN or FP)
- Reviews can have multiple quads (multiple rows with same Review_ID)

## Output Format

SAFE adds three columns to your input data:

| Column | Type | Description | Values |
|--------|------|-------------|--------|
| Classification | str | Quadruple classification | TP, FP, FN, N/A |
| Quad_SAFE | float | Semantic similarity score (0-100) | 0.00 to 100.00 |
| Review_SAFE_F1 | float | F1 score for entire review | 0.00 to 100.00 |

**Classification Rules:**
- **TP (True Positive):** Both gold and prediction exist, SAFE ≥ τ
- **FP (False Positive):** Prediction exists but no matching gold, OR SAFE < τ
- **FN (False Negative):** Gold exists but no matching prediction
- **N/A:** Neither gold nor prediction exists (empty row)

**Review_SAFE_F1:** Calculated per review, shown only on first row of each review.

## Reproducibility Details

### Exact Model Specifications

**Embedding Model:**
- Name: `sentence-transformers/all-MiniLM-L6-v2`
- Architecture: 6-layer MiniLM distilled from BERT
- Embedding dimension: 384
- Vocabulary size: 30,522 WordPiece tokens
- Model size: 80 MB
- Download: Automatic on first use (cached locally)

**Similarity Computation:**
```python
# Pseudocode for exact reproduction
embeddings1 = model.encode(text1, convert_to_tensor=True)
embeddings2 = model.encode(text2, convert_to_tensor=True)
cosine_sim = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
normalized_sim = (cosine_sim + 1) / 2  # Map [-1, 1] to [0, 1]
```

**SAFE Score Formula:**
```
SAFE(q̂, q) = λ₁·Sim(aspect) + λ₂·Sim(category) + λ₃·Sim(opinion) + λ₄·Sim(sentiment)
where λ₁ = λ₂ = λ₃ = λ₄ = 0.25
```

**F1 Calculation:**
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### Deterministic Execution

SAFE produces **deterministic results** (same input → same output) because:
1. Sentence embeddings are deterministic (no dropout at inference)
2. No random sampling in similarity computation
3. Classification logic is rule-based (τ threshold)

**Verification:** Run twice on same data, compare outputs:
```bash
python scripts/evaluate_dataset.py examples/sample_input.csv out1.csv --tau 70
python scripts/evaluate_dataset.py examples/sample_input.csv out2.csv --tau 70
diff out1.csv out2.csv  # Should show no differences
```

### Hardware/Software Requirements

**Minimum Requirements:**
- CPU: Any modern processor (2+ cores recommended)
- RAM: 4 GB
- Storage: 2 GB free space
- OS: Windows 10+, macOS 10.14+, Ubuntu 18.04+

**Tested Configurations:**
- Windows 11 + Python 3.14.3 ✓
- macOS Sonoma + Python 3.10 ✓
- Ubuntu 22.04 + Python 3.11 ✓

**GPU:** Optional (CPU inference is fast enough for datasets <10,000 quads)

### Expected Runtimes

| Dataset Size | Time (CPU) | Time (GPU) |
|--------------|------------|------------|
| 10 quads | <1 second | <1 second |
| 100 quads | ~3 seconds | ~1 second |
| 1,000 quads | ~30 seconds | ~5 seconds |
| 10,000 quads | ~5 minutes | ~30 seconds |

**Paper dataset (109 quads):** ~3-5 seconds on standard laptop

## How SAFE Works

### 1. Similarity Calculation

For each quadruple element (aspect, category, opinion, sentiment):

```
Sim(text1, text2) = (cos_similarity(embed(text1), embed(text2)) + 1) / 2
```

Normalized to [0, 1] range.

**Special cases:**
- Identical text (case-insensitive): 1.0
- Both empty: 1.0
- One empty, one non-empty: 0.0

### 2. SAFE Score (Soft Variant)

```
SAFE(gold, pred) = λ₁·Sim(aspect) + λ₂·Sim(category) + λ₃·Sim(opinion) + λ₄·Sim(sentiment)
```

Where λ₁ = λ₂ = λ₃ = λ₄ = 0.25 (equal weighting)

### 3. Classification Logic

For each quadruple pair:

- **TP** (True Positive): Both gold and prediction exist, SAFE ≥ τ
- **FP** (False Positive): Prediction exists but no matching gold, OR SAFE < τ
- **FN** (False Negative): Gold exists but no matching prediction

### 4. F1 Calculation

Standard precision/recall/F1 from TP, FP, FN counts:

```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 × Precision × Recall / (Precision + Recall)
```

## Repository Structure

```
safe-quad-absa/
├── safe_metric/              # Core library (importable)
│   ├── __init__.py          # Package initialization
│   ├── core.py              # SAFEMetric class (similarity, scoring, F1)
│   └── evaluation.py        # QuadEvaluator class (dataset processing)
├── scripts/
│   └── evaluate_dataset.py  # Command-line interface
├── notebooks/               # Jupyter tutorials
│   └── 01_basic_usage.ipynb
├── tests/                   # Unit tests (pytest)
│   └── test_core.py         # 24 test cases
├── examples/                # Sample data
│   ├── sample_input.csv     # Synthetic example (9 quads)
│   └── real_data_sample.csv # Paper dataset (109 quads)
├── requirements.txt         # Python dependencies
├── setup.py                 # pip installation config
├── README.md                # This file
├── LICENSE                  # MIT License
└── .gitignore               # Git ignore rules
```

## Configuration Parameters

### Threshold (τ)

Controls the strictness of matching:

- **τ = 70** (default): Balanced precision/recall
- **τ = 80**: Stricter (fewer false positives, more false negatives)
- **τ = 60**: Looser (more false positives, fewer false negatives)

Selection criteria:
1. Human correlation (primary)
2. Metric stability (secondary)
3. Discriminative power (tertiary)

### Embedding Model

Default: `sentence-transformers/all-MiniLM-L6-v2`
- Fast, lightweight (80MB)
- Good semantic coverage for sentiment analysis

Alternatives:
- `all-mpnet-base-v2`: Slower but higher quality
- `paraphrase-multilingual-MiniLM-L12-v2`: For non-English text

### Lambda Weights

Default: λ₁ = λ₂ = λ₃ = λ₄ = 0.25 (equal weighting)

Grounded in maximum entropy principle (Jaynes, 1957) — no element should be privileged without prior knowledge.

## Dataset

Human evaluation dataset available at:

**Harvard Dataverse**: https://doi.org/10.7910/DVN/HISKDV

Contains:
- 40 hospitality reviews
- 109 quadruple comparison pairs
- Human similarity ratings (5-point Likert scale)
- Gold standard annotations + LLM predictions

**Download:** Click "Access Dataset" → Download all files as ZIP

**Files included:**
- `HumEvalEthic.csv` - Human evaluation responses
- `review_corpus.csv` - Full review texts
- `metadata.txt` - Dataset documentation
- `croissant.json` - Machine-readable metadata

## Citation

```bibtex
@inproceedings{safe2026,
  title={Does Semantic Evaluation Align with Human Judgement? 
         Validating Semantic Evaluation Metric Against Human Scores 
         in Quadruple Aspect-Based Sentiment Analysis},
  author={[Authors]},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)
             — Evaluations \& Datasets Track},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Built with [sentence-transformers](https://www.sbert.net/)
- Dataset from hospitality review corpus (WISE 2024, Alharbi et al., 2025)
- Inspired by BERTScore (Zhang et al., 2020)

## Contact

For questions or issues, please:
- Open a GitHub issue
- Email: To be updated after publication

---

**Note**: This is research software accompanying a NeurIPS 2026 E&D Track submission. All code and data are provided for reproducibility and transparency.
