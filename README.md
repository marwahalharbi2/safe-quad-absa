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

### From Source (Recommended for Reviewers)

```bash
git clone https://github.com/marwahalharbi2/safe-quad-absa.git
cd safe-quad-absa
pip install -e .
```

### Using pip (After Publication)

```bash
pip install safe-metric
```

## Quick Start

### Python API

```python
from safe_metric import SAFEMetric, QuadEvaluator
import pandas as pd

# Initialize SAFE metric with threshold
safe = SAFEMetric(tau=70.0)

# Evaluate single quad pair
gold = ("service", "quality", "excellent", "positive")
pred = ("service", "quality", "very good", "positive")
score = safe.calculate_quad_score(gold, pred)
print(f"SAFE score: {score:.2f}")  # ~98.5

# Evaluate entire dataset
df = pd.read_csv("data/human_eval.csv")
evaluator = QuadEvaluator(safe)
results_df, metrics = evaluator.evaluate_dataset(df)

print(f"Global SAFE F1: {metrics['f1']:.2f}%")
print(f"Precision: {metrics['precision']:.2f}%")
print(f"Recall: {metrics['recall']:.2f}%")
```

### Command-Line Interface

```bash
# Basic usage
python scripts/evaluate_dataset.py input.csv output.csv

# With custom threshold and verbose output
python scripts/evaluate_dataset.py \
    data/human_eval.csv \
    results/safe_output.csv \
    --tau 70 \
    --append-metrics \
    --verbose

# Using different embedding model
python scripts/evaluate_dataset.py input.csv output.csv \
    --model sentence-transformers/all-mpnet-base-v2 \
    --tau 75
```

## Input Data Format

Your CSV should contain these columns (names configurable):

| Review_ID | Aspect | Category | Opinion | Sentiment | Aspect.1 | Category.1 | Opinion.1 | Sentiment.1 |
|-----------|--------|----------|---------|-----------|----------|------------|-----------|-------------|
| 1 | service | quality | excellent | positive | service | quality | very good | positive |
| 1 | room | cleanliness | clean | positive | room | cleanliness | spotless | positive |

- **First 4 columns**: Gold standard (human annotations)
- **Next 4 columns**: Model predictions (LLM outputs)
- **Review_ID**: Groups quads belonging to same review

## Output Format

SAFE adds three columns to your input data:

| ... | Classification | Quad_SAFE | Review_SAFE_F1 |
|-----|----------------|-----------|----------------|
| ... | TP | 98.5 | 85.3 |
| ... | FP | 45.2 | |

- **Classification**: TP (true positive), FP (false positive), FN (false negative), N/A
- **Quad_SAFE**: Semantic similarity score (0-100) for this quad pair
- **Review_SAFE_F1**: F1 score for the entire review (only on first row)

## How SAFE Works

### 1. Similarity Calculation

For each quadruple element (aspect, category, opinion, sentiment):

```
Sim(text1, text2) = (cos_similarity(embed(text1), embed(text2)) + 1) / 2
```

Normalized to [0, 1] range.

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
│   ├── __init__.py
│   ├── core.py              # SAFEMetric class
│   └── evaluation.py        # QuadEvaluator class
├── scripts/
│   └── evaluate_dataset.py  # CLI tool
├── notebooks/               # Usage examples
│   ├── 01_basic_usage.ipynb
│   └── 02_reproduce_paper_results.ipynb
├── tests/                   # Unit tests
│   ├── test_core.py
│   └── test_evaluation.py
├── examples/                # Sample data
│   ├── sample_input.csv
│   └── sample_output.csv
├── requirements.txt
├── setup.py
└── README.md
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
