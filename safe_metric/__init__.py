"""
SAFE: Semantic-Aware Flexible Evaluation for Quad-ABSA
=======================================================

A semantic embedding-based evaluation metric that addresses the limitations
of exact-match F1 when evaluating LLM-generated quadruple aspect-based
sentiment analysis outputs.

Quick Start:
    >>> from safe_metric import SAFEMetric, QuadEvaluator
    >>> 
    >>> # Initialize metric
    >>> safe = SAFEMetric(tau=70.0)
    >>> 
    >>> # Evaluate single quad pair
    >>> gold = ("service", "quality", "excellent", "positive")
    >>> pred = ("service", "quality", "very good", "positive")
    >>> score = safe.calculate_quad_score(gold, pred)
    >>> 
    >>> # Evaluate entire dataset
    >>> import pandas as pd
    >>> df = pd.read_csv("data.csv")
    >>> evaluator = QuadEvaluator(safe)
    >>> results_df, metrics = evaluator.evaluate_dataset(df)
    >>> print(f"Global SAFE F1: {metrics['f1']:.2f}")

Components:
    - SAFEMetric: Core similarity and scoring functions
    - QuadEvaluator: Dataset-level evaluation with matching logic

Reference:
    Alharbi et al. (2026). Does Semantic Evaluation Align with Human Judgement?
    Validating Semantic Evaluation Metric Against Human Scores in Quadruple
    Aspect-Based Sentiment Analysis. NeurIPS Evaluations & Datasets Track.
"""

__version__ = "1.0.0"
__author__ = "SAFE Research Team"
__license__ = "MIT"

from .core import SAFEMetric
from .evaluation import QuadEvaluator

__all__ = ["SAFEMetric", "QuadEvaluator"]
