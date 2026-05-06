"""
SAFE (Semantic-Aware Flexible Evaluation) Core Implementation
==============================================================

This module implements the core SAFE metric for evaluating Quad-ABSA predictions.

SAFE uses semantic similarity instead of exact string matching to evaluate
quadruple aspect-based sentiment analysis outputs.

Formula (Soft SAFE):
    SAFE_Soft(q̂, q) = λ₁·Sim(yₐ, ŷₐ) + λ₂·Sim(yₒ, ŷₒ) + λ₃·Sim(yc, ŷc) + λ₄·Sim(ys, ŷs)
    
    where:
    - λ₁ = λ₂ = λ₃ = λ₄ = 0.25 (equal weighting)
    - Sim() is cosine similarity between sentence embeddings
    - q = (aspect, category, opinion, sentiment) is the gold standard quad
    - q̂ is the predicted quad

Classification:
    - TP: Model prediction matches gold standard with SAFE ≥ τ
    - FP: Model prediction exists but no matching gold standard (or SAFE < τ)
    - FN: Gold standard exists but no matching model prediction

Reference:
    Alharbi et al. (2026). Does Semantic Evaluation Align with Human Judgement?
    NeurIPS Evaluations & Datasets Track.
"""

from typing import Tuple, Optional
import pandas as pd
from sentence_transformers import SentenceTransformer, util


class SAFEMetric:
    """
    SAFE (Semantic-Aware Flexible Evaluation) metric calculator.
    
    Attributes:
        model: SentenceTransformer model for computing embeddings
        lambda_weights: Tuple of weights for (aspect, category, opinion, sentiment)
        tau: Similarity threshold for TP classification (0-100)
    
    Example:
        >>> safe = SAFEMetric(tau=70.0)
        >>> gold = ("service", "quality", "excellent", "positive")
        >>> pred = ("service", "quality", "very good", "positive")
        >>> score = safe.calculate_quad_score(gold, pred)
        >>> print(f"SAFE score: {score:.2f}")  # ~95-99
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        lambda_weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
        tau: float = 70.0
    ):
        """
        Initialize SAFE metric calculator.
        
        Args:
            model_name: HuggingFace model identifier for sentence embeddings
            lambda_weights: Weights for (aspect, category, opinion, sentiment)
            tau: Similarity threshold (0-100) for TP classification
        """
        self.model = SentenceTransformer(model_name)
        self.lambda_weights = lambda_weights
        self.tau = tau
        
        # Validate weights sum to 1.0
        if abs(sum(lambda_weights) - 1.0) > 1e-6:
            raise ValueError(f"Lambda weights must sum to 1.0, got {sum(lambda_weights)}")
    
    def compute_similarity(self, text1: Optional[str], text2: Optional[str]) -> float:
        """
        Compute semantic similarity between two texts.
        
        Uses cosine similarity of sentence embeddings, normalized to [0, 1].
        
        Args:
            text1: First text (can be None or empty)
            text2: Second text (can be None or empty)
        
        Returns:
            Similarity score in [0, 1]:
            - 1.0 if texts are identical (case-insensitive) or both empty
            - 0.0 if one text is empty and the other is not
            - Normalized cosine similarity otherwise
        """
        # Normalize inputs
        text1 = str(text1).strip() if pd.notna(text1) else ""
        text2 = str(text2).strip() if pd.notna(text2) else ""
        
        # Handle exact match and empty cases
        if text1.lower() == text2.lower():
            return 1.0
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        
        # Compute embedding similarity
        emb1 = self.model.encode(text1, convert_to_tensor=True)
        emb2 = self.model.encode(text2, convert_to_tensor=True)
        cos_sim = util.cos_sim(emb1, emb2).item()
        
        # Normalize from [-1, 1] to [0, 1]
        return (cos_sim + 1) / 2.0
    
    def calculate_quad_score(
        self,
        gold_quad: Tuple[str, str, str, str],
        pred_quad: Tuple[str, str, str, str]
    ) -> float:
        """
        Calculate SAFE score between gold standard and predicted quadruples.
        
        Args:
            gold_quad: (aspect, category, opinion, sentiment) from human annotation
            pred_quad: (aspect, category, opinion, sentiment) from model prediction
        
        Returns:
            SAFE score in [0, 100] (percentage)
        
        Example:
            >>> gold = ("bed", "furniture", "comfortable", "positive")
            >>> pred = ("bed", "furniture", "very comfortable", "positive")
            >>> score = safe.calculate_quad_score(gold, pred)
            >>> print(score)  # ~95-98
        """
        g_aspect, g_category, g_opinion, g_sentiment = gold_quad
        p_aspect, p_category, p_opinion, p_sentiment = pred_quad
        
        # Compute element-wise similarities
        sim_aspect = self.compute_similarity(g_aspect, p_aspect)
        sim_category = self.compute_similarity(g_category, p_category)
        sim_opinion = self.compute_similarity(g_opinion, p_opinion)
        sim_sentiment = self.compute_similarity(g_sentiment, p_sentiment)
        
        # Weighted average
        λ1, λ2, λ3, λ4 = self.lambda_weights
        safe_score = (
            λ1 * sim_aspect +
            λ2 * sim_category +
            λ3 * sim_opinion +
            λ4 * sim_sentiment
        ) * 100
        
        return round(safe_score, 2)
    
    def classify_quad(
        self,
        has_gold: bool,
        has_pred: bool,
        safe_score: Optional[float] = None
    ) -> str:
        """
        Classify a quadruple pair as TP, FP, FN, or N/A.
        
        Classification logic:
        - Gold present, Pred absent → FN
        - Pred present, Gold absent → FP
        - Both present, SAFE ≥ τ → TP
        - Both present, SAFE < τ → FP (and contributes to FN count)
        - Neither present → N/A
        
        Args:
            has_gold: Whether gold standard quad exists
            has_pred: Whether predicted quad exists
            safe_score: SAFE similarity score (required if both quads present)
        
        Returns:
            Classification: "TP", "FP", "FN", or "N/A"
        """
        if has_gold and not has_pred:
            return "FN"
        elif has_pred and not has_gold:
            return "FP"
        elif not has_gold and not has_pred:
            return "N/A"
        elif has_gold and has_pred:
            if safe_score is None:
                raise ValueError("safe_score required when both quads present")
            return "TP" if safe_score >= self.tau else "FP"
        else:
            return "N/A"
    
    def calculate_f1(self, tp: int, fp: int, fn: int) -> float:
        """
        Calculate F1 score from TP, FP, FN counts.
        
        Args:
            tp: True positive count
            fp: False positive count
            fn: False negative count
        
        Returns:
            F1 score in [0, 100] (percentage)
        """
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return round(f1 * 100, 2)
