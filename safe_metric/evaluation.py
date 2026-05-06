"""
Dataset-Level Evaluation for Quad-ABSA
======================================

This module handles the evaluation of complete datasets, managing the matching
logic between gold standard and predicted quadruples within reviews.
"""

from typing import Dict, Tuple, Set, List
import pandas as pd
from .core import SAFEMetric


class QuadEvaluator:
    """
    Evaluates Quad-ABSA predictions at the dataset level.
    
    Handles the complex matching logic:
    - Multiple quads per review
    - Best-match assignment between gold and predicted quads
    - Classification and counting for F1 calculation
    """
    
    def __init__(
        self,
        safe_metric: SAFEMetric,
        gold_columns: List[str] = ["Aspect", "Category", "Opinion", "Sentiment"],
        pred_columns: List[str] = ["Aspect.1", "Category.1", "Opinion.1", "Sentiment.1"]
    ):
        """
        Initialize evaluator.
        
        Args:
            safe_metric: SAFEMetric instance for scoring
            gold_columns: Column names for gold standard quads
            pred_columns: Column names for predicted quads
        """
        self.safe = safe_metric
        self.gold_columns = gold_columns
        self.pred_columns = pred_columns
    
    def has_quad(self, row: pd.Series, columns: List[str]) -> bool:
        """
        Check if a row contains a valid quad (at least one non-empty element).
        
        Args:
            row: DataFrame row
            columns: Column names to check
        
        Returns:
            True if at least one column has a non-empty value
        """
        for col in columns:
            val = row[col] if col in row and pd.notna(row[col]) else ""
            if str(val).strip():
                return True
        return False
    
    def get_quad(self, row: pd.Series, columns: List[str]) -> Tuple[str, str, str, str]:
        """
        Extract quadruple from a row.
        
        Args:
            row: DataFrame row
            columns: Column names in order (aspect, category, opinion, sentiment)
        
        Returns:
            Tuple of (aspect, category, opinion, sentiment)
        """
        return tuple(
            row[col] if col in row and pd.notna(row[col]) else ""
            for col in columns
        )
    
    def classify_review_quads(
        self,
        review_group: pd.DataFrame
    ) -> Tuple[Dict[int, str], Dict[int, float], int, int, int]:
        """
        Classify all quads in a single review.
        
        Implements the matching logic:
        1. Identify rows with gold-only, pred-only, or both
        2. For rows with both, find best-matching pairs using SAFE score
        3. Classify each quad as TP/FP/FN based on threshold
        4. Track unmatched gold quads as FN
        
        Args:
            review_group: DataFrame containing all rows for one review
        
        Returns:
            Tuple of:
            - classifications: Dict mapping row index to classification (TP/FP/FN/N/A)
            - quad_scores: Dict mapping row index to SAFE score
            - tp_count: True positive count for this review
            - fp_count: False positive count
            - fn_count: False negative count
        """
        classifications = {}
        quad_scores = {}
        tp_count = 0
        fp_count = 0
        fn_count = 0
        
        matched_gold_indices: Set[int] = set()
        
        # Collect rows with gold and/or predicted quads
        rows_with_gold = []
        rows_with_pred = []
        row_info = {}
        
        for idx, row in review_group.iterrows():
            has_gold = self.has_quad(row, self.gold_columns)
            has_pred = self.has_quad(row, self.pred_columns)
            
            row_info[idx] = {'has_gold': has_gold, 'has_pred': has_pred}
            
            if has_gold:
                rows_with_gold.append((idx, self.get_quad(row, self.gold_columns)))
            if has_pred:
                rows_with_pred.append((idx, self.get_quad(row, self.pred_columns)))
            
            # Immediate classification for rows with only one type
            if has_gold and not has_pred:
                # Gold exists, prediction missing → FN
                classifications[idx] = 'FN'
                quad_scores[idx] = 0.0
                fn_count += 1
            elif has_pred and not has_gold:
                # Prediction exists, gold missing → FP
                classifications[idx] = 'FP'
                quad_scores[idx] = 0.0
                fp_count += 1
            elif not has_gold and not has_pred:
                # Neither exists → N/A
                classifications[idx] = 'N/A'
                quad_scores[idx] = pd.NA
        
        # Handle rows with BOTH gold and predicted quads
        rows_with_both = [
            (idx, info) for idx, info in row_info.items()
            if info['has_gold'] and info['has_pred']
        ]
        
        if rows_with_both:
            # For each prediction, find best matching gold quad
            for pred_idx, pred_quad in rows_with_pred:
                # Skip if already classified (pred-only case)
                if pred_idx in classifications:
                    continue
                
                best_score = 0.0
                best_gold_idx = None
                
                # Find best matching gold quad
                for gold_idx, gold_quad in rows_with_gold:
                    score = self.safe.calculate_quad_score(gold_quad, pred_quad)
                    if score > best_score:
                        best_score = score
                        best_gold_idx = gold_idx
                
                quad_scores[pred_idx] = best_score
                
                # Classify based on threshold
                if best_score >= self.safe.tau:
                    # TP: Prediction matches gold
                    classifications[pred_idx] = 'TP'
                    matched_gold_indices.add(best_gold_idx)
                    tp_count += 1
                else:
                    # Both present but SAFE < τ: wrong prediction (FP + FN)
                    classifications[pred_idx] = 'FP'
                    fp_count += 1
                    fn_count += 1
            
            # Find unmatched gold quads (additional FNs)
            for gold_idx, gold_quad in rows_with_gold:
                if gold_idx not in matched_gold_indices:
                    if gold_idx not in classifications:
                        classifications[gold_idx] = 'FN'
                        quad_scores[gold_idx] = 0.0
                        fn_count += 1
        
        return classifications, quad_scores, tp_count, fp_count, fn_count
    
    def evaluate_dataset(
        self,
        df: pd.DataFrame,
        review_id_column: str = "Review_ID"
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Evaluate entire dataset and compute metrics.
        
        Args:
            df: Input DataFrame with gold and predicted quads
            review_id_column: Name of column containing review IDs
        
        Returns:
            Tuple of:
            - Updated DataFrame with Classification, Quad_SAFE, and Review_SAFE_F1 columns
            - Dictionary of global metrics (precision, recall, F1, counts)
        """
        # Initialize new columns
        df = df.copy()
        df['Classification'] = ""
        df['Quad_SAFE'] = pd.NA
        df['Quad_SAFE'] = df['Quad_SAFE'].astype('Float64')
        df['Review_SAFE_F1'] = pd.NA
        df['Review_SAFE_F1'] = df['Review_SAFE_F1'].astype('Float64')
        
        # Get unique reviews
        unique_reviews = df[review_id_column].dropna().unique()
        
        # Global counters
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        # Process each review
        for review_id in unique_reviews:
            review_mask = df[review_id_column] == review_id
            review_group = df[review_mask]
            
            # Classify quads and get counts
            classifications, quad_scores, tp, fp, fn = self.classify_review_quads(review_group)
            
            # Calculate F1 for this review
            review_f1 = self.safe.calculate_f1(tp, fp, fn)
            
            # Update DataFrame
            for idx, classification in classifications.items():
                df.loc[idx, 'Classification'] = classification
            
            for idx, score in quad_scores.items():
                df.loc[idx, 'Quad_SAFE'] = score
            
            # Set Review_SAFE_F1 only on first row of review
            first_row_idx = review_group.index[0]
            df.loc[first_row_idx, 'Review_SAFE_F1'] = review_f1
            
            # Update global totals
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        # Calculate global metrics
        global_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        global_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        global_f1 = (
            2 * global_precision * global_recall / (global_precision + global_recall)
            if (global_precision + global_recall) > 0 else 0
        )
        
        metrics = {
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'precision': global_precision * 100,
            'recall': global_recall * 100,
            'f1': global_f1 * 100,
            'threshold': self.safe.tau,
            'num_reviews': len(unique_reviews)
        }
        
        return df, metrics
