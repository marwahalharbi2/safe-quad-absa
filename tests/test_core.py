"""
Unit tests for SAFE core module.
"""

import pytest
from safe_metric.core import SAFEMetric


class TestSAFEMetric:
    """Test suite for SAFEMetric class."""
    
    @pytest.fixture
    def safe(self):
        """Create SAFEMetric instance for testing."""
        return SAFEMetric(tau=70.0)
    
    def test_initialization(self):
        """Test metric initialization with default parameters."""
        safe = SAFEMetric()
        assert safe.tau == 70.0
        assert safe.lambda_weights == (0.25, 0.25, 0.25, 0.25)
    
    def test_initialization_custom_params(self):
        """Test metric initialization with custom parameters."""
        safe = SAFEMetric(tau=80.0, lambda_weights=(0.3, 0.3, 0.2, 0.2))
        assert safe.tau == 80.0
        assert safe.lambda_weights == (0.3, 0.3, 0.2, 0.2)
    
    def test_lambda_weights_validation(self):
        """Test that lambda weights must sum to 1.0."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            SAFEMetric(lambda_weights=(0.5, 0.5, 0.5, 0.5))
    
    def test_compute_similarity_identical(self, safe):
        """Test similarity of identical texts."""
        sim = safe.compute_similarity("excellent", "excellent")
        assert sim == 1.0
    
    def test_compute_similarity_case_insensitive(self, safe):
        """Test that similarity is case-insensitive for exact matches."""
        sim = safe.compute_similarity("Excellent", "excellent")
        assert sim == 1.0
    
    def test_compute_similarity_both_empty(self, safe):
        """Test similarity when both texts are empty."""
        sim = safe.compute_similarity("", "")
        assert sim == 1.0
        
        sim = safe.compute_similarity(None, None)
        assert sim == 1.0
    
    def test_compute_similarity_one_empty(self, safe):
        """Test similarity when one text is empty."""
        sim = safe.compute_similarity("excellent", "")
        assert sim == 0.0
        
        sim = safe.compute_similarity("", "excellent")
        assert sim == 0.0
    
    def test_compute_similarity_semantic(self, safe):
        """Test semantic similarity between related terms."""
        # Should be high but not perfect
        sim = safe.compute_similarity("excellent", "very good")
        assert 0.7 < sim < 1.0
        
        # Different sentiments should be low
        sim = safe.compute_similarity("excellent", "terrible")
        assert 0.0 < sim < 0.5
    
    def test_calculate_quad_score_perfect_match(self, safe):
        """Test SAFE score for perfect quad match."""
        gold = ("service", "quality", "excellent", "positive")
        pred = ("service", "quality", "excellent", "positive")
        score = safe.calculate_quad_score(gold, pred)
        assert score == 100.0
    
    def test_calculate_quad_score_semantic_match(self, safe):
        """Test SAFE score for semantically similar quads."""
        gold = ("service", "quality", "excellent", "positive")
        pred = ("service", "quality", "very good", "positive")
        score = safe.calculate_quad_score(gold, pred)
        # Should be high due to semantic similarity
        assert 80.0 < score < 100.0
    
    def test_calculate_quad_score_complete_mismatch(self, safe):
        """Test SAFE score for completely different quads."""
        gold = ("service", "quality", "excellent", "positive")
        pred = ("room", "cleanliness", "dirty", "negative")
        score = safe.calculate_quad_score(gold, pred)
        # Should be very low
        assert score < 50.0
    
    def test_classify_quad_fn(self, safe):
        """Test FN classification (gold present, pred absent)."""
        classification = safe.classify_quad(has_gold=True, has_pred=False)
        assert classification == "FN"
    
    def test_classify_quad_fp(self, safe):
        """Test FP classification (pred present, gold absent)."""
        classification = safe.classify_quad(has_gold=False, has_pred=True)
        assert classification == "FP"
    
    def test_classify_quad_na(self, safe):
        """Test N/A classification (neither present)."""
        classification = safe.classify_quad(has_gold=False, has_pred=False)
        assert classification == "N/A"
    
    def test_classify_quad_tp(self, safe):
        """Test TP classification (both present, score ≥ τ)."""
        classification = safe.classify_quad(
            has_gold=True,
            has_pred=True,
            safe_score=75.0  # Above τ=70
        )
        assert classification == "TP"
    
    def test_classify_quad_fp_below_threshold(self, safe):
        """Test FP classification (both present, score < τ)."""
        classification = safe.classify_quad(
            has_gold=True,
            has_pred=True,
            safe_score=65.0  # Below τ=70
        )
        assert classification == "FP"
    
    def test_classify_quad_missing_score(self, safe):
        """Test that classification requires score when both quads present."""
        with pytest.raises(ValueError, match="safe_score required"):
            safe.classify_quad(has_gold=True, has_pred=True, safe_score=None)
    
    def test_calculate_f1_perfect(self, safe):
        """Test F1 calculation with perfect predictions."""
        f1 = safe.calculate_f1(tp=10, fp=0, fn=0)
        assert f1 == 100.0
    
    def test_calculate_f1_balanced(self, safe):
        """Test F1 calculation with balanced errors."""
        f1 = safe.calculate_f1(tp=8, fp=2, fn=2)
        # Precision = 8/10 = 0.8, Recall = 8/10 = 0.8, F1 = 0.8
        assert abs(f1 - 80.0) < 0.1
    
    def test_calculate_f1_zero_tp(self, safe):
        """Test F1 calculation when no true positives."""
        f1 = safe.calculate_f1(tp=0, fp=5, fn=5)
        assert f1 == 0.0
    
    def test_calculate_f1_zero_predictions(self, safe):
        """Test F1 calculation when no predictions made."""
        f1 = safe.calculate_f1(tp=0, fp=0, fn=10)
        assert f1 == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
