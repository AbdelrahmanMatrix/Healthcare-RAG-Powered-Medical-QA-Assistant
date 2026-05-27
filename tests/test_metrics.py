"""
Tests for src/evaluation/metrics.py.

Covers all public functions:
  - compute_bleu()
  - compute_rouge()
  - compute_bertscore()
  - compute_faithfulness()
  - compute_improvement()
  - evaluate_pair()
  - evaluate_full()
"""

from unittest.mock import patch, MagicMock
import numpy as np
import pytest


# ==============================================================================
# compute_bleu
# ==============================================================================

class TestComputeBleu:
    """Tests for compute_bleu()."""

    def test_basic_score(self):
        """Identical strings give BLEU close to 1.0."""
        from src.evaluation.metrics import compute_bleu

        score = compute_bleu(["the cat sat on the mat"], ["the cat sat on the mat"])
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_no_overlap(self):
        """Completely different strings — score should be low."""
        from src.evaluation.metrics import compute_bleu

        score = compute_bleu(["abc"], ["xyz"])
        assert 0.0 <= score < 0.5

    def test_multiple_pairs(self):
        """Averaging across multiple prediction-reference pairs."""
        from src.evaluation.metrics import compute_bleu

        score = compute_bleu(["the cat", "hello world"], ["the cat", "goodbye world"])
        assert isinstance(score, float)

    def test_empty_prediction_string(self):
        """An empty prediction string yields 0.0 for that pair."""
        from src.evaluation.metrics import compute_bleu

        score = compute_bleu([""], ["reference text"])
        assert score == 0.0

    def test_empty_reference_string(self):
        """An empty reference string yields 0.0 for that pair."""
        from src.evaluation.metrics import compute_bleu

        score = compute_bleu(["prediction"], [""])
        assert score == 0.0


# ==============================================================================
# compute_rouge
# ==============================================================================

class TestComputeRouge:
    """Tests for compute_rouge()."""

    def test_empty_predictions(self):
        """Empty predictions list → 0.0."""
        from src.evaluation.metrics import compute_rouge

        assert compute_rouge([], ["ref"]) == 0.0

    def test_empty_references(self):
        """Empty references list → 0.0."""
        from src.evaluation.metrics import compute_rouge

        assert compute_rouge(["pred"], []) == 0.0

    def test_both_empty(self):
        """Both empty → 0.0."""
        from src.evaluation.metrics import compute_rouge

        assert compute_rouge([], []) == 0.0

    def test_with_empty_string_in_pair(self):
        """An empty string in a pair yields 0.0 for that pair (lines 87-88)."""
        from src.evaluation.metrics import compute_rouge

        score = compute_rouge(["valid pred", ""], ["valid ref", "ref2"])
        assert isinstance(score, float)
        assert score > 0.0  # Only one valid pair contributes

    def test_smoke_with_real_scorer(self):
        """Smoke test with the real rouge_scorer if installed."""
        from src.evaluation.metrics import compute_rouge

        score = compute_rouge(
            ["the cat sat on the mat"],
            ["the cat sat on the mat"],
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ==============================================================================
# compute_bertscore
# ==============================================================================

class TestComputeBertscore:
    """Tests for compute_bertscore()."""

    def test_import_error_path(self):
        """When bert-score is not importable, returns 0.0."""
        from src.evaluation.metrics import compute_bertscore

        real_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "bert_score":
                raise ImportError("mock: bert_score not available")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = compute_bertscore(["test"], ["test"])
            assert result == 0.0

    def test_compute_bertscore_success_path(self):
        """BERTScore success path: mock bert_score.score and verify F1 mean."""
        from src.evaluation.metrics import compute_bertscore

        mock_score_fn = MagicMock(return_value=(None, None, np.array([0.85, 0.90])))
        real_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "bert_score":
                mock_mod = MagicMock()
                mock_mod.score = mock_score_fn
                return mock_mod
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = compute_bertscore(["pred1"], ["ref1"])
            assert result == pytest.approx(0.875, abs=1e-4)

    def test_compute_bertscore_success_single_value(self):
        """BERTScore success path with single prediction pair."""
        from src.evaluation.metrics import compute_bertscore

        mock_score_fn = MagicMock(return_value=(None, None, np.array([0.92])))
        real_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "bert_score":
                mock_mod = MagicMock()
                mock_mod.score = mock_score_fn
                return mock_mod
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = compute_bertscore(["excellent match"], ["excellent match"])
            assert result == pytest.approx(0.92, abs=1e-4)


# ==============================================================================
# compute_faithfulness
# ==============================================================================

class TestComputeFaithfulness:
    """Tests for compute_faithfulness()."""

    def test_sentence_transformers_not_installed(self):
        """When sentence-transformers can't be imported, returns 0.0."""
        from src.evaluation.metrics import compute_faithfulness

        real_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("mock: not installed")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = compute_faithfulness(["test"], [["context"]])
            assert result == 0.0

    def test_model_load_exception_handled(self):
        """When _get_nli_model raises a general exception, returns 0.0."""
        from src.evaluation.metrics import compute_faithfulness

        with patch("src.evaluation.metrics._get_nli_model") as mock_fn:
            mock_fn.side_effect = Exception("model download failed")
            result = compute_faithfulness(["test"], [["context"]])
            assert result == 0.0

    def test_empty_predictions(self):
        """Empty predictions list → 0.0."""
        from src.evaluation.metrics import compute_faithfulness

        # Patch _get_nli_model to avoid loading CrossEncoder
        with patch("src.evaluation.metrics._get_nli_model") as mock_model:
            mock_model.return_value = MagicMock()
            result = compute_faithfulness([], [["context"]])
            assert result == 0.0

    def test_empty_pred_in_pair(self):
        """Empty pred string in a pair is skipped."""
        from src.evaluation.metrics import compute_faithfulness

        with patch("src.evaluation.metrics._get_nli_model") as mock_model:
            mock_model.return_value = MagicMock()
            result = compute_faithfulness([""], [["context"]])
            assert result == 0.0

    def test_empty_context_list_in_pair(self):
        """Empty context list in a pair is skipped."""
        from src.evaluation.metrics import compute_faithfulness

        with patch("src.evaluation.metrics._get_nli_model") as mock_model:
            mock_model.return_value = MagicMock()
            result = compute_faithfulness(["pred"], [[]])
            assert result == 0.0

    def test_predict_exception_handled(self):
        """When model.predict raises an exception, pair is not counted as faithful."""
        from src.evaluation.metrics import compute_faithfulness

        with patch("src.evaluation.metrics._get_nli_model") as mock_fn:
            mock_model = MagicMock()
            mock_model.predict.side_effect = Exception("predict failed")
            mock_fn.return_value = mock_model
            # 1 prediction, model.predict fails → 0 faithful out of 1
            result = compute_faithfulness(["test pred"], [["test context"]])
            assert result == 0.0

    def test_compute_faithfulness_entailment_above_threshold(self):
        """Faithfulness counts a pair as faithful when entailment >= threshold."""
        from src.evaluation.metrics import compute_faithfulness

        with patch("src.evaluation.metrics._get_nli_model") as mock_nli:
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([[0.85, 0.10, 0.05]])
            mock_nli.return_value = mock_model

            result = compute_faithfulness(["test answer"], [[u"test context"]])
            assert result == 1.0

    def test_compute_faithfulness_entailment_below_threshold(self):
        """Faithfulness does NOT count a pair when entailment < threshold."""
        from src.evaluation.metrics import compute_faithfulness

        with patch("src.evaluation.metrics._get_nli_model") as mock_nli:
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([[0.30, 0.40, 0.30]])
            mock_nli.return_value = mock_model

            result = compute_faithfulness(["test answer"], [[u"test context"]])
            assert result == 0.0

    def test_compute_faithfulness_multiple_pairs(self):
        """Faithfulness averages across multiple prediction-context pairs."""
        from src.evaluation.metrics import compute_faithfulness

        with patch("src.evaluation.metrics._get_nli_model") as mock_nli:
            mock_model = MagicMock()
            mock_model.predict.side_effect = [
                np.array([[0.90, 0.05, 0.05], [0.20, 0.60, 0.20]]),
                np.array([[0.20, 0.60, 0.20]]),
            ]
            mock_nli.return_value = mock_model

            result = compute_faithfulness(
                ["answer 1", "answer 2"],
                [[u"ctx 1a", u"ctx 1b"], [u"ctx 2"]],
            )
            assert result == 0.5


# ==============================================================================
# compute_improvement
# ==============================================================================

# ==============================================================================
# _get_nli_model
# ==============================================================================

class TestGetNliModel:
    """Tests for _get_nli_model() singleton."""

    def test_get_nli_model_loads_cross_encoder(self):
        """_get_nli_model instantiates CrossEncoder when cache is empty."""
        import src.evaluation.metrics as metrics_mod
        metrics_mod._FaithfulnessModel = None

        mock_ce = MagicMock()
        mock_cross_encoder_cls = MagicMock(return_value=mock_ce)
        real_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                mock_mod = MagicMock()
                mock_mod.CrossEncoder = mock_cross_encoder_cls
                return mock_mod
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            model = metrics_mod._get_nli_model()
            assert model is mock_ce
            mock_cross_encoder_cls.assert_called_once_with(
                "cross-encoder/nli-deberta-v3-base"
            )

    def test_get_nli_model_returns_cached_instance(self):
        """_get_nli_model returns cached model on subsequent calls."""
        import src.evaluation.metrics as metrics_mod

        cached = MagicMock()
        metrics_mod._FaithfulnessModel = cached

        with patch("builtins.__import__") as mock_import:
            model = metrics_mod._get_nli_model()
            assert model is cached
            mock_import.assert_not_called()

        metrics_mod._FaithfulnessModel = None


# ==============================================================================
# compute_improvement
# ==============================================================================

class TestComputeImprovement:
    """Tests for compute_improvement()."""

    def test_positive_improvement(self):
        """Baseline → improved yields positive percentage."""
        from src.evaluation.metrics import compute_improvement

        result = compute_improvement(0.5, 0.75)
        assert result == 50.0

    def test_no_improvement(self):
        """No change → 0%."""
        from src.evaluation.metrics import compute_improvement

        assert compute_improvement(0.5, 0.5) == 0.0

    def test_degradation(self):
        """Negative improvement."""
        from src.evaluation.metrics import compute_improvement

        result = compute_improvement(0.8, 0.6)
        assert result == pytest.approx(-25.0)

    def test_baseline_zero_improvement_positive(self):
        """Baseline 0, improved > 0 → infinity."""
        from src.evaluation.metrics import compute_improvement

        assert compute_improvement(0.0, 0.5) == float("inf")

    def test_baseline_zero_improvement_zero(self):
        """Both zero → 0.0."""
        from src.evaluation.metrics import compute_improvement

        assert compute_improvement(0.0, 0.0) == 0.0

    def test_baseline_zero_improvement_negative(self):
        """Baseline 0, improved < 0 → 0.0."""
        from src.evaluation.metrics import compute_improvement

        assert compute_improvement(0.0, -0.1) == 0.0


# ==============================================================================
# evaluate_pair
# ==============================================================================

class TestEvaluatePair:
    """Tests for evaluate_pair()."""

    @patch("src.evaluation.metrics.compute_bleu", return_value=0.5)
    @patch("src.evaluation.metrics.compute_rouge", return_value=0.3)
    def test_basic_structure(self, mock_rouge, mock_bleu):
        """Returns dict with expected keys."""
        from src.evaluation.metrics import evaluate_pair

        result = evaluate_pair(["pred"], ["ref"], label="TestModel")
        assert result == {
            "label": "TestModel",
            "bleu": 0.5,
            "rouge_l": 0.3,
            "n_samples": 1,
        }

    @patch("src.evaluation.metrics.compute_bleu", return_value=0.8)
    @patch("src.evaluation.metrics.compute_rouge", return_value=0.9)
    def test_default_label(self, mock_rouge, mock_bleu):
        """Default label is 'Model'."""
        from src.evaluation.metrics import evaluate_pair

        result = evaluate_pair(["a"], ["b"])
        assert result["label"] == "Model"


# ==============================================================================
# evaluate_full
# ==============================================================================

class TestEvaluateFull:
    """Tests for evaluate_full()."""

    @patch("src.evaluation.metrics.compute_bleu", return_value=0.5)
    @patch("src.evaluation.metrics.compute_rouge", return_value=0.3)
    @patch("src.evaluation.metrics.compute_bertscore", return_value=0.85)
    def test_without_contexts(self, mock_bert, mock_rouge, mock_bleu):
        """Without contexts, faithfulness is not included."""
        from src.evaluation.metrics import evaluate_full

        result = evaluate_full(["pred"], ["ref"], label="RAG")
        assert result == {
            "label": "RAG",
            "bleu": 0.5,
            "rouge_l": 0.3,
            "bertscore_f1": 0.85,
            "n_samples": 1,
        }
        assert "faithfulness" not in result

    @patch("src.evaluation.metrics.compute_bleu", return_value=0.5)
    @patch("src.evaluation.metrics.compute_rouge", return_value=0.3)
    @patch("src.evaluation.metrics.compute_bertscore", return_value=0.85)
    @patch("src.evaluation.metrics.compute_faithfulness", return_value=0.95)
    def test_with_contexts(self, mock_faith, mock_bert, mock_rouge, mock_bleu):
        """With contexts, faithfulness is included."""
        from src.evaluation.metrics import evaluate_full

        result = evaluate_full(
            ["pred"], ["ref"], contexts=[["chunk1", "chunk2"]], label="RAG"
        )
        assert result["faithfulness"] == 0.95

    @patch("src.evaluation.metrics.compute_bleu", return_value=0.5)
    @patch("src.evaluation.metrics.compute_rouge", return_value=0.3)
    @patch("src.evaluation.metrics.compute_bertscore", return_value=0.85)
    def test_default_label_rag(self, mock_bert, mock_rouge, mock_bleu):
        """Default label is 'RAG'."""
        from src.evaluation.metrics import evaluate_full

        result = evaluate_full(["pred"], ["ref"])
        assert result["label"] == "RAG"
