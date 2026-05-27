"""
Unit tests for src/classification/classifier.py.

Mocks torch and transformers so tests run without GPU or model files.
Covers: MedicalClassifier.__init__ (local, HF, fallback paths),
        predict, predict_with_confidence, predict_batch,
        load_classifier singleton (thread-safe), module-level predict(), constants.

Strategy:
  - Local-path tests create a real temp dir with a placeholder model file.
  - HF-path tests pass a nonexistent model_path, naturally skipping the local check.
  - All torch/transformers ops are mocked via patch() on the real modules.
"""

import threading
from unittest.mock import MagicMock, patch, call
import pytest
import numpy as np

# Pre-load torch.cuda DLLs at import time so that subsequent
# patch("torch.cuda.is_available") doesn't trigger a DLL initialization
# failure when another test has already loaded torch into the process.
import torch
try:
    torch.cuda.is_available()
except Exception:
    pass


# ==============================================================================
# ── Constants ─────────────────────────────────────────────────────────────────
# ==============================================================================

ID2LABEL = {
    0: "Symptoms", 1: "Diagnosis", 2: "Treatment",
    3: "Medication", 4: "Prevention", 5: "General",
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# A path guaranteed not to exist — forces the HF download path in the constructor
NONEXISTENT_PATH = "/__nonexistent__path__"


# ==============================================================================
# ── Helpers ───────────────────────────────────────────────────────────────────
# ==============================================================================


def _make_mock_model():
    """Build a mocked model with config.id2label/label2id and callable returning outputs."""
    mock_model = MagicMock()
    mock_model.config.id2label = ID2LABEL
    mock_model.config.label2id = LABEL2ID
    mock_model.to = MagicMock(return_value=mock_model)
    mock_model.eval = MagicMock(return_value=None)
    mock_outputs = MagicMock()  # model(**inputs) returns this
    mock_model.return_value = mock_outputs
    return mock_model, mock_outputs


def _make_mock_tokenizer():
    """Build a mocked tokenizer where tokenizer(text, ...) returns an object with .to()."""
    mock_tokenizer = MagicMock()
    mock_tokenized = MagicMock()
    mock_tokenizer.return_value = mock_tokenized
    mock_tokenized.to = MagicMock(return_value=mock_tokenized)
    return mock_tokenizer


# ==============================================================================
# ── Fixtures ──────────────────────────────────────────────────────────────────
# ==============================================================================


@pytest.fixture
def mock_model():
    """Returns a mocked model + outputs."""
    return _make_mock_model()


@pytest.fixture
def mock_tokenizer():
    """Returns a mocked tokenizer."""
    return _make_mock_tokenizer()


@pytest.fixture
def classifier_mod():
    """Import the classifier module. Reset singleton before each test."""
    import src.classification.classifier as mod
    mod._classifier_instance = None
    return mod


@pytest.fixture
def model_dir(tmp_path):
    """Create a temporary directory with a dummy model file."""
    d = tmp_path / "test_model"
    d.mkdir(parents=True)
    (d / "pytorch_model.bin").write_text("dummy")
    return str(d)


# ==============================================================================
# ── MedicalClassifier.__init__ ───────────────────────────────────────────────
# ==============================================================================

class TestMedicalClassifierInit:
    """Tests for MedicalClassifier.__init__()."""

    def test_init_local_path_bin(self, classifier_mod, model_dir, mock_model, mock_tokenizer):
        """Loads from local path when .bin model files exist."""
        model, _ = mock_model

        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
            patch("transformers.AutoModelForSequenceClassification.from_pretrained",
                  return_value=model),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.device", return_value=MagicMock()),
        ):
            clf = classifier_mod.MedicalClassifier(model_path=model_dir)

            model.to.assert_called_once()
            model.eval.assert_called_once()
            assert clf.id2label == ID2LABEL
            assert clf.label2id == LABEL2ID

    def test_init_local_path_safetensors(self, classifier_mod, tmp_path, mock_model, mock_tokenizer):
        """Loads from local path when .safetensors model files exist."""
        model, _ = mock_model
        d = tmp_path / "test_model_safe"
        d.mkdir(parents=True)
        (d / "model.safetensors").write_text("dummy")

        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
            patch("transformers.AutoModelForSequenceClassification.from_pretrained", return_value=model),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.device", return_value=MagicMock()),
        ):
            clf = classifier_mod.MedicalClassifier(model_path=str(d))
            assert clf is not None

    def test_init_not_a_directory_uses_hf(self, classifier_mod, tmp_path, mock_model, mock_tokenizer):
        """When path exists but is not a directory, falls back to HF."""
        model, _ = mock_model
        f = tmp_path / "not_a_dir"
        f.write_text("i am a file, not a directory")

        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
            patch("transformers.AutoModelForSequenceClassification.from_pretrained", return_value=model),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.device", return_value=MagicMock()),
        ):
            clf = classifier_mod.MedicalClassifier(model_path=str(f))
            assert clf is not None

    def test_init_no_local_path_uses_hf(self, classifier_mod, mock_model, mock_tokenizer):
        """No local model -> loads from HuggingFace."""
        model, _ = mock_model

        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
            patch("transformers.AutoModelForSequenceClassification.from_pretrained",
                  return_value=model),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.device", return_value=MagicMock()),
        ):
            clf = classifier_mod.MedicalClassifier(model_path=NONEXISTENT_PATH)
            assert clf is not None

    def test_init_custom_model_path(self, classifier_mod, tmp_path, mock_model, mock_tokenizer):
        """Custom model_path is used instead of default."""
        model, _ = mock_model
        d = tmp_path / "custom_model"
        d.mkdir(parents=True)
        (d / "model.safetensors").write_text("dummy")

        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
            patch("transformers.AutoModelForSequenceClassification.from_pretrained",
                  return_value=model),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.device", return_value=MagicMock()),
        ):
            clf = classifier_mod.MedicalClassifier(model_path=str(d))
            assert clf is not None

    def test_init_hf_fallback_on_exception(self, classifier_mod, mock_model, mock_tokenizer):
        """When HF primary repo fails, falls back to HF_REPO_ID_FALLBACK.

        Execution flow:
          1. try:   AutoTokenizer(HF_REPO_ID) → OSError  → jumps to except
          2. except: AutoTokenizer(HF_REPO_ID_FALLBACK) → mock_tokenizer
          3. except: AutoModel(HF_REPO_ID_FALLBACK)     → model (called once)
        """
        model, _ = mock_model

        with (
            patch("transformers.AutoTokenizer.from_pretrained",
                  side_effect=[OSError("not found"), mock_tokenizer]),
            patch("transformers.AutoModelForSequenceClassification.from_pretrained",
                  return_value=model),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.device", return_value=MagicMock()),
        ):
            clf = classifier_mod.MedicalClassifier(model_path=NONEXISTENT_PATH)
            assert clf is not None
            model.to.assert_called_once()
            model.eval.assert_called_once()
            assert clf.id2label == ID2LABEL
            assert clf.label2id == LABEL2ID

    def test_init_hf_fallback_repo_ids(self, classifier_mod, mock_model, mock_tokenizer):
        """Fallback path tries HF_REPO_ID first, then HF_REPO_ID_FALLBACK.

        AutoTokenizer is called twice (try: HF_REPO_ID, except: FALLBACK).
        AutoModel is called once (except: FALLBACK) because the try block
        raises at AutoTokenizer before reaching AutoModel.
        """
        model, _ = mock_model
        hf_id = classifier_mod.HF_REPO_ID
        fallback_id = classifier_mod.HF_REPO_ID_FALLBACK

        with (
            patch("transformers.AutoTokenizer.from_pretrained",
                  side_effect=[OSError("not found"), mock_tokenizer]) as mock_tok,
            patch("transformers.AutoModelForSequenceClassification.from_pretrained",
                  return_value=model) as mock_model_cls,
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.device", return_value=MagicMock()),
        ):
            classifier_mod.MedicalClassifier(model_path=NONEXISTENT_PATH)

            assert mock_tok.call_args_list[0] == call(hf_id)
            assert mock_tok.call_args_list[1] == call(fallback_id)
            # AutoModel is only called once, inside the except block, with fallback
            mock_model_cls.assert_called_once_with(fallback_id)

    def test_init_device_cpu(self, classifier_mod, mock_model, mock_tokenizer):
        """When CUDA is not available, device is CPU."""
        model, _ = mock_model
        mock_device = MagicMock()
        mock_device.__str__ = MagicMock(return_value="cpu")

        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
            patch("transformers.AutoModelForSequenceClassification.from_pretrained",
                  return_value=model),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.device", return_value=mock_device),
        ):
            clf = classifier_mod.MedicalClassifier(model_path=NONEXISTENT_PATH)
            assert str(clf.device) == "cpu"

    def test_init_device_cuda(self, classifier_mod, mock_model, mock_tokenizer):
        """When CUDA is available, device is CUDA."""
        model, _ = mock_model
        mock_device = MagicMock()
        mock_device.__str__ = MagicMock(return_value="cuda")

        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
            patch("transformers.AutoModelForSequenceClassification.from_pretrained",
                  return_value=model),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.device", return_value=mock_device),
        ):
            clf = classifier_mod.MedicalClassifier(model_path=NONEXISTENT_PATH)
            assert str(clf.device) == "cuda"

    def test_init_prints_confirmation(self, classifier_mod, mock_model, mock_tokenizer, capsys):
        """Init prints a confirmation message."""
        model, _ = mock_model

        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
            patch("transformers.AutoModelForSequenceClassification.from_pretrained",
                  return_value=model),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.device", return_value=MagicMock()),
        ):
            classifier_mod.MedicalClassifier(model_path=NONEXISTENT_PATH)
            captured = capsys.readouterr()
            assert "Classifier loaded" in captured.out
            assert "Symptoms" in captured.out

    def test_init_prints_local_source(self, classifier_mod, model_dir, mock_model,
                                       mock_tokenizer, capsys):  # noqa: E127
        """Init prints local path when loading from disk."""
        model, _ = mock_model

        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
            patch("transformers.AutoModelForSequenceClassification.from_pretrained",
                  return_value=model),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.device", return_value=MagicMock()),
        ):
            classifier_mod.MedicalClassifier(model_path=model_dir)
            captured = capsys.readouterr()
            assert "Loading classifier from local" in captured.out

    def test_init_prints_hf_download(self, classifier_mod, mock_model, mock_tokenizer, capsys):
        """Init prints download message when loading from HF."""
        model, _ = mock_model

        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
            patch("transformers.AutoModelForSequenceClassification.from_pretrained",
                  return_value=model),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.device", return_value=MagicMock()),
        ):
            classifier_mod.MedicalClassifier(model_path=NONEXISTENT_PATH)
            captured = capsys.readouterr()
            assert "Downloading from HuggingFace" in captured.out


# ==============================================================================
# ── MedicalClassifier.predict ────────────────────────────────────────────────
# ==============================================================================

class TestMedicalClassifierPredict:
    """Tests for MedicalClassifier.predict()."""

    @pytest.fixture
    def clf(self, classifier_mod):
        """Build a mocked classifier for predict tests."""
        model, outputs = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        argmax_result = MagicMock()
        argmax_result.item = MagicMock(return_value=2)  # Treatment

        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
            patch("transformers.AutoModelForSequenceClassification.from_pretrained",
                  return_value=model),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.device", return_value=MagicMock()),
            patch("torch.argmax", return_value=argmax_result),
        ):
            clf = classifier_mod.MedicalClassifier(model_path=NONEXISTENT_PATH)
            yield clf

    def test_predict_returns_string(self, clf):
        """predict() returns a string category."""
        result = clf.predict("What are the symptoms of diabetes?")
        assert isinstance(result, str)
        assert result in ID2LABEL.values()

    def test_predict_returns_correct_category(self, clf):
        """predict returns the category from argmax result."""
        result = clf.predict("What is the treatment for hypertension?")
        assert result == "Treatment"  # argmax returns 2 -> Treatment

    def test_predict_uses_tokenizer(self, clf):
        """predict calls tokenizer with correct arguments."""
        clf.predict("test query")
        clf.tokenizer.assert_called_with(
            "test query",
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        )

    def test_predict_empty_string(self, clf):
        """predict handles empty string without error."""
        result = clf.predict("")
        assert isinstance(result, str)


# ==============================================================================
# ── MedicalClassifier.predict_with_confidence ────────────────────────────────
# ==============================================================================

class TestMedicalClassifierPredictWithConfidence:
    """Tests for MedicalClassifier.predict_with_confidence()."""

    @pytest.fixture
    def clf(self, classifier_mod):
        """Build a mocked classifier for confidence tests.

        torch.softmax returns a 2D array (batch=1, classes=6) so that
        probs = softmax(...)[0] yields a 1D array of 6 probabilities.
        """
        model, outputs = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        argmax_result = MagicMock()
        argmax_result.item = MagicMock(return_value=2)  # Treatment
        probs = np.array([[0.05, 0.10, 0.70, 0.05, 0.05, 0.05]], dtype=np.float32)

        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
            patch("transformers.AutoModelForSequenceClassification.from_pretrained",
                  return_value=model),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.device", return_value=MagicMock()),
            patch("torch.argmax", return_value=argmax_result),
            patch("torch.softmax", return_value=probs),
        ):
            clf = classifier_mod.MedicalClassifier(model_path=NONEXISTENT_PATH)
            yield clf

    def test_returns_dict_with_keys(self, clf):
        """predict_with_confidence returns dict with category, confidence, all_scores."""
        result = clf.predict_with_confidence("What are the side effects of ibuprofen?")
        assert isinstance(result, dict)
        assert "category" in result
        assert "confidence" in result
        assert "all_scores" in result

    def test_confidence_is_float(self, clf):
        """confidence is a float in [0, 1]."""
        result = clf.predict_with_confidence("test")
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_all_scores_has_all_categories(self, clf):
        """all_scores dict has one entry per category."""
        result = clf.predict_with_confidence("test")
        assert isinstance(result["all_scores"], dict)
        assert len(result["all_scores"]) == 6
        for cat in ID2LABEL.values():
            assert cat in result["all_scores"]

    def test_category_in_valid_set(self, clf):
        """category is one of the 6 valid categories."""
        result = clf.predict_with_confidence("test")
        assert result["category"] in ID2LABEL.values()

    def test_confidence_score_matches_softmax(self, clf):
        """confidence equals the probability of the predicted class."""
        result = clf.predict_with_confidence("test")
        # argmax returns 2 (Treatment), probs[0][2] = 0.70
        assert result["category"] == "Treatment"
        assert result["confidence"] == pytest.approx(0.70, abs=1e-6)
        assert result["all_scores"]["Treatment"] == pytest.approx(0.70, abs=1e-6)
        assert result["all_scores"]["Symptoms"] == pytest.approx(0.05, abs=1e-6)


# ==============================================================================
# ── MedicalClassifier.predict_batch ──────────────────────────────────────────
# ==============================================================================

class TestMedicalClassifierPredictBatch:
    """Tests for MedicalClassifier.predict_batch()."""

    @pytest.fixture
    def clf(self, classifier_mod):
        """Build a mocked classifier for batch tests."""
        model, outputs = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        argmax_result = MagicMock()
        argmax_result.item = MagicMock(return_value=2)  # Treatment

        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
            patch("transformers.AutoModelForSequenceClassification.from_pretrained",
                  return_value=model),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.device", return_value=MagicMock()),
            patch("torch.argmax", return_value=argmax_result),
        ):
            clf = classifier_mod.MedicalClassifier(model_path=NONEXISTENT_PATH)
            yield clf

    def test_returns_list_of_strings(self, clf):
        """predict_batch returns a list of strings (one per input)."""
        texts = ["symptom question?", "treatment question?"]
        results = clf.predict_batch(texts)
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)

    def test_empty_list(self, clf):
        """predict_batch with empty list returns empty list."""
        results = clf.predict_batch([])
        assert results == []

    def test_single_item(self, clf):
        """predict_batch with single item returns list of one."""
        results = clf.predict_batch(["test query"])
        assert len(results) == 1
        assert isinstance(results[0], str)


# ==============================================================================
# ── load_classifier singleton ─────────────────────────────────────────────────
# ==============================================================================

class TestLoadClassifier:
    """Tests for module-level load_classifier() singleton."""

    def test_returns_medical_classifier_instance(self, classifier_mod):
        """load_classifier() returns a MedicalClassifier instance."""
        with patch.object(classifier_mod, "MedicalClassifier") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            classifier_mod._classifier_instance = None

            clf = classifier_mod.load_classifier()
            assert clf is mock_instance

    def test_singleton_returns_same_instance(self, classifier_mod):
        """Subsequent calls return the same cached instance."""
        with patch.object(classifier_mod, "MedicalClassifier") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            classifier_mod._classifier_instance = None

            clf1 = classifier_mod.load_classifier()
            clf2 = classifier_mod.load_classifier()
            assert clf1 is clf2
            mock_cls.assert_called_once()

    def test_singleton_called_once(self, classifier_mod):
        """MedicalClassifier is only constructed once."""
        with patch.object(classifier_mod, "MedicalClassifier") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            classifier_mod._classifier_instance = None

            classifier_mod.load_classifier()
            classifier_mod.load_classifier()
            classifier_mod.load_classifier()

            mock_cls.assert_called_once()

    def test_thread_safety(self, classifier_mod):
        """Concurrent calls to load_classifier return the same instance."""
        with patch.object(classifier_mod, "MedicalClassifier") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            classifier_mod._classifier_instance = None

            results = []

            def get_instance():
                inst = classifier_mod.load_classifier()
                results.append(inst)

            threads = [threading.Thread(target=get_instance) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert all(r is results[0] for r in results)
            mock_cls.assert_called_once()

    def test_kwargs_passed_to_constructor(self, classifier_mod):
        """load_classifier passes kwargs to MedicalClassifier."""
        with patch.object(classifier_mod, "MedicalClassifier") as mock_cls:
            mock_cls.return_value = MagicMock()
            classifier_mod._classifier_instance = None
            classifier_mod.load_classifier(model_path="/test/path")
            mock_cls.assert_called_once_with(model_path="/test/path")


# ==============================================================================
# ── Module-level predict() ───────────────────────────────────────────────────
# ==============================================================================

class TestModuleLevelPredict:
    """Tests for module-level predict() convenience function."""

    def test_with_explicit_classifier(self, classifier_mod):
        """predict(text, classifier=...) uses provided classifier."""
        mock_clf = MagicMock(spec=classifier_mod.MedicalClassifier)
        mock_clf.predict.return_value = "Symptoms"

        result = classifier_mod.predict("test query", classifier=mock_clf)

        mock_clf.predict.assert_called_once_with("test query")
        assert result == "Symptoms"

    def test_without_classifier_uses_singleton(self, classifier_mod):
        """predict(text) without classifier uses global singleton."""
        with patch.object(classifier_mod, "load_classifier") as mock_load:
            mock_clf = MagicMock()
            mock_clf.predict.return_value = "Diagnosis"
            mock_load.return_value = mock_clf
            classifier_mod._classifier_instance = None

            result = classifier_mod.predict("test query")

            mock_load.assert_called_once()
            mock_clf.predict.assert_called_once_with("test query")
            assert result == "Diagnosis"

    def test_reuses_cached_singleton(self, classifier_mod):
        """predict() reuses already-loaded singleton without calling load_classifier again."""
        with patch.object(classifier_mod, "load_classifier") as mock_load:
            mock_clf = MagicMock()
            mock_clf.predict.return_value = "Symptoms"
            mock_load.return_value = mock_clf

            classifier_mod._classifier_instance = mock_clf

            result = classifier_mod.predict("what is diabetes?", classifier=None)
            assert result == "Symptoms"
            mock_load.assert_not_called()


# ==============================================================================
# ── Constants ─────────────────────────────────────────────────────────────────
# ==============================================================================

class TestConstants:
    """Tests for module-level constants."""

    def test_default_path_is_under_models(self, classifier_mod):
        """DEFAULT_LOCAL_PATH ends with models/classifier/biobert_classifier."""
        assert "models" in str(classifier_mod.DEFAULT_LOCAL_PATH)
        assert "classifier" in str(classifier_mod.DEFAULT_LOCAL_PATH)
        assert "biobert_classifier" in str(classifier_mod.DEFAULT_LOCAL_PATH)

    def test_hf_repo_id(self, classifier_mod):
        """HF_REPO_ID is set correctly."""
        assert classifier_mod.HF_REPO_ID == "AbdoMatrix/biobert-medical-classifier"

    def test_hf_repo_id_fallback(self, classifier_mod):
        """HF_REPO_ID_FALLBACK is set correctly."""
        assert classifier_mod.HF_REPO_ID_FALLBACK == "AbdoMatrix/distilbert-medical-classifier"

    def test_project_root_is_resolved(self, classifier_mod):
        """PROJECT_ROOT is an absolute Path."""
        from pathlib import Path
        assert isinstance(classifier_mod.PROJECT_ROOT, Path)
        assert classifier_mod.PROJECT_ROOT.is_absolute()
