"""
Tests for src/data modules.

Covers:
  - src/data/hub.py      (check_data_exists, download_file, download_all_data)
  - src/data/loader.py    (load_raw_data, load_cleaned_data, load_labelled_data,
                           load_dataset_from_hub)
  - src/data/preprocessor.py  (additional edge cases)
  - src/data/labeller.py       (additional edge cases)
"""

from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import pandas as pd


# ==============================================================================
# ── Fixtures ──────────────────────────────────────────────────────────────────
# ==============================================================================

MOCK_CSV_DATA = pd.DataFrame({
    "pubid": ["1", "2"],
    "question": ["q1", "q2"],
    "context": ["ctx1", "ctx2"],
    "long_answer": ["ans1", "ans2"],
    "final_decision": ["yes", "no"],
})


def _patch_hub(test_root):
    """Return context-managers that patch src.data.hub globals to use test_root."""
    return [
        patch("src.data.hub.PROJECT_ROOT", test_root),
    ]


# ==============================================================================
# ── hub.py ────────────────────────────────────────────────────────────────────
# ==============================================================================

class TestHubCheckDataExists:
    """Tests for hub.check_data_exists()."""

    def test_all_missing(self, tmp_path):
        """When no data files exist, all statuses are False."""
        fake_required = [
            ("remote/file1.csv", tmp_path / "data" / "raw" / "file1.csv"),
            ("remote/file2.csv", tmp_path / "data" / "processed" / "file2.csv"),
        ]
        with patch("src.data.hub.REQUIRED_FILES", fake_required):
            from src.data.hub import check_data_exists

            status = check_data_exists()
            assert isinstance(status, dict)
            for path_str, exists in status.items():
                assert exists is False, f"{path_str} should be False"

    def test_mixed_status(self, tmp_path):
        """Some files exist, some don't."""
        existing = tmp_path / "data" / "raw" / "exists.csv"
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_text("dummy")

        fake_required = [
            ("remote/exists.csv", existing),
            ("remote/missing.csv", tmp_path / "data" / "processed" / "missing.csv"),
        ]
        with patch("src.data.hub.REQUIRED_FILES", fake_required):
            from src.data.hub import check_data_exists

            status = check_data_exists()
            assert status[str(existing)] is True
            true_count = sum(1 for v in status.values() if v)
            assert true_count == 1

    def test_default_required_file_rejects_stub(self, tmp_path):
        """Default required artifacts smaller than the minimum size are treated as missing."""
        from src.data.hub import check_data_exists

        stub = tmp_path / "data" / "raw" / "pubmedqa_raw.csv"
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_text("dummy")

        with patch("src.data.hub.REQUIRED_FILES", [("raw/pubmedqa_raw.csv", stub)]):
            with patch.dict("src.data.hub.MIN_FILE_BYTES", {str(stub): 1024}, clear=True):
                status = check_data_exists()

        assert status[str(stub)] is False


class TestHubDownloadFile:
    """Tests for hub.download_file()."""

    def test_huggingface_hub_not_installed(self):
        """When huggingface-hub is not importable, returns False."""
        from src.data.hub import download_file

        real_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "huggingface_hub":
                raise ImportError("mock: not installed")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = download_file("test/file.csv", Path("/tmp/test.csv"))
            assert result is False

    def test_download_success(self, tmp_path):
        """Successful download returns True."""
        from src.data.hub import download_file

        # All paths must be under the same root for .relative_to() to work
        test_root = tmp_path / "project"
        test_root.mkdir()
        dest = test_root / "data" / "raw" / "downloaded.csv"

        # Create a temp cache file that hf_hub_download will "return"
        cache_file = tmp_path / "cache" / "downloaded.csv"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text("dummy content that is long enough for size calc")

        mock_hf_download = MagicMock(return_value=str(cache_file))

        real_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "huggingface_hub":
                mock_mod = MagicMock()
                mock_mod.hf_hub_download = mock_hf_download
                return mock_mod
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with patch("src.data.hub.PROJECT_ROOT", test_root):
                with patch("src.data.hub.os.getenv", return_value=None):
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    result = download_file("data/raw/downloaded.csv", dest)

                assert result is True, f"download failed: dest={dest}, exists={dest.exists()}"

                mock_hf_download.assert_called_once_with(
                    repo_id="AbdoMatrix/healthcare-rag-data",
                    filename="data/raw/downloaded.csv",
                    repo_type="dataset",
                    token=None,
                )
                # Verify the file was copied to dest
                assert dest.exists()
                assert dest.read_text() == "dummy content that is long enough for size calc"

    def test_download_failure(self, tmp_path):
        """When download raises an exception, returns False."""
        from src.data.hub import download_file

        test_root = tmp_path / "project"
        test_root.mkdir()
        dest = test_root / "data" / "raw" / "failed.csv"
        mock_hf_download = MagicMock(side_effect=Exception("Network error"))

        real_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "huggingface_hub":
                mock_mod = MagicMock()
                mock_mod.hf_hub_download = mock_hf_download
                return mock_mod
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with patch("src.data.hub.PROJECT_ROOT", test_root):
                result = download_file("data/raw/failed.csv", dest)
                assert result is False


class TestHubDownloadAllData:
    """Tests for hub.download_all_data()."""

    def test_empty_required_list(self):
        """No required files → all zeros."""
        from src.data.hub import download_all_data

        with patch("src.data.hub.REQUIRED_FILES", []):
            result = download_all_data()
            assert result == {"downloaded": 0, "skipped": 0, "failed": 0}

    def test_skip_existing_files(self, tmp_path):
        """Files that already exist are skipped."""
        test_root = tmp_path / "project"
        test_root.mkdir()
        existing = test_root / "existing.csv"
        existing.write_text("dummy")

        with patch("src.data.hub.REQUIRED_FILES", [("remote", existing)]):
            with patch("src.data.hub.PROJECT_ROOT", test_root):
                from src.data.hub import download_all_data

                with patch("src.data.hub.download_file") as mock_dl:
                    result = download_all_data()
                    assert result == {"downloaded": 0, "skipped": 1, "failed": 0}
                    mock_dl.assert_not_called()

    def test_download_missing_file_success(self, tmp_path):
        """Missing files are downloaded and counted."""
        test_root = tmp_path / "project"
        test_root.mkdir()
        missing = test_root / "missing.csv"

        with patch("src.data.hub.REQUIRED_FILES", [("remote", missing)]):
            with patch("src.data.hub.PROJECT_ROOT", test_root):
                from src.data.hub import download_all_data

                with patch("src.data.hub.download_file", return_value=True) as mock_dl:
                    result = download_all_data()
                    assert result == {"downloaded": 1, "skipped": 0, "failed": 0}
                    mock_dl.assert_called_once()

    def test_download_missing_file_fails(self, tmp_path):
        """Missing files that fail are counted as failed."""
        test_root = tmp_path / "project"
        test_root.mkdir()
        missing = test_root / "missing.csv"

        with patch("src.data.hub.REQUIRED_FILES", [("remote", missing)]):
            with patch("src.data.hub.PROJECT_ROOT", test_root):
                from src.data.hub import download_all_data

                with patch("src.data.hub.download_file", return_value=False) as mock_dl:
                    result = download_all_data()
                    assert result == {"downloaded": 0, "skipped": 0, "failed": 1}
                    mock_dl.assert_called_once()


# ==============================================================================
# ── loader.py ─────────────────────────────────────────────────────────────────
# ==============================================================================

class TestLoadRawData:
    """Tests for loader.load_raw_data()."""

    def test_file_not_found(self, tmp_path):
        """Raises FileNotFoundError when the file does not exist."""
        from src.data.loader import load_raw_data

        with pytest.raises(FileNotFoundError, match="not found"):
            load_raw_data(str(tmp_path / "nonexistent.csv"))

    def test_loads_csv(self, tmp_path):
        """Returns a DataFrame when the file exists."""
        from src.data.loader import load_raw_data

        csv_path = tmp_path / "raw.csv"
        MOCK_CSV_DATA.to_csv(csv_path, index=False)

        df = load_raw_data(str(csv_path))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2


class TestLoadCleanedData:
    """Tests for loader.load_cleaned_data()."""

    def test_file_not_found(self):
        """Raises FileNotFoundError."""
        from src.data.loader import load_cleaned_data

        with pytest.raises(FileNotFoundError, match="not found"):
            load_cleaned_data("/nonexistent/path.csv")

    def test_loads_csv(self, tmp_path):
        """Returns a DataFrame when the file exists."""
        from src.data.loader import load_cleaned_data

        csv_path = tmp_path / "cleaned.csv"
        df_data = pd.DataFrame({
            "question": ["q1"], "context": ["ctx1"], "answer": ["ans1"]
        })
        df_data.to_csv(csv_path, index=False)

        df = load_cleaned_data(str(csv_path))
        assert isinstance(df, pd.DataFrame)
        assert "question" in df.columns


class TestLoadLabelledData:
    """Tests for loader.load_labelled_data()."""

    def test_file_not_found(self):
        """Raises FileNotFoundError."""
        from src.data.loader import load_labelled_data

        with pytest.raises(FileNotFoundError, match="not found"):
            load_labelled_data("/nonexistent/path.csv")

    def test_loads_csv(self, tmp_path):
        """Returns a DataFrame when the file exists."""
        from src.data.loader import load_labelled_data

        csv_path = tmp_path / "labelled.csv"
        df_data = pd.DataFrame({
            "question": ["q1"], "context": ["ctx1"],
            "answer": ["ans1"], "category": ["Symptoms"]
        })
        df_data.to_csv(csv_path, index=False)

        df = load_labelled_data(str(csv_path))
        assert isinstance(df, pd.DataFrame)
        assert "category" in df.columns


class TestLoadDatasetFromHub:
    """Tests for loader.load_dataset_from_hub()."""

    def test_import_error(self):
        """When 'datasets' is not installed, raises ImportError."""
        from src.data.loader import load_dataset_from_hub

        real_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "datasets":
                raise ImportError("No module named 'datasets'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="datasets"):
                load_dataset_from_hub()

    def test_successful_load(self):
        """Returns a DataFrame when datasets is available."""
        from src.data.loader import load_dataset_from_hub

        mock_ds = MagicMock()
        mock_ds.to_pandas.return_value = pd.DataFrame({
            "pubid": ["1"], "question": ["test q"],
            "context": ["test ctx"], "long_answer": ["ans"],
            "final_decision": ["yes"],
        })

        mock_load_dataset = MagicMock(return_value=mock_ds)
        real_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "datasets":
                mock_mod = MagicMock()
                mock_mod.load_dataset = mock_load_dataset
                return mock_mod
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            df = load_dataset_from_hub()
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 1
            assert "question" in df.columns


# ==============================================================================
# ── preprocessor.py (additional edge cases) ──────────────────────────────────
# ==============================================================================

class TestPreprocessorExtended:
    """Additional coverage for preprocessor.py."""

    def test_clean_text_non_ascii_removal(self):
        """remove_non_ascii=True strips non-ASCII characters."""
        from src.data.preprocessor import clean_text

        text = "café résumé"
        result = clean_text(text, remove_non_ascii=True)
        assert "é" not in result
        assert isinstance(result, str)

    def test_clean_text_none_input(self):
        """None input returns empty string."""
        from src.data.preprocessor import clean_text

        assert clean_text(None) == ""

    def test_clean_text_non_string_input(self):
        """Non-string input returns empty string."""
        from src.data.preprocessor import clean_text

        assert clean_text(123) == ""

    def test_clean_text_empty_string(self):
        """Empty string returns empty string."""
        from src.data.preprocessor import clean_text

        assert clean_text("") == ""

    def test_extract_question_empty(self):
        """Empty input returns empty string."""
        from src.data.preprocessor import extract_question

        assert extract_question(None) == ""
        assert extract_question("") == ""

    def test_extract_question_plain(self):
        """Plain text (no prefix) returns as-is."""
        from src.data.preprocessor import extract_question

        assert extract_question("hello world") == "hello world"

    def test_extract_question_dict_string(self):
        """Dict-string format extracts question field."""
        from src.data.preprocessor import extract_question

        text = "{'question': 'What causes fever?', 'context': '...'}"
        result = extract_question(text)
        assert result == "What causes fever?"

    def test_extract_context_empty(self):
        """Empty input returns empty string."""
        from src.data.preprocessor import extract_context

        assert extract_context(None) == ""
        assert extract_context("") == ""

    def test_extract_context_plain_text(self):
        """Plain text with 'context:' prefix."""
        from src.data.preprocessor import extract_context

        result = extract_context("context: the patient has fever")
        assert result == "the patient has fever"

    def test_extract_context_regex_fallback(self):
        """Regex fallback for dict-string (ast.literal_eval fails, regex matches)."""
        from src.data.preprocessor import extract_context

        # Malformed dict that fails literal_eval but matches _CONTEXTS_PATTERN
        text = "{'contexts': ['symptom A.', 'symptom B.'], 'invalid syntax"
        result = extract_context(text)
        assert "symptom A" in result
        assert "symptom B" in result

    def test_extract_context_plain_text_fallback(self):
        """Plain text without 'context:' prefix returns as-is (line 141)."""
        from src.data.preprocessor import extract_context

        result = extract_context("just a plain string")
        assert result == "just a plain string"

    def test_extract_context_no_contexts_key(self):
        """Dict without 'contexts' key returns as-is."""
        from src.data.preprocessor import extract_context

        result = extract_context("{'labels': ['yes'], 'meshes': []}")
        assert result == "{'labels': ['yes'], 'meshes': []}"

    def test_extract_question_with_question_prefix(self):
        """'Question:' prefix is stripped."""
        from src.data.preprocessor import extract_question

        result = extract_question("Question: What causes diabetes?")
        assert result == "What causes diabetes?"


# ==============================================================================
# ── labeller.py (additional edge cases) ──────────────────────────────────────
# ==============================================================================

class TestLabellerExtended:
    """Additional coverage for labeller.py."""

    def test_assign_category_none(self):
        """None input returns 'General'."""
        from src.data.labeller import assign_category

        assert assign_category(None) == "General"

    def test_assign_category_empty(self):
        """Empty string returns 'General'."""
        from src.data.labeller import assign_category

        assert assign_category("") == "General"

    def test_assign_category_non_string(self):
        """Non-string returns 'General'."""
        from src.data.labeller import assign_category

        assert assign_category(123) == "General"

    def test_assign_category_all_categories(self):
        """All 6 categories are reachable."""
        from src.data.labeller import assign_category

        tests = {
            "symptom": "Symptoms",
            "diagnosis": "Diagnosis",
            "treatment": "Treatment",
            "medication": "Medication",
            "prevention": "Prevention",
            "what is": "General",
        }
        for query, expected in tests.items():
            assert assign_category(query) == expected, f"Failed for '{query}' → expected {expected}"

    def test_assign_category_batch(self):
        """Batch assignment returns list of categories."""
        from src.data.labeller import assign_category_batch

        questions = [
            "What are the symptoms of diabetes?",
            "What is the diagnosis for pneumonia?",
            "What is 2+2?",
        ]
        results = assign_category_batch(questions)
        assert len(results) == 3
        assert results[0] == "Symptoms"
        assert results[1] == "Diagnosis"
        assert results[2] == "General"

    def test_assign_category_batch_empty(self):
        """Empty list returns empty list."""
        from src.data.labeller import assign_category_batch

        assert assign_category_batch([]) == []


# ==============================================================================
# ── metrics.py extended coverage ──────────────────────────────────────────────
# ==============================================================================

class TestComputeImprovementExtended:
    """More edge cases for compute_improvement."""

    def test_from_zero_to_infinity(self):
        """Baseline 0, improved > 0 returns inf."""
        from src.evaluation.metrics import compute_improvement

        assert compute_improvement(0.0, 1.0) == float("inf")
        assert compute_improvement(0.0, 0.0001) == float("inf")


class TestComputeBleuExtended:
    """More edge cases for compute_bleu."""

    def test_single_word_identical(self):
        """Single identical word gives non-zero BLEU."""
        from src.evaluation.metrics import compute_bleu

        score = compute_bleu(["hello"], ["hello"])
        assert score > 0.0

    def test_mixed_empty_and_valid_pairs(self):
        """Mix of empty and valid strings."""
        from src.evaluation.metrics import compute_bleu

        score = compute_bleu(["valid", ""], ["valid", "ref"])
        assert isinstance(score, float)
        assert score > 0.0


class TestComputeRougeExtended:
    """More edge cases for compute_rouge."""

    def test_identical_strings(self):
        """Identical strings give ROUGE-L close to 1.0."""
        from src.evaluation.metrics import compute_rouge

        score = compute_rouge(["the cat"], ["the cat"])
        assert score == pytest.approx(1.0, abs=0.1)

    def test_completely_different(self):
        """Completely different strings give low ROUGE-L."""
        from src.evaluation.metrics import compute_rouge

        score = compute_rouge(["abc def"], ["xyz"])
        assert 0.0 <= score < 0.5


class TestEvaluatePairExtended:
    """More edge cases for evaluate_pair."""

    @patch("src.evaluation.metrics.compute_bleu", return_value=0.9)
    @patch("src.evaluation.metrics.compute_rouge", return_value=0.85)
    def test_multiple_samples(self, mock_rouge, mock_bleu):
        """n_samples reflects the list length."""
        from src.evaluation.metrics import evaluate_pair

        result = evaluate_pair(["a", "b", "c"], ["x", "y", "z"])
        assert result["n_samples"] == 3

    @patch("src.evaluation.metrics.compute_bleu", return_value=0.0)
    @patch("src.evaluation.metrics.compute_rouge", return_value=0.0)
    def test_zero_scores(self, mock_rouge, mock_bleu):
        """Zero scores propagate correctly."""
        from src.evaluation.metrics import evaluate_pair

        result = evaluate_pair(["abc"], ["xyz"])
        assert result["bleu"] == 0.0
        assert result["rouge_l"] == 0.0


# ==============================================================================
# ── hub.py — upload_file ─────────────────────────────────────────────────────
# ==============================================================================

class TestHubUploadFile:
    """Tests for hub.upload_file()."""

    def test_huggingface_hub_not_installed(self):
        """When huggingface-hub is not importable, returns False."""
        from src.data.hub import upload_file

        real_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "huggingface_hub":
                raise ImportError("mock: not installed")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = upload_file("test/file.csv", "remote/file.csv")
            assert result is False

    def test_local_file_not_found(self, tmp_path):
        """When the local file does not exist, returns False."""
        test_root = tmp_path / "project"
        test_root.mkdir()

        from src.data.hub import upload_file

        with patch("src.data.hub.PROJECT_ROOT", test_root):
            result = upload_file("nonexistent/file.csv", "remote/file.csv")
            assert result is False

    def test_upload_success(self, tmp_path):
        """Successful upload with HF_TOKEN returns True and calls HfApi correctly."""
        from src.data.hub import upload_file

        test_root = tmp_path / "project"
        test_root.mkdir()

        local_file = test_root / "data" / "test.csv"
        local_file.parent.mkdir(parents=True, exist_ok=True)
        local_file.write_text("dummy upload content")

        mock_api_instance = MagicMock()
        mock_api_class = MagicMock(return_value=mock_api_instance)

        real_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "huggingface_hub":
                mock_mod = MagicMock()
                mock_mod.HfApi = mock_api_class
                return mock_mod
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with patch("src.data.hub.PROJECT_ROOT", test_root):
                with patch("src.data.hub.os.getenv", return_value="hf_fake_token"):
                    result = upload_file("data/test.csv", "remote/test.csv")
                    assert result is True
                    mock_api_instance.upload_file.assert_called_once_with(
                        path_or_fileobj=str(local_file),
                        path_in_repo="remote/test.csv",
                        repo_id="AbdoMatrix/healthcare-rag-data",
                        repo_type="dataset",
                        token="hf_fake_token",
                    )

    def test_upload_success_without_token(self, tmp_path):
        """Upload succeeds without HF_TOKEN (falls back to cached CLI login)."""
        from src.data.hub import upload_file

        test_root = tmp_path / "project"
        test_root.mkdir()

        local_file = test_root / "data" / "test.csv"
        local_file.parent.mkdir(parents=True, exist_ok=True)
        local_file.write_text("dummy content")

        mock_api_instance = MagicMock()
        mock_api_class = MagicMock(return_value=mock_api_instance)

        real_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "huggingface_hub":
                mock_mod = MagicMock()
                mock_mod.HfApi = mock_api_class
                return mock_mod
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with patch("src.data.hub.PROJECT_ROOT", test_root):
                with patch("src.data.hub.os.getenv", return_value=None):
                    result = upload_file("data/test.csv", "remote/test.csv")
                    assert result is True
                    # Token should be passed as None when not set in env
                    mock_api_instance.upload_file.assert_called_once_with(
                        path_or_fileobj=str(local_file),
                        path_in_repo="remote/test.csv",
                        repo_id="AbdoMatrix/healthcare-rag-data",
                        repo_type="dataset",
                        token=None,
                    )

    def test_upload_failure(self, tmp_path):
        """When HfApi().upload_file() raises an exception, returns False."""
        from src.data.hub import upload_file

        test_root = tmp_path / "project"
        test_root.mkdir()

        local_file = test_root / "data" / "test.csv"
        local_file.parent.mkdir(parents=True, exist_ok=True)
        local_file.write_text("dummy content")

        mock_api_instance = MagicMock()
        mock_api_instance.upload_file.side_effect = Exception("Upload failed: network error")
        mock_api_class = MagicMock(return_value=mock_api_instance)

        real_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "huggingface_hub":
                mock_mod = MagicMock()
                mock_mod.HfApi = mock_api_class
                return mock_mod
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with patch("src.data.hub.PROJECT_ROOT", test_root):
                with patch("src.data.hub.os.getenv", return_value="hf_fake_token"):
                    result = upload_file("data/test.csv", "remote/test.csv")
                    assert result is False
