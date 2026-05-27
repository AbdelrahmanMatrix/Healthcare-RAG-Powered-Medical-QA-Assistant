"""
Coverage gap tests — targets lines that coverage.py cannot trace
through importlib.reload or sys.modules patching.

Kept in a separate file because they use fragile patching strategies
(importlib.reload, builtins.__import__) that don't fit the cleaner
patterns in the main test files.

Modules covered here:
  - src/data/hub.py              (stdout reconfigure, download_file edge cases)
  - src/classification/classifier.py  (module-level settings path branches)
"""

import importlib
import os
import sys
from pathlib import Path
from unittest.mock import patch


# ==============================================================================
# ── src/data/hub.py ─────────────────────────────────────────────────────────
# ==============================================================================

class TestHubCoverageGaps:
    """Closes 3 missed lines in hub.py: 28-29 (stdout encoding), 133 (download size check)."""

    def test_stdout_reconfigure_exception_handled(self):
        """Exercises the except AttributeError: pass on lines 28-29.

        Patches sys.stdout.reconfigure to raise AttributeError, then
        reloads the module to trigger the exception handler.
        """
        import src.data.hub as hub_mod

        orig_reconfigure = sys.stdout.reconfigure
        try:
            def failing_reconfigure(*args, **kwargs):
                raise AttributeError("mock: no reconfigure")
            sys.stdout.reconfigure = failing_reconfigure

            importlib.reload(hub_mod)

            assert hasattr(hub_mod, "download_file")
        finally:
            sys.stdout.reconfigure = orig_reconfigure
            importlib.reload(hub_mod)

    def test_download_file_success_with_size_check(self, tmp_path):
        """download_file success path: covers local_path.exists() check on line 133.

        Uses patch("huggingface_hub.hf_hub_download") directly instead of
        sys.modules patching, so coverage.py can trace the hub.py lines.
        """
        from src.data.hub import download_file

        dest = tmp_path / "dest" / "size_check.csv"
        cache_file = tmp_path / "cache" / "size_check.csv"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text("x" * 20480)  # 20 KB file

        with patch("huggingface_hub.hf_hub_download", return_value=str(cache_file)):
            with patch("src.data.hub.PROJECT_ROOT", tmp_path):
                with patch("src.data.hub.os.getenv", return_value=None):
                    result = download_file("data/raw/size_check.csv", dest)
                    assert result is True
                    assert dest.exists()
                    assert dest.read_text() == "x" * 20480

    def test_download_file_copy_fails_returns_false(self, tmp_path):
        """download_file returns False when copy2 fails."""
        from src.data.hub import download_file

        dest = tmp_path / "dest" / "copy_fail.csv"

        with patch("huggingface_hub.hf_hub_download", return_value="/nonexistent/cache/file.csv"):
            with patch("src.data.hub.PROJECT_ROOT", tmp_path):
                with patch("src.data.hub.os.getenv", return_value=None):
                    result = download_file("data/raw/copy_fail.csv", dest)
                    assert result is False


# ==============================================================================
# ── src/classification/classifier.py ─────────────────────────────────────────
# ==============================================================================

class TestClassifierCoverageGaps:
    """Closes 4 missed lines in classifier.py: 30-31, 43-44.

    Lines 43-44 (module-level os.path.isabs branch) are exercised via
    importlib.reload, which coverage.py cannot trace.
    """

    def test_stdout_reconfigure_exception_handled(self):
        """Exercises the except (AttributeError, ValueError): pass at lines 30-31.

        Patches sys.stdout.reconfigure to raise AttributeError, then
        reloads the module to trigger the exception handler.
        """
        import src.classification.classifier as classifier_mod

        orig_reconfigure = sys.stdout.reconfigure
        try:
            def failing_reconfigure(*args, **kwargs):
                raise AttributeError("mock: no reconfigure")
            sys.stdout.reconfigure = failing_reconfigure

            importlib.reload(classifier_mod)

            assert hasattr(classifier_mod, "MedicalClassifier")
        finally:
            sys.stdout.reconfigure = orig_reconfigure
            importlib.reload(classifier_mod)

    def test_settings_classifier_path_absolute_branch(self):
        """Exercises the os.path.isabs() branch (line 43-44)."""
        import src.classification.classifier as classifier_mod
        from config.settings import settings

        original_path = settings.CLASSIFIER_PATH
        try:
            abs_path = "C:/tmp/custom_models/biobert" if os.name == "nt" else "/tmp/custom_models/biobert"
            settings.CLASSIFIER_PATH = abs_path

            classifier_mod._classifier_instance = None
            importlib.reload(classifier_mod)

            assert str(classifier_mod.DEFAULT_LOCAL_PATH) == str(Path(abs_path).resolve())
        finally:
            settings.CLASSIFIER_PATH = original_path
            importlib.reload(classifier_mod)

    def test_settings_classifier_path_relative_default(self):
        """Verifies DEFAULT_LOCAL_PATH uses PROJECT_ROOT for relative paths."""
        import src.classification.classifier as classifier_mod
        assert classifier_mod.DEFAULT_LOCAL_PATH is not None
        assert "biobert_classifier" in str(classifier_mod.DEFAULT_LOCAL_PATH)

    def test_classifier_module_imports_successfully(self):
        """Basic smoke test that the module loaded correctly."""
        import src.classification.classifier as classifier_mod
        assert hasattr(classifier_mod, "MedicalClassifier")
        assert hasattr(classifier_mod, "load_classifier")
