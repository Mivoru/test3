"""
Test suite for modules.forensics.ForensicAnalyzer

Vytvoří syntetický testovací obrázek a ověří všechny metody.
Spuštění:  python -m pytest tests/test_forensics.py -v
"""

import os
import sys
import json
import tempfile

import numpy as np
from PIL import Image

# Přidat kořen projektu na sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.forensics import ForensicAnalyzer


# ──────────────────────────────────── helpers ──────────────────────────────────

def _create_test_image(path: str, width: int = 200, height: int = 200) -> str:
    """
    Vytvoří syntetický JPEG s:
      - modrým pozadím
      - červeným čtvercem uprostřed (simuluje manipulovanou oblast)
      - tmavým rohem (pro AI-gen noise test)
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Modré pozadí
    img[:, :] = (100, 130, 200)

    # Červený čtverec uprostřed
    cx, cy = width // 2, height // 2
    img[cy - 20 : cy + 20, cx - 20 : cx + 20] = (220, 50, 50)

    # Tmavý roh s náhodným šumem (dark-area noise)
    dark_region = np.random.randint(0, 15, (50, 50, 3), dtype=np.uint8)
    img[:50, :50] = dark_region

    pil_img = Image.fromarray(img)
    pil_img.save(path, "JPEG", quality=98)
    return path


# ──────────────────────────────────── tests ───────────────────────────────────

class TestForensicAnalyzer:
    """Testy pro ForensicAnalyzer."""

    @classmethod
    def setup_class(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="forensic_test_")
        cls.test_image = _create_test_image(
            os.path.join(cls.tmpdir, "test_photo.jpg")
        )
        cls.analyzer = ForensicAnalyzer(cls.test_image)

    # ---- __init__ --------------------------------------------------------

    def test_init_valid_file(self):
        assert os.path.isfile(self.analyzer.image_path)
        assert self.analyzer.filename == "test_photo.jpg"

    def test_init_invalid_file(self):
        try:
            ForensicAnalyzer("/nonexistent/image.jpg")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass

    # ---- extract_metadata ------------------------------------------------

    def test_extract_metadata_returns_dict(self):
        meta = self.analyzer.extract_metadata()
        assert isinstance(meta, dict)

    def test_extract_metadata_keys(self):
        meta = self.analyzer.extract_metadata()
        expected_keys = {
            "gps", "datetime_original", "make", "model",
            "software", "exposure_bias",
        }
        assert expected_keys == set(meta.keys())

    def test_metadata_json_serializable(self):
        meta = self.analyzer.extract_metadata()
        serialized = json.dumps(meta, default=str)
        assert isinstance(serialized, str)

    # ---- error_level_analysis --------------------------------------------

    def test_ela_returns_dict(self):
        ela = self.analyzer.error_level_analysis()
        assert isinstance(ela, dict)

    def test_ela_required_keys(self):
        ela = self.analyzer.error_level_analysis()
        for key in ("ela_image_path", "max_error", "mean_error",
                     "suspicious_regions", "quality_used"):
            assert key in ela, f"Missing key: {key}"

    def test_ela_image_created(self):
        ela = self.analyzer.error_level_analysis()
        assert os.path.isfile(ela["ela_image_path"])

    def test_ela_error_values(self):
        ela = self.analyzer.error_level_analysis()
        assert isinstance(ela["max_error"], int)
        assert isinstance(ela["mean_error"], float)
        assert ela["max_error"] >= 0
        assert ela["mean_error"] >= 0

    # ---- check_ai_generation ---------------------------------------------

    def test_ai_check_returns_dict(self):
        result = self.analyzer.check_ai_generation()
        assert isinstance(result, dict)

    def test_ai_check_required_keys(self):
        result = self.analyzer.check_ai_generation()
        for key in ("verdict", "confidence", "noise_stddev",
                     "dark_pixel_ratio", "laplacian_variance",
                     "uniformity_score", "reasons"):
            assert key in result, f"Missing key: {key}"

    def test_ai_check_verdict_values(self):
        result = self.analyzer.check_ai_generation()
        valid = {"likely_authentic", "possibly_ai_generated", "likely_ai_generated"}
        assert result["verdict"] in valid

    def test_ai_check_confidence_range(self):
        result = self.analyzer.check_ai_generation()
        assert 0.0 <= result["confidence"] <= 1.0

    # ---- run_full_analysis -----------------------------------------------

    def test_full_analysis_returns_dict(self):
        full = self.analyzer.run_full_analysis()
        assert isinstance(full, dict)

    def test_full_analysis_all_sections(self):
        full = self.analyzer.run_full_analysis()
        for section in ("file", "metadata", "ela",
                        "ai_generation_check", "analysis_time_sec"):
            assert section in full, f"Missing section: {section}"

    def test_full_analysis_no_errors(self):
        full = self.analyzer.run_full_analysis()
        assert full["errors"] == [], f"Unexpected errors: {full['errors']}"

    def test_full_analysis_json_serializable(self):
        full = self.analyzer.run_full_analysis()
        serialized = json.dumps(full, default=str)
        assert isinstance(serialized, str)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
