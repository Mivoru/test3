"""
Ověřovací testy pro GeoTimeAnalyzer (modules/environment.py).
Spouštění:  python -m pytest tests/test_environment.py -v
"""

import sys
import os
import math
import tempfile

import numpy as np
import cv2

# Přidání kořenového adresáře projektu do sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.environment import GeoTimeAnalyzer


def _create_test_image(w: int = 640, h: int = 480) -> str:
    """Vytvoří dočasný testovací obrázek (modrá obloha nahoře, zelená zem dole)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # horní polovina – modrá obloha
    img[0 : h // 2, :] = (220, 160, 60)  # BGR – světle modrá
    # dolní polovina – zelená
    img[h // 2 :, :] = (50, 140, 50)
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, img)
    return tmp.name


def test_shadow_analysis_basic():
    """Ověří, že shadow_angle_deg odpovídá tan(h/s)."""
    analyzer = GeoTimeAnalyzer()
    img_path = _create_test_image()
    try:
        result = analyzer.analyze_shadow(
            image_path=img_path,
            object_height_px=100,
            shadow_length_px=100,
            known_datetime="2025-06-21T12:00:00+00:00",
            known_longitude=15.0,
        )
        expected_angle = math.degrees(math.atan(100 / 100))  # 45°
        assert abs(result["shadow_angle_deg"] - expected_angle) < 0.01
        assert "candidate_locations" in result
        assert isinstance(result["candidate_locations"], list)
        print(f"  shadow_angle_deg = {result['shadow_angle_deg']}  (expected {expected_angle:.2f})")
        print(f"  candidates found = {len(result['candidate_locations'])}")
    finally:
        os.unlink(img_path)


def test_extract_visual_features():
    """Ověří, že ORB detekce vrátí keypoints."""
    analyzer = GeoTimeAnalyzer()
    img_path = _create_test_image()
    try:
        result = analyzer.extract_visual_features(img_path)
        assert "orb_keypoints_count" in result
        assert result["orb_keypoints_count"] >= 0
        assert "image_size" in result
        print(f"  ORB keypoints = {result['orb_keypoints_count']}")
        print(f"  SIFT available = {result['sift_available']}")
    finally:
        os.unlink(img_path)


def test_analyze_sky_classification():
    """Ověří klasifikaci jasného modrého nebe."""
    analyzer = GeoTimeAnalyzer()
    img_path = _create_test_image()
    try:
        result = analyzer.analyze_sky(img_path)
        assert "sky_classification" in result
        assert result["sky_classification"] in (
            "clear", "overcast", "partly_cloudy", "sunset_sunrise", "night"
        )
        print(f"  sky_classification = {result['sky_classification']}")
        print(f"  mean HSV = ({result['mean_hue']}, {result['mean_saturation']}, {result['mean_value']})")
    finally:
        os.unlink(img_path)


def test_compare_sky_weather():
    """Ověří logiku porovnání sky vs weather data."""
    analyzer = GeoTimeAnalyzer()
    sky = {"sky_classification": "clear", "mean_hue": 110, "mean_saturation": 80, "mean_value": 180}
    weather = {"clouds_pct": 10, "weather_description": "clear sky"}
    result = analyzer.compare_sky_weather(sky, weather)
    assert result["match"] is True
    assert result["confidence"] > 0.5
    print(f"  match = {result['match']}, confidence = {result['confidence']}")
    print(f"  details = {result['details']}")


def test_search_landmarks_no_key():
    """Bez API klíče vrátí lokální rysy a note."""
    analyzer = GeoTimeAnalyzer()
    img_path = _create_test_image()
    try:
        result = analyzer.search_landmarks(img_path, api_key=None)
        assert result["search_results"] is None
        assert "note" in result
        print(f"  note = {result['note']}")
    finally:
        os.unlink(img_path)


if __name__ == "__main__":
    tests = [
        ("Shadow Analysis", test_shadow_analysis_basic),
        ("Visual Features", test_extract_visual_features),
        ("Sky Classification", test_analyze_sky_classification),
        ("Sky-Weather Comparison", test_compare_sky_weather),
        ("Landmarks (no key)", test_search_landmarks_no_key),
    ]
    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n{'='*50}")
        print(f"TEST: {name}")
        print('='*50)
        try:
            fn()
            print("  [OK] PASSED")
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] FAILED: {exc}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print('='*50)
