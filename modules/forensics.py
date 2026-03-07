# Modul pro digitální forenzní analýzu snímků

import os
import json
import time
import tempfile
import logging
from typing import Any, Optional

import exifread
import numpy as np
from PIL import Image
import cv2

from modules.schemas import (
    MetadataResult, GPSData, SuspiciousRegion, BBox, ELAResult, AIGenResult,
    FileInfo, FullForensicReport
)

logger = logging.getLogger(__name__)


class ForensicAnalyzer:
    """
    Hloubková forenzní analýza obrazových souborů.

    Poskytuje tři analytické metody:
      1. extract_metadata()      – EXIF metadata (GPS, DateTime, Make/Model …)
      2. error_level_analysis()   – ELA detekce potenciálních manipulací
      3. check_ai_generation()    – analýza šumu v tmavých oblastech (AI-gen indikátor)
    """

    # ------------------------------------------------------------------ init
    def __init__(self, image_path: str) -> None:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        self.image_path: str = os.path.abspath(image_path)
        self.filename: str = os.path.basename(image_path)

        # Output directory for ELA artefacts
        self._ela_output_dir = os.path.join(
            os.path.dirname(os.path.dirname(self.image_path)),
            "data", "ela_output",
        )
        os.makedirs(self._ela_output_dir, exist_ok=True)

    # ============================================================
    #  1. METADATA EXTRACTION
    # ============================================================
    def extract_metadata(self) -> MetadataResult:
        """
        Čte EXIF tagy pomocí exifread a vrací Pydantic model
        s metadaty, včetně možných GPS záznamů.
        """
        with open(self.image_path, "rb") as f:
            tags = exifread.process_file(f, details=False)

        # Build dict first
        gps_data = self._extract_gps(tags)
        dt_orig = self._tag_str(tags, "EXIF DateTimeOriginal")
        make = self._tag_str(tags, "Image Make")
        model = self._tag_str(tags, "Image Model")
        software = self._tag_str(tags, "Image Software")
        expo = self._tag_str(tags, "EXIF ExposureBiasValue")

        # Create model directly
        return MetadataResult(
            gps=GPSData(latitude=gps_data["latitude"], longitude=gps_data["longitude"]) if gps_data else None,
            datetime_original=dt_orig,
            make=make,
            model=model,
            software=software,
            exposure_bias=expo
        )

    # ---- GPS helpers ---------------------------------------------------
    @staticmethod
    def _gps_to_decimal(values: list, ref: str) -> float:
        """Převede EXIF GPS racionální hodnoty na desetinné stupně."""
        d = float(values[0])
        m = float(values[1])
        s = float(values[2])
        decimal = d + m / 60.0 + s / 3600.0
        if ref in ("S", "W"):
            decimal = -decimal
        return round(decimal, 7)

    def _extract_gps(self, tags: dict) -> dict[str, float] | None:
        """Vrátí {latitude, longitude} nebo None, pokud GPS data neexistují."""
        lat_tag = tags.get("GPS GPSLatitude")
        lat_ref = tags.get("GPS GPSLatitudeRef")
        lon_tag = tags.get("GPS GPSLongitude")
        lon_ref = tags.get("GPS GPSLongitudeRef")

        if not all([lat_tag, lat_ref, lon_tag, lon_ref]):
            return None

        try:
            latitude = self._gps_to_decimal(
                lat_tag.values, str(lat_ref)
            )
            longitude = self._gps_to_decimal(
                lon_tag.values, str(lon_ref)
            )
            return {"latitude": latitude, "longitude": longitude}
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            logger.warning("GPS parsing failed: %s", exc)
            return None

    @staticmethod
    def _tag_str(tags: dict, key: str) -> str | None:
        """Bezpečně vrátí stringovou hodnotu EXIF tagu."""
        tag = tags.get(key)
        return str(tag) if tag else None

    # ============================================================
    #  2. ERROR LEVEL ANALYSIS  (ELA)
    # ============================================================
    def error_level_analysis(self, quality: int = 95, max_dim: int = 2000) -> ELAResult:
        """
        Provede ELA, s omezením velikosti snýmku (max_dim) pro rychlost:
          1. Uloží dočasný JPEG s danou kvalitou.
          2. Porovná s originálem (absolutní pixel-diff).
          3. Identifikuje oblasti s vysokým rozdílem (potenciální manipulace).
        """
        # -- Načti originál přes PIL a re-uložení jako JPEG ----------------
        original_pil = Image.open(self.image_path).convert("RGB")
        
        # Optimize size
        orig_w, orig_h = original_pil.size
        scale = 1.0
        if max(orig_w, orig_h) > max_dim:
            scale = max_dim / max(orig_w, orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            original_pil = original_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(tmp_fd)
        try:
            original_pil.save(tmp_path, "JPEG", quality=quality)

            # -- Načti oba obrázky přes cv2 --------------------------------
            # If resized, we need original_cv to match the resized PIL image, so just convert the resized PIL to cv2
            original_cv = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)
            resaved_cv = cv2.imread(tmp_path)

            if original_cv is None or resaved_cv is None:
                raise ValueError("Failed to load images via OpenCV")

            # Ensure matching dimensions
            if original_cv.shape != resaved_cv.shape:
                resaved_cv = cv2.resize(
                    resaved_cv,
                    (original_cv.shape[1], original_cv.shape[0]),
                )

            # -- Absolutní rozdíl + zesílení (×20) -------------------------
            diff = cv2.absdiff(original_cv, resaved_cv)
            ela_image = cv2.multiply(diff, np.array([20.0]))
            ela_image = np.clip(ela_image, 0, 255).astype(np.uint8)

            # -- Ulož ELA výstup -------------------------------------------
            ela_filename = f"ela_{self.filename}"
            if not ela_filename.lower().endswith((".jpg", ".jpeg", ".png")):
                ela_filename += ".png"
            ela_path = os.path.join(self._ela_output_dir, ela_filename)
            cv2.imwrite(ela_path, ela_image)

            # -- Statistiky a detekce podezřelých oblastí ------------------
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            max_error = int(np.max(gray_diff))
            mean_error = float(np.mean(gray_diff))

            suspicious_regions_raw = self._find_suspicious_regions(gray_diff)
            susp_mapped = []
            for sr in suspicious_regions_raw:
                susp_mapped.append(SuspiciousRegion(
                    bbox=BBox(**sr["bbox"]),
                    area=sr["area"],
                    mean_error=sr["mean_error"],
                    max_error=sr["max_error"]
                ))

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return ELAResult(
            ela_image_path=ela_path,
            max_error=max_error,
            mean_error=round(mean_error, 4),
            suspicious_regions=susp_mapped,
            quality_used=quality
        )

    @staticmethod
    def _find_suspicious_regions(
        gray_diff: np.ndarray,
        threshold: int = 25,
        min_area: int = 100,
    ) -> list[dict[str, Any]]:
        """Najde spojité oblasti, vrací raw dicty konvertované později v calleru."""
        _, binary = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        regions: list[dict[str, Any]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            roi = gray_diff[y : y + h, x : x + w]
            regions.append({
                "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "area": int(area),
                "mean_error": round(float(np.mean(roi)), 2),
                "max_error": int(np.max(roi)),
            })

        regions.sort(key=lambda r: r["area"], reverse=True)
        return regions

    # ============================================================
    #  3. AI GENERATION CHECK  (offline — noise analysis)
    # ============================================================
    def check_ai_generation(self) -> AIGenResult:
        """
        Detekce potenciálně AI-generovaného obrázku.
        """
        img = cv2.imread(self.image_path)
        if img is None:
            raise ValueError(f"Cannot open image: {self.image_path}")
            
        # Optimize size for noise analysis
        max_dim = 1000
        orig_h, orig_w = img.shape[:2]
        if max(orig_h, orig_w) > max_dim:
            scale = max_dim / max(orig_h, orig_w)
            img = cv2.resize(img, (int(orig_w * scale), int(orig_h * scale)))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        dark_mask = gray < 30
        dark_pixel_count = int(np.sum(dark_mask))
        total_pixels = h * w
        dark_pixel_ratio = dark_pixel_count / total_pixels if total_pixels else 0

        if dark_pixel_count > 100:
            dark_region = gray[dark_mask].astype(np.float64)
            noise_stddev = float(np.std(dark_region))
        else:
            noise_stddev = 0.0

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = float(np.var(laplacian))

        block_size = 64
        block_stds: list[float] = []
        for by in range(0, h - block_size, block_size):
            for bx in range(0, w - block_size, block_size):
                block = gray[by : by + block_size, bx : bx + block_size]
                block_stds.append(float(np.std(block)))

        uniformity_score = 0.0
        if block_stds:
            uniformity_score = float(np.std(block_stds))

        score = 0.0
        reasons: list[str] = []

        if noise_stddev > 5.0:
            score += 0.3
            reasons.append(f"Elevated noise in dark regions (stddev={noise_stddev:.2f})")

        if laplacian_var < 50:
            score += 0.25
            reasons.append(f"Very low Laplacian variance ({laplacian_var:.1f}) — unusually smooth image")
        elif laplacian_var > 5000:
            score += 0.15
            reasons.append(f"High Laplacian variance ({laplacian_var:.1f}) — possible over-sharpening artefact")

        if uniformity_score < 10.0 and total_pixels > 10000:
            score += 0.2
            reasons.append(f"Unnaturally uniform noise distribution (uniformity_score={uniformity_score:.2f})")

        if dark_pixel_ratio < 0.01:
            score += 0.1
            reasons.append("Very few dark pixels — limited noise analysis data")

        confidence = min(score, 1.0)
        if confidence >= 0.6:
            verdict = "likely_ai_generated"
        elif confidence >= 0.3:
            verdict = "possibly_ai_generated"
        else:
            verdict = "likely_authentic"

        return AIGenResult(
            verdict=verdict,
            confidence=round(confidence, 3),
            noise_stddev=round(noise_stddev, 4),
            dark_pixel_ratio=round(dark_pixel_ratio, 4),
            laplacian_variance=round(laplacian_var, 2),
            uniformity_score=round(uniformity_score, 2),
            reasons=reasons
        )

    # ============================================================
    #  FULL ANALYSIS ORCHESTRATOR
    # ============================================================
    def run_full_analysis(self) -> FullForensicReport:
        """
        Spustí všechny tři analytické metody a vrátí sloučený model.
        """
        t0 = time.perf_counter()

        file_info = FileInfo(
            path=self.image_path,
            filename=self.filename,
            size_bytes=os.path.getsize(self.image_path)
        )
        metadata_res = None
        ela_res = None
        aigen_res = None
        errors = []

        try:
            metadata_res = self.extract_metadata()
        except Exception as exc:
            logger.error("Metadata extraction failed: %s", exc)
            errors.append(f"metadata: {exc}")

        try:
            ela_res = self.error_level_analysis()
        except Exception as exc:
            logger.error("ELA failed: %s", exc)
            errors.append(f"ela: {exc}")

        try:
            aigen_res = self.check_ai_generation()
        except Exception as exc:
            logger.error("AI generation check failed: %s", exc)
            errors.append(f"ai_generation_check: {exc}")

        elapsed = time.perf_counter() - t0

        return FullForensicReport(
            file=file_info,
            metadata=metadata_res,
            ela=ela_res,
            ai_generation_check=aigen_res,
            errors=errors,
            analysis_time_sec=round(elapsed, 3)
        )
