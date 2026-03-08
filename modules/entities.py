# Modul pro detekci a analýzu entit (osoby, objekty, text)
# Integrity Check - Anti-Sycophancy Agent

import logging
import os
import base64
import requests
from typing import List, Dict, Any

import cv2
import numpy as np

from .schemas import EntityAnalyzeResult, EntityIdentity

logger = logging.getLogger(__name__)


class EntityAnalyzer:
    """
    Analýza osob, objektů a textových entit na obrázku.
    Obsahuje Anti-Sycophancy logiku pro zamezení nepodložených identifikací osob.
    """

    def __init__(self, confidence_threshold: float = 0.90):
        # Hranice spolehlivosti pro přijetí identity osoby
        self.confidence_threshold = confidence_threshold

    def _ensure_integrity(self, persons: List[Dict[str, Any]]) -> List[EntityIdentity]:
        """
        Anti-Sycophancy check:
        Pokud identifikace osoby nedosahuje prahové spolehlivosti (např. 90%),
        identita (jméno) je vymazána a osoba je označena pouze anonymním popisem.
        """
        verified_persons = []
        for p in persons:
            confidence = float(p.get("confidence", 0.0))
            name = p.get("name")
            description = p.get("description", "Neznámá osoba")

            if name and confidence < self.confidence_threshold:
                logger.warning(
                    f"Anti-Sycophancy zásah: Identifikace '{name}' smazána "
                    f"kvůli nízké spolehlivosti ({confidence:.2f} < {self.confidence_threshold:.2f})."
                )
                # Smažeme jméno
                name = None

            verified_persons.append(
                EntityIdentity(
                    name=name,
                    description=description,
                    confidence=confidence
                )
            )

        return verified_persons

    def analyze_entities(self, raw_persons: List[Dict[str, Any]], raw_objects: List[str], raw_texts: List[str]) -> EntityAnalyzeResult:
        """
        Zpracuje nalezené entity z externího rozpoznávače a uplatní integritní kontroly.
        
        Parametry:
            raw_persons: seznam dictů s klíči 'name', 'description', 'confidence'.
            raw_objects: seznam stringů (nalezené objekty).
            raw_texts:   seznam stringů (nalezený text a OCR).
        """
        verified_persons = self._ensure_integrity(raw_persons)

        return EntityAnalyzeResult(
            persons=verified_persons,
            objects=raw_objects,
            texts=raw_texts
        )

    def process_image(self, image_path: str) -> EntityAnalyzeResult:
        """
        Integrace Google Cloud Vision API (LANDMARK_DETECTION, OBJECT_LOCALIZATION)
        """
        api_key = os.getenv("GOOGLE_VISION_API_KEY")
        raw_persons: List[Dict[str, Any]] = []
        raw_objects: List[str] = []
        raw_texts: List[str] = []

        if not api_key:
            logger.warning("GOOGLE_VISION_API_KEY is not set.")
            return self.analyze_entities(raw_persons, raw_objects, raw_texts)

        try:
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Nelze nacist obrazek pro Vision API: {e}")
            return self.analyze_entities(raw_persons, raw_objects, raw_texts)

        endpoint = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
        payload = {
            "requests": [
                {
                    "image": {"content": image_b64},
                    "features": [
                        {"type": "LANDMARK_DETECTION", "maxResults": 10},
                        {"type": "OBJECT_LOCALIZATION", "maxResults": 10}
                    ]
                }
            ]
        }

        try:
            resp = requests.post(endpoint, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            responses = data.get("responses", [{}])[0]

            # 1. Landmark Detection
            landmarks = responses.get("landmarkAnnotations", [])
            for lm in landmarks:
                desc = lm.get("description", "Landmark")
                conf = lm.get("score", 0.0)
                locations = lm.get("locations", [])
                
                # Ulozeni informace o poloze do description nebo name
                loc_str = ""
                if locations:
                    lat_lng = locations[0].get("latLng", {})
                    lat = lat_lng.get("latitude")
                    lng = lat_lng.get("longitude")
                    if lat and lng:
                        loc_str = f" [Lat: {lat:.5f}, Lng: {lng:.5f}]"

                raw_objects.append(f"Stavba/Landmark: {desc}{loc_str}")

            # 2. Object Localization (detekce zvirat, techniky atd.)
            objects = responses.get("localizedObjectAnnotations", [])
            for obj in objects:
                name = obj.get("name", "Neznamy objekt")
                conf = obj.get("score", 0.0)
                
                if name.lower() in ("person", "man", "woman", "boy", "girl"):
                    raw_persons.append({
                        "name": None,
                        "description": f"Osoba ({name})",
                        "confidence": conf
                    })
                else:
                    if name not in raw_objects:
                        raw_objects.append(name)

        except requests.RequestException as exc:
            if exc.response is not None and exc.response.status_code in (401, 403):
                logger.error(f"Google Vision API Access Error: {exc}")
                res = self.analyze_entities(raw_persons, raw_objects, raw_texts)
                res.status = "skipped"
                res.reason = "API_KEY_ERROR"
                return res
            logger.error(f"Google Vision API Error: {exc}")
        except Exception as exc:
            logger.error(f"Google Vision API Error: {exc}")

        return self.analyze_entities(raw_persons, raw_objects, raw_texts)
