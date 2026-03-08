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
            # Try offline fallback detection
            return self._offline_entity_detection(image_path)

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
                        {"type": "OBJECT_LOCALIZATION", "maxResults": 15},  # Increased for more objects
                        {"type": "LABEL_DETECTION", "maxResults": 20}     # Add label detection for better descriptions
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

            # 2. Object Localization (detekce zvirat, techniky, rostlin atd.)
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
                    # Enhanced object descriptions
                    obj_desc = self._enhance_object_description(name, conf)
                    if obj_desc not in raw_objects:
                        raw_objects.append(obj_desc)

            # 3. Label Detection (pro doplneni informaci o rostlinach, prostredí atd.)
            labels = responses.get("labelAnnotations", [])
            vegetation_labels = []
            for label in labels:
                desc = label.get("description", "").lower()
                conf = label.get("score", 0.0)
                
                # Collect vegetation-related labels
                if any(keyword in desc for keyword in ["plant", "tree", "flower", "grass", "bush", "vegetation", "forest", "wood", "leaf", "garden"]):
                    if conf > 0.5:  # Only high confidence vegetation labels
                        vegetation_labels.append(f"Rostlina/Příroda: {label.get('description', 'Neznámá rostlina')}")

            # Add unique vegetation labels
            for veg_label in vegetation_labels[:5]:  # Limit to top 5
                if veg_label not in raw_objects:
                    raw_objects.append(veg_label)

        except requests.RequestException as exc:
            if exc.response is not None and exc.response.status_code in (401, 403):
                logger.error(f"Google Vision API Access Error: {exc}")
                res = self.analyze_entities(raw_persons, raw_objects, raw_texts)
                res.status = "skipped"
                res.reason = "API_KEY_ERROR"
                return res
            logger.error(f"Google Vision API Error: {exc}")
            res = self.analyze_entities(raw_persons, raw_objects, raw_texts)
            res.status = "skipped"
            res.reason = "API_ERROR"
            return res
        except Exception as exc:
            logger.error(f"Google Vision API Error: {exc}")
            res = self.analyze_entities(raw_persons, raw_objects, raw_texts)
            res.status = "skipped"
            res.reason = "UNKNOWN_ERROR"
            return res

    def _offline_entity_detection(self, image_path: str) -> EntityAnalyzeResult:
        """
        Offline entity detection using basic OpenCV techniques when API is unavailable.
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                result = self.analyze_entities([], [], [])
                result.status = "skipped"
                result.reason = "IMAGE_LOAD_ERROR"
                return result
            
            height, width = img.shape[:2]
            raw_objects = []
            
            # Basic color analysis for vegetation detection
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Green color range for vegetation (more inclusive)
            lower_green = np.array([25, 40, 40])
            upper_green = np.array([90, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            green_ratio = np.sum(green_mask > 0) / (height * width)
            
            # Brown/earth color range
            lower_brown = np.array([5, 30, 30])
            upper_brown = np.array([25, 255, 200])
            brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
            brown_ratio = np.sum(brown_mask > 0) / (height * width)
            
            # Blue color range for sky/water
            lower_blue = np.array([85, 30, 30])
            upper_blue = np.array([135, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            blue_ratio = np.sum(blue_mask > 0) / (height * width)
            
            # Analyze dominant colors and patterns
            if green_ratio > 0.05:  # Lower threshold for vegetation
                if green_ratio > 0.15:
                    raw_objects.append("Rostlina/Příroda: Bohatá vegetace/zelené prostředí (vysoká jistota)")
                elif green_ratio > 0.10:
                    raw_objects.append("Rostlina/Příroda: Vegetace přítomna (střední jistota)")
                else:
                    raw_objects.append("Rostlina/Příroda: Možná vegetace (nízká jistota)")
            
            if blue_ratio > 0.10:  # Lower threshold for sky/water
                if blue_ratio > 0.25:
                    raw_objects.append("Prostředí: Dominantní modré prvky - obloha/voda (vysoká jistota)")
                else:
                    raw_objects.append("Prostředí: Modré prvky přítomny - obloha/voda (střední jistota)")
            
            if brown_ratio > 0.05:
                raw_objects.append("Prostředí: Hnědé prvky - země/kámen přítomny")
            
            # Edge analysis for structural complexity
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)  # More sensitive edge detection
            edge_ratio = np.sum(edges > 0) / (height * width)
            
            if edge_ratio > 0.02:  # Lower threshold
                if edge_ratio > 0.10:
                    raw_objects.append("Prostředí: Vysoce strukturované prostředí (budovy/město/les)")
                elif edge_ratio > 0.05:
                    raw_objects.append("Prostředí: Středně strukturované prostředí")
                else:
                    raw_objects.append("Prostředí: Mírně strukturované prostředí")
            
            # Brightness analysis
            brightness = np.mean(gray)
            if brightness < 50:
                raw_objects.append("Prostředí: Tmavé prostředí (noc/interiér)")
            elif brightness > 200:
                raw_objects.append("Prostředí: Velmi světlé prostředí (sníh/jasné světlo)")
            
            result = self.analyze_entities([], raw_objects, [])
            result.status = "offline_fallback"
            result.reason = "API_UNAVAILABLE"
            return result
            
        except Exception as e:
            logger.error(f"Offline entity detection failed: {e}")
            result = self.analyze_entities([], [], [])
            result.status = "skipped"
            result.reason = "OFFLINE_ERROR"
            return result

    def _enhance_object_description(self, name: str, confidence: float) -> str:
        """
        Vytvoří detailnější popis objektu na základě jeho typu a spolehlivosti.
        """
        name_lower = name.lower()
        
        # Categorize objects
        if any(keyword in name_lower for keyword in ["car", "vehicle", "truck", "bus", "motorcycle", "bicycle"]):
            category = "Doprava"
        elif any(keyword in name_lower for keyword in ["building", "house", "tower", "bridge", "church", "castle"]):
            category = "Stavba"
        elif any(keyword in name_lower for keyword in ["tree", "plant", "flower", "bush", "grass"]):
            category = "Rostlina/Příroda"
        elif any(keyword in name_lower for keyword in ["animal", "dog", "cat", "bird", "horse", "cow"]):
            category = "Živočich"
        elif any(keyword in name_lower for keyword in ["chair", "table", "furniture", "sofa", "bed"]):
            category = "Nábytek"
        elif any(keyword in name_lower for keyword in ["phone", "computer", "laptop", "device", "electronics"]):
            category = "Elektronika"
        else:
            category = "Objekt"
        
        # Add confidence indicator
        conf_text = ""
        if confidence > 0.9:
            conf_text = " (vysoká jistota)"
        elif confidence > 0.7:
            conf_text = " (střední jistota)"
        elif confidence < 0.5:
            conf_text = " (nízká jistota)"
        
        return f"{category}: {name}{conf_text}"
