# Modul pro detekci a analýzu entit (osoby, objekty, text)
# Integrity Check - Anti-Sycophancy Agent

import logging
from typing import List, Dict, Any

from modules.schemas import EntityAnalyzeResult, EntityIdentity

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
