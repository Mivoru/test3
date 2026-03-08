import logging
import math
from typing import List, Optional

from .schemas import (
    FullForensicReport, FullEnvironmentReport, EntityAnalyzeResult,
    SynthesisReport, CandidateLocation
)

logger = logging.getLogger(__name__)

class SynthesisEngine:
    """
    Systém pro syntézu a objektivní hodnocení dat ze všech tří agentů.
    Váží důvěryhodnost, detekuje manipulace a vytváří finální zprávu.
    """

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Spočítá vzdálenost dvou bodů na Zemi v kilometrech."""
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def run_synthesis(
        self,
        forensic: FullForensicReport,
        environment: FullEnvironmentReport,
        entities: EntityAnalyzeResult
    ) -> SynthesisReport:
        """Vyhodnotí všechny dostupné reporty a sestaví finální hodnocení."""
        inconsistencies: List[str] = []
        reliability_notes: List[str] = []
        
        score: float = 1.0  # Výchozí skóre důvěryhodnosti (1.0 = perfektní)

        # ---------------- 0. Data Availability Check ----------------
        available_sources = 0
        total_sources = 6 # metadata, ela, aigen, shadows, weather, entities
        
        if forensic.metadata and forensic.metadata.status != "no_metadata": available_sources += 1
        if forensic.ela and not forensic.ela.max_error == 0: available_sources += 1
        if forensic.ai_generation_check: available_sources += 1
        if environment.shadow_analysis and environment.shadow_analysis.candidate_locations: available_sources += 1
        if environment.weather_data and environment.weather_data.error is None: available_sources += 1
        if entities.status != "skipped": available_sources += 1
            
        data_availability_index = float(round((available_sources / total_sources) * 100, 1))

        # ---------------- 1. AI Generation Check ----------------
        if forensic.ai_generation_check:
            aigen = forensic.ai_generation_check
            if aigen.verdict == "likely_ai_generated":
                inconsistencies.append("Forensics hlásí vysokou pravděpodobnost AI generace snýmku.")
                score = score - 0.5
            elif aigen.verdict == "possibly_ai_generated":
                reliability_notes.append("Forensics hlásí možnou AI manipulaci (nižší sebejistota).")
                score = score - 0.2

        # ---------------- 2. ELA Check ----------------
        if forensic.ela and forensic.ela.suspicious_regions:
            inconsistencies.append(f"Forensics ELA detekovalo {len(forensic.ela.suspicious_regions)} podezřelých oblastí manipulace.")
            score = score - (0.1 * min(len(forensic.ela.suspicious_regions), 5))

        # ---------------- 3. Cloud / Weather Pattern Matching ----------------
        if environment.sky_weather_match:
            match_res = environment.sky_weather_match
            if match_res.weather_discrepancy_warning:
                inconsistencies.append("Environment: Kritický nesoulad počasí! "
                                       "Historická data neodpovídají vizuální analýze oblohy.")
                score = score - 0.4
            elif match_res.match is False:
                inconsistencies.append("Environment: Částečný nesoulad počasí mezi snímkem a historickými daty.")
                score = score - 0.15

        # ---------------- 4. Location Synthesis (GPS vs Candidates) ----------------
        final_loc: Optional[CandidateLocation] = None
        has_exif_gps = False

        if forensic.metadata and forensic.metadata.gps:
            has_exif_gps = True
            exif_gps = forensic.metadata.gps
            final_loc = CandidateLocation(
                latitude=exif_gps.latitude,
                longitude=exif_gps.longitude,
                predicted_elevation_deg=0.0,
                predicted_azimuth_deg=0.0,
                elevation_error_deg=0.0
            )

        # Pokud máme stíny a EXIF GPS, provedeme křížovou kontrolu
        if forensic.metadata and forensic.metadata.gps and environment.shadow_analysis and environment.shadow_analysis.candidate_locations:
            exif_lat = forensic.metadata.gps.latitude
            exif_lon = forensic.metadata.gps.longitude
            shadow_candidates = environment.shadow_analysis.candidate_locations
            
            # Zjistíme, jestli EXIF GPS vůbec odpovídá stínům
            best_dist = float('inf')
            for sc in shadow_candidates:
                dist = self._haversine_distance(exif_lat, exif_lon, sc.latitude, sc.longitude)
                if dist < best_dist:
                    best_dist = dist
                    
            if best_dist > 500.0: # Větší odchylka než 500km
                inconsistencies.append(f"Critical: GPS metadata inconsistent with visual evidence (Shadows). Distance {best_dist:.0f}km.")
                score = score - 0.4
            else:
                reliability_notes.append("EXIF GPS poloha koresponduje se stínovou analýzou.")
                
        elif not (forensic.metadata and forensic.metadata.gps):
            reliability_notes.append("No GPS metadata available for cross-validation")
            if environment.shadow_analysis and environment.shadow_analysis.candidate_locations:
                # Nemáme GPS, ale máme odhad polohy podle stínů => vezmeme prvního kandidáta
                final_loc = environment.shadow_analysis.candidate_locations[0]
                reliability_notes.append("Poloha určena ze stínů, chybí EXIF GPS.")

        # Porovnání EXIF GPS a Google Vision Landmark Detection (> 500m)
        if forensic.metadata and forensic.metadata.gps and entities.objects:
            exif_lat = forensic.metadata.gps.latitude
            exif_lon = forensic.metadata.gps.longitude
            for obj_str in entities.objects:
                if "Stavba/Landmark:" in obj_str and "[Lat:" in obj_str:
                    try:
                        parts = obj_str.split("[Lat:")[1].replace("]", "")
                        lat_str, lng_str = parts.split(",")
                        v_lat = float(lat_str.strip())
                        v_lng = float(lng_str.replace("Lng:", "").strip())
                        
                        v_dist = self._haversine_distance(exif_lat, exif_lon, v_lat, v_lng)
                        if v_dist > 0.5: # 500 metrů = 0.5 km
                            inconsistencies.append(f"Critical: GPS metadata inconsistent with visual evidence (Landmarks). Distance {v_dist*1000:.0f}m.")
                            score = score - 0.4
                    except Exception as e:
                        logger.warning(f"Chyba pri parsovani souradnic z objektu: {e}")

        # ---------------- 5. Time Synthesis ----------------
        final_time = None
        if forensic.metadata and forensic.metadata.datetime_original:
            final_time = forensic.metadata.datetime_original
        elif environment.shadow_analysis and environment.shadow_analysis.search_datetime:
            final_time = environment.shadow_analysis.search_datetime

        # ---------------- Final Clamp ----------------
        score = float(max(0.0, min(1.0, float(score))))

        return SynthesisReport(
            is_authentic=(score >= 0.7),
            authenticity_score=round(score, 2),
            data_availability_index=data_availability_index,
            inconsistencies=inconsistencies,
            final_location=final_loc,
            final_time=final_time,
            reliability_notes=reliability_notes
        )
