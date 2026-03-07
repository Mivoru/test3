# Modul pro analýzu prostředí a odhad lokace/času
# Environmental Agent – GeoTimeAnalyzer

import os
import math
import base64
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple

import cv2
import numpy as np
import requests
from PIL import Image

from modules.schemas import (
    CandidateLocation, ShadowAnalysisResult, ImageSize,
    VisualFeaturesResult, LandmarkResult, SkyAnalysisResult,
    WeatherDataResult, SkyWeatherMatchResult, FullEnvironmentReport
)

try:
    from pysolar.solar import get_altitude, get_azimuth
except ImportError:
    get_altitude = None
    get_azimuth = None

logger = logging.getLogger(__name__)


class GeoTimeAnalyzer:
    """
    Třída pro environmentální OSINT analýzu snímků.

    Poskytuje tři hlavní funkce:
      1. Shadow Analysis  – odhad zeměpisné šířky z délky stínu a pozice slunce
      2. Visual Geolocation – extrakce vizuálních rysů (SIFT/ORB) a vyhledání landmarků
      3. Sky/Weather Correlation – analýza nebe a korelace s historickými meteodaty
    """

    # ------------------------------------------------------------------ #
    #  PRIVATE HELPERS                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_image(image_path: str) -> np.ndarray:
        """Načte obrázek přes OpenCV, vrátí BGR numpy pole."""
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Soubor nenalezen: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Nelze dekódovat obrázek: {image_path}")
        return img

    @staticmethod
    def _encode_image_base64(image_path: str) -> str:
        """Vrátí base64-kódovaný obsah souboru."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # ------------------------------------------------------------------ #
    #  1.  SHADOW ANALYSIS                                                 #
    # ------------------------------------------------------------------ #

    def analyze_shadow(
        self,
        image_path: str,
        object_height_px: float,
        shadow_length_px: float,
        known_datetime: str,
        known_longitude: Optional[float] = None,
        latitude_range: Tuple[float, float] = (-66.0, 66.0),
        latitude_step: float = 0.5,
        tolerance_deg: float = 2.0,
    ) -> ShadowAnalysisResult:
        """
        Odhad zeměpisné šířky na základě stínu referenčního objektu.

        Parametry:
            image_path:       cesta ke snímku (pro metadata / kontext)
            object_height_px: výška referenčního objektu v pixelech
            shadow_length_px: délka stínu v pixelech
            known_datetime:   ISO 8601 řetězec s datem a časem pořízení
            known_longitude:  volitelná známá zeměpisná délka (°); pokud
                              není zadána, iteruje se přes několik hodnot
            latitude_range:   rozsah prohledávaných šířek (°)
            latitude_step:    krok iterace v stupních
            tolerance_deg:    maximální odchylka elevace (°)

        Vrací:
            ShadowAnalysisResult model
        """
        if get_altitude is None or get_azimuth is None:
            return ShadowAnalysisResult(
                shadow_ratio=0.0, shadow_angle_deg=0.0, search_datetime="",
                candidate_locations=[], note="", error="pysolar není nainstalován – shadow analysis nedostupná."
            )

        # --- Výpočet úhlu stínu z poměru výška/stín ---
        if shadow_length_px <= 0:
            return ShadowAnalysisResult(
                shadow_ratio=0.0, shadow_angle_deg=0.0, search_datetime="",
                candidate_locations=[], note="", error="shadow_length_px musí být > 0"
            )

        shadow_ratio = object_height_px / shadow_length_px
        observed_elevation_deg = math.degrees(math.atan(shadow_ratio))

        # --- Parsování datetime ---
        try:
            dt = datetime.fromisoformat(known_datetime)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return ShadowAnalysisResult(
                shadow_ratio=0.0, shadow_angle_deg=0.0, search_datetime="",
                candidate_locations=[], note="", error=f"Neplatný formát datetime: {known_datetime}"
            )

        # --- Kandidáti na zeměpisnou délku ---
        longitudes = (
            [known_longitude]
            if known_longitude is not None
            else list(range(-180, 181, 15))
        )

        candidates: List[CandidateLocation] = []

        lat = latitude_range[0]
        while lat <= latitude_range[1]:
            for lon in longitudes:
                try:
                    predicted_elevation = get_altitude(lat, lon, dt)
                except Exception:
                    lat += latitude_step
                    continue

                if predicted_elevation <= 0:
                    # Slunce pod obzorem – nelze vrhat stín
                    continue

                if abs(predicted_elevation - observed_elevation_deg) <= tolerance_deg:
                    predicted_azimuth = get_azimuth(lat, lon, dt)
                    candidates.append(
                        CandidateLocation(
                            latitude=round(lat, 2),
                            longitude=round(lon, 2),
                            predicted_elevation_deg=round(predicted_elevation, 2),
                            predicted_azimuth_deg=round(predicted_azimuth, 2),
                            elevation_error_deg=round(
                                abs(predicted_elevation - observed_elevation_deg), 2
                            )
                        )
                    )
            lat += latitude_step

        return ShadowAnalysisResult(
            shadow_ratio=round(shadow_ratio, 4),
            shadow_angle_deg=round(observed_elevation_deg, 2),
            search_datetime=dt.isoformat(),
            candidate_locations=candidates,
            note=(
                f"Nalezeno {len(candidates)} kandidátních lokací "
                f"(tolerance ±{tolerance_deg}°)."
            ),
        )

    def validate_shadows(self, lat: float, lon: float, date_str: str, shadow_angle: float, tolerance: float = 3.0) -> bool:
        """
        Křížová kontrola: vypočítá skutečnou elevaci slunce a porovná s vizuální (shadow_angle).
        Vrátí True, pokud se shodují v rámci tolerance.
        """
        if get_altitude is None:
            logger.warning("pysolar není k dispozici, nelze validovat stíny.")
            return True

        try:
            dt = datetime.fromisoformat(date_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            predicted_elev = get_altitude(lat, lon, dt)
            return abs(predicted_elev - shadow_angle) <= tolerance
        except Exception as exc:
            logger.warning("validate_shadows failed %s", exc)
            return False

    # ------------------------------------------------------------------ #
    #  2.  VISUAL GEOLOCATION                                              #
    # ------------------------------------------------------------------ #

    def extract_visual_features(
        self,
        image_path: str,
        max_features: int = 1000,
        max_dim: int = 1000,
    ) -> VisualFeaturesResult:
        """
        Extrahuje unikátní vizuální rysy snímku pomocí ORB a SIFT.
        Optimalizace: omezuje velikost snímku pro zrychlení výpočtu.
        """
        img = self._load_image(image_path)
        
        # Optimize size
        orig_h, orig_w = img.shape[:2]
        if max(orig_h, orig_w) > max_dim:
            scale = max_dim / max(orig_h, orig_w)
            img = cv2.resize(img, (int(orig_w * scale), int(orig_h * scale)))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        image_size = ImageSize(width=w, height=h)
        sift_available = hasattr(cv2, "SIFT_create")

        # --- ORB (vždy k dispozici) ---
        orb = cv2.ORB_create(nfeatures=max_features)
        kp_orb, desc_orb = orb.detectAndCompute(gray, None)
        orb_kps = len(kp_orb)
        orb_shp = list(desc_orb.shape) if desc_orb is not None else None

        # --- SIFT (nemusí být k dispozici ve všech buildech OpenCV) ---
        sift_kps = 0
        sift_shp = None
        if sift_available:
            sift = cv2.SIFT_create(nfeatures=max_features)
            kp_sift, desc_sift = sift.detectAndCompute(gray, None)
            sift_kps = len(kp_sift)
            sift_shp = list(desc_sift.shape) if desc_sift is not None else None

        return VisualFeaturesResult(
            image_size=image_size,
            orb_keypoints_count=orb_kps,
            orb_descriptor_shape=orb_shp,
            sift_keypoints_count=sift_kps,
            sift_descriptor_shape=sift_shp,
            sift_available=sift_available
        )

    def search_landmarks(
        self,
        image_path: str,
        api_key: Optional[str] = None,
        engine: str = "serper",
        serper_endpoint: str = "https://google.serper.dev/images",
    ) -> LandmarkResult:
        """
        Vyhledá landmarky odpovídající vizuálním rysům snímku.
        """
        features = self.extract_visual_features(image_path)

        if api_key is None:
            logger.warning(
                "API klíč nebyl poskytnut – vracím pouze lokální rysy."
            )
            return LandmarkResult(
                local_features=features,
                search_results=None,
                note="API klíč nebyl poskytnut; vyhledávání neprovedeno."
            )

        image_b64 = self._encode_image_base64(image_path)

        if engine == "serper":
            return self._search_serper(image_b64, api_key, serper_endpoint, features)
        elif engine == "google_vision":
            return self._search_google_vision(image_b64, api_key, features)
        else:
            return LandmarkResult(
                local_features=features,
                error=f"Neznámý engine: {engine}"
            )

    # ---------- private search helpers ----------

    @staticmethod
    def _search_serper(
        image_b64: str,
        api_key: str,
        endpoint: str,
        features: VisualFeaturesResult,
    ) -> LandmarkResult:
        """Odešle obrázek na Serper Images API."""
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "q": "landmark identification",
            "images": [f"data:image/jpeg;base64,{image_b64}"],
        }
        try:
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            return LandmarkResult(
                local_features=features,
                search_results=resp.json(),
                engine="serper"
            )
        except requests.RequestException as exc:
            logger.error("Serper API chyba: %s", exc)
            return LandmarkResult(
                local_features=features,
                error=str(exc),
                engine="serper"
            )

    @staticmethod
    def _search_google_vision(
        image_b64: str,
        api_key: str,
        features: VisualFeaturesResult,
    ) -> LandmarkResult:
        """Odešle obrázek na Google Cloud Vision API (LANDMARK_DETECTION)."""
        endpoint = (
            f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
        )
        payload = {
            "requests": [
                {
                    "image": {"content": image_b64},
                    "features": [{"type": "LANDMARK_DETECTION", "maxResults": 10}],
                }
            ]
        }
        try:
            resp = requests.post(endpoint, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            landmarks = (
                data.get("responses", [{}])[0].get("landmarkAnnotations", [])
            )
            return LandmarkResult(
                local_features=features,
                search_results=landmarks,
                engine="google_vision"
            )
        except requests.RequestException as exc:
            logger.error("Google Vision API chyba: %s", exc)
            return LandmarkResult(
                local_features=features,
                error=str(exc),
                engine="google_vision"
            )

    # ------------------------------------------------------------------ #
    #  3.  SKY / WEATHER CORRELATION                                       #
    # ------------------------------------------------------------------ #

    def analyze_sky(self, image_path: str, sky_fraction: float = 0.30) -> SkyAnalysisResult:
        """
        Analyzuje barvu a texturu nebe v horní části snímku.
        """
        img = self._load_image(image_path)
        h, w = img.shape[:2]

        sky_h = max(1, int(h * sky_fraction))
        sky_region = img[0:sky_h, :]

        hsv = cv2.cvtColor(sky_region, cv2.COLOR_BGR2HSV)
        mean_h, mean_s, mean_v = cv2.mean(hsv)[:3]

        # --- Klasifikace ---
        classification = self._classify_sky(mean_h, mean_s, mean_v)

        # --- Textura / kontrast (směrodatná odchylka jasu) ---
        gray_sky = cv2.cvtColor(sky_region, cv2.COLOR_BGR2GRAY)
        brightness_std = float(np.std(gray_sky))

        return SkyAnalysisResult(
            mean_hue=round(mean_h, 2),
            mean_saturation=round(mean_s, 2),
            mean_value=round(mean_v, 2),
            brightness=round(mean_v, 2),
            brightness_std=round(brightness_std, 2),
            sky_classification=classification,
            sky_region_size=ImageSize(width=w, height=sky_h)
        )

    @staticmethod
    def _classify_sky(hue: float, saturation: float, value: float) -> str:
        """
        Jednoduchá heuristická klasifikace nebe podle HSV hodnot.

        Vrací: 'clear' | 'overcast' | 'partly_cloudy' |
               'sunset_sunrise' | 'night'
        """
        if value < 50:
            return "night"

        # Západ / východ – teplé odstíny (H < 25 nebo H > 160)
        if (hue < 25 or hue > 160) and saturation > 80 and value > 80:
            return "sunset_sunrise"

        # Jasná modrá obloha – H mezi 90–130, S > 60
        if 90 <= hue <= 130 and saturation > 60:
            return "clear"

        # Zataženo – nízká saturace, střední až vysoký jas
        if saturation < 40 and value > 100:
            return "overcast"

        return "partly_cloudy"

    def correlate_weather(
        self,
        lat: float,
        lon: float,
        datetime_str: str,
        api_key: str,
        endpoint: str = "https://api.openweathermap.org/data/3.0/onecall/timemachine",
    ) -> WeatherDataResult:
        """
        Stáhne historická meteorologická data z OpenWeather One Call 3.0.
        """
        try:
            dt = datetime.fromisoformat(datetime_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            unix_ts = int(dt.timestamp())
        except ValueError:
            return WeatherDataResult(
                weather_description="", error=f"Neplatný formát datetime: {datetime_str}"
            )

        params = {
            "lat": lat,
            "lon": lon,
            "dt": unix_ts,
            "appid": api_key,
            "units": "metric",
        }

        try:
            resp = requests.get(endpoint, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.error("OpenWeather API chyba: %s", exc)
            return WeatherDataResult(weather_description="", error=str(exc))

        # Extrakce dat z odpovědi
        current = data.get("data", [{}])[0] if "data" in data else data
        weather_list = current.get("weather", [{}])
        weather_desc = weather_list[0].get("description", "N/A") if weather_list else "N/A"

        return WeatherDataResult(
            clouds_pct=current.get("clouds", None),
            weather_description=weather_desc,
            temperature_c=current.get("temp", None),
            wind_speed_ms=current.get("wind_speed", None),
            humidity_pct=current.get("humidity", None),
            raw_response=data
        )

    def compare_sky_weather(
        self,
        sky_analysis: SkyAnalysisResult,
        weather_data: WeatherDataResult,
    ) -> SkyWeatherMatchResult:
        """
        Porovná vizuální klasifikaci nebe s meteorologickými daty.
        """
        if weather_data.error:
            return SkyWeatherMatchResult(
                confidence=0.0,
                details=f"Weather data error: {weather_data.error}"
            )

        sky_class = sky_analysis.sky_classification
        clouds_pct = weather_data.clouds_pct
        weather_desc = weather_data.weather_description.lower()

        if clouds_pct is None:
            return SkyWeatherMatchResult(
                confidence=0.0,
                details="Chybí údaj o oblačnosti z API."
            )

        # --- Pravidla korelace ---
        confidence = 0.5  # baseline
        reasons: List[str] = []
        warning = False

        if sky_class == "clear":
            if clouds_pct < 25:
                confidence += 0.35
                reasons.append(f"Obloha klasifikována jako jasná, oblačnost {clouds_pct}% – shoda.")
            else:
                confidence -= 0.25
                reasons.append(f"Obloha klasifikována jako jasná, ale oblačnost {clouds_pct}%.")
                if clouds_pct > 80:
                    warning = True
                    reasons.append("VAROVÁNÍ: Historická data hlásí zataženo/bouřku, ale na fotce je jasno.")

        elif sky_class == "overcast":
            if clouds_pct > 70:
                confidence += 0.35
                reasons.append(f"Zataženo – oblačnost {clouds_pct}% – shoda.")
            else:
                confidence -= 0.25
                reasons.append(f"Zataženo vizuálně, ale oblačnost jen {clouds_pct}%.")

        elif sky_class == "partly_cloudy":
            if 20 <= clouds_pct <= 80:
                confidence += 0.25
                reasons.append(f"Polojasno – oblačnost {clouds_pct}% – přijatelné.")
            else:
                confidence -= 0.10
                reasons.append(f"Polojasno vizuálně, oblačnost {clouds_pct}%.")

        elif sky_class == "night":
            # V noci těžko korelovat oblačnost
            confidence = 0.3
            reasons.append("Noční snímek – korelace omezená.")

        elif sky_class == "sunset_sunrise":
            if "sunset" in weather_desc or "sunrise" in weather_desc or "clear" in weather_desc:
                confidence += 0.20
                reasons.append(f"Západ/východ slunce – popis počasí: '{weather_desc}'.")
            else:
                reasons.append(f"Západ/východ slunce – popis: '{weather_desc}' (nelze ověřit).")

        confidence = max(0.0, min(1.0, confidence))

        return SkyWeatherMatchResult(
            match=confidence >= 0.6,
            confidence=round(confidence, 2),
            details=" | ".join(reasons) if reasons else "Nedostatek dat pro porovnání.",
            weather_discrepancy_warning=warning if warning else None
        )

    # ------------------------------------------------------------------ #
    #  CONVENIENCE – spuštění všech analýz najednou                        #
    # ------------------------------------------------------------------ #

    def full_analysis(
        self,
        image_path: str,
        object_height_px: Optional[float] = None,
        shadow_length_px: Optional[float] = None,
        known_datetime: Optional[str] = None,
        known_longitude: Optional[float] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        search_api_key: Optional[str] = None,
        weather_api_key: Optional[str] = None,
        search_engine: str = "serper",
    ) -> FullEnvironmentReport:
        """
        Provede kompletní environmentální analýzu snímku.
        """
        shadow_res = None
        vis_res = None
        land_res = None
        sky_res = None
        weather_res = None
        match_res = None

        # --- Shadow ---
        if object_height_px and shadow_length_px and known_datetime:
            shadow_res = self.analyze_shadow(
                image_path=image_path,
                object_height_px=object_height_px,
                shadow_length_px=shadow_length_px,
                known_datetime=known_datetime,
                known_longitude=known_longitude,
            )

        # --- Visual Geolocation ---
        vis_res = self.extract_visual_features(image_path)
        land_res = self.search_landmarks(
            image_path=image_path,
            api_key=search_api_key,
            engine=search_engine,
        )

        # --- Sky / Weather ---
        sky_res = self.analyze_sky(image_path)

        if lat is not None and lon is not None and known_datetime and weather_api_key:
            weather_res = self.correlate_weather(lat, lon, known_datetime, weather_api_key)
            match_res = self.compare_sky_weather(
                sky_res, weather_res
            )

        return FullEnvironmentReport(
            image_path=image_path,
            shadow_analysis=shadow_res,
            visual_features=vis_res,
            landmark_search=land_res,
            sky_analysis=sky_res,
            weather_data=weather_res,
            sky_weather_match=match_res
        )
