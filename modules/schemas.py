from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

# ---------------- FORENSICS SCHEMAS ---------------- #

class GPSData(BaseModel):
    latitude: float
    longitude: float

class MetadataResult(BaseModel):
    gps: Optional[GPSData] = None
    datetime_original: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None
    software: Optional[str] = None
    exposure_bias: Optional[str] = None

class BBox(BaseModel):
    x: int
    y: int
    w: int
    h: int

class SuspiciousRegion(BaseModel):
    bbox: BBox
    area: int
    mean_error: float
    max_error: int

class ELAResult(BaseModel):
    ela_image_path: str
    max_error: int
    mean_error: float
    suspicious_regions: List[SuspiciousRegion]
    quality_used: int

class AIGenResult(BaseModel):
    verdict: str
    confidence: float
    noise_stddev: float
    dark_pixel_ratio: float
    laplacian_variance: float
    uniformity_score: float
    reasons: List[str]

class FileInfo(BaseModel):
    path: str
    filename: str
    size_bytes: int

class FullForensicReport(BaseModel):
    file: FileInfo
    metadata: Optional[MetadataResult] = None
    ela: Optional[ELAResult] = None
    ai_generation_check: Optional[AIGenResult] = None
    errors: List[str] = Field(default_factory=list)
    analysis_time_sec: float

# ---------------- ENVIRONMENT SCHEMAS ---------------- #

class CandidateLocation(BaseModel):
    latitude: float
    longitude: float
    predicted_elevation_deg: float
    predicted_azimuth_deg: float
    elevation_error_deg: float

class ShadowAnalysisResult(BaseModel):
    shadow_ratio: float
    shadow_angle_deg: float
    search_datetime: str
    candidate_locations: List[CandidateLocation]
    is_valid_shadow: Optional[bool] = None
    note: str
    error: Optional[str] = None

class ImageSize(BaseModel):
    width: int
    height: int

class VisualFeaturesResult(BaseModel):
    image_size: ImageSize
    orb_keypoints_count: int
    orb_descriptor_shape: Optional[List[int]] = None
    sift_keypoints_count: int
    sift_descriptor_shape: Optional[List[int]] = None
    sift_available: bool

class LandmarkResult(BaseModel):
    local_features: VisualFeaturesResult
    search_results: Optional[Any] = None
    engine: Optional[str] = None
    note: Optional[str] = None
    error: Optional[str] = None

class SkyAnalysisResult(BaseModel):
    mean_hue: float
    mean_saturation: float
    mean_value: float
    brightness: float
    brightness_std: float
    sky_classification: str
    sky_region_size: ImageSize

class WeatherDataResult(BaseModel):
    clouds_pct: Optional[int] = None
    weather_description: str
    temperature_c: Optional[float] = None
    wind_speed_ms: Optional[float] = None
    humidity_pct: Optional[int] = None
    raw_response: Optional[Any] = None
    error: Optional[str] = None

class SkyWeatherMatchResult(BaseModel):
    match: Optional[bool] = None
    confidence: float
    details: str
    weather_discrepancy_warning: Optional[bool] = None

class FullEnvironmentReport(BaseModel):
    image_path: str
    shadow_analysis: Optional[ShadowAnalysisResult] = None
    visual_features: Optional[VisualFeaturesResult] = None
    landmark_search: Optional[LandmarkResult] = None
    sky_analysis: Optional[SkyAnalysisResult] = None
    weather_data: Optional[WeatherDataResult] = None
    sky_weather_match: Optional[SkyWeatherMatchResult] = None

# ---------------- ENTITIES SCHEMAS ---------------- #

class EntityIdentity(BaseModel):
    name: Optional[str] = None
    description: str
    confidence: float

class EntityAnalyzeResult(BaseModel):
    persons: List[EntityIdentity] = Field(default_factory=list)
    objects: List[str] = Field(default_factory=list)
    texts: List[str] = Field(default_factory=list)

# ---------------- SYNTHESIS SCHEMAS ---------------- #

class SynthesisReport(BaseModel):
    is_authentic: bool
    authenticity_score: float
    inconsistencies: List[str]
    final_location: Optional[CandidateLocation] = None
    final_time: Optional[str] = None
    reliability_notes: List[str]
