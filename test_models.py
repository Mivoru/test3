from modules.schemas import (
    FullForensicReport, FileInfo, MetadataResult, GPSData,
    FullEnvironmentReport, CandidateLocation, ShadowAnalysisResult,
    EntityAnalyzeResult, EntityIdentity, AIGenResult
)
from modules.synthesis_engine import SynthesisEngine

def test_synthesis():
    forensic = FullForensicReport(
        file=FileInfo(path="/tmp/img.jpg", filename="img.jpg", size_bytes=1000),
        metadata=MetadataResult(
            gps=GPSData(latitude=50.0, longitude=14.0),
            datetime_original="2023-01-01T12:00:00Z"
        ),
        ai_generation_check=AIGenResult(
            verdict="likely_authentic",
            confidence=0.9,
            noise_stddev=1.0,
            dark_pixel_ratio=0.1,
            laplacian_variance=500.0,
            uniformity_score=15.0,
            reasons=[]
        ),
        analysis_time_sec=1.0
    )

    environment = FullEnvironmentReport(
        image_path="/tmp/img.jpg",
        shadow_analysis=ShadowAnalysisResult(
            shadow_ratio=1.0,
            shadow_angle_deg=45.0,
            search_datetime="2023-01-01T12:00:00Z",
            candidate_locations=[
                CandidateLocation(
                    latitude=50.1,
                    longitude=14.1,
                    predicted_elevation_deg=45.0,
                    predicted_azimuth_deg=180.0,
                    elevation_error_deg=0.0
                )
            ],
            note="OK"
        )
    )

    entities = EntityAnalyzeResult(
        persons=[EntityIdentity(name="John Doe", description="A man", confidence=0.95)]
    )

    engine = SynthesisEngine()
    report = engine.run_synthesis(forensic, environment, entities)
    print("Mismatches:", report.inconsistencies)
    print("Score:", report.authenticity_score)
    print("Authentic:", report.is_authentic)

if __name__ == "__main__":
    test_synthesis()
    print("Smoke tests passed successfully.")
