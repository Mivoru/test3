import os
import asyncio
from datetime import datetime, timezone

# Load env variables for local testing
if os.path.exists(".env"):
    with open(".env", "r", encoding="utf-8") as env_file:
        for env_line in env_file:
            env_line = env_line.strip()
            if "=" in env_line and not env_line.startswith("#"):
                env_key, env_val = env_line.split("=", 1)
                os.environ[env_key.strip()] = env_val.strip()

# Clear API keys for testing offline functionality
for key in ['OPENWEATHER_API_KEY', 'GOOGLE_VISION_API_KEY', 'SERPER_API_KEY', 'SERP_API_KEY']:
    os.environ.pop(key, None)

from PIL import Image

def run_tests():
    os.makedirs("data", exist_ok=True)
    test_image_path = "data/test_image.jpg"
    img = Image.new('RGB', (800, 600), color=(135, 206, 235)) # Sky blue color
    img.save(test_image_path)
    
    print("Test image created.")
    
    # 1. ForensicAnalyzer
    from modules.forensics import ForensicAnalyzer
    print("--- Testing ForensicAnalyzer ---")
    forensic = ForensicAnalyzer(test_image_path)
    f_report = forensic.run_full_analysis()
    print(f"Forensic Metadata: {f_report.metadata}")
    
    # 2. GeoTimeAnalyzer
    from modules.environment import GeoTimeAnalyzer
    print("--- Testing GeoTimeAnalyzer ---")
    env = GeoTimeAnalyzer()
    e_report = env.full_analysis(test_image_path, lat=50.08, lon=14.42, known_datetime=datetime.now(timezone.utc).isoformat())
    print(f"Env Weather: {e_report.weather_data}")
    if e_report.landmark_search:
        print(f"Env Landmarks: Engine={e_report.landmark_search.engine}, Error={e_report.landmark_search.error}")
    
    # 3. EntityAnalyzer
    from modules.entities import EntityAnalyzer
    print("--- Testing EntityAnalyzer ---")
    ent = EntityAnalyzer()
    ent_report = ent.process_image(test_image_path)
    print(f"Entities: Objects={ent_report.objects}, Persons={ent_report.persons}")
    
    # 4. SynthesisEngine
    from modules.synthesis_engine import SynthesisEngine
    print("--- Testing SynthesisEngine ---")
    synth = SynthesisEngine()
    s_report = synth.run_synthesis(f_report, e_report, ent_report)
    print(f"Synthesis Authentic: {s_report.is_authentic}, Score: {s_report.authenticity_score}")
    
    print("\n--- All modules executed successfully ---")

if __name__ == "__main__":
    run_tests()
