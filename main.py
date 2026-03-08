import asyncio
import os
import uuid
import json
import functools
from typing import Dict, Any, Optional

# Nacteni .env (bez externich knihoven)
if os.path.exists(".env"):
    with open(".env", "r", encoding="utf-8") as env_file:
        for env_line in env_file:
            env_line = env_line.strip()
            if "=" in env_line and not env_line.startswith("#"):
                env_key, env_val = env_line.split("=", 1)
                os.environ[env_key.strip()] = env_val.strip()

from fastapi import FastAPI, File, UploadFile, Request, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn

from modules.forensics import ForensicAnalyzer
from modules.environment import GeoTimeAnalyzer
from modules.entities import EntityAnalyzer
from modules.synthesis_engine import SynthesisEngine

app = FastAPI(title="Ultimate OSINT Image Analyzer Dashboard")

# Nastavení složek
os.makedirs("data", exist_ok=True)
os.makedirs("data/reports", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Připojení statických souborů a šablon
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/data", StaticFiles(directory="data"), name="data")
templates = Jinja2Templates(directory="templates")

# Globální úložiště stavů a WebSocket spojení
# V produkci by to měl být Redis Pub/Sub nebo podobný broker.
analysis_tasks: Dict[str, Dict[str, Any]] = {}
active_connections: Dict[str, WebSocket] = {}

class ProgressManager:
    """Pomocná třída pro správu progresu a odesílání přes WebSocket"""
    def __init__(self, task_id: str):
        self.task_id = task_id
        
    async def update(self, status_msg: str, step: float, data: Optional[dict] = None):
        """Aktualizuje stav a pošle ho klientovi v reálném čase."""
        if self.task_id not in analysis_tasks:
            analysis_tasks[self.task_id] = {"status": status_msg, "progress": step, "data": data or {}}
        else:
            analysis_tasks[self.task_id].update({"status": status_msg, "progress": step})
            if data:
                analysis_tasks[self.task_id]["data"].update(data)
                
        # Odeslání přes WebSocket, pokud je klient připojen
        if self.task_id in active_connections:
            ws = active_connections[self.task_id]
            try:
                msg = {
                    "task_id": self.task_id,
                    "status": status_msg,
                    "progress": step,
                    "data": analysis_tasks[self.task_id].get("data", {})
                }
                await ws.send_json(msg)
            except Exception as e:
                print(f"Failed to send to WS: {e}")

async def cleanup_files(task_id: str, file_path: str):
    """
    Automatické smazání souborů po 1 hodině.
    """
    await asyncio.sleep(3600)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            
        file_ext = os.path.splitext(file_path)[1]
        filename = os.path.basename(file_path)
        ela_path = os.path.join("static", "ela_output", f"ela_{filename}")
        if not ela_path.lower().endswith((".jpg", ".jpeg", ".png")):
            ela_path += ".png"
            
        if os.path.exists(ela_path):
            os.remove(ela_path)
            
        pdf_path = f"data/reports/{task_id}.pdf"
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
    except Exception as e:
        print(f"Cleanup error for {task_id}: {e}")

async def run_analysis_pipeline(task_id: str, file_path: str):
    pm = ProgressManager(task_id)
    try:
        from modules.schemas import FullForensicReport, FullEnvironmentReport, EntityAnalyzeResult, FileInfo, SynthesisReport
        
        await pm.update("Analyzuji metadata a Exif data...", 10)
        
        # 1. Forensics
        forensic_report = None
        forensic_data = {}
        try:
            forensic_analyzer = ForensicAnalyzer(file_path)
            forensic_report = await asyncio.to_thread(forensic_analyzer.run_full_analysis)
            
            forensic_data = {
                "metadata": forensic_report.metadata.model_dump() if forensic_report.metadata else {},
                "ai_generation_check": forensic_report.ai_generation_check.model_dump() if forensic_report.ai_generation_check else {},
                "ela": forensic_report.ela.model_dump() if forensic_report.ela else {}
            }
        except Exception as e:
            print(f"Forensics chyba: {e}")
            forensic_report = FullForensicReport(file=FileInfo(path=file_path, filename=os.path.basename(file_path), size_bytes=0), errors=[str(e)])
        
        await pm.update("Forensics dokončeno. Geolokuji...", 30, {"forensic": forensic_data})
        
        # 2. Environment
        environment_report = None
        environment_data = {}
        try:
            lat = None
            lon = None
            known_datetime = None
            if forensic_report and forensic_report.metadata:
                if forensic_report.metadata.gps:
                    lat = forensic_report.metadata.gps.latitude
                    lon = forensic_report.metadata.gps.longitude
                known_datetime = forensic_report.metadata.datetime_original

            geo_analyzer = GeoTimeAnalyzer()
            
            def run_env_analysis():
                return geo_analyzer.full_analysis(
                    image_path=file_path,
                    lat=lat,
                    lon=lon,
                    known_datetime=known_datetime,
                    search_api_key=os.environ.get("SERPER_API_KEY"),
                    weather_api_key=os.environ.get("OPENWEATHER_API_KEY"),
                    search_engine="serper"
                )
                
            environment_report = await asyncio.to_thread(run_env_analysis)
            
            environment_data = {
                "shadow_analysis": environment_report.shadow_analysis.model_dump() if environment_report.shadow_analysis else {},
                "sky_weather_match": environment_report.sky_weather_match.model_dump() if environment_report.sky_weather_match else {}
            }
        except Exception as e:
            print(f"Environment chyba: {e}")
            environment_report = FullEnvironmentReport(image_path=file_path)

        await pm.update("Geolokace dokončena. Extrahuji entity...", 60, {"environment": environment_data})
        
        # 3. Entities
        entities_report = None
        entity_data = {}
        try:
            entity_analyzer = EntityAnalyzer()
            entities_report = await asyncio.to_thread(entity_analyzer.process_image, file_path)
            
            entity_data = {
                "persons": [p.model_dump() for p in entities_report.persons] if entities_report.persons else [],
                "objects": entities_report.objects,
                "texts": entities_report.texts
            }
        except Exception as e:
            print(f"Entities chyba: {e}")
            entities_report = EntityAnalyzeResult(persons=[], objects=[], texts=[])

        await pm.update("Entity identifikovány. Vytvářím syntetickou zprávu...", 80, {"entities": entity_data})
        
        # 4. Synthesis
        synthesis_report = None
        synthesis_data = {}
        try:
            synthesis_engine = SynthesisEngine()
            synthesis_report = await asyncio.to_thread(synthesis_engine.run_synthesis, forensic_report, environment_report, entities_report)
            synthesis_data = synthesis_report.model_dump()
        except Exception as e:
            print(f"Synthesis chyba: {e}")
            synthesis_data = {}

        await pm.update("Generuji PDF report...", 90, {"synthesis": synthesis_data})
        
        all_data = {
            "forensic": forensic_data,
            "environment": environment_data,
            "entities": entity_data,
            "synthesis": synthesis_data
        }
        
        try:
            from utils.report_generator import generate_report
            report_url = await asyncio.to_thread(generate_report, task_id, all_data)
        except Exception as report_err:
            print(f"Chyba při generování reportu: {report_err}")
            report_url = None
        
        await pm.update("Analýza dokončena", 100, {
            "report_url": report_url, 
            "complete": True,
            "forensic": forensic_data,
            "environment": environment_data,
            "entities": entity_data,
            "synthesis": synthesis_data
        })
        
    except Exception as e:
        await pm.update(f"Chyba při analýze: {str(e)}", -1)

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def handle_analyze(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # Validace
    allowed_extensions = [".jpg", ".jpeg", ".png", ".tiff"]
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_extensions:
        return {"error": "Nepodporovaný formát. Povoleno pouze .jpg, .png, .tiff"}
        
    # Unikátní ID a uložení
    task_id = str(uuid.uuid4())
    file_path = os.path.join("data", f"{task_id}{ext}")
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
        
    # Inicializace state
    analysis_tasks[task_id] = {"status": "Soubor přijat", "progress": 0, "data": {}}
    
    # Asynchronní spuštění tasku
    background_tasks.add_task(run_analysis_pipeline, task_id, file_path)
    background_tasks.add_task(cleanup_files, task_id, file_path)
    
    return {"task_id": task_id, "status": "Task naplánován"}

@app.websocket("/ws/progress/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    await websocket.accept()
    active_connections[task_id] = websocket
    try:
        # Odešleme hned první stav
        if task_id in analysis_tasks:
            await websocket.send_json({
                "task_id": task_id,
                "status": analysis_tasks[task_id]["status"],
                "progress": analysis_tasks[task_id]["progress"],
                "data": analysis_tasks[task_id].get("data", {})
            })
        while True:
            # Udržení spojení na živu, případně čekání na zprávy od klienta
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        if task_id in active_connections:
            del active_connections[task_id]

@app.get("/download-report/{task_id}")
async def download_report(task_id: str):
    # Pro účely příkladu, později zde bude vazba na vygenerované PDF
    pdf_path = f"data/reports/{task_id}.pdf"
    if os.path.exists(pdf_path):
        return FileResponse(path=pdf_path, filename=f"SessionReport_{task_id}.pdf", media_type='application/pdf')
    return {"error": "Report zatím neexistuje nebo nebyl vygenerován."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
