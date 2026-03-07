import asyncio
import os
import uuid
import json
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, Request, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="Ultimate OSINT Image Analyzer Dashboard")

# Nastavení složek
os.makedirs("data", exist_ok=True)
os.makedirs("data/reports", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Připojení statických souborů a šablon
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Globální úložiště stavů a WebSocket spojení
# V produkci by to měl být Redis Pub/Sub nebo podobný broker.
analysis_tasks: Dict[str, Dict[str, Any]] = {}
active_connections: Dict[str, WebSocket] = {}

class ProgressManager:
    """Pomocná třída pro správu progresu a odesílání přes WebSocket"""
    def __init__(self, task_id: str):
        self.task_id = task_id
        
    async def update(self, status_msg: str, step: float, data: dict = None):
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

async def run_analysis_pipeline(task_id: str, file_path: str):
    """
    Simulace spuštění agentů a asynchronních úloh pro demonstraci pipeline.
    Opravdová integrace zavolá ForensicAnalyzer, GeoTimeAnalyzer apod.
    """
    pm = ProgressManager(task_id)
    
    try:
        await pm.update("Analyzuji metadata a Exif data...", 10)
        await asyncio.sleep(1.5) # Simulace zpracování
        
        # MOCK DATA from ForensicAnalyzer
        forensic_data = {
            "metadata": {"datetime_original": "2023-10-14 12:45:00", "make": "Sony", "model": "A7III"},
            "ai_generation_check": {"verdict": "authentic", "confidence": 0.98}
        }
        await pm.update("Forensics dokončeno. Geolokuji pomocí stínů...", 30, {"forensic": forensic_data})
        await asyncio.sleep(2)
        
        # MOCK DATA from EnvironmentAnalyzer
        environment_data = {
            "shadow_analysis": {
                "candidate_locations": [{"latitude": 50.0755, "longitude": 14.4378}, {"latitude": 48.8566, "longitude": 2.3522}]
            },
            "sky_weather_match": {"match": True, "confidence": 0.85, "details": "Oblačnost odpovídá 60% v Praze."}
        }
        await pm.update("Geolokace dokončena. Extrahuji entity...", 60, {"environment": environment_data})
        await asyncio.sleep(1.5)
        
        # MOCK DATA from EntityAnalyzer
        entity_data = {
            "persons": [{"description": "Osoba v červené bundě", "confidence": 0.92}],
            "objects": ["Auto", "Dopravní značka"]
        }
        await pm.update("Entity identifikovány. Vytvářím syntetickou zprávu...", 80, {"entities": entity_data})
        await asyncio.sleep(1)
        
        # MOCK DATA from SynthesisEngine
        synthesis_data = {
            "is_authentic": True,
            "authenticity_score": 0.95,
            "final_location": {"latitude": 50.0755, "longitude": 14.4378},
            "reliability_notes": ["EXIF GPS poloha koresponduje se stínovou analýzou."]
        }
        await pm.update("Generuji PDF report...", 90, {"synthesis": synthesis_data})
        await asyncio.sleep(1)
        
        # Zavoláme utils.report_generator na reálná sesbíraná data
        from utils.report_generator import generate_report
        report_url = generate_report(task_id, analysis_tasks[task_id].get("data", {}))

        
        await pm.update("Analýza dokončena", 100, {"report_url": report_url, "complete": True})
        
    except Exception as e:
        await pm.update(f"Chyba při analýze: {str(e)}", -1)

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def handle_analyze(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # Validace
    allowed_extensions = [".jpg", ".jpeg", ".png", ".tiff"]
    ext = os.path.splitext(file.filename)[1].lower()
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
