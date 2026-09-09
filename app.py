import logging
import cv2
import asyncio
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response, status, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from config import settings
from audio.tts import tts_manager
from services import init_services, get_face_service
from perception.perception_manager import get_perception_manager
from intelligence.assistant import get_assistant_brain
from scene.scene_state import SceneState, TrackedObject, DetectedFace, DetectedText
import time
from pydantic import BaseModel

# Configure Logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.PROJECT_NAME} version {settings.VERSION}")
    
    init_services()
    tts_manager.start()
    
    get_perception_manager()
    get_assistant_brain()
        
    yield 
    
    logger.info("Shutting down services...")
    tts_manager.stop()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="FRIDAY-like AI-powered assistant",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    await websocket.accept()
    perception = get_perception_manager()
    brain = get_assistant_brain()
    
    try:
        while True:
            data = await websocket.receive_bytes()
            
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
                
            t0 = time.time()
            obs = await perception.analyze_frame(frame)
            latency = (time.time() - t0) * 1000
            
            if not obs:
                continue
                
            tracked_objs = [
                TrackedObject(o["track_id"], o["class_name"], o["box"], "UNKNOWN", 0.0) 
                for o in obs["objects"]
            ]
            for d in obs["distances"]:
                # mapping by class for demo purposes
                for o in tracked_objs:
                    if o.class_name == d["object"]:
                        o.distance_category = d["distance_category"]
                        o.distance_val = d["distance_val"]
                        
            faces = [DetectedFace(f["name"], f["box"], f["confidence"]) for f in obs["faces"]]
            texts = [DetectedText(t["text"], t["box"]) for t in obs["texts"]]
            
            # Correlate faces to person objects
            for face in faces:
                fx1, fy1, fx2, fy2 = face.box
                fcx, fcy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
                for obj in tracked_objs:
                    if obj.class_name == "person":
                        ox1, oy1, ox2, oy2 = obj.box
                        if ox1 <= fcx <= ox2 and oy1 <= fcy <= oy2:
                            obj.class_name = face.name
                            break
                            
            state = SceneState(
                timestamp=obs["timestamp"],
                objects=tracked_objs,
                faces=faces,
                texts=texts,
                raw_observation={"frame": frame, "scene_json": obs}
            )
            
            brain.process_new_scene(state)
            
            frontend_detections = []
            for o in tracked_objs:
                frontend_detections.append({
                    "box": o.box,
                    "confidence": 1.0,
                    "class_name": f"[{o.track_id}] {o.class_name}"
                })
            for t in obs["texts"]:
                bbox = t["bbox"]
                xs = [pt[0] for pt in bbox]
                ys = [pt[1] for pt in bbox]
                frontend_detections.append({
                    "box": [min(xs), min(ys), max(xs), max(ys)],
                    "confidence": t["confidence"],
                    "class_name": f'text: "{t["text"]}"'
                })
            for f in obs["faces"]:
                frontend_detections.append({
                    "box": f["box"],
                    "confidence": f["confidence"],
                    "class_name": f'face: {f["name"]}'
                })
                
            frontend_distances = obs["distances"]
            speech_msgs = brain.response.get_pending_messages()

            await websocket.send_json({
                "detections": frontend_detections,
                "distance": frontend_distances,
                "depth_map": obs["depth_map_base64"],
                "latency_ms": round(latency, 1),
                "mode": settings.ASSISTANT_MODE,
                "speech": speech_msgs
            })
            
    except WebSocketDisconnect:
        logger.info("Client disconnected from WebSocket")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

class VoiceQuery(BaseModel):
    text: str

@app.post("/api/query")
async def assistant_query(query: VoiceQuery):
    brain = get_assistant_brain()
    
    # Immediately interrupt any ongoing speech before processing
    brain.audio_tools.interrupt()
    
    import threading
    threading.Thread(target=brain.process_user_query, args=(query.text,)).start()
    return {"success": True, "message": "Query received"}

@app.get("/api/describe_scene")
async def describe_scene_api():
    brain = get_assistant_brain()
    state = brain.scene_memory.get_latest_state()
    
    if not state or not state.raw_observation:
        return {"success": True, "description": "I don't have enough visual context right now."}
        
    frame = state.raw_observation.get("frame")
    scene_json = state.raw_observation.get("scene_json")
    
    if frame is None or not scene_json:
        return {"success": True, "description": "I can't see the scene clearly."}
        
    description = await asyncio.to_thread(
        brain.vision_tools.describe_scene_complex, frame, scene_json
    )
    
    return {"success": True, "description": description}

@app.post("/api/register_face_multi")
async def register_face_multi_api(name: str = Form(...), files: list[UploadFile] = File(...)):
    face_service = get_face_service()
    if face_service is None:
        return {"success": False, "message": "Face service not initialized"}
        
    try:
        frames = []
        for file in files:
            data = await file.read()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                frames.append(frame)
        
        if not frames:
            return {"success": False, "message": "Failed to decode any images"}
            
        success, message = await asyncio.to_thread(face_service.register_face_multi, name, frames)
        return {"success": success, "message": message}
    except Exception as e:
        logger.error(f"Multi-face registration API error: {e}")
        return {"success": False, "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)
