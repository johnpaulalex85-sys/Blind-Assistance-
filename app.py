import logging
import cv2
import asyncio
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response, status, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from config import settings
from audio import tts_manager
from services import init_services, get_yolo_detector, get_ocr_service, get_face_service, get_depth_service, get_describe_service
from utils.scene_builder import build_scene_json
import json

# Configure Logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global state to hold the latest frame and scene data for description generation
latest_state = {
    "frame": None,
    "scene_json": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"Starting {settings.PROJECT_NAME} version {settings.VERSION}")
    
    # Initialize AI models/services
    init_services()
    
    # Start the audio TTS thread
    tts_manager.start()
        
    yield  # Hand over control to FastAPI to serve requests
    
    # Shutdown
    logger.info("Shutting down services...")
    tts_manager.stop()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AI-powered assistant for visually impaired people",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """WebSocket endpoint to receive frames from client and return YOLO, OCR detections, and depth map."""
    await websocket.accept()
    yolo_detector = get_yolo_detector()
    ocr_service = get_ocr_service()
    face_service = get_face_service()
    depth_service = get_depth_service()
    
    try:
        while True:
            # Receive image frame as bytes
            data = await websocket.receive_bytes()
            
            if yolo_detector is None:
                await websocket.send_json({"detections": []})
                continue
                
            # Decode the image bytes into a cv2 frame
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
                
            # Run YOLO, OCR, and Face Recognition concurrently in threadpools
            async def run_ocr():
                if ocr_service is None:
                    return {"texts": []}
                return await asyncio.to_thread(ocr_service.detect_text, frame, 0.5)
                
            async def run_face():
                if face_service is None:
                    return {"faces": []}
                return await asyncio.to_thread(face_service.recognize_face, frame, 0.6)

            yolo_task = asyncio.to_thread(yolo_detector.detect, frame, 0.4)
            ocr_task = run_ocr()
            face_task = run_face()
            
            detections, ocr_data, face_data = await asyncio.gather(yolo_task, ocr_task, face_task)
            
            # Save original YOLO detections for the Scene Builder before appending other data
            yolo_raw = list(detections)
            
            # Format OCR results to match YOLO detections format for the frontend
            for item in ocr_data.get("texts", []):
                # OCR bbox is [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                bbox = item["bbox"]
                xs = [pt[0] for pt in bbox]
                ys = [pt[1] for pt in bbox]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
                
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": item["confidence"],
                    "class_name": f'"{item["text"]}"'
                })
                
            # Format Face results to match YOLO detections format for the frontend
            for item in face_data.get("faces", []):
                detections.append({
                    "box": item["box"],
                    "confidence": item["confidence"],
                    "class_name": f'Face: {item["name"]}'
                })
                
            # Run depth estimation using the final detections list to calculate distances
            distance_data = []
            depth_map = None
            if depth_service is not None:
                distance_data, depth_map = await asyncio.to_thread(depth_service.estimate_depth, frame, detections)
                
            # Build structured scene JSON using the modular scene builder
            scene_json = build_scene_json(
                yolo_raw,
                ocr_data, 
                face_data, 
                distance_data
            )
            
            # Store in global state for the Qwen endpoint
            latest_state["frame"] = frame.copy()
            latest_state["scene_json"] = scene_json
            
            # Send results back to the client browser
            await websocket.send_json({
                "detections": detections,
                "distance": distance_data,
                "depth_map": depth_map
            })

            
    except WebSocketDisconnect:
        logger.info("Client disconnected from WebSocket")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

@app.get("/audio/speak")
def speak_text(text: str = "Hello, I am your vision assistant."):
    """Test endpoint for Text-to-Speech."""
    tts_manager.speak(text)
    return {"status": "speaking", "text": text}

@app.get("/api/describe_scene")
async def describe_scene_api():
    """Endpoint to generate a description of the current scene."""
    describe_service = get_describe_service()
    if describe_service is None:
        return {"description": "Describe service not initialized.", "error": True}
        
    frame = latest_state.get("frame")
    scene_json = latest_state.get("scene_json")
    
    if frame is None or scene_json is None:
        return {"description": "No active camera feed or scene data.", "error": True}
        
    try:
        # Run Rule-based describe in a threadpool (or direct, it's instant)
        description = describe_service.generate_description(frame, scene_json)
        
        # Speak the description automatically
        tts_manager.speak(description)
        
        return {"description": description, "success": True}
    except Exception as e:
        logger.error(f"Describe scene API error: {e}")
        return {"description": str(e), "error": True}

@app.post("/api/ocr")
async def detect_text_api(file: UploadFile = File(...)):
    """API endpoint to receive an image and return text detections."""
    ocr_service = get_ocr_service()
    if ocr_service is None:
        return {"texts": [], "error": "OCR service not initialized"}
        
    try:
        data = await file.read()
        nparr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"texts": [], "error": "Failed to decode image"}
            
        # Run OCR in a threadpool to avoid blocking
        result = await asyncio.to_thread(ocr_service.detect_text, frame, 0.5)
        return result
    except Exception as e:
        logger.error(f"OCR API error: {e}")
        return {"texts": [], "error": str(e)}

@app.post("/api/register_face")
async def register_face_api(name: str = Form(...), file: UploadFile = File(...)):
    """API endpoint to register a new face."""
    face_service = get_face_service()
    if face_service is None:
        return {"success": False, "message": "Face service not initialized"}
        
    try:
        data = await file.read()
        nparr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"success": False, "message": "Failed to decode image"}
            
        success, message = await asyncio.to_thread(face_service.register_face, name, frame)
        return {"success": success, "message": message}
    except Exception as e:
        logger.error(f"Face registration API error: {e}")
        return {"success": False, "message": str(e)}

@app.post("/api/register_face_multi")
async def register_face_multi_api(name: str = Form(...), files: list[UploadFile] = File(...)):
    """API endpoint to register a new face with multiple angles."""
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
