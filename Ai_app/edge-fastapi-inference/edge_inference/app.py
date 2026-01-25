from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
import io
from ultralytics import YOLO
from utils.postprocess import postprocess_yolo_results
import os

app = FastAPI()

# Load YOLOv11 model (prefer TensorRT engine if available)
TRT_ENGINE_PATH = os.path.join(os.path.dirname(__file__), "model", "yolo11n.engine")
PT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "yolo11n.pt")

MODEL_PATH = TRT_ENGINE_PATH if os.path.exists(TRT_ENGINE_PATH) else PT_MODEL_PATH
model = YOLO(MODEL_PATH, task="detect")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Read the HTML file
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r") as f:
        return f.read()

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    # 1. Read the image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    # 2. Run YOLOv11 inference
    # results is a list of Results objects
    results = model(image)
    
    # 3. Postprocess
    detections = postprocess_yolo_results(results)

    return {
        "status": "success", 
        "detections": detections
    }

