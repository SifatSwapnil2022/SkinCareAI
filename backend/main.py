from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from bson import ObjectId
import io, os, sys

sys.path.insert(0, os.path.dirname(__file__))

from database.mongo import analyses_collection, create_indexes
from auth import router as auth_router, get_current_user
from llm.grok_advisor import get_recommendations
from utils.preprocess import image_to_base64
from utils.pdf_generator import PDF_generator_report

import models.efficientnet as efficientnet_model
import models.mobilenet    as mobilenet_model
import models.resnet50     as resnet50_model
import models.yolov8       as yolov8_model

app = FastAPI(title="Skin Disease Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)



@app.on_event("startup")
async def startup():
    await create_indexes()
    print("⏳ Loading EfficientNetB0...")
    efficientnet_model.load_model()
    print("⏳ Loading MobileNetV2...")
    mobilenet_model.load_model()
    print("⏳ Loading ResNet50...")
    resnet50_model.load_model()
    print("⏳ Loading YOLOv8...")
    yolov8_model.load_model()
    print("✅ All models loaded!")


MODEL_MAP = {
    "EfficientNetB0": efficientnet_model.predict,
    "MobileNetV2":    mobilenet_model.predict,
    "ResNet50":       resnet50_model.predict,
    "YOLOv8":         yolov8_model.predict,
}


#  Analyze Endpoint 
@app.post("/analyze_skin")
async def analyze_skin(
    file:       UploadFile = File(...),
    model_name: str        = Form("EfficientNetB0"),
    current_user: dict     = Depends(get_current_user)
):
    if model_name not in MODEL_MAP:
        raise HTTPException(400, f"Model must be one of {list(MODEL_MAP.keys())}")

    image_bytes = await file.read()

    result = MODEL_MAP[model_name](image_bytes)
    llm = get_recommendations(result["disease"], result["confidence"])

    doc = {
        "user_id":         str(current_user["_id"]),
        "timestamp":       datetime.utcnow(),
        "model_used":      model_name,
        "image_base64":    image_to_base64(image_bytes),
        "disease":         result["disease"],
        "confidence":      result["confidence"],
        "all_predictions": result["all_predictions"],
        **llm
    }
    inserted = await analyses_collection.insert_one(doc)

    return {
        "analysis_id":     str(inserted.inserted_id),
        "disease":         result["disease"],
        "confidence":      result["confidence"],
        "all_predictions": result["all_predictions"],
        "model_used":      model_name,
        **llm
    }


# History Endpoints 
@app.get("/history")
async def get_history(current_user: dict = Depends(get_current_user)):
    cursor = analyses_collection.find(
        {"user_id": str(current_user["_id"])},
        {"image_base64": 0}
    ).sort("timestamp", -1)

    history = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        history.append(doc)
    return history


@app.get("/history/{analysis_id}")
async def get_analysis(analysis_id: str,
                       current_user: dict = Depends(get_current_user)):
    doc = await analyses_collection.find_one({
        "_id":     ObjectId(analysis_id),
        "user_id": str(current_user["_id"])
    })
    if not doc:
        raise HTTPException(404, "Analysis not found")
    doc["_id"] = str(doc["_id"])
    return doc


@app.delete("/history/{analysis_id}")
async def delete_analysis(analysis_id: str,
                          current_user: dict = Depends(get_current_user)):
    result = await analyses_collection.delete_one({
        "_id":     ObjectId(analysis_id),
        "user_id": str(current_user["_id"])
    })
    if result.deleted_count == 0:
        raise HTTPException(404, "Analysis not found")
    return {"message": "Deleted successfully"}


# PDF Report 
@app.get("/report/{analysis_id}")
async def download_report(analysis_id: str,
                          current_user: dict = Depends(get_current_user)):
    doc = await analyses_collection.find_one({
        "_id":     ObjectId(analysis_id),
        "user_id": str(current_user["_id"])
    })
    if not doc:
        raise HTTPException(404, "Analysis not found")

    pdf_bytes = PDF_generator_report(
        user_name=       current_user["name"],
        user_email=      current_user["email"],
        disease=         doc["disease"],
        confidence=      doc["confidence"],
        all_predictions= doc["all_predictions"],
        
        recommendations= doc.get("recommendations", ""),
        next_steps=      doc.get("next_steps", ""),
        tips=            doc.get("tips", ""),
        model_used=      doc["model_used"],
        image_base64=    doc.get("image_base64", ""),
        analysis_date=   doc["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
    )

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition":
                 f"attachment; filename=skin_report_{analysis_id}.pdf"}
    )


@app.get("/")
async def root():
    return {"message": "Skin Disease Detection API is running 🚀"}