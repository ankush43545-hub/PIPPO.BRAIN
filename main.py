"""
Pippo Backend - Stable Production Version
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import httpx
import asyncio
import json
from datetime import datetime
import os
from enum import Enum
import logging
import sys
from motor.motor_asyncio import AsyncIOMotorClient

# =====================================================
# LOGGING (Clean & Crisp)
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Pippo")

# =====================================================
# CONFIGURATION
# =====================================================

class Config:
    # Get Token from Render Environment
    HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
    HF_API_BASE = "https://router.huggingface.co/models/"
    
    # MongoDB (Optional)
    MONGO_URL = os.getenv("MONGO_URL", "")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "pippo_db")
    
    # Server
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    # Flags
    USE_MONGODB = bool(MONGO_URL)

config = Config()

# =====================================================
# ROBUST MODEL SELECTION
# =====================================================

MODELS = {
    # Zephyr is reliable, free, and ungated (no permission errors)
    "brain": "HuggingFaceH4/zephyr-7b-beta", 
    "vision": "Salesforce/blip2-opt-2.7b",
    "tts": "facebook/mms-tts-eng",
    "image_gen": "stabilityai/stable-diffusion-2-1",
    "ocr": "microsoft/trocr-base-printed"
}

# =====================================================
# DATABASE (Safe Mode)
# =====================================================

class Database:
    client: Optional[AsyncIOMotorClient] = None
    db = None
    
    @classmethod
    async def connect_db(cls):
        if not config.USE_MONGODB:
            logger.warning("⚠️ MongoDB not configured. Pippo is running in Memory-Only mode.")
            return
        try:
            cls.client = AsyncIOMotorClient(config.MONGO_URL)
            cls.db = cls.client[config.MONGO_DB_NAME]
            await cls.client.admin.command('ping')
            logger.info("✅ Connected to MongoDB")
        except Exception as e:
            logger.error(f"❌ MongoDB Failed: {e}")
            cls.client = None # Fallback to no-DB

db = Database()

# =====================================================
# APP SETUP
# =====================================================

app = FastAPI(title="Pippo Backend", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# DATA MODELS
# =====================================================

class Message(BaseModel):
    content: str
    role: str = "user"

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Message]] = []
    image_data: Optional[str] = None

# =====================================================
# INTELLIGENCE LAYER
# =====================================================

async def query_huggingface(model_key: str, payload: dict):
    """Generic safe requester for HF API"""
    if not config.HF_API_TOKEN:
        raise Exception("HF_API_TOKEN is missing in Environment Variables!")

    url = f"{config.HF_API_BASE}{MODELS[model_key]}"
    headers = {"Authorization": f"Bearer {config.HF_API_TOKEN}"}
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        
    if response.status_code != 200:
        logger.error(f"HF Error ({model_key}): {response.text}")
        if response.status_code == 503:
            raise Exception("Model is loading (cold start). Try again in 20 seconds.")
        if response.status_code == 401:
            raise Exception("Invalid API Token. Check Render settings.")
        raise Exception(f"API Error: {response.status_code}")
        
    return response.json()

async def get_brain_response(prompt: str):
    # Zephyr/Mistral Prompt Format
    formatted_prompt = f"<|system|>\nYou are Pippo, a helpful AI.<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    
    payload = {
        "inputs": formatted_prompt,
        "parameters": {"max_new_tokens": 256, "temperature": 0.7, "return_full_text": False}
    }
    
    result = await query_huggingface("brain", payload)
    
    if isinstance(result, list) and "generated_text" in result[0]:
        return result[0]["generated_text"].strip()
    return str(result)

# =====================================================
# ENDPOINTS
# =====================================================

@app.on_event("startup")
async def startup():
    logger.info("🚀 Pippo Starting Up...")
    await db.connect_db()

@app.get("/")
async def root():
    return {
        "status": "Online",
        "model": MODELS["brain"],
        "token_set": bool(config.HF_API_TOKEN)
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # 1. Check for Images
        if request.image_data:
            logger.info("Processing Image...")
            vision_res = await query_huggingface("vision", {
                "inputs": {"image": request.image_data, "question": request.message}
            })
            description = vision_res[0].get("generated_text", "image") if isinstance(vision_res, list) else str(vision_res)
            
            # Synthesize
            final_answer = await get_brain_response(f"User showed an image of: {description}. User asked: {request.message}")
            return {"response": final_answer, "intent": "image_analysis"}

        # 2. Normal Chat
        logger.info(f"Processing Text: {request.message[:50]}...")
        response_text = await get_brain_response(request.message)
        
        return {
            "response": response_text,
            "intent": "chat",
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Crash: {str(e)}")
        # RETURN THE REAL ERROR TO THE USER
        raise HTTPException(status_code=500, detail=f"System Error: {str(e)}")

# =====================================================
# RUNNER
# =====================================================

if __name__ == "__main__":
    import uvicorn
    # reload=False prevents the log spam and file system errors
    uvicorn.run(app, host=config.HOST, port=config.PORT, reload=False)
    
