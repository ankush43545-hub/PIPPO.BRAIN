"""
Pippo Backend - Final Fix
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import httpx
import os
import logging
import sys
from motor.motor_asyncio import AsyncIOMotorClient

# =====================================================
# LOGGING
# =====================================================
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("Pippo")

# =====================================================
# CONFIGURATION
# =====================================================
class Config:
    HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
    # SWITCHED TO STANDARD ENDPOINT (More reliable)
    HF_API_BASE = "https://router.huggingface.co/models/"
    
    MONGO_URL = os.getenv("MONGO_URL", "")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "pippo_db")
    HOST = "0.0.0.0"
    PORT = int(os.getenv("PORT", 8000))

config = Config()

# =====================================================
# MODELS
# =====================================================
MODELS = {
    "brain": "HuggingFaceH4/zephyr-7b-beta",
    "vision": "Salesforce/blip2-opt-2.7b",
}

# =====================================================
# DATABASE (Auto-Disable on Failure)
# =====================================================
class Database:
    client = None
    
    @classmethod
    async def connect(cls):
        if not config.MONGO_URL:
            logger.warning("⚠️ No MONGO_URL found. Running in Memory-Only mode.")
            return
        try:
            cls.client = AsyncIOMotorClient(config.MONGO_URL)
            await cls.client.admin.command('ping')
            logger.info("✅ Connected to MongoDB")
        except Exception as e:
            logger.error(f"❌ MongoDB Auth Failed: {e}")
            logger.warning("⚠️ Disabling Database to keep Pippo alive.")
            cls.client = None 

db = Database()

# =====================================================
# APP
# =====================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    image_data: Optional[str] = None

@app.on_event("startup")
async def startup():
    await db.connect()

@app.get("/")
async def root():
    return {"status": "Pippo is Online", "model": MODELS["brain"]}

@app.post("/chat")
async def chat(request: ChatRequest):
    if not config.HF_API_TOKEN:
        raise HTTPException(500, "Missing HF_API_TOKEN in Render Environment!")

    headers = {"Authorization": f"Bearer {config.HF_API_TOKEN}"}
    
    # 1. Simple Chat Logic
    payload = {
        "inputs": f"<|system|>\nYou are Pippo.<|end|>\n<|user|>\n{request.message}<|end|>\n<|assistant|>",
        "parameters": {"max_new_tokens": 200, "return_full_text": False}
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{config.HF_API_BASE}{MODELS['brain']}",
            json=payload,
            headers=headers
        )
        
        if response.status_code != 200:
            # Return the EXACT error from Hugging Face so we can see it
            return {"error": f"HF Error {response.status_code}", "details": response.text}

        result = response.json()
        answer = result[0]['generated_text'] if isinstance(result, list) else str(result)
        
        return {"response": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
    
