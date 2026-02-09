"""
PIPPO Backend - Final Production (Router API + Zephyr)
"""
import os
import json
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
import httpx
import uvicorn

app = FastAPI(title="PIPPO Orchestrator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
MONGO_URL = os.getenv("MONGO_URL", "")

# *** CRITICAL FIX: NEW HUGGING FACE ROUTER URL ***
HF_API_BASE = "https://router.huggingface.co/hf-inference/models/"

# *** CRITICAL FIX: SWAPPING TO ZEPHYR (Free & Reliable) ***
MODELS = {
    "brain": "HuggingFaceH4/zephyr-7b-beta",
    "image_gen": "stabilityai/stable-diffusion-2-1",
}

# --- MEMORY (Safe Mode) ---
class SafeMemory:
    def __init__(self):
        self.client = None
        self.db = None
        
    def connect(self):
        if not MONGO_URL: return
        try:
            self.client = AsyncIOMotorClient(MONGO_URL, serverSelectionTimeoutMS=5000)
            self.db = self.client.get_database("pippo_db")
            print("✅ PIPPO Connected to Cloud Memory")
        except:
            print("❌ Database Init Error")

    async def save(self, user, bot, cid):
        if self.db is None: return
        try:
            await self.db.conversations.insert_one({
                "conversation_id": cid,
                "timestamp": datetime.utcnow(),
                "user": user,
                "pippo": bot
            })
        except:
            pass
            
    async def get_history_objects(self, conversation_id: str, limit: int = 6):
        if self.db is None: return []
        try:
            cursor = self.db.conversations.find({"conversation_id": conversation_id}).sort("timestamp", -1).limit(limit)
            history_data = await cursor.to_list(length=limit)
            history_data.reverse()
            messages = []
            for h in history_data:
                messages.append(Message(role="user", content=h.get("user", "")))
                messages.append(Message(role="assistant", content=h.get("pippo", "")))
            return messages
        except:
            return []

memory = SafeMemory()

@app.on_event("startup")
async def startup():
    memory.connect()

# --- DATA MODELS ---
class Message(BaseModel):
    content: str
    role: str = "user"

class ChatRequest(BaseModel):
    message: str
    conversation_id: str = "default"
    history: Optional[List[Message]] = [] 

# --- BRAIN ---
class StrategicOrchestrator:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=60.0)

    async def _query_brain(self, prompt: str) -> str:
        # Use the NEW Router URL
        url = f"{HF_API_BASE}{MODELS['brain']}"
        
        payload = {
            "inputs": prompt, 
            "parameters": {"max_new_tokens": 512, "temperature": 0.7, "return_full_text": False}
        }
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        
        try:
            # Print URL to logs so we can debug if it fails
            print(f"Attempting to connect to: {url}")
            
            response = await self.client.post(url, json=payload, headers=headers)
            
            # If 200 OK, return text
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list): return result[0].get("generated_text", "")
                return str(result)
            
            # If 503, Model is Loading
            if response.status_code == 503:
                return "I'm waking up... please ask me again in 20 seconds."
            
            # If 404, Model Not Found on Router
            if response.status_code == 404:
                return f"Error: Model not found at {url}. Check HF_API_BASE."
                
            return f"Brain Error {response.status_code}: {response.text}"
            
        except Exception as e:
            return f"Connection Error: {e}"

    async def process(self, request):
        db_history = await memory.get_history_objects(request.conversation_id)
        full_history = db_history + (request.history or [])
        context = "\n".join([f"{msg.role}: {msg.content}" for msg in full_history[-4:]])
        
        # Zephyr Prompt Format
        prompt = f"<|system|>\nYou are PIPPO, a helpful AI.<|user|>\nContext: {context}\n\n{request.message}<|assistant|>"
        
        response = await self._query_brain(prompt)
        await memory.save(request.message, response, request.conversation_id)
        return response

orchestrator = StrategicOrchestrator()

@app.post("/chat")
async def chat(request: ChatRequest):
    response = await orchestrator.process(request)
    return {"response": response}

@app.get("/health")
async def health():
    return {"status": "PIPPO is alive (Zephyr + Router)"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
    
