"""
PIPPO Backend - Final Fixed Version (New Router API)
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

# CORS
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

# Model endpoints
MODELS = {
    "brain": "mistralai/Mistral-7B-Instruct-v0.2",
    "vision": "Salesforce/blip2-opt-2.7b",
    "image_gen": "stabilityai/stable-diffusion-2-1",
}

# --- SAFE MEMORY SYSTEM ---
class SafeMemory:
    def __init__(self):
        self.client = None
        self.db = None
        
    def connect(self):
        if not MONGO_URL:
            print("⚠️ Memory Disabled: No MONGO_URL found")
            return
        try:
            self.client = AsyncIOMotorClient(MONGO_URL, serverSelectionTimeoutMS=5000)
            self.db = self.client.get_database("pippo_db")
            print("✅ PIPPO Connected to Cloud Memory")
        except Exception as e:
            print(f"❌ Database Init Error: {e}")

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
        # Use the NEW URL structure
        url = f"{HF_API_BASE}{MODELS['brain']}"
        payload = {
            "inputs": prompt, 
            "parameters": {"max_new_tokens": 512, "temperature": 0.7, "return_full_text": False}
        }
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        
        try:
            response = await self.client.post(url, json=payload, headers=headers)
            
            # If 200 OK, return text
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list): return result[0].get("generated_text", "")
                return str(result)
            
            # If Model is Loading (503), tell user to wait
            if response.status_code == 503:
                return "I'm waking up... please ask me again in 30 seconds."
                
            return f"Brain Error {response.status_code}: {response.text}"
            
        except Exception as e:
            return f"Connection Error: {e}"

    async def process(self, request):
        # 1. Get Context
        db_history = await memory.get_history_objects(request.conversation_id)
        full_history = db_history + (request.history or [])
        context = "\n".join([f"{msg.role}: {msg.content}" for msg in full_history[-4:]])
        
        # 2. Ask Brain
        prompt = f"<s>[INST] You are PIPPO. Context: {context} User: {request.message} [/INST]"
        response = await self._query_brain(prompt)
        
        # 3. Save
        await memory.save(request.message, response, request.conversation_id)
        return response

orchestrator = StrategicOrchestrator()

@app.post("/chat")
async def chat(request: ChatRequest):
    response = await orchestrator.process(request)
    return {"response": response}

@app.get("/health")
async def health():
    return {"status": "PIPPO is alive (Router API Updated)"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
        
