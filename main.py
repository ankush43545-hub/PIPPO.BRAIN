"""
PIPPO Backend - Strategic AI Orchestrator with Cloud Memory
"""
import os
import json
import asyncio
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime

# Database & Server Imports
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import uvicorn

# --- CONFIGURATION ---
app = FastAPI(title="PIPPO Backend")

# Allow connections from anywhere (Required for mobile apps/frontend testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Secrets (Set these in Render Environment Variables)
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
MONGO_URL = os.getenv("MONGO_URL", "")
HF_API_BASE = "https://api-inference.huggingface.co/models/"

# PIPPO's Toolkit
MODELS = {
    "brain": "mistralai/Mistral-7B-Instruct-v0.2",
    "image_gen": "stabilityai/stable-diffusion-2-1",
}

# --- MEMORY SYSTEM (MongoDB) ---
class MemoryBank:
    def __init__(self):
        self.client = None
        self.db = None
        
    def connect(self):
        if not MONGO_URL:
            print("⚠️ WARNING: MONGO_URL not found. PIPPO has no long-term memory.")
            return
        try:
            self.client = AsyncIOMotorClient(MONGO_URL)
            self.db = self.client.get_database("pippo_db")
            print("✅ PIPPO connected to Cloud Memory")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")

    async def save_interaction(self, user_msg: str, bot_resp: str, conversation_id: str):
        if self.db is None: return
        
        doc = {
            "conversation_id": conversation_id,
            "timestamp": datetime.utcnow(),
            "role": "interaction",
            "user": user_msg,
            "pippo": bot_resp
        }
        await self.db.conversations.insert_one(doc)

    async def get_context(self, conversation_id: str, limit: int = 5):
        """Fetches recent chat history to maintain context"""
        if self.db is None: return ""
        
        cursor = self.db.conversations.find(
            {"conversation_id": conversation_id}
        ).sort("timestamp", -1).limit(limit)
        
        history = await cursor.to_list(length=limit)
        # Reverse to chronological order (oldest -> newest)
        history.reverse()
        
        context_str = ""
        for h in history:
            context_str += f"User: {h['user']}\nPIPPO: {h['pippo']}\n"
        return context_str

memory = MemoryBank()

@app.on_event("startup")
async def startup():
    memory.connect()

@app.on_event("shutdown")
async def shutdown():
    if memory.client:
        memory.client.close()

# --- DATA MODELS ---
class ChatRequest(BaseModel):
    message: str
    conversation_id: str = "default" 

# --- STRATEGIC ENGINE ---
class PippoBrain:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=60.0)

    async def think_and_reply(self, message: str, context: str) -> str:
        """Decides if the user wants an image or text, then executes."""
        
        # 1. Quick Intent Check
        msg_lower = message.lower()
        if any(w in msg_lower for w in ["draw", "generate image", "picture of", "create an image"]):
            return await self._generate_image(message)
        
        # 2. Text Response (with Context)
        system_prompt = (
            "You are PIPPO, a helpful and strategic AI assistant. "
            "Keep answers concise. Use the context below to continue the conversation.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"USER: {message}\n"
            "PIPPO:"
        )
        return await self._query_hf(MODELS['brain'], {"inputs": system_prompt})

    async def _generate_image(self, prompt: str) -> str:
        clean_prompt = prompt.replace("draw", "").replace("generate image", "").strip()
        try:
            payload = {"inputs": clean_prompt}
            headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
            response = await self.client.post(
                f"{HF_API_BASE}{MODELS['image_gen']}", 
                json=payload, 
                headers=headers
            )
            if response.status_code == 200:
                import base64
                img_str = base64.b64encode(response.content).decode()
                return f"IMAGE_GENERATED:{img_str}"
            return "I tried to draw that, but my canvas is blank (API Error)."
        except Exception as e:
            return f"Image generation failed: {e}"

    async def _query_hf(self, model_url: str, payload: dict) -> str:
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        # Add generation parameters for better text
        if "inputs" in payload and isinstance(payload["inputs"], str):
             payload["parameters"] = {"max_new_tokens": 512, "temperature": 0.7, "return_full_text": False}

        try:
            response = await self.client.post(f"{HF_API_BASE}{model_url}", json=payload, headers=headers)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list):
                    return result[0].get("generated_text", "")
                return str(result)
        except:
            pass
        return "My thoughts are cloudy right now. Please try again."

brain = PippoBrain()

# --- API ENDPOINTS ---

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # 1. Get Memory
    context = await memory.get_context(request.conversation_id)
    
    # 2. Generate Response
    response = await brain.think_and_reply(request.message, context)
    
    # 3. Save to Memory
    await memory.save_interaction(request.message, response, request.conversation_id)
    
    return {
        "response": response,
        "conversation_id": request.conversation_id
    }

@app.get("/health")
async def health_check():
    return {
        "status": "PIPPO is alive", 
        "memory": "Online" if memory.db is not None else "Offline"
    }

if __name__ == "__main__":
    # Render provides the PORT variable
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
