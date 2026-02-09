"""
PIPPO Backend - INDESTRUCTIBLE CHAT
Auto-switches between Router/API and Models until it works.
"""
import os
import json
from datetime import datetime
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
import httpx
import uvicorn

app = FastAPI(title="PIPPO Chat")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CONFIG
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
MONGO_URL = os.getenv("MONGO_URL", "")

# --- ROBUST MODEL CONFIG ---
# We try these URLS in order until one works.
# 1. New Router (Zephyr)
# 2. Old API (Zephyr)
# 3. New Router (Phi-3.5 - Backup)
ENDPOINTS = [
    {
        "url": "https://router.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
        "name": "Router (Zephyr)"
    },
    {
        "url": "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
        "name": "Legacy API (Zephyr)"
    },
    {
        "url": "https://router.huggingface.co/models/microsoft/Phi-3.5-mini-instruct",
        "name": "Router (Phi-3.5)"
    }
]

# --- MEMORY ---
class SafeMemory:
    def __init__(self):
        self.client = None
        self.db = None
        
    def connect(self):
        if not MONGO_URL: return
        try:
            self.client = AsyncIOMotorClient(MONGO_URL, serverSelectionTimeoutMS=3000)
            self.db = self.client.get_database("pippo_db")
            print("✅ Memory Connected")
        except:
            print("⚠️ Memory Disabled")

    async def save(self, user, bot, cid):
        if self.db is None: return
        try:
            await self.db.conversations.insert_one({
                "conversation_id": cid,
                "timestamp": datetime.utcnow(),
                "user": user,
                "bot": bot
            })
        except:
            pass

    async def get_context(self, cid):
        if self.db is None: return ""
        try:
            cursor = self.db.conversations.find({"conversation_id": cid}).sort("timestamp", -1).limit(3)
            history = await cursor.to_list(length=3)
            history.reverse()
            return "\n".join([f"User: {h['user']}\nAI: {h['bot']}" for h in history])
        except:
            return ""

memory = SafeMemory()

@app.on_event("startup")
async def startup():
    memory.connect()

# --- CHAT LOGIC ---
class ChatRequest(BaseModel):
    message: str
    conversation_id: str = "default"

@app.post("/chat")
async def chat(request: ChatRequest):
    # 1. Get Context
    context = await memory.get_context(request.conversation_id)
    
    # 2. Prompt Engineering
    # We use a generic format that works for both Zephyr and Phi
    prompt = f"<|user|>\nContext:\n{context}\n\n{request.message} <|end|>\n<|assistant|>"

    # 3. Try Endpoints Loop
    last_error = ""
    
    async with httpx.AsyncClient() as client:
        for endpoint in ENDPOINTS:
            try:
                print(f"🔄 Trying {endpoint['name']}...")
                headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
                payload = {
                    "inputs": prompt, 
                    "parameters": {"max_new_tokens": 512, "temperature": 0.7, "return_full_text": False}
                }
                
                response = await client.post(endpoint['url'], json=payload, headers=headers, timeout=25.0)
                
                # If Loading (503), fail fast so we can try next or tell user
                if response.status_code == 503:
                    last_error = "Model is loading (503)"
                    continue 

                # If Success (200)
                if response.status_code == 200:
                    result = response.json()
                    ai_reply = result[0].get("generated_text", "") if isinstance(result, list) else str(result)
                    
                    # Save & Return
                    await memory.save(request.message, ai_reply, request.conversation_id)
                    return {"response": ai_reply, "model": endpoint['name']}
                
                # If Auth Error (401), stop immediately (no point trying others)
                if response.status_code == 401:
                    return {"response": "Error: Invalid HF_API_TOKEN. Check Render settings."}

                last_error = f"Error {response.status_code}: {response.text}"
                
            except Exception as e:
                last_error = f"Connection Error: {str(e)}"
                continue

    # 4. If all failed
    return {"response": f"All models failed. Last error: {last_error}"}

@app.get("/health")
async def health():
    return {"status": "Online"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
    
