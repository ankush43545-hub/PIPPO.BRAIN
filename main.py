"""
PIPPO Backend - Final Fixed Version
"""

import os
import json
import traceback
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
import httpx
import uvicorn

app = FastAPI(title="PIPPO Orchestrator")

# CORS for mobile/web access
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
HF_API_BASE = "https://api-inference.huggingface.co/models/"

# Model endpoints
MODELS = {
    "brain": "mistralai/Mistral-7B-Instruct-v0.2",
    "vision": "Salesforce/blip2-opt-2.7b",
    "speech": "openai/whisper-small",
    "tts": "coqui/XTTS-v2",
    "image_gen": "stabilityai/stable-diffusion-2-1",
    "ocr": "facebook/nougat-base"
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

    async def get_history_objects(self, conversation_id: str, limit: int = 6):
        # FIX: Check if None explicitly
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
        except Exception as e:
            print(f"⚠️ Fetch Error (Ignored): {e}")
            return []

    async def save(self, user, bot, cid):
        # FIX: Check if None explicitly
        if self.db is None: return
        try:
            await self.db.conversations.insert_one({
                "conversation_id": cid,
                "timestamp": datetime.utcnow(),
                "user": user,
                "pippo": bot
            })
        except Exception as e:
            print(f"⚠️ Save Error (Ignored): {e}")

memory = SafeMemory()

@app.on_event("startup")
async def startup():
    memory.connect()

@app.on_event("shutdown")
async def shutdown():
    if memory.client:
        memory.client.close()

# --- DATA MODELS ---

class IntentType(str, Enum):
    CHAT = "chat"
    CODE = "code"
    IMAGE_UNDERSTAND = "image_understand"
    IMAGE_GENERATE = "image_generate"
    VOICE_INPUT = "voice_input"
    VOICE_OUTPUT = "voice_output"
    DOCUMENT_READ = "document_read"
    CLARIFICATION_NEEDED = "clarification_needed"

class Message(BaseModel):
    content: str
    role: str = "user"
    timestamp: Optional[str] = None
    
class ChatRequest(BaseModel):
    message: str
    conversation_id: str = "default"
    history: Optional[List[Message]] = [] 
    voice_enabled: bool = False
    image_data: Optional[str] = None
    document_data: Optional[str] = None

class IntentAnalysis(BaseModel):
    intent: IntentType
    confidence: float
    needs_clarification: bool
    clarification_question: Optional[str]
    required_models: List[str]
    expected_response_type: str

# =====================================================
# STRATEGIC THINKING ENGINE
# =====================================================

class StrategicOrchestrator:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=60.0)
        
    async def analyze_intent(self, message: str, history: List[Message]) -> IntentAnalysis:
        msg_lower = message.lower()
        if any(w in msg_lower for w in ["draw", "generate image", "picture of"]):
            return IntentAnalysis(intent=IntentType.IMAGE_GENERATE, confidence=0.9, needs_clarification=False, clarification_question=None, required_models=["brain", "image_gen"], expected_response_type="visual")
        
        return IntentAnalysis(
            intent=IntentType.CHAT,
            confidence=0.8,
            needs_clarification=False,
            clarification_question=None,
            required_models=["brain"],
            expected_response_type="brief"
        )
    
    async def strategic_response(self, message: str, intent: IntentAnalysis, history: List[Message], additional_data: Dict[str, Any] = None) -> str:
        if intent.intent == IntentType.IMAGE_GENERATE:
            return await self._handle_image_generation(message)
        else:
            return await self._handle_chat(message, history, intent)
    
    async def _handle_image_generation(self, message: str) -> str:
        clean_prompt = message.replace("draw", "").replace("generate image", "").strip()
        try:
            image_data = await self._query_image_gen(clean_prompt)
            return f"IMAGE_GENERATED:{image_data}"
        except Exception as e:
            return f"I couldn't generate that image. Error: {e}"

    async def _handle_chat(self, message: str, history: List[Message], intent: IntentAnalysis) -> str:
        context = self._build_context(history)
        prompt = f"<s>[INST] You are PIPPO. Be helpful and concise. Context: {context} User: {message} [/INST]"
        return await self._query_brain(prompt)

    async def _query_brain(self, prompt: str) -> str:
        try:
            payload = {"inputs": prompt, "parameters": {"max_new_tokens": 512, "temperature": 0.7}}
            headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
            response = await self.client.post(f"{HF_API_BASE}{MODELS['brain']}", json=payload, headers=headers)
            if response.status_code == 200:
                result = response.json()
                return result[0].get("generated_text", "") if isinstance(result, list) else str(result)
            return "Thinking failed."
        except Exception as e:
            return f"Brain Error: {e}"

    async def _query_image_gen(self, prompt: str) -> str:
        payload = {"inputs": prompt}
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        response = await self.client.post(f"{HF_API_BASE}{MODELS['image_gen']}", json=payload, headers=headers)
        if response.status_code == 200:
            import base64
            return base64.b64encode(response.content).decode()
        raise Exception("Image Gen Failed")

    def _build_context(self, history: List[Message]) -> str:
        return "\n".join([f"{msg.role}: {msg.content}" for msg in history[-4:]])

orchestrator = StrategicOrchestrator()

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        db_history = await memory.get_history_objects(request.conversation_id)
        full_history = db_history + (request.history or [])
        intent = await orchestrator.analyze_intent(request.message, full_history)
        
        response = await orchestrator.strategic_response(
            request.message, intent, full_history, {}
        )
        
        await memory.save(request.message, response, request.conversation_id)
        
        return {"response": response, "intent": intent.intent}
        
    except Exception as e:
        return {"response": f"Critical Logic Error: {str(e)}", "intent": "error"}

@app.get("/health")
async def health():
    return {"status": "PIPPO is alive"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
    
