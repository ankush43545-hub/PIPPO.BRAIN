"""
PIPPO Backend - Strategic Orchestrator with Safe Cloud Memory
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

# --- SAFE MEMORY SYSTEM (Crash-Proof) ---
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
        """Returns history as a list of Message objects for the Orchestrator"""
        if not self.db: return []
        try:
            cursor = self.db.conversations.find({"conversation_id": conversation_id}).sort("timestamp", -1).limit(limit)
            history_data = await cursor.to_list(length=limit)
            history_data.reverse()
            
            # Convert DB format to Message objects
            messages = []
            for h in history_data:
                messages.append(Message(role="user", content=h.get("user", "")))
                messages.append(Message(role="assistant", content=h.get("pippo", "")))
            return messages
        except Exception as e:
            print(f"⚠️ Fetch Error (Ignored): {e}")
            return []

    async def save(self, user, bot, cid):
        if not self.db: return
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
    # History is now optional because we fetch it from DB
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
# STRATEGIC THINKING ENGINE (The Brain)
# =====================================================

class StrategicOrchestrator:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=60.0)
        
    async def analyze_intent(self, message: str, history: List[Message]) -> IntentAnalysis:
        # 1. Quick Keyword Checks (Saves API tokens and time)
        msg_lower = message.lower()
        if any(w in msg_lower for w in ["draw", "generate image", "picture of"]):
            return IntentAnalysis(intent=IntentType.IMAGE_GENERATE, confidence=0.9, needs_clarification=False, clarification_question=None, required_models=["brain", "image_gen"], expected_response_type="visual")
        if any(w in msg_lower for w in ["read this", "summarize pdf", "document"]):
            return IntentAnalysis(intent=IntentType.DOCUMENT_READ, confidence=0.9, needs_clarification=False, clarification_question=None, required_models=["ocr", "brain"], expected_response_type="brief")
        
        # 2. Default to Chat/Code
        intent_type = IntentType.CODE if "code" in msg_lower else IntentType.CHAT
        
        return IntentAnalysis(
            intent=intent_type,
            confidence=0.8,
            needs_clarification=False,
            clarification_question=None,
            required_models=["brain"],
            expected_response_type="detailed" if intent_type == IntentType.CODE else "brief"
        )
    
    async def strategic_response(self, message: str, intent: IntentAnalysis, history: List[Message], additional_data: Dict[str, Any] = None) -> str:
        if intent.intent == IntentType.CODE:
            return await self._handle_code_request(message, history)
        elif intent.intent == IntentType.IMAGE_UNDERSTAND:
            return await self._handle_image_understanding(message, additional_data.get("image"))
        elif intent.intent == IntentType.IMAGE_GENERATE:
            return await self._handle_image_generation(message)
        elif intent.intent == IntentType.DOCUMENT_READ:
            return await self._handle_document_reading(message, additional_data.get("document"))
        else:
            return await self._handle_chat(message, history, intent)
    
    async def _handle_code_request(self, message: str, history: List[Message]) -> str:
        context = self._build_context(history)
        prompt = f"<s>[INST] You are a coding assistant. Context: {context} User: {message} Provide code and brief explanation. [/INST]"
        return await self._query_brain(prompt)
    
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

    async def _handle_image_understanding(self, message: str, image_data: str) -> str:
        if not image_data: return "Please upload an image."
        caption = await self._query_vision(image_data, message)
        return f"I see: {caption}"

    async def _handle_document_reading(self, message: str, doc_data: str) -> str:
        if not doc_data: return "Please upload a document."
        text = await self._query_ocr(doc_data)
        return f"Document Content: {text[:500]}..."

    # --- MODEL CALLS ---
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

    async def _query_vision(self, image_data: str, prompt: str) -> str:
        # Simplified vision call
        payload = {"inputs": prompt, "image": image_data}
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        response = await self.client.post(f"{HF_API_BASE}{MODELS['vision']}", json=payload, headers=headers)
        if response.status_code == 200:
            return response.json()[0].get("generated_text", "")
        raise Exception("Vision Failed")
    
    async def _query_ocr(self, doc_data: str) -> str:
        # Simplified OCR call
        payload = {"inputs": doc_data}
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        response = await self.client.post(f"{HF_API_BASE}{MODELS['ocr']}", json=payload, headers=headers)
        if response.status_code == 200:
            return response.json()[0].get("generated_text", "")
        raise Exception("OCR Failed")

    def _build_context(self, history: List[Message]) -> str:
        return "\n".join([f"{msg.role}: {msg.content}" for msg in history[-4:]])

orchestrator = StrategicOrchestrator()

# =====================================================
# API ENDPOINTS
# =====================================================

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # 1. Fetch Cloud History (Safe Mode)
        # If DB is broken, this returns empty list [] instead of crashing
        db_history = await memory.get_history_objects(request.conversation_id)
        
        # Combine local history (if any) with Cloud history
        full_history = db_history + (request.history or [])
        
        # 2. Analyze Intent
        intent = await orchestrator.analyze_intent(request.message, full_history)
        
        additional_data = {
            "image": request.image_data,
            "document": request.document_data
        }
        
        # 3. Generate Response
        response = await orchestrator.strategic_response(
            request.message,
            intent,
            full_history,
            additional_data
        )
        
        # 4. Save to Cloud (Safe Mode)
        # If DB is broken, this skips saving instead of crashing
        await memory.save(request.message, response, request.conversation_id)
        
        return {
            "response": response,
            "intent": intent.intent,
            "confidence": intent.confidence
        }
        
    except Exception as e:
        # Ultimate Fallback: If code logic fails, tell user why
        return {"response": f"Critical Logic Error: {str(e)}", "intent": "error"}

@app.post("/voice-to-text")
async def voice_to_text(audio: UploadFile = File(...)):
    # Simple pass-through to Whisper
    try:
        audio_data = await audio.read()
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        response = await httpx.AsyncClient().post(
            f"{HF_API_BASE}{MODELS['speech']}",
            headers=headers,
            files={"file": audio_data}
        )
        return response.json() if response.status_code == 200 else {"text": "Voice Error"}
    except Exception as e:
        return {"text": f"Error: {e}"}

@app.get("/health")
async def health():
    return {
        "status": "PIPPO is alive", 
        "memory": "Online" if memory.db is not None else "Offline (Safe Mode Active)"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
        
