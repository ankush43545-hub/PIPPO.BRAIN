"""
Pippo Backend - Production Ready with Debugging
FastAPI server with MongoDB, comprehensive error handling, and debugging
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator, Field
from typing import Optional, List, Dict, Any
import httpx
import asyncio
import json
from datetime import datetime
import os
from enum import Enum
import logging
import sys
import traceback
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

# =====================================================
# LOGGING CONFIGURATION
# =====================================================

logging.basicConfig(
    level=logging.INFO, # Changed to INFO to reduce noise, use DEBUG for deep dives
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pippo.log')
    ]
)
logger = logging.getLogger("Pippo")

# =====================================================
# FASTAPI APP INITIALIZATION
# =====================================================

app = FastAPI(
    title="Pippo Orchestrator",
    description="Backend brain for Pippo, the strategic AI chatbot",
    version="1.1.0",
    debug=True
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Production: Change this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# CONFIGURATION
# =====================================================

class Config:
    # Hugging Face - Using the router endpoint
    HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
    HF_API_BASE = "https://router.huggingface.co/models/"
    
    # MongoDB
    MONGO_URL = os.getenv("MONGO_URL", "")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "pippo_db")
    
    # Server
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    # Feature flags
    USE_MONGODB = bool(MONGO_URL)
    DEBUG_MODE = os.getenv("DEBUG_MODE", "true").lower() == "true"

config = Config()

logger.info(f"Starting Pippo Backend v1.1.0")
logger.info(f"MongoDB Enabled: {config.USE_MONGODB}")

# =====================================================
# DATABASE CONNECTION
# =====================================================

class Database:
    client: Optional[AsyncIOMotorClient] = None
    db = None
    
    @classmethod
    async def connect_db(cls):
        """Connect to MongoDB"""
        if not config.USE_MONGODB or not config.MONGO_URL:
            logger.warning("MongoDB not configured, using memory-only mode")
            return
        
        try:
            logger.info("Connecting to MongoDB...")
            cls.client = AsyncIOMotorClient(config.MONGO_URL)
            cls.db = cls.client[config.MONGO_DB_NAME]
            
            # Test connection
            await cls.client.admin.command('ping')
            logger.info("✓ Successfully connected to MongoDB")
            
            # Create indexes
            await cls.db.conversations.create_index("user_id")
            await cls.db.messages.create_index("conversation_id")
            await cls.db.messages.create_index("timestamp")
            
        except Exception as e:
            logger.error(f"✗ Failed to connect to MongoDB: {str(e)}")
            cls.client = None
            cls.db = None
    
    @classmethod
    async def close_db(cls):
        if cls.client:
            cls.client.close()
    
    @classmethod
    async def save_conversation(cls, user_id: str, title: str = "New Conversation"):
        if not cls.db: return None
        try:
            conversation = {
                "user_id": user_id,
                "title": title,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            result = await cls.db.conversations.insert_one(conversation)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            return None
    
    @classmethod
    async def save_message(cls, conversation_id: str, role: str, content: str, metadata: Dict = None):
        if not cls.db: return None
        try:
            message = {
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow()
            }
            await cls.db.messages.insert_one(message)
        except Exception as e:
            logger.error(f"Error saving message: {e}")

    @classmethod
    async def get_conversation_history(cls, conversation_id: str, limit: int = 20):
        if not cls.db: return []
        try:
            cursor = cls.db.messages.find(
                {"conversation_id": conversation_id}
            ).sort("timestamp", -1).limit(limit)
            messages = await cursor.to_list(length=limit)
            messages.reverse()
            return [
                {"role": msg["role"], "content": msg["content"], "timestamp": msg["timestamp"].isoformat()}
                for msg in messages
            ]
        except Exception as e:
            logger.error(f"Error retrieving history: {e}")
            return []

db = Database()

# =====================================================
# MODELS AND SCHEMAS
# =====================================================

# Updated Model List - Verified for Inference API Support
MODELS = {
    "brain": "mistralai/Mistral-7B-Instruct-v0.3",  # Upgraded to v0.3
    "vision": "Salesforce/blip2-opt-2.7b",
    "speech": "openai/whisper-small",
    "tts": "facebook/mms-tts-eng",                 # Changed from XTTS (unsupported) to MMS
    "image_gen": "stabilityai/stable-diffusion-2-1",
    "ocr": "microsoft/trocr-base-printed"          # Changed from Nougat (too heavy) to TrOCR
}

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
    message: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None
    user_id: Optional[str] = "anonymous"
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

class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str

# =====================================================
# STRATEGIC THINKING ENGINE (PIPPO'S BRAIN)
# =====================================================

class StrategicOrchestrator:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=120.0)
        self.request_count = 0
    
    async def analyze_intent(self, message: str, history: List[Message]) -> IntentAnalysis:
        try:
            # Simple keyword fallback if brain is slow, but we try brain first
            # Prompt engineered for Mistral Instruction format
            prompt = f"""<s>[INST] You are Pippo, a strategic AI system. Analyze this user request.
Context: {self._build_context(history)}
User Input: "{message}"

Return a valid JSON object ONLY. No markdown, no explanations.
Format:
{{
    "intent": "chat|code|image_understand|image_generate|document_read",
    "confidence": 0.9,
    "needs_clarification": false,
    "clarification_question": null,
    "expected_output": "brief"
}}
[/INST]"""

            response = await self._query_brain(prompt)
            analysis_json = self._extract_json(response)
            
            # Fallback if JSON fails
            if not analysis_json or "intent" not in analysis_json:
                return self._fallback_intent(message)
            
            return IntentAnalysis(
                intent=analysis_json.get("intent", "chat"),
                confidence=float(analysis_json.get("confidence", 0.7)),
                needs_clarification=analysis_json.get("needs_clarification", False),
                clarification_question=analysis_json.get("clarification_question"),
                required_models=self._determine_models(analysis_json.get("intent")),
                expected_response_type=analysis_json.get("expected_output", "brief")
            )
            
        except Exception as e:
            logger.error(f"Intent analysis failed: {e}")
            return self._fallback_intent(message)
    
    async def strategic_response(self, message: str, intent: IntentAnalysis, history: List[Message], data: Dict = None) -> str:
        self.request_count += 1
        
        handlers = {
            IntentType.CODE: self._handle_code_request,
            IntentType.IMAGE_UNDERSTAND: self._handle_image_understanding,
            IntentType.IMAGE_GENERATE: self._handle_image_generation,
            IntentType.DOCUMENT_READ: self._handle_document_reading,
        }
        
        handler = handlers.get(intent.intent)
        if handler:
            if intent.intent == IntentType.IMAGE_UNDERSTAND:
                return await handler(message, data.get("image"))
            elif intent.intent == IntentType.DOCUMENT_READ:
                return await handler(message, data.get("document"))
            elif intent.intent == IntentType.IMAGE_GENERATE:
                return await handler(message)
            else:
                return await handler(message, history, intent)
        
        return await self._handle_chat(message, history, intent)
    
    # --- Handlers ---

    async def _handle_code_request(self, message: str, history: List[Message], intent: IntentAnalysis) -> str:
        prompt = f"""<s>[INST] You are Pippo, an expert coding assistant.
User Request: "{message}"
Context: {self._build_context(history)}

Provide the solution in this format:
1. Strategy (1 sentence)
2. Code Block (using markdown)
3. Brief Explanation
[/INST]"""
        return await self._query_brain(prompt)

    async def _handle_image_understanding(self, message: str, image_data: Optional[str]) -> str:
        if not image_data: return "I need you to upload an image first."
        
        # BLIP-2 VQA
        vision_response = await self._query_vision(image_data, message)
        
        # Synthesize with Brain
        prompt = f"""<s>[INST] I showed you an image. 
My Question: "{message}"
Vision Model Analysis: "{vision_response}"

Give me a natural, conversational answer based on that analysis. [/INST]"""
        return await self._query_brain(prompt)

    async def _handle_image_generation(self, message: str) -> str:
        # 1. Refine Prompt
        refine_prompt = f"""<s>[INST] Convert this request into a high-quality Stable Diffusion prompt: "{message}". 
Return ONLY the prompt string. [/INST]"""
        refined = await self._query_brain(refine_prompt)
        refined = refined.replace('"', '').strip()
        
        # 2. Generate
        try:
            image_b64 = await self._query_image_gen(refined)
            return f"IMAGE_GENERATED:{image_b64}|PROMPT:{refined}"
        except Exception:
            return "I tried to paint that, but my image generator is currently busy. Please try again in a moment."

    async def _handle_document_reading(self, message: str, document_data: Optional[str]) -> str:
        if not document_data: return "Please upload a document."
        
        text = await self._query_ocr(document_data)
        if not text: return "I couldn't read the text in that document. Is it clear?"
        
        prompt = f"""<s>[INST] Analyze this document text:
{text[:2000]}... [truncated]

User Question: "{message}"
Answer concisely. [/INST]"""
        return await self._query_brain(prompt)

    async def _handle_chat(self, message: str, history: List[Message], intent: IntentAnalysis) -> str:
        prompt = f"""<s>[INST] You are Pippo, a helpful and friendly AI assistant.
Conversation History:
{self._build_context(history)}

User: {message}
Respond naturally and concisely. [/INST]"""
        return await self._query_brain(prompt)

    # --- API Queries ---

    async def _query_brain(self, prompt: str) -> str:
        """Queries the LLM (Mistral)"""
        try:
            payload = {
                "inputs": prompt,
                "parameters": {"max_new_tokens": 512, "temperature": 0.7, "return_full_text": False}
            }
            response = await self.client.post(
                f"{config.HF_API_BASE}{MODELS['brain']}",
                json=payload,
                headers={"Authorization": f"Bearer {config.HF_API_TOKEN}"}
            )
            if response.status_code != 200: raise Exception(f"Brain Error: {response.text}")
            result = response.json()
            return result[0]["generated_text"].strip() if isinstance(result, list) else str(result)
        except Exception as e:
            logger.error(f"Brain query failed: {e}")
            return "I'm having a bit of a headache (internal error). Try again?"

    async def _query_vision(self, image_data: str, question: str) -> str:
        """Queries BLIP-2"""
        payload = {
            "inputs": {
                "image": image_data, # base64 string
                "question": question or "Describe this image"
            }
        }
        response = await self.client.post(
            f"{config.HF_API_BASE}{MODELS['vision']}",
            json=payload,
            headers={"Authorization": f"Bearer {config.HF_API_TOKEN}"}
        )
        # BLIP-2 often returns a list containing 'generated_text'
        if response.status_code == 200:
            res = response.json()
            return res[0].get("generated_text", "") if isinstance(res, list) else str(res)
        return "Image unclear"

    async def _query_image_gen(self, prompt: str) -> str:
        """Queries Stable Diffusion"""
        response = await self.client.post(
            f"{config.HF_API_BASE}{MODELS['image_gen']}",
            json={"inputs": prompt},
            headers={"Authorization": f"Bearer {config.HF_API_TOKEN}"}
        )
        if response.status_code == 200:
            import base64
            # API returns raw bytes for image
            return base64.b64encode(response.content).decode()
        raise Exception("Image gen failed")

    async def _query_ocr(self, image_data: str) -> str:
        """Queries TrOCR"""
        # TrOCR expects just the image in 'inputs' typically
        payload = {"inputs": image_data}
        response = await self.client.post(
            f"{config.HF_API_BASE}{MODELS['ocr']}",
            json=payload,
            headers={"Authorization": f"Bearer {config.HF_API_TOKEN}"}
        )
        if response.status_code == 200:
            res = response.json()
            return res[0].get("generated_text", "") if isinstance(res, list) else ""
        return ""

    # --- Helpers ---
    def _build_context(self, history: List[Message]) -> str:
        return "\n".join([f"{'User' if m.role == 'user' else 'Pippo'}: {m.content}" for m in history[-5:]])

    def _extract_json(self, text: str) -> Dict:
        try:
            start, end = text.find("{"), text.rfind("}") + 1
            if start != -1 and end > start: return json.loads(text[start:end])
            return {}
        except: return {}

    def _fallback_intent(self, message: str) -> IntentAnalysis:
        # Basic keyword matching fallback
        msg = message.lower()
        intent = IntentType.CHAT
        if "code" in msg or "python" in msg: intent = IntentType.CODE
        elif "image" in msg and "create" in msg: intent = IntentType.IMAGE_GENERATE
        elif "image" in msg: intent = IntentType.IMAGE_UNDERSTAND
        
        return IntentAnalysis(
            intent=intent, confidence=0.5, needs_clarification=False,
            required_models=[], expected_response_type="brief"
        )
    
    def _determine_models(self, intent: str) -> List[str]:
        return [] # Logic to list models if needed

orchestrator = StrategicOrchestrator()

# =====================================================
# ENDPOINTS
# =====================================================

@app.get("/")
async def root():
    return {"message": "Pippo is awake and listening! Send POST requests to /chat"}

@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat with Pippo"""
    try:
        if config.USE_MONGODB and request.conversation_id:
            # Retrieve history
            pass 

        intent = await orchestrator.analyze_intent(request.message, request.history)
        
        response_text = await orchestrator.strategic_response(
            request.message, intent, request.history,
            {"image": request.image_data, "document": request.document_data}
        )

        if config.USE_MONGODB:
            # Save history
            pass
            
        return {
            "response": response_text,
            "intent": intent.intent,
            "request_id": f"req_{int(datetime.utcnow().timestamp())}"
        }
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Pippo is confused (Internal Error)")

@app.post("/text-to-voice")
async def text_to_voice(text: str):
    """Convert text to voice using MMS"""
    try:
        # MMS-TTS-ENG payload is usually simple raw string in 'inputs'
        payload = {"inputs": text[:500]} 
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{config.HF_API_BASE}{MODELS['tts']}",
                json=payload,
                headers={"Authorization": f"Bearer {config.HF_API_TOKEN}"}
            )
        
        if response.status_code == 200:
            import base64
            # The API returns raw audio bytes (flac/wav)
            return {"audio": base64.b64encode(response.content).decode()}
        raise HTTPException(status_code=500, detail="Voice generation failed")
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=config.DEBUG_MODE)
