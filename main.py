"""
PIPPO Backend - DIAGNOSTIC MODE
"""
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from motor.motor_asyncio import AsyncIOMotorClient

app = FastAPI(title="PIPPO Diagnostic")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DEBUGGING SECRETS ---
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
MONGO_URL = os.getenv("MONGO_URL", "")
# Let's switch to a smaller, faster model for testing
MODELS = {"brain": "HuggingFaceH4/zephyr-7b-beta"} 
HF_API_BASE = "https://api-inference.huggingface.co/models/"

# --- DB SETUP ---
class SafeMemory:
    def __init__(self):
        self.client = None
        self.db = None
    
    def connect(self):
        if not MONGO_URL: return
        try:
            self.client = AsyncIOMotorClient(MONGO_URL, serverSelectionTimeoutMS=5000)
            self.db = self.client.get_database("pippo_db")
            print("✅ DB Connected")
        except:
            print("❌ DB Failed")

    async def save(self, user, bot, cid):
        if self.db is None: return
        try:
            await self.db.conversations.insert_one({"user": user, "bot": bot})
        except:
            pass

memory = SafeMemory()

@app.on_event("startup")
async def startup():
    memory.connect()

# --- DIAGNOSTIC BRAIN ---
class ChatRequest(BaseModel):
    message: str
    conversation_id: str = "default"

@app.post("/chat")
async def chat(request: ChatRequest):
    # 1. Check Token Visibility
    token_status = "MISSING"
    if HF_API_TOKEN:
        token_status = f"PRESENT (Starts with {HF_API_TOKEN[:4]}...)"
    
    # 2. Try to Query Brain
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": request.message}
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{HF_API_BASE}{MODELS['brain']}", 
                json=payload, 
                headers=headers, 
                timeout=20.0
            )
            
            # SUCCESS
            if response.status_code == 200:
                ai_reply = response.json()[0].get("generated_text", "No text")
                # Clean up the reply (remove the user prompt if it repeats it)
                if "User:" in ai_reply: 
                    ai_reply = ai_reply.split("User:")[-1]
                
                await memory.save(request.message, ai_reply, request.conversation_id)
                return {"response": ai_reply, "status": "Success"}
            
            # FAILURE - RETURN THE EXACT ERROR
            else:
                return {
                    "response": f"HF Error {response.status_code}: {response.text}",
                    "token_debug": token_status,
                    "model_used": MODELS['brain']
                }
                
        except Exception as e:
            return {"response": f"Crash: {str(e)}", "token_debug": token_status}

@app.get("/health")
async def health():
    return {"status": "Online"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
    
