"""
Pippo Backend - Powered by Llama 3.3 70B
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
import logging
import sys

# 1. Setup Logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("Pippo")

# 2. Configuration
# We need the HF Token to access Llama 3.3
HF_API_TOKEN = os.getenv("HF_API_TOKEN") 

# Using the serverless inference endpoint for Llama 3.3 70B Instruct
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.3-70B-Instruct"

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

@app.get("/")
def root():
    return {
        "status": "Online", 
        "model": "Meta-llama/Llama-3.3-70B-Instruct",
        "token_set": bool(HF_API_TOKEN)
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    if not HF_API_TOKEN:
        raise HTTPException(status_code=500, detail="Missing HF_API_TOKEN in Render Environment.")

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    
    # Llama 3 specific prompt format
    # <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are Pippo, a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
    
    formatted_prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"You are Pippo, a smart and helpful AI assistant.<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{request.message}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "return_full_text": False
        }
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(API_URL, json=payload, headers=headers)
            
            if response.status_code != 200:
                logger.error(f"HF Error: {response.status_code} - {response.text}")
                # Check for specific Llama 3 loading error
                if response.status_code == 503:
                    return {"response": "I'm waking up (Model Loading). Please ask again in 20 seconds."}
                return {"error": f"HF Error {response.status_code}", "details": response.text}

            result = response.json()
            
            # Robust parsing
            if isinstance(result, list) and len(result) > 0:
                return {"response": result[0].get('generated_text', "").strip()}
            elif isinstance(result, dict) and 'generated_text' in result:
                 return {"response": result['generated_text'].strip()}
            else:
                return {"response": str(result)}
                
        except httpx.TimeoutException:
            return {"error": "Request timed out. Llama 70B might be too busy for the free tier right now."}
        except Exception as e:
            return {"error": f"System Error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
    
