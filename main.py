from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import json

# --- CONFIG ---
API_KEY = os.environ.get("GEMINI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CheckRequest(BaseModel):
    text: str

def get_allowed_models():
    """Ask Google what models are actually available for this Key"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        # Filter for models that support 'generateContent'
        valid_models = []
        if "models" in data:
            for m in data["models"]:
                if "generateContent" in m.get("supportedGenerationMethods", []):
                    valid_models.append(m["name"])
        return valid_models
    except Exception as e:
        return [f"Error listing models: {str(e)}"]

@app.post("/check")
async def check_trust(request: CheckRequest):
    print(f"\n--- ANALYZING: {request.text[:30]}... ---")
    
    # 1. Try the most standard model first (Gemini 1.5 Flash)
    target_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"
    
    system_prompt = """
    You are a fact-checking system. Return ONLY valid JSON:
    { "trust_score": number, "reason": "short string" }
    """
    
    payload = {
        "contents": [{"parts": [{"text": f"{system_prompt}\n\nText:\n{request.text}"}]}]
    }

    try:
        response = requests.post(target_url, json=payload, headers={"Content-Type": "application/json"})
        data = response.json()

        # --- IF GOOGLE SAYS "NOT FOUND", WE DIAGNOSE ---
        if "error" in data:
            error_msg = data['error'].get('message', 'Unknown')
            print(f"❌ MODEL FAILED: {error_msg}")
            
            # RUN THE DIAGNOSTIC
            print("\n🔍 ASKING GOOGLE FOR VALID MODELS...")
            allowed = get_allowed_models()
            print(f"✅ GOOGLE SAYS YOU CAN USE THESE MODELS: {allowed}")
            print("--------------------------------------------------")
            
            return {"score": 0, "color": "Red", "reason": "Check Logs for Valid Model List"}

        # --- IF SUCCESS ---
        ai_text = data["candidates"][0]["content"]["parts"][0]["text"]
        ai_text = ai_text.replace("```json", "").replace("```", "").strip()
        result = json.loads(ai_text)
        
        score = result.get("trust_score", 0)
        color = "Green" if score >= 80 else "Yellow" if score >= 50 else "Red"
        
        return {"score": score, "color": color, "reason": result.get("reason")}

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return {"score": 0, "color": "Red", "reason": "Connection Error"}
