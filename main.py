from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import json

# --- CONFIG ---
API_KEY = os.environ.get("GEMINI_API_KEY")

# LIST OF ENDPOINTS TO TRY (If one fails, we try the next)
# We prioritize Flash (Fast/High Quota), then fallback to Pro (Stable).
ENDPOINTS = [
    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}",
    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={API_KEY}",
    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.0-pro:generateContent?key={API_KEY}"
]

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

def query_google(endpoint, payload):
    """Helper function to send request to a specific URL"""
    try:
        response = requests.post(endpoint, json=payload, headers={"Content-Type": "application/json"})
        return response.json()
    except:
        return {"error": "Connection Failed"}

@app.post("/check")
async def check_trust(request: CheckRequest):
    print(f"\n--- ANALYZING: {request.text[:30]}... ---")
    
    system_prompt = """
    You are a fact-checking system. Analyze the text.
    Return ONLY valid JSON:
    { "trust_score": number, "reason": "short string" }
    """

    payload = {
        "contents": [{"parts": [{"text": f"{system_prompt}\n\nText:\n{request.text}"}]}]
    }

    # --- THE MAGIC LOOP ---
    # We try each URL until one works.
    final_data = None
    
    for url in ENDPOINTS:
        print(f"Trying model: {url.split('models/')[1].split(':')[0]} ...")
        data = query_google(url, payload)
        
        # If we see an error, print it and try the NEXT model
        if "error" in data:
            print(f"FAILED: {data['error'].get('message', 'Unknown Error')}")
            continue # Try next URL
        
        # If success, save data and break the loop!
        final_data = data
        print("SUCCESS! Model found.")
        break
    
    # If all 3 failed:
    if not final_data or "error" in final_data:
        return {"score": 0, "color": "Red", "reason": "All AI models failed. Check API Key."}

    # --- PARSE RESULT ---
    try:
        ai_text = final_data["candidates"][0]["content"]["parts"][0]["text"]
        ai_text = ai_text.replace("```json", "").replace("```", "").strip()
        result = json.loads(ai_text)
        
        score = result.get("trust_score", 0)
        color = "Red"
        if score >= 80: color = "Green"
        elif score >= 50: color = "Yellow"

        return {
            "score": score,
            "color": color,
            "reason": result.get("reason", "Analysis complete.")
        }
    except Exception as e:
        print(f"PARSING ERROR: {e}")
        return {"score": 0, "color": "Red", "reason": "Error reading AI response."}
