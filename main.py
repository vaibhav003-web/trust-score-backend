from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os

# --- SETUP ---
# We use the raw API URL for Gemini 1.5 Flash (The high-quota model)
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("CRITICAL: API Key is missing.")

# DIRECT URL - No library confusion
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"

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

@app.post("/check")
async def check_trust(request: CheckRequest):
    print(f"\n--- ANALYZING: {request.text[:30]}... ---")
    
    # 1. Prepare the payload (The message to Google)
    payload = {
        "contents": [{
            "parts": [{
                "text": f"""
                Analyze this text for misinformation: "{request.text}"
                
                Return the response in this EXACT format (no markdown):
                Score: [0-100]
                Color: [Red/Yellow/Green]
                Reason: [Short explanation]
                """
            }]
        }]
    }

    # 2. Send via Direct "Requests" (Bypassing the Google Library)
    try:
        response = requests.post(URL, json=payload, headers={"Content-Type": "application/json"})
        data = response.json()
        
        # Check for errors in the raw response
        if "error" in data:
            print(f"GOOGLE ERROR: {data['error']}")
            return {"score": 0, "color": "Red", "reason": f"API Error: {data['error'].get('message', 'Unknown')}"}

        # 3. Parse the success message
        ai_text = data["candidates"][0]["content"]["parts"][0]["text"]
        print(f"AI Said: {ai_text}")

        lines = ai_text.split('\n')
        score = 0
        color = "Red"
        reason = "Could not parse"

        for line in lines:
            if "Score:" in line:
                score_str = line.split(':')[1].strip()
                score = int(''.join(filter(str.isdigit, score_str)))
            if "Color:" in line:
                color = line.split(':')[1].strip()
            if "Reason:" in line:
                reason = line.split(':')[1].strip()

        return {"score": score, "color": color, "reason": reason}

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return {"score": 0, "color": "Red", "reason": "Connection failed."}
