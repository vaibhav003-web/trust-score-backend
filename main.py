from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import json

# --- CONFIG ---
API_KEY = os.environ.get("GEMINI_API_KEY")

# FIX: Switched from 'v1beta' to 'v1' (Stable Endpoint)
# This solves the 404 "Model Not Found" error.
URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={API_KEY}"

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
    
    # YOUR STRICT JSON PROMPT
    system_prompt = """
    You are a fact-checking and credibility analysis system.

    Analyze the following text and evaluate how trustworthy it is.

    Criteria:
    - Factual accuracy
    - Logical consistency
    - Emotional manipulation or bias
    - Misinformation or exaggeration
    - Source reliability (if implied)

    Return ONLY a valid JSON object in this exact format:
    {
      "trust_score": number, 
      "reason": "short explanation"
    }

    Be strict. Assume the text is untrusted unless strong evidence supports it.
    """

    # Prepare payload for Gemini
    payload = {
        "contents": [{
            "parts": [{
                "text": f"{system_prompt}\n\nText to analyze:\n\"\"\"{request.text}\"\"\""
            }]
        }]
    }

    try:
        # Send Request to v1 Endpoint
        response = requests.post(URL, json=payload, headers={"Content-Type": "application/json"})
        data = response.json()

        # 1. Check for API Errors
        if "error" in data:
            print(f"GOOGLE ERROR: {data['error']}")
            return {"score": 0, "color": "Red", "reason": f"API Error: {data['error'].get('message', 'Unknown')}"}

        # 2. Extract Text
        ai_text = data["candidates"][0]["content"]["parts"][0]["text"]
        
        # 3. Clean JSON (Remove ```json ... ``` wrappers if present)
        ai_text = ai_text.replace("```json", "").replace("```", "").strip()
        
        # 4. Parse JSON
        result = json.loads(ai_text)
        
        # 5. Determine Color based on Score
        score = result.get("trust_score", 0)
        color = "Red"
        if score >= 80:
            color = "Green"
        elif score >= 50:
            color = "Yellow"

        return {
            "score": score,
            "color": color,
            "reason": result.get("reason", "No reason provided.")
        }

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return {"score": 0, "color": "Red", "reason": "Error parsing AI response."}
