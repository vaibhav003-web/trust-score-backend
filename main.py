from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import json

# --- CONFIG ---
API_KEY = os.environ.get("GEMINI_API_KEY")
# Direct URL to Google (Bypasses the buggy Python library)
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
    
    # --- YOUR NEW PROMPT ---
    system_prompt = f"""
    You are an impartial, analytical AI system.
    Your task is to evaluate the trustworthiness of the given content.
    Respond strictly in JSON format only.

    Analyze the following content:
    \"\"\"
    {request.text}
    \"\"\"

    Instructions:
    1. Detect factual accuracy, logical consistency, and emotional manipulation.
    2. Output JSON format:
    {{
      "trust_score": number (0-100),
      "recommendation": "trust" | "verify" | "avoid",
      "summary": "Short explanation (max 1 sentence)"
    }}
    """

    payload = {
        "contents": [{"parts": [{"text": system_prompt}]}]
    }

    try:
        # Direct Request (No Library)
        response = requests.post(URL, json=payload, headers={"Content-Type": "application/json"})
        data = response.json()

        # Error Handling
        if "error" in data:
            print(f"GOOGLE ERROR: {data['error']}")
            return {"score": 0, "color": "Red", "reason": "API Error. Please try again."}

        # Parse AI Response
        ai_text = data["candidates"][0]["content"]["parts"][0]["text"]
        
        # Clean up JSON (sometimes AI adds ```json marks)
        ai_text = ai_text.replace("```json", "").replace("```", "").strip()
        result = json.loads(ai_text)

        # Map your JSON to the Frontend (Popup) format
        # Your prompt returns "recommendation", we convert it to "Color"
        color_map = {
            "trust": "Green",
            "verify": "Yellow",
            "avoid": "Red"
        }
        
        return {
            "score": result.get("trust_score", 0),
            "color": color_map.get(result.get("recommendation", "verify"), "Yellow"),
            "reason": result.get("summary", "Analysis complete.")
        }

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return {"score": 0, "color": "Red", "reason": "Connection Error."}
