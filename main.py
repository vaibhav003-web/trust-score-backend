from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os

# --- 1. SETUP API KEY ---
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("API Key not found!")

genai.configure(api_key=api_key)

# --- 2. SETUP MODEL (Standard Stable Version) ---
# We use the standard flash model which works everywhere.
model = genai.GenerativeModel('gemini-1.5-flash')

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
    print(f"\n--- ANALYZING: {request.text[:50]}... ---")
    
    prompt = f"""
    Analyze this text for misinformation: "{request.text}"
    
    Return the response in this EXACT format (no markdown):
    Score: [0-100] (Where 0 is Fake/Misinformation, 100 is True/Verified)
    Color: [Red/Yellow/Green]
    Reason: [Short explanation]
    """

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        print(f"AI Said: {text}") 
        
        lines = text.split('\n')
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
        # Default fallback so the user always sees something
        return {"score": 0, "color": "Red", "reason": "Error analyzing text."}
