from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os

# --- SETUP ---
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("CRITICAL: API Key is missing.")

genai.configure(api_key=api_key)

# WE ARE SWITCHING TO THE STANDARD MODEL
# 'gemini-pro' is the most stable version.
model = genai.GenerativeModel('gemini-pro')

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
    
    prompt = f"""
    Analyze this text for misinformation: "{request.text}"
    
    Return the response in this EXACT format (no markdown):
    Score: [0-100] (Where 0 is Fake/Misinformation, 100 is True/Verified)
    Color: [Red/Yellow/Green]
    Reason: [Short explanation]
    """

    try:
        response = model.generate_content(prompt)
        # Check if response was blocked or empty
        if not response.text:
            return {"score": 50, "color": "Yellow", "reason": "AI response was empty."}
            
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
        print(f"ERROR: {e}")
        # If the AI fails, return a safe fallback so the user doesn't see a crash
        return {"score": 0, "color": "Red", "reason": "Connection Error. Please try again."}
