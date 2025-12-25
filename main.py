from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
import os

# --- SECURITY UPDATE ---
# 1. Try to get key from Cloud Environment (Safe Box)
api_key = os.environ.get("GEMINI_API_KEY")

# 2. If not found (running on laptop), use this backup key
if not api_key:
    api_key = "AIzaSyBKjIJKHbliSyaxxMyo79SDQu5z8QPDS3E" 

client = genai.Client(api_key=api_key)

app = FastAPI()

# --- CONNECTION FIX (CORS) ---
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
    print(f"\n--- ANALYZING: {request.text} ---")
    
    prompt = f"""
    Analyze this text for misinformation: "{request.text}"
    
    Return the response in this EXACT format (no markdown):
    Score: [0-100] (Where 0 is Fake/Misinformation, 100 is True/Verified)
    Color: [Red/Yellow/Green]
    Reason: [Short explanation]
    """

    try:
        response = client.models.generate_content(
            model="gemini-flash-latest", 
            contents=prompt
        )
        
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
        return {"score": 0, "color": "Red", "reason": f"Error: {str(e)}"}
