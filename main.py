from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse # NEW
from pydantic import BaseModel
import os
import json
from groq import Groq

# 1. Setup
api_key = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=api_key)

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

# --- NEW: Serve the Website at the Root URL ---
@app.get("/")
async def read_root():
    # When someone visits your URL, show them the website
    return FileResponse("index.html")

# --- EXISTING: The API for Extension AND Website ---
@app.post("/check")
async def check_trust(request: CheckRequest):
    print(f"Analyzing: {request.text[:30]}...")

    system_prompt = """
    You are a strict fact-checker. Analyze the text for credibility, bias, and evidence.
    SCORING RULES:
    - If "No Evidence", "Rumors", "Conspiracy" -> Score < 40.
    - If "Highly Biased" -> Score < 60.
    - Only give 80+ for cited, neutral facts.

    Return STRICT JSON:
    {
      "trust_score": (int 0-100),
      "reason": (string max 15 words),
      "bias_rating": (string),
      "flags": (list of strings)
    }
    """

    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.text}
            ],
            model="llama-3.1-8b-instant",
            temperature=0,
            response_format={"type": "json_object"}
        )

        result = json.loads(completion.choices[0].message.content)
        
        score = result.get("trust_score", 0)
        color = "Red"
        if score >= 80: color = "Green"
        elif score >= 50: color = "Yellow"

        return {
            "score": score,
            "color": color,
            "reason": result.get("reason"),
            "bias": result.get("bias_rating"),
            "flags": result.get("flags", [])
        }

    except Exception as e:
        print(f"Error: {e}")
        return {"score": 0, "color": "Red", "reason": "AI Error"}
