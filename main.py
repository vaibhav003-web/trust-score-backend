from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

@app.post("/check")
async def check_trust(request: CheckRequest):
    print(f"Analyzing: {request.text[:30]}...")

    # 2. PROMPT - We now ask for specific JSON fields: score, reason, bias, flags
    system_prompt = """
    You are an expert fact-checker. Analyze the text for credibility, bias, and logical fallacies.
    Return a STRICT JSON object with these exact fields:
    {
      "trust_score": (integer 0-100),
      "reason": (string, max 15 words summary),
      "bias_rating": (string: "Neutral", "Slight Bias", or "High Bias"),
      "flags": (list of strings, e.g. ["Sensationalism", "Political Propaganda", "Logical Fallacy", "No Evidence"])
    }
    """

    try:
        # 3. Call AI
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.text}
            ],
            model="llama-3.1-8b-instant",
            temperature=0,
            response_format={"type": "json_object"}
        )

        # 4. Parse Data
        result = json.loads(completion.choices[0].message.content)
        
        # Calculate Color
        score = result.get("trust_score", 0)
        color = "Red"
        if score >= 80: color = "Green"
        elif score >= 50: color = "Yellow"

        # Return Expanded Data
        return {
            "score": score,
            "color": color,
            "reason": result.get("reason", "Analysis complete."),
            "bias": result.get("bias_rating", "Unknown"),
            "flags": result.get("flags", [])
        }

    except Exception as e:
        print(f"Error: {e}")
        return {"score": 0, "color": "Red", "reason": "AI Error. Try again."}
