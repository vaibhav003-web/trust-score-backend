from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
from groq import Groq

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
    print(f"Analyzing: {request.text[:50]}...")

    # UPGRADED SYSTEM PROMPT
    # We now ask for 'bias_rating' and 'flags' in the JSON
    system_prompt = """
    You are an expert fact-checker. Analyze the text for credibility, tone, and logic.
    Return a STRICT JSON object with these fields:
    {
      "trust_score": (integer 0-100),
      "reason": (string, max 15 words summary),
      "bias_rating": (string: "Neutral", "Slight Bias", or "Highly Biased"),
      "flags": (list of strings, e.g. ["Sensationalism", "Political Propaganda", "No Sources"])
    }
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.text}
            ],
            model="llama-3.1-8b-instant",
            temperature=0,
            response_format={"type": "json_object"}
        )

        result = json.loads(chat_completion.choices[0].message.content)
        
        # Color Logic
        score = result.get("trust_score", 0)
        color = "Red"
        if score >= 80: color = "Green"
        elif score >= 50: color = "Yellow"

        return {
            "score": score,
            "color": color,
            "reason": result.get("reason"),
            "bias": result.get("bias_rating", "Unknown"),
            "flags": result.get("flags", [])
        }

    except Exception as e:
        print(f"Error: {e}")
        return {"score": 0, "color": "Red", "reason": "AI Error"}
