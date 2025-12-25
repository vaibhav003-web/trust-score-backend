from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
from groq import Groq

# --- CONFIG ---
# We now look for the GROQ_API_KEY
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("CRITICAL: GROQ_API_KEY is missing.")

# Initialize the Groq Client
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
    print(f"\n--- ANALYZING (VIA GROQ): {request.text[:30]}... ---")
    
    # SYSTEM PROMPT (Strict JSON)
    system_prompt = """
    You are a fact-checking and credibility analysis system.
    Analyze the submitted text.
    Return ONLY a valid JSON object in this exact format:
    {
      "trust_score": number (0-100), 
      "reason": "short explanation (max 15 words)"
    }
    """

    try:
        # Send request to Llama 3 on Groq
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": request.text,
                }
            ],
            # We use Llama 3 8B (Fast & Smart)
            model="llama3-8b-8192",
            temperature=0, # Keep it strict
            response_format={"type": "json_object"} # Force JSON mode
        )

        # Parse Response
        ai_response = chat_completion.choices[0].message.content
        result = json.loads(ai_response)
        
        score = result.get("trust_score", 0)
        
        # Determine Color
        color = "Red"
        if score >= 80: color = "Green"
        elif score >= 50: color = "Yellow"

        print(f"✅ SUCCESS: Score {score} - {color}")
        return {
            "score": score,
            "color": color,
            "reason": result.get("reason", "Analysis complete.")
        }

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return {"score": 0, "color": "Red", "reason": "AI Error. Try again."}
