from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import json
import base64
import re
from datetime import datetime
from groq import Groq

# --- CONFIGURATION ---
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

@app.get("/")
async def read_root():
    return FileResponse("index.html")

def encode_image(image_file):
    return base64.b64encode(image_file).decode('utf-8')

# --- HEURISTIC ENGINE (Rule-Based Checks) ---
def run_safety_checks(text: str, current_flags: list, verdict: str) -> list:
    text_lower = text.lower()
    new_flags = []

    # 1. Virality / Forwarded Detection
    viral_keywords = ["forward", "share fast", "share this", "breaking news", "viral", "maximum share"]
    if any(k in text_lower for k in viral_keywords):
        new_flags.append("Likely viral or forwarded content")
    
    # Check for ALL CAPS (if text is long enough)
    if len(text) > 20 and sum(1 for c in text if c.isupper()) / len(text) > 0.6:
        new_flags.append("Excessive capitalization (Viral Pattern)")

    # 2. Overconfidence Detection
    absolute_words = ["guaranteed", "100%", "confirmed", "official proof", "proven"]
    if verdict == "Unverified" and any(w in text_lower for w in absolute_words):
        new_flags.append("Overconfident language without verification")

    # 3. Time-Sensitive Detection
    time_words = ["today", "tomorrow", "tonight", "yesterday", "this week"]
    if verdict == "Unverified" and any(w in text_lower for w in time_words):
        new_flags.append("Time-sensitive claim — accuracy decays quickly")

    # 4. Multi-Claim Detection (Simple Sentence Count)
    # Rough check: if inputs have multiple distinct sentences, risk increases.
    sentences = [s for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 10]
    if len(sentences) > 3:
        new_flags.append("Multiple claims detected — analysis may be incomplete")

    return list(set(current_flags + new_flags))

@app.post("/check")
async def check_trust(text: str = Form(...), image: UploadFile = File(None)):
    print(f"Request: {len(text)} chars | Image: {image.filename if image else 'None'}")

    try:
        current_date = datetime.now().strftime("%A, %d %B %Y")
        
        # --- SYSTEM PROMPT ---
        system_text = f"""
        You are a Credibility Assessment Tool (Offline Mode). 
        Current Date: {current_date}.
        
        GOAL: Estimate credibility based on INTERNAL KNOWLEDGE only. Do not hallucinate external search results.

        ### CLASSIFICATION
        - 'Fact', 'Rumor', 'Opinion', 'Satire', 'Breaking-News'.

        ### VERDICT RULES
        1. 'History Match': Confirmed by your internal training data (e.g., established science/history).
        2. 'No Match': Not found in training data (Rumors, Recent events). DEFAULT.
        3. 'False': Contradicts internal training data.

        ### SCORING
        - Established Fact: 90-100
        - Unverified / No Match: 40-50
        - Proven False: 0-20

        ### OUTPUT JSON:
        {{
          "claim_type": "string",
          "trust_score": (int 0-100),
          "verdict": "History Match | No Match | False | Misleading",
          "explanation": "Brief logic. If 'No Match', admit you cannot verify recent events.",
          "bias_rating": "Neutral | Left | Right | Propaganda",
          "flags": ["list", "of", "content", "risks"],
          "estimated_sources": ["Source 1"] (Only if 'History Match')
        }}
        """

        messages = []
        model = "llama-3.1-8b-instant" 

        if image:
            model = "llama-3.2-11b-vision-preview" 
            contents = await image.read()
            base64_image = encode_image(contents)
            messages = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Assess: {text}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ]
        else:
            messages = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": text}
            ]

        completion = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        result = json.loads(completion.choices[0].message.content)

        # --- PYTHON LOGIC ENFORCEMENT ---
        
        # 1. Extract & Sanitize
        score = result.get("trust_score", 50)
        verdict = result.get("verdict", "No Match")
        explanation = result.get("explanation", "")
        flags = result.get("flags", [])

        # 2. Run Heuristics (Viral, Time, etc.)
        flags = run_safety_checks(text, flags, verdict)

        # 3. Fail-Safe: Explanation too short?
        if len(explanation.split()) < 5:
            verdict = "No Match"
            score = 50
            explanation = "Analysis inconclusive due to insufficient data."
            flags.append("Automated Fail-Safe Triggered")

        # 4. Calculate Confidence Band
        confidence_band = "Medium"
        if score <= 30: confidence_band = "Low"
        elif score >= 61: confidence_band = "High"

        # 5. Offline Mode Enforcer
        verification_mode = "Offline"
        
        # 6. Breaking News / Rumor Safety Cap
        if result.get("claim_type") in ["Breaking-News", "Rumor"] and verdict == "History Match":
             # If AI claims to know breaking news (hallucination risk), downgrade it
             if not result.get("estimated_sources"):
                 verdict = "No Match"
                 score = 45
                 explanation += " (Downgraded: Recent events cannot be verified offline.)"

        # 7. Final JSON Assembly
        final_response = {
            "claim_type": result.get("claim_type", "Unknown"),
            "trust_score": score,
            "confidence_band": confidence_band,
            "verdict": verdict,
            "verification_mode": verification_mode,
            "explanation": explanation,
            "bias_rating": result.get("bias_rating", "Neutral"),
            "flags": flags,
            "estimated_sources": result.get("estimated_sources", [])
        }
        
        return final_response

    except Exception as e:
        print(f"Error: {e}")
        return {
            "trust_score": 50,
            "confidence_band": "Low",
            "verdict": "System Error",
            "explanation": "Please try again later.",
            "flags": ["Server Error"],
            "estimated_sources": []
        }
