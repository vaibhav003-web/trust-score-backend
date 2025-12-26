from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import json
import base64
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

# --- SERVE UI ---
@app.get("/")
async def read_root():
    return FileResponse("index.html")

# --- IMAGE ENCODER ---
def encode_image(image_file):
    return base64.b64encode(image_file).decode('utf-8')

# --- MAIN ENGINE ---
@app.post("/check")
async def check_trust(text: str = Form(...), image: UploadFile = File(None)):
    print(f"Request: {len(text)} chars | Image: {image.filename if image else 'None'}")

    try:
        # 1. GET CONTEXT
        current_date = datetime.now().strftime("%A, %d %B %Y")
        
        # 2. DEFINE THE AI PERSONA (CREDIBILITY ANALYST)
        # This prompt forces the AI to analyze STRUCTURE and PROOF, not just guess.
        system_text = f"""
        You are a Credibility Intelligence System (CIS). Your job is to analyze the structure, verifiability, and risk of a claim.
        
        CURRENT CONTEXT:
        - Today's Date: {current_date}
        - You do NOT have live internet access. You cannot verify breaking news from the last 24 hours.
        
        ### STEP 1: CLASSIFY THE INPUT
        Determine the 'claim_type':
        - 'Fact': A specific, verifiable statement about the world (e.g., "Water boils at 100C").
        - 'Rumor': Unverified information circulating without sources (e.g., "I heard the PM is resigning").
        - 'Opinion': Subjective views, rants, or preferences (e.g., "This policy is terrible").
        - 'Satire': Humor/Parody not meant to be taken literally.
        - 'Prediction': Guesses about the future.
        - 'Breaking-News': Time-sensitive claims about recent events.

        ### STEP 2: DETERMINE VERDICT (NO GUESSING)
        - 'Verified': You are 100% certain this is an established fact found in your training data.
        - 'False': You are 100% certain this is proven incorrect/debunked.
        - 'Unverified': (DEFAULT) You cannot find proof in your training data, or it is too recent/obscure. 
        - 'Misleading': Technically true but manipulated context.

        ### STEP 3: LOGIC-BASED SCORING (0-100)
        Start at 50. Apply modifiers:
        - (+20) Mentions specific, verifiable data/dates/locations.
        - (+20) Neutral, professional tone.
        - (-30) Uses emotional/trigger words ("Shocking", "Betrayal", "You won't believe").
        - (-30) "Forwarded many times" style or ALL CAPS.
        - (-40) Anonymous/Social Media sourcing ("A friend told me").
        
        ### STEP 4: SOURCE PROTOCOL
        - IF Verdict is 'Verified': You MUST list the established entity (e.g., 'NASA', 'Reuters', 'Official History').
        - IF Verdict is 'Unverified'/'Rumor': verified_sources MUST be empty []. DO NOT HALLUCINATE.

        ### OUTPUT FORMAT (STRICT JSON):
        {{
          "claim_type": "Fact | Rumor | Opinion | Satire | Prediction | Breaking-News",
          "trust_score": (int 0-100),
          "verdict": "Verified | Unverified | False | Misleading",
          "explanation": "Short, objective analysis of WHY you gave this score.",
          "risk_level": "Low | Medium | High",
          "flags": ["list", "of", "risk", "indicators"],
          "verified_sources": ["Only if 100% certain"],
          "confidence_note": "Explain strictly based on logic (e.g., 'High emotional bias detected' or 'Established scientific fact')"
        }}
        """

        messages = []
        model = "llama-3.1-8b-instant" 

        # 3. BUILD REQUEST
        if image:
            model = "llama-3.2-11b-vision-preview" 
            contents = await image.read()
            base64_image = encode_image(contents)
            messages = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Analyze this content for credibility: {text}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ]
        else:
            messages = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": text}
            ]

        # 4. EXECUTE ANALYSIS
        completion = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0.1, # Keep strictly logical
            response_format={"type": "json_object"}
        )

        result = json.loads(completion.choices[0].message.content)
        
        # 5. SAFEGUARD DEFAULTS
        return {
            "claim_type": result.get("claim_type", "Unknown"),
            "score": result.get("trust_score", 50),
            "verdict": result.get("verdict", "Unverified"),
            "reason": result.get("explanation", "Analysis complete."), # Mapping 'explanation' to 'reason' for frontend compat
            "risk_level": result.get("risk_level", "Medium"),
            "flags": result.get("flags", []),
            "sources": result.get("verified_sources", [])
        }

    except Exception as e:
        print(f"Error: {e}")
        return {
            "score": 0, 
            "verdict": "Error", 
            "reason": "System overloaded. Please try again.", 
            "flags": ["Server Error"], 
            "sources": []
        }
