from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os, json, base64
from datetime import datetime
from groq import Groq

# ---------------- CONFIG ----------------
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return FileResponse("index.html")

def encode_image(img):
    return base64.b64encode(img).decode("utf-8")

# -------- DEFAULT SAFE RESPONSE ----------
def safe_response():
    return {
        "claim_type": "Unknown",
        "verdict": "Unverified",
        "trust_score": 50,
        "confidence_band": "Medium",
        "risk_level": "Medium",
        "explanation": "No reliable confirmation available.",
        "bias_rating": "Neutral",
        "flags": [],
        "estimated_sources": []
    }

# ----------------------------------------
@app.post("/check")
async def check_trust(text: str = Form(...), image: UploadFile = File(None)):

    base = safe_response()
    today = datetime.now().strftime("%d %B %Y")

    system_prompt = f"""
You are an OFFLINE credibility assessment engine.
You do NOT browse the internet.

TASK:
Classify and assess credibility.

Rules:
- Opinion → Unverified (NOT Fake)
- Rumor / Prediction → Unverified
- Breaking-News → Unverified
- VERIFIED requires strong historical certainty
- If unsure → Unverified

Return STRICT JSON with ALL fields filled.

JSON FORMAT:
{{
 "claim_type": "",
 "verdict": "",
 "trust_score": 0,
 "risk_level": "",
 "explanation": "",
 "bias_rating": "",
 "flags": [],
 "estimated_sources": []
}}
"""

    # ---- 1. HANDLE IMAGE VS TEXT PROMPT ----
    messages = []
    
    if image:
        # If image exists, use Llama 3.2 Vision
        model = "llama-3.2-11b-vision-preview"
        contents = await image.read()
        base64_image = encode_image(contents)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": f"Analyze this image and text: {text}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ]
    else:
        # Text only
        model = "llama-3.1-8b-instant"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        ai = json.loads(completion.choices[0].message.content)

        # ---- SAFE MERGE ----
        # Start with base defaults, overwrite with AI data only if valid
        result = base.copy()
        result.update({k: ai.get(k, base[k]) for k in base})

        # ---- SCORE SAFETY ----
        if not result["estimated_sources"]:
            result["trust_score"] = min(result["trust_score"], 50)
            if result["verdict"] == "Verified":
                result["verdict"] = "Unverified"
                result["explanation"] += " (Downgraded: No sources found in memory.)"

        # ---- OPINION RULE ----
        if result["claim_type"] == "Opinion":
            result["verdict"] = "Unverified"
            result["trust_score"] = 50
            result["risk_level"] = "Medium"

        # ---- CONFIDENCE BAND ----
        s = result["trust_score"]
        if s <= 30:
            result["confidence_band"] = "Low"
        elif s <= 60:
            result["confidence_band"] = "Medium"
        else:
            result["confidence_band"] = "High"

        # ---- FINAL GUARANTEE ----
        # Ensure no None values exist
        for key in base:
            if result.get(key) is None:
                result[key] = base[key]

        return result

    except Exception as e:
        print(f"Error: {e}")
        return safe_response()
