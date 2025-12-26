from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import os, json, base64
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
def home():
    return FileResponse("index.html")

def encode_image(img):
    return base64.b64encode(img).decode("utf-8")

# -------- DEFAULT SAFE RESPONSE ----------
def base_response():
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

@app.post("/check")
async def check_trust(text: str = Form(...), image: UploadFile = File(None)):

    result = base_response()

    system_prompt = """
You are an OFFLINE credibility assessment AI.
You DO NOT browse the internet.

Rules:
- Opinion → Unverified (not Fake)
- Rumor / Prediction → Unverified
- Breaking News → Unverified
- False → only if clearly debunked historically
- Verified → only for well-known historical facts

Return STRICT JSON with ALL fields:

{
 "claim_type": "",
 "verdict": "",
 "trust_score": 0,
 "risk_level": "",
 "explanation": "",
 "bias_rating": "",
 "flags": [],
 "estimated_sources": []
}
"""

    try:
        # ---------- IMAGE OR TEXT ----------
        if image:
            model = "llama-3.2-11b-vision-preview"
            img_bytes = await image.read()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Assess credibility: {text}"},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_bytes)}"}}
                ]}
            ]
        else:
            model = "llama-3.1-8b-instant"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ]

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        ai = json.loads(completion.choices[0].message.content)

        # ---------- SAFE MERGE ----------
        for key in result:
            if key in ai and ai[key] not in [None, ""]:
                result[key] = ai[key]

        # ---------- LOGIC ENFORCEMENT ----------

        # Opinion rule
        if result["claim_type"] == "Opinion":
            result["verdict"] = "Unverified"
            result["trust_score"] = 50
            result["risk_level"] = "Medium"

        # Rumor / Prediction rule
        if result["claim_type"] in ["Rumor", "Prediction", "Breaking-News"]:
            result["verdict"] = "Unverified"
            result["trust_score"] = min(result["trust_score"], 50)
            result["risk_level"] = "Medium"

        # Verified requires sources
        if not result["estimated_sources"]:
            result["trust_score"] = min(result["trust_score"], 50)
            if result["verdict"] == "Verified":
                result["verdict"] = "Unverified"

        # Confidence band
        score = result["trust_score"]
        if score <= 30:
            result["confidence_band"] = "Low"
        elif score <= 60:
            result["confidence_band"] = "Medium"
        else:
            result["confidence_band"] = "High"

        # Risk fallback
        if not result["risk_level"]:
            result["risk_level"] = "Medium"

        return JSONResponse(result)

    except Exception as e:
        print("ERROR:", e)
        return JSONResponse(base_response())
