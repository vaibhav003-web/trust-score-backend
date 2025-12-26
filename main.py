from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import os, json, base64, re
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
def home():
    return FileResponse("index.html")

def encode_image(img):
    return base64.b64encode(img).decode("utf-8")

# -------- HEURISTIC ENGINE (The "Safety Net") ----------
# This runs purely on code, not AI. It catches what AI misses.
def run_heuristics(text, current_flags):
    flags = current_flags.copy()
    text_lower = text.lower()
    
    # 1. Viral / Panic Patterns
    if any(x in text_lower for x in ["forwarded", "share fast", "viral", "whatsapp", "maximum share"]):
        flags.append("Likely viral/forwarded content")
    
    # 2. Time Sensitive / Decay
    if any(x in text_lower for x in ["today", "tomorrow", "breaking", "tonight"]):
        flags.append("Time-sensitive (accuracy decays quickly)")
        
    # 3. Overconfidence / Manipulation
    if any(x in text_lower for x in ["100%", "guaranteed", "proven fact", "official proof"]):
        flags.append("Manipulative/High-confidence language detected")

    return list(set(flags))

# -------- DEFAULT SAFE RESPONSE ----------
def safe_response():
    return {
        "claim_type": "Unknown",
        "verdict": "Unverified",
        "trust_score": 50,
        "confidence_band": "Medium",
        "risk_level": "Medium",
        "explanation": "Analysis inconclusive. Treat as unverified.",
        "bias_rating": "Neutral",
        "flags": [],
        "estimated_sources": []
    }

@app.post("/check")
async def check_trust(text: str = Form(...), image: UploadFile = File(None)):
    
    base = safe_response()
    
    # --- STRICT PROMPT (The "Reasoning Engine") ---
    # This prompts the AI to think like a professional analyst.
    system_prompt = f"""
    You are an OFFLINE credibility assessment AI.
    Current Date: {datetime.now().strftime("%d %B %Y")}
    
    TASK: Analyze the text/image for logical consistency, historical fact, and manipulation.
    
    RULES:
    1. Opinion/Rant → Verdict: Unverified (Score 50). Do not call it Fake.
    2. Rumor/Prediction → Verdict: Unverified.
    3. Breaking News (<24h) → Verdict: Unverified (Admit you have no live search).
    4. VERIFIED requires established historical certainty (e.g. Science, Geography, History).
    5. FALSE requires definitive proof in your training data.
    
    OUTPUT JSON:
    {{
     "claim_type": "Fact|Rumor|Opinion|Satire|Breaking-News",
     "verdict": "Verified|Unverified|False|Misleading",
     "trust_score": (0-100),
     "risk_level": "Low|Medium|High",
     "explanation": "Clear, logic-based reasoning. Avoid generic phrases.",
     "bias_rating": "Neutral|Left|Right|Propaganda",
     "flags": ["list", "any", "manipulation", "tactics"],
     "estimated_sources": ["Source1"] (Only if historically certain)
    }}
    """

    try:
        messages = []
        # Handle Image (Llama 3.2 Vision) or Text (Llama 3.1)
        if image:
            model = "llama-3.2-11b-vision-preview"
            img_bytes = await image.read()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Analyze this image and text for credibility: {text}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_bytes)}"}}
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
            temperature=0.0, # Zero creativity = Maximum logic
            response_format={"type": "json_object"}
        )

        ai = json.loads(completion.choices[0].message.content)

        # ---------- SAFE MERGE ----------
        result = base.copy()
        result.update({k: ai.get(k, base[k]) for k in base})

        # ---------- LOGIC ENFORCEMENT (The "99% Reliability" Layer) ----------

        # 1. Run Python Heuristics (Viral checks)
        result["flags"] = run_heuristics(text, result["flags"])

        # 2. Opinion Safety Rule
        if result["claim_type"] == "Opinion":
            result["verdict"] = "Unverified"
            result["trust_score"] = 50
            result["risk_level"] = "Medium"

        # 3. Breaking News / Rumor Safety Rule
        # If AI guesses "Verified" on breaking news without sources, we force it down.
        if result["claim_type"] in ["Rumor", "Prediction", "Breaking-News"]:
            if result["verdict"] == "Verified" and not result["estimated_sources"]:
                result["verdict"] = "Unverified"
                result["trust_score"] = 45
                result["explanation"] += " (Downgraded: Breaking news requires live verification.)"

        # 4. Source Enforcement
        # If no sources exist, you cannot have a high score.
        if not result["estimated_sources"]:
            result["trust_score"] = min(result["trust_score"], 50)
            if result["verdict"] == "Verified":
                result["verdict"] = "Unverified"
                result["explanation"] += " (Downgraded: No verifiable sources in memory.)"

        # 5. Calculate Confidence Band
        s = result["trust_score"]
        result["confidence_band"] = "High" if s > 65 else ("Low" if s <= 35 else "Medium")

        return JSONResponse(result)

    except Exception as e:
        print("ERROR:", e)
        return JSONResponse(base_response())
