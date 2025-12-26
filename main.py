from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import os, json
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

# -------- SAFETY HEURISTICS ----------
def run_heuristics(text, current_flags):
    flags = current_flags.copy()
    text_lower = text.lower()
    
    # 1. Viral / Panic Patterns
    if any(x in text_lower for x in ["forwarded", "share fast", "viral", "whatsapp", "maximum share"]):
        flags.append("Likely viral/forwarded content")
    
    # 2. Time Sensitive
    if any(x in text_lower for x in ["today", "tomorrow", "breaking", "tonight"]):
        flags.append("Time-sensitive (accuracy decays quickly)")
        
    # 3. Overconfidence
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
async def check_trust(text: str = Form(...)):
    
    base = safe_response()
    
    # --- STRICT PROMPT ---
    system_prompt = f"""
    You are an OFFLINE credibility assessment AI.
    Current Date: {datetime.now().strftime("%d %B %Y")}
    
    TASK: Analyze the text for logical consistency, historical fact, and manipulation.
    
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
     "explanation": "Clear, logic-based reasoning.",
     "bias_rating": "Neutral|Left|Right|Propaganda",
     "flags": ["list", "any", "manipulation"],
     "estimated_sources": ["Source1"] (Only if historically certain)
    }}
    """

    try:
        # Text Only Model
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
        result = base.copy()
        result.update({k: ai.get(k, base[k]) for k in base})

        # ---------- LOGIC ENFORCEMENT ----------

        # 1. Run Python Heuristics
        result["flags"] = run_heuristics(text, result["flags"])

        # 2. Opinion Safety Rule
        if result["claim_type"] == "Opinion":
            result["verdict"] = "Unverified"
            result["trust_score"] = 50
            result["risk_level"] = "Medium"

        # 3. Breaking News / Rumor Safety Rule
        if result["claim_type"] in ["Rumor", "Prediction", "Breaking-News"]:
            if result["verdict"] == "Verified" and not result["estimated_sources"]:
                result["verdict"] = "Unverified"
                result["trust_score"] = 45
                result["explanation"] += " (Downgraded: Breaking news requires live verification.)"

        # 4. Source Enforcement
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
