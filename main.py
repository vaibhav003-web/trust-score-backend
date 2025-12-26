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
    if any(x in text_lower for x in ["breaking", "developing story"]):
        flags.append("Time-sensitive (accuracy decays quickly)")
        
    # 3. Overconfidence
    if any(x in text_lower for x in ["100%", "guaranteed", "proven fact", "official proof"]):
        flags.append("Manipulative/High-confidence language detected")

    return list(set(flags))

# -------- SPECIAL DATE CHECKER (Python Logic) ----------
def check_date_claim(text):
    text_lower = text.lower()
    today = datetime.now()
    current_day = today.strftime("%A").lower() # e.g. "friday"
    
    # Check if user says "Today is [Day]"
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for day in days:
        if f"today is {day}" in text_lower:
            if day == current_day:
                return True, "Verified", "System Calendar"
            else:
                return True, "False", "System Calendar"
    return False, None, None

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
    
    # 1. RUN SPECIAL DATE CHECK FIRST
    is_date_claim, date_verdict, date_source = check_date_claim(text)
    if is_date_claim:
        return JSONResponse({
            "claim_type": "Fact",
            "verdict": date_verdict,
            "trust_score": 100 if date_verdict == "Verified" else 0,
            "risk_level": "Low",
            "confidence_band": "High",
            "explanation": f"Validated against system date: {datetime.now().strftime('%A, %d %B %Y')}.",
            "bias_rating": "Neutral",
            "flags": [],
            "estimated_sources": [date_source]
        })

    # --- STRICT PROMPT (Tweaked for Myths & History) ---
    system_prompt = f"""
    You are an OFFLINE credibility assessment AI.
    Current Date: {datetime.now().strftime("%A, %d %B %Y")}
    
    RULES:
    1. HISTORICAL FACTS (e.g., "TikTok banned in India", "Earth is round") → Verdict: Verified. Source: "Official Records" or "General Knowledge".
    2. COMMON MYTHS (e.g., "Hot water kills COVID") → Verdict: False. Source: "Scientific Consensus".
    3. Opinion/Rant → Verdict: Unverified (Score 50).
    4. Rumor/Prediction → Verdict: Unverified.
    5. Breaking News (<24h) → Verdict: Unverified.
    
    OUTPUT JSON:
    {{
     "claim_type": "Fact|Rumor|Opinion|Satire|Breaking-News",
     "verdict": "Verified|Unverified|False|Misleading",
     "trust_score": (0-100),
     "risk_level": "Low|Medium|High",
     "explanation": "Clear, logic-based reasoning.",
     "bias_rating": "Neutral|Left|Right",
     "flags": ["list", "flags"],
     "estimated_sources": ["Source1"] 
    }}
    """

    try:
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

        # 1. Run Heuristics
        result["flags"] = run_heuristics(text, result["flags"])

        # 2. Breaking News / Rumor Safety Rule
        # We relax this slightly: If AI is VERY confident (Verified) and has a source, we let it pass.
        # We only downgrade if it marks a Rumor as Verified WITHOUT a source.
        if result["claim_type"] in ["Rumor", "Prediction", "Breaking-News"]:
            if result["verdict"] == "Verified" and not result["estimated_sources"]:
                result["verdict"] = "Unverified"
                result["trust_score"] = 45
                result["explanation"] += " (Downgraded: Claims require sources.)"

        # 3. Source Enforcement (Relaxed for False Claims)
        # If something is FALSE (like a myth), we don't strictly require a specific source citation if the AI explains it well.
        # But for VERIFIED facts, we still want a source.
        if result["verdict"] == "Verified" and not result["estimated_sources"]:
             # Last chance: check if it's a known historical fact? 
             # If not, downgrade.
             result["verdict"] = "Unverified"
             result["trust_score"] = 50
             result["explanation"] += " (Downgraded: No source found in offline memory.)"

        # 4. Calculate Confidence Band
        s = result["trust_score"]
        result["confidence_band"] = "High" if s > 65 or s < 20 else ("Low" if s <= 35 else "Medium")

        return JSONResponse(result)

    except Exception as e:
        print("ERROR:", e)
        return JSONResponse(base_response())
