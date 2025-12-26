from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import os, json
from datetime import datetime
from groq import Groq
from tavily import TavilyClient

# ---------------- CONFIG ----------------
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
tavily = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

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

# -------- 1. PYTHON DATE CHECKER ----------
def check_date_claim(text):
    text_lower = text.lower()
    now = datetime.now()
    today_day = now.strftime("%A").lower()
    today_date = now.strftime("%d-%m-%Y")
    
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for day in days:
        if f"today is {day}" in text_lower or f"day is {day}" in text_lower:
            if day == today_day:
                return True, "Verified", 100, f"Correct. Today is {today_day}, {today_date}."
            else:
                return True, "False", 0, f"Incorrect. Today is actually {today_day}, {today_date}."

    if "today date is" in text_lower or "date is" in text_lower:
        if today_date not in text: 
            return True, "False", 0, f"Incorrect. The real system date is {today_date}."
        else:
             return True, "Verified", 100, f"Correct. System date is {today_date}."

    return False, None, 0, None

# -------- 2. VIRAL HEURISTICS ----------
def run_heuristics(text, current_flags):
    flags = current_flags.copy()
    text_lower = text.lower()
    
    if any(x in text_lower for x in ["forwarded", "share fast", "viral", "whatsapp"]):
        flags.append("Likely viral/forwarded content")
    if any(x in text_lower for x in ["100%", "guaranteed", "proven fact"]):
        flags.append("High-confidence/Manipulative language")
        
    return list(set(flags))

# -------- 3. LIVE WEB SEARCH ----------
def search_web(query):
    try:
        response = tavily.search(query=query, search_depth="basic", max_results=3)
        context = []
        sources = []
        for result in response.get('results', []):
            context.append(f"- {result['content']}")
            sources.append(result['url'])
        return "\n".join(context), sources
    except:
        return None, []

# -------- DEFAULT RESPONSE ----------
def safe_response():
    return {
        "claim_type": "Unknown",
        "verdict": "Unverified",
        "trust_score": 50,
        "confidence_band": "Medium",
        "risk_level": "Medium",
        "explanation": "Analysis inconclusive.",
        "bias_rating": "Neutral",
        "flags": [],
        "estimated_sources": []
    }

@app.post("/check")
async def check_trust(text: str = Form(...)):
    
    base = safe_response()
    
    # --- PHASE 1: PYTHON LOGIC CHECK ---
    is_date, d_verdict, d_score, d_expl = check_date_claim(text)
    if is_date:
        return JSONResponse({
            "claim_type": "Fact",
            "verdict": d_verdict,
            "trust_score": d_score,
            "risk_level": "Low" if d_verdict == "Verified" else "High",
            "confidence_band": "High",
            "explanation": d_expl,
            "bias_rating": "Neutral",
            "flags": [],
            "estimated_sources": ["System Calendar"]
        })

    # --- PHASE 2: LIVE SEARCH ---
    web_context, web_sources = search_web(text)

    # --- PHASE 3: AI ANALYSIS ---
    system_prompt = f"""
    You are a Fact-Checking AI.
    Current Date: {datetime.now().strftime("%d %B %Y")}

    WEB CONTEXT:
    {web_context if web_context else "No search results. Rely on logical analysis."}

    RULES:
    1. If Web Context confirms claim -> Verified.
    2. If Web Context debunks claim -> False.
    3. If Web Context is empty -> Unverified.
    
    OUTPUT JSON:
    {{
     "claim_type": "Fact|Rumor|Opinion",
     "verdict": "Verified|Unverified|False",
     "trust_score": (0-100),
     "risk_level": "Low|Medium|High",
     "explanation": "Reasoning based on context.",
     "bias_rating": "Neutral|Left|Right",
     "flags": [],
     "estimated_sources": []
    }}
    """

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        ai = json.loads(completion.choices[0].message.content)
        
        result = base.copy()
        result.update({k: ai.get(k, base[k]) for k in base})

        if web_sources:
            clean_sources = [url.split('/')[2] for url in web_sources]
            result["estimated_sources"] = clean_sources

        result["flags"] = run_heuristics(text, result["flags"])

        # --- SCORE SYNC ---
        verdict = result["verdict"]
        if verdict == "False":
            result["trust_score"] = min(result["trust_score"], 20)
            result["risk_level"] = "High"
        elif verdict == "Verified":
            result["trust_score"] = max(result["trust_score"], 80)
            result["risk_level"] = "Low"
        elif verdict == "Unverified":
            result["trust_score"] = 50
            result["risk_level"] = "Medium"

        return JSONResponse(result)

    except Exception as e:
        print("ERROR:", e)
        return JSONResponse(base_response())
