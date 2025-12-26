from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import os, json
from datetime import datetime
from groq import Groq
from tavily import TavilyClient

# ---------------- CONFIG ----------------
# You need BOTH keys now
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

# -------- 1. LIVE WEB SEARCH ----------
def search_web(query):
    try:
        # Search for context (limited to 3 results for speed)
        response = tavily.search(query=query, search_depth="basic", max_results=3)
        context = []
        sources = []
        for result in response.get('results', []):
            context.append(f"- {result['content']}")
            sources.append(result['url'])
        return "\n".join(context), sources
    except Exception as e:
        print(f"Search Error: {e}")
        return None, []

# -------- DEFAULT RESPONSE ----------
def safe_response():
    return {
        "claim_type": "Unknown",
        "verdict": "Unverified",
        "trust_score": 50,
        "confidence_band": "Medium",
        "risk_level": "Medium",
        "explanation": "Could not verify claim at this time.",
        "bias_rating": "Neutral",
        "flags": [],
        "estimated_sources": []
    }

@app.post("/check")
async def check_trust(text: str = Form(...)):
    
    base = safe_response()
    current_date = datetime.now().strftime("%d %B %Y")

    # STEP 1: PERFORM LIVE SEARCH
    # We search the web for the user's text to get the latest facts.
    print(f"Searching web for: {text[:50]}...")
    web_context, web_sources = search_web(text)

    # STEP 2: FEED CONTEXT TO AI
    system_prompt = f"""
    You are a Fact-Checking AI with LIVE INTERNET ACCESS.
    Current Date: {current_date}

    CONTEXT FROM WEB SEARCH:
    {web_context if web_context else "No search results found. Rely on internal memory."}

    TASK:
    Analyze the user's claim based on the WEB CONTEXT above.

    RULES:
    1. If the Web Context confirms the claim -> Verdict: Verified.
    2. If the Web Context debunks the claim -> Verdict: False.
    3. If the Web Context is empty/unclear -> Verdict: Unverified.
    4. CITE SOURCES: Use the domain names from the context (e.g. bbc.com, reuters.com).

    OUTPUT JSON:
    {{
     "claim_type": "Fact|Rumor|Opinion|Satire|Breaking-News",
     "verdict": "Verified|Unverified|False|Misleading",
     "trust_score": (0-100),
     "risk_level": "Low|Medium|High",
     "explanation": "Explain using the search results.",
     "bias_rating": "Neutral|Left|Right",
     "flags": ["list", "flags"],
     "estimated_sources": ["List of sources found"]
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

        # STEP 3: MERGE & FORMAT
        result = base.copy()
        result.update({k: ai.get(k, base[k]) for k in base})

        # Override sources with REAL links if available
        if web_sources:
            # Just show domain names to keep it clean, or full URLs
            clean_sources = [url.split('/')[2] for url in web_sources]
            result["estimated_sources"] = clean_sources

        # Confidence Calculation
        if web_context:
            result["confidence_band"] = "High" # We have real data!
        else:
            result["confidence_band"] = "Low"

        return JSONResponse(result)

    except Exception as e:
        print("ERROR:", e)
        return JSONResponse(base_response())
