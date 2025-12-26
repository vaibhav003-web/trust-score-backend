from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import json
import base64
from datetime import datetime # <--- NEW: Import Clock
from groq import Groq

# 1. Setup
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

# --- Serve the Website ---
@app.get("/")
async def read_root():
    return FileResponse("index.html")

# --- Helper to process images ---
def encode_image(image_file):
    return base64.b64encode(image_file).decode('utf-8')

# --- The Analysis Endpoint ---
@app.post("/check")
async def check_trust(text: str = Form(...), image: UploadFile = File(None)):
    print(f"Request Received. Text len: {len(text)}")

    try:
        # --- NEW: GET CURRENT DATE ---
        current_date = datetime.now().strftime("%A, %d %B %Y") # e.g., "Thursday, 26 December 2025"
        
        messages = []
        model = "llama-3.1-8b-instant" 

        # --- SYSTEM PROMPT WITH DATE AWARENESS ---
        system_text = f"""
        You are a ruthless, cynical fact-checking AI. 
        CONTEXT: Today's date is {current_date}. 
        Use this date to verify time-sensitive claims (e.g., "Today is Monday").

        RULES:
        1. START WITH A SCORE OF 0. Only give points for verified facts.
        2. IF CLAIM CONTRADICTS DATE: Score 0. Flag as "False Information".
        3. IF VAGUE/UNVERIFIED: Score < 40. Flag as "Unsubstantiated".
        4. IF OPINION/RANT: Score < 30. Flag as "Subjective Opinion".
        5. IF TRUE: You MUST provide the likely SOURCE (e.g., 'Matches reports from BBC, Reuters').
        6. NEVER give 100 unless it is a universal scientific truth (like 'Water is H2O').
        
        Return STRICT JSON format:
        {{
          "trust_score": (int 0-100),
          "reason": (string, max 15 words, very direct),
          "bias_rating": (string: 'Neutral', 'Left-Wing', 'Right-Wing', 'Propaganda'),
          "flags": (list of strings),
          "sources": (list of strings)
        }}
        """
        
        # 1. IMAGE HANDLING
        if image:
            model = "llama-3.2-11b-vision-preview" 
            contents = await image.read()
            base64_image = encode_image(contents)
            
            messages = [
                {"role": "system", "content": system_text},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Strictly audit this image and text: {text}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
        
        # 2. TEXT HANDLING
        else:
            messages = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": text}
            ]

        # CALL AI
        completion = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        result = json.loads(completion.choices[0].message.content)
        
        return {
            "score": result.get("trust_score", 0),
            "reason": result.get("reason"),
            "bias": result.get("bias_rating"),
            "flags": result.get("flags", []),
            "sources": result.get("sources", ["None"])
        }

    except Exception as e:
        print(f"Error: {e}")
        return {"score": 0, "reason": "Server Error", "bias": "Error", "flags": ["Try again"], "sources": []}
