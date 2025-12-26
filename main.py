from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import json
import base64
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

# --- The Analysis Endpoint (Handles Text AND Images) ---
@app.post("/check")
async def check_trust(text: str = Form(...), image: UploadFile = File(None)):
    print(f"Request Received. Text length: {len(text)}. Image: {image.filename if image else 'None'}")

    try:
        messages = []
        model = "llama-3.1-8b-instant" # Default to fast text model

        # SYSTEM PROMPT
        system_text = """
        You are a strict fact-checker. 
        If Image is provided: Analyze for deepfakes, AI generation errors (hands/eyes), photoshop, or context mismatch.
        If Text is provided: Analyze for bias, logical fallacies, and propaganda.
        
        Return STRICT JSON:
        {
          "trust_score": (int 0-100),
          "reason": (string max 20 words),
          "bias_rating": (string),
          "flags": (list of strings)
        }
        """
        
        # 1. IF IMAGE IS PRESENT -> USE VISION MODEL
        if image:
            model = "llama-3.2-11b-vision-preview" 
            contents = await image.read()
            base64_image = encode_image(contents)
            
            messages = [
                {"role": "system", "content": system_text},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Check this content: {text}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        
        # 2. IF TEXT ONLY -> USE TEXT MODEL
        else:
            messages = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": text}
            ]

        # CALL AI
        completion = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0,
            response_format={"type": "json_object"}
        )

        result = json.loads(completion.choices[0].message.content)
        
        return {
            "score": result.get("trust_score", 0),
            "reason": result.get("reason"),
            "bias": result.get("bias_rating"),
            "flags": result.get("flags", [])
        }

    except Exception as e:
        print(f"Error: {e}")
        return {"score": 0, "reason": "Server Error or Invalid Input", "bias": "Error", "flags": ["Try again"]}
