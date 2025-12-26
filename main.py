from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import json
import base64
from groq import Groq

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

@app.get("/")
async def read_root():
    return FileResponse("index.html")

# Function to encode image to Base64
def encode_image(image_file):
    return base64.b64encode(image_file).decode('utf-8')

@app.post("/check")
async def check_trust(text: str = Form(...), image: UploadFile = File(None)):
    print(f"Analyzing Request... Text: {len(text)} chars. Image: {image.filename if image else 'None'}")

    try:
        messages = []
        model = "llama-3.1-8b-instant" # Default to text model

        # SYSTEM PROMPT
        system_text = """
        You are a strict fact-checker. 
        If Image is provided: Analyze for deepfakes, AI generation, photoshop errors, or mismatch with text.
        If Text is provided: Analyze for bias, logical fallacies, and lack of evidence.
        
        Return STRICT JSON:
        {
          "trust_score": (int 0-100),
          "reason": (string max 20 words),
          "bias_rating": (string),
          "flags": (list of strings)
        }
        """
        
        # 1. HANDLE IMAGE + TEXT
        if image:
            model = "llama-3.2-11b-vision-preview" # Switch to Vision Model
            contents = await image.read()
            base64_image = encode_image(contents)
            
            messages = [
                {"role": "system", "content": system_text},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Analyze this text and image credibility: {text}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        
        # 2. HANDLE TEXT ONLY
        else:
            messages = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": text}
            ]

        # CALL API
        completion = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0,
            response_format={"type": "json_object"}
        )

        result = json.loads(completion.choices[0].message.content)
        
        score = result.get("trust_score", 0)
        return {
            "score": score,
            "reason": result.get("reason"),
            "bias": result.get("bias_rating"),
            "flags": result.get("flags", [])
        }

    except Exception as e:
        print(f"Error: {e}")
        return {"score": 0, "reason": "Error processing request.", "bias": "Error", "flags": ["Server Error"]}
