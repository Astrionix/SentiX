from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
import json

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")

client = Groq(api_key=GROQ_API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str


class AnalyzeResponse(BaseModel):
    sentiment: str
    confidence: float


def build_prompt(text: str) -> str:
    return f"""
You are a sentiment analysis engine for code-mixed text (e.g., English + Hindi).

Analyze the sentiment of the following message and respond ONLY with a STRICT JSON object.

Message:
{text}

Your response MUST be valid JSON with this exact structure:

{{
  "sentiment": "positive" | "negative" | "neutral",
  "confidence": float between 0 and 1
}}

Rules:
- Do NOT include any explanation.
- Do NOT include any extra keys.
- Do NOT include backticks or markdown.
- Make sure it's valid JSON.
"""


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_sentiment(payload: AnalyzeRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    prompt = build_prompt(payload.text)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        # Groq client returns a ChatCompletion with choices, each having a message object.
        # Access the content via the attribute, not as a dict.
        result = response.choices[0].message.content

        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500,
                detail="Model did not return valid JSON.",
            )

        sentiment = str(data.get("sentiment", "")).lower()
        if sentiment not in {"positive", "negative", "neutral"}:
            raise HTTPException(
                status_code=500,
                detail="Invalid sentiment value returned by model.",
            )

        confidence_raw = data.get("confidence", 0.0)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0

        confidence = max(0.0, min(1.0, confidence))

        return AnalyzeResponse(sentiment=sentiment, confidence=confidence)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling Groq API: {e}")
