import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
import base64
import io
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from gtts import gTTS

from agent_graph import graph


logger = logging.getLogger("voice-proxy")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="voice-proxy-langgraph-merged")

# CORS (local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Stop commands
STOP_COMMANDS = {
    "en": ["stop", "stop speaking", "ok stop", "stop now", "please stop", "that's enough"],
    "hi": ["रुको", "बोलना बंद करो", "रुकिए", "बंद करो"],
    "te": ["ఆపు", "మాట్లాడటం ఆపు", "ఆపండి", "ఆపు ఇప్పుడు"],
    "ta": ["நிறுத்து", "பேசுவதை நிறுத்து", "நிறுத்துங்கள்"],
    "gu": ["રોકો", "બોલવું બંધ કરો", "બંધ કરો"],
}


def LANG_CODE_FROM_LABEL(label: str) -> str:
    lookup = {
        "English": "en",
        "Hindi": "hi",
        "Telugu": "te",
        "Tamil": "ta",
        "Gujarati": "gu",
        "en": "en",
        "hi": "hi",
        "te": "te",
        "ta": "ta",
        "gu": "gu",
    }
    return lookup.get(label, "en")


def GTTS_LANG_FROM_LABEL(label: str) -> str:
    cfg = {
        "English": "en",
        "Hindi": "hi",
        "Telugu": "te",
        "Tamil": "ta",
        "Gujarati": "gu",
    }
    return cfg.get(label, "en")


def detect_stop_phrase(text: str, lang_label: str) -> bool:
    if not text:
        return False
    lang_code = LANG_CODE_FROM_LABEL(lang_label)
    stop_list = STOP_COMMANDS.get(lang_code, [])
    return text.strip().lower() in [s.lower() for s in stop_list]


def shorten_text(text: str, max_chars: int = 350) -> str:
    if not text:
        return ""
    t = " ".join(text.split()).strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rsplit(" ", 1)[0] + "..."


async def tts_synthesize_mp3_gtts(text: str, lang: str = "en") -> Optional[str]:
    """
    Generates MP3 using gTTS and returns base64 string.
    """
    try:
        if not text.strip():
            return None

        mp3_fp = io.BytesIO()
        tts = gTTS(text=text, lang=lang)
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)

        return base64.b64encode(mp3_fp.read()).decode("utf-8")
    except Exception:
        logger.exception("gTTS failed")
        return None


@app.post("/voice")
async def voice_endpoint(payload: dict):
    """
    Expected payload:
      {
        "text": "I have fever",
        "lang": "English",
        "messages": [{"role":"user","content":"..."}, ...]   (optional)
      }

    Returns:
      {"reply":"...", "audio":"<base64 mp3>"} or {"reply":"__STOP__"}
    """
    if not payload:
        raise HTTPException(status_code=400, detail="Missing JSON body")

    text = str(payload.get("text") or "").strip()
    lang_label = payload.get("lang") or "English"
    messages = payload.get("messages") or []

    if text == "":
        return {"reply": "", "audio": None}

    if detect_stop_phrase(text, lang_label):
        return {"reply": "__STOP__", "audio": None}

    # Build memory + new user message
    full_messages = []
    if isinstance(messages, list):
        full_messages.extend(messages)

    full_messages.append({"role": "user", "content": text})

    # Call LangGraph agent
    try:
        result = graph.invoke({"messages": full_messages})
        reply = (result.get("reply") or "").strip()
    except Exception as e:
        reply = f"Sorry, I am not able to answer right now. Please try again. ({e})"

    reply = shorten_text(reply)

    # Convert reply to MP3 audio (gTTS)
    gtts_lang = GTTS_LANG_FROM_LABEL(lang_label)
    audio_b64 = await tts_synthesize_mp3_gtts(reply, lang=gtts_lang)

    return {"reply": reply, "audio": audio_b64}
from fastapi import UploadFile, File, Form

@app.post("/vision")
async def vision_endpoint(
    image: UploadFile = File(...),
    question: str = Form(""),
    lang: str = Form("English")
):
    try:
        img_bytes = await image.read()

        result = graph.invoke({
            "messages": [
                {"role": "user", "content": f"{question}\n\nAnalyze the uploaded image and give safe medical advice."}
            ],
            "image_bytes": img_bytes
        })

        reply = (result.get("reply") or "").strip()

        if not reply:
            reply = "Sorry, I could not analyze the image. Please try again."

        return {"reply": reply}

    except Exception as e:
        return {"reply": f"Vision error: {e}"}

@app.get("/health")
async def health():
    return {"status": "ok", "merged": True}
