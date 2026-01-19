# voice_proxy.py
# Multilingual Voice Proxy using GEMINI + gTTS (FREE audio)
#
# Supports: English, Hindi, Telugu, Tamil, Gujarati
#
# Endpoints:
#  - POST /voice -> {"text":"...", "lang":"Hindi"} -> {"reply":"...", "audio":"<base64_mp3>"}
#  - GET  /health
#
# Run local:
#   uvicorn voice_proxy:app --host 127.0.0.1 --port 8765

import os
import re
import io
import time
import base64
import logging
from typing import Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from gtts import gTTS

# ✅ Gemini NEW SDK
from google import genai

logger = logging.getLogger("voice-proxy")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="voice-proxy-gemini-multilingual-tts")

# -------------------------
# CORS (Streamlit / Website / Any)
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (easy for Streamlit + web)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Gemini setup
# -------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not GEMINI_API_KEY:
    raise RuntimeError("❌ Missing GEMINI_API_KEY environment variable.")

# ✅ Your requested model
# ⚠️ gemini-2.5-flash sometimes overloaded, so we keep fallback too
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-1.5-flash")

client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_INSTRUCTION = (
    "You are a medical assistant. "
    "Always reply in 2–3 short lines only. "
    "Be simple, safe, non-diagnostic, actionable. "
    "Avoid long paragraphs. "
    "If emergency symptoms, suggest doctor visit."
)

STOP_COMMANDS = {
    "en": ["stop", "stop speaking", "ok stop", "stop now", "please stop", "that's enough"],
    "hi": ["रुको", "बोलना बंद करो", "रुकिए", "बंद करो"],
    "te": ["ఆపు", "మాట్లాడటం ఆపు", "ఆపండి", "ఆపు ఇప్పుడు"],
    "ta": ["நிறுத்து", "பேசுவதை நிறுத்து", "நிறுத்துங்கள்"],
    "gu": ["રોકો", "બોલવું બંધ કરો", "બંધ કરો"],
}

# -------------------------
# Helpers
# -------------------------
def shorten_text_to_sentences(text: str, max_sentences: int = 2) -> str:
    """Keep answer short for better TTS + UI."""
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text).strip()
    if not t:
        return ""

    sents = re.split(r"(?<=[.!?])\s+", t)
    if len(sents) >= max_sentences:
        return " ".join(sents[:max_sentences]).strip()

    return t if len(t) <= 350 else t[:347].rsplit(" ", 1)[0] + "..."


def LANG_CODE_FROM_LABEL(label: str) -> str:
    """Frontend label -> language code"""
    if not label:
        return "en"
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
    return lookup.get(str(label).strip(), "en")


def GTTS_LANG_FROM_LABEL(label: str) -> str:
    """Frontend label -> gTTS language code"""
    cfg = {
        "English": "en",
        "Hindi": "hi",
        "Telugu": "te",
        "Tamil": "ta",
        "Gujarati": "gu",
    }
    return cfg.get(str(label).strip(), "en")


def detect_stop_phrase(text: str, lang_label: str) -> bool:
    if not text:
        return False
    lang_code = LANG_CODE_FROM_LABEL(lang_label)
    stop_list = STOP_COMMANDS.get(lang_code, [])
    txt = text.strip().lower()
    return txt in [s.lower() for s in stop_list]


def language_instruction_from_code(lang_code: str) -> str:
    return {
        "en": "Reply strictly in English.",
        "hi": "Reply strictly in Hindi (हिंदी).",
        "te": "Reply strictly in Telugu (తెలుగు).",
        "ta": "Reply strictly in Tamil (தமிழ்).",
        "gu": "Reply strictly in Gujarati (ગુજરાતી).",
    }.get(lang_code, "Reply strictly in English.")


# -------------------------
# Gemini generation (with retry + fallback model)
# -------------------------
async def call_gemini_generate(user_text: str, lang_label: str = "English") -> Tuple[int, str]:
    """
    Calls Gemini and returns (status_code, reply_text)
    - Retries on overload
    - Falls back to gemini-1.5-flash if 2.5-flash fails
    """
    lang_code = LANG_CODE_FROM_LABEL(lang_label)

    prompt = (
        f"{SYSTEM_INSTRUCTION}\n"
        f"{language_instruction_from_code(lang_code)}\n\n"
        f"User question: {user_text}\n\n"
        f"Answer:"
    )

    # 1) Try main model with retries
    last_error = ""
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            text = (resp.text or "").strip()
            if text:
                return 200, text
            last_error = "Empty response from Gemini"
        except Exception as e:
            last_error = str(e)
            logger.warning(f"[Gemini main model failed] attempt={attempt+1} error={last_error}")
            time.sleep(2)

    # 2) Fallback model (more stable)
    for attempt in range(2):
        try:
            resp = client.models.generate_content(
                model=FALLBACK_MODEL,
                contents=prompt,
            )
            text = (resp.text or "").strip()
            if text:
                return 200, text
            last_error = "Empty response from Gemini fallback"
        except Exception as e:
            last_error = str(e)
            logger.warning(f"[Gemini fallback failed] attempt={attempt+1} error={last_error}")
            time.sleep(2)

    return 500, last_error


# -------------------------
# gTTS audio (FREE)
# -------------------------
async def tts_synthesize_mp3_gtts(text: str, lang: str = "en") -> Tuple[int, Optional[str]]:
    """
    Generates MP3 using gTTS and returns base64 string.
    """
    try:
        if not text or not text.strip():
            return 200, None

        mp3_fp = io.BytesIO()
        tts = gTTS(text=text, lang=lang)
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)

        audio_b64 = base64.b64encode(mp3_fp.read()).decode("utf-8")
        return 200, audio_b64

    except Exception as e:
        logger.exception("gTTS failed")
        return 500, str(e)


# -------------------------
# Endpoints
# -------------------------
@app.post("/voice")
async def voice_endpoint(payload: dict):
    """
    Payload:
      {"text":"...", "lang":"Telugu"}
    Returns:
      {"reply":"...", "audio":"<base64_mp3>"} or {"reply":"__STOP__"}
    """
    if not payload:
        raise HTTPException(status_code=400, detail="Missing JSON body")

    text = str(payload.get("text") or payload.get("prompt") or "").strip()
    lang_label = str(payload.get("lang") or payload.get("language") or "English").strip()

    if text == "":
        return {"reply": "", "audio": None}

    # stop command
    if detect_stop_phrase(text, lang_label):
        return {"reply": "__STOP__", "audio": None}

    # Gemini reply
    status, gen_text = await call_gemini_generate(text, lang_label=lang_label)

    # If Gemini fails -> still return fallback reply + voice
    if status != 200 or not gen_text.strip():
        fallback_reply = "Sorry, AI is busy right now. Please try again in 10 seconds."
        gtts_lang = GTTS_LANG_FROM_LABEL(lang_label)
        _, audio_b64 = await tts_synthesize_mp3_gtts(fallback_reply, lang=gtts_lang)
        return {"reply": fallback_reply, "audio": audio_b64, "error": f"GEMINI_{status}", "detail": gen_text}

    short_reply = shorten_text_to_sentences(gen_text, max_sentences=2)

    # gTTS voice in same language
    gtts_lang = GTTS_LANG_FROM_LABEL(lang_label)
    tts_status, audio_b64_or_err = await tts_synthesize_mp3_gtts(short_reply, lang=gtts_lang)

    if tts_status != 200:
        # If TTS fails, still return reply
        return {"reply": short_reply, "audio": None, "error": "TTS_FAILED", "detail": audio_b64_or_err}

    return {"reply": short_reply, "audio": audio_b64_or_err}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": GEMINI_MODEL,
        "fallback_model": FALLBACK_MODEL
    }
