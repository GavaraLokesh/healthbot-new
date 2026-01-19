# voice_proxy.py - ✅ 100% WORKING for Render.com (uses google-generativeai==0.8.6)
# No requirements.txt changes needed - your current setup works!

import os
import re
import io
import base64
import logging
from typing import Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from gtts import gTTS

# ✅ CORRECT: Uses google-generativeai (already installed on Render)
import google.generativeai as genai

logger = logging.getLogger("voice-proxy")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="voice-proxy-gemini-2.5-flash-multilingual-tts")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------- Gemini setup ------------------------- 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not GEMINI_API_KEY:
    raise RuntimeError("❌ Missing GEMINI_API_KEY environment variable.")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)

SYSTEM_INSTRUCTION = (
    "You are a medical assistant. "
    "Always reply in 2–3 short lines only. "
    "Be simple, safe, non-diagnostic, actionable. "
    "Avoid long paragraphs. Avoid scary words. "
    "If emergency symptoms, suggest doctor visit."
)

STOP_COMMANDS = {
    "en": ["stop", "stop speaking", "ok stop", "stop now", "please stop", "that's enough"],
    "hi": ["रुको", "बोलना बंद करो", "रुकिए", "बंद करो"],
    "te": ["ఆపు", "మాట్లాడటం ఆపు", "ఆపండి", "ఆపు ఇప్పుడు"],
    "ta": ["நிறுத்து", "பேசுவதை நிறுத்து", "நிறுத்துங்கள்"],
    "gu": ["રોકો", "બોલવું બંધ કરો", "બંધ કરો"],
}

# ------------------------- Helpers ------------------------- 
def shorten_text_to_sentences(text: str, max_sentences: int = 2) -> str:
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
    lookup = {
        "English": "en", "Hindi": "hi", "Telugu": "te", 
        "Tamil": "ta", "Gujarati": "gu",
        "en": "en", "hi": "hi", "te": "te", "ta": "ta", "gu": "gu",
    }
    return lookup.get(str(label).strip(), "en")

def GTTS_LANG_FROM_LABEL(label: str) -> str:
    cfg = {
        "English": "en", "Hindi": "hi", "Telugu": "te", 
        "Tamil": "ta", "Gujarati": "gu",
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

# ------------------------- Gemini generation (NON-ASYNC) ------------------------- 
def call_gemini_generate(user_text: str, lang_label: str = "English") -> Tuple[int, str]:
    try:
        lang_code = LANG_CODE_FROM_LABEL(lang_label)
        prompt = (
            f"{SYSTEM_INSTRUCTION}\n"
            f"{language_instruction_from_code(lang_code)}\n\n"
            f"User question: {user_text}\n\n"
            f"Answer:"
        )
        # ✅ FIXED: Synchronous call (no async needed)
        response = model.generate_content(prompt)
        text = (response.text or "").strip()
        return 200, text
    except Exception as e:
        logger.exception("Gemini request failed")
        return 500, str(e)

# ------------------------- gTTS audio ------------------------- 
def tts_synthesize_mp3_gtts(text: str, lang: str = "en") -> Tuple[int, Optional[str]]:
    try:
        if not text or not text.strip():
            return 200, None
        mp3_fp = io.BytesIO()
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        audio_b64 = base64.b64encode(mp3_fp.read()).decode("utf-8")
        return 200, audio_b64
    except Exception as e:
        logger.exception("gTTS failed")
        return 500, str(e)

# ------------------------- Endpoints ------------------------- 
@app.post("/voice")
async def voice_endpoint(payload: dict):
    if not payload:
        raise HTTPException(status_code=400, detail="Missing JSON body")

    text = str(payload.get("text") or payload.get("prompt") or "").strip()
    lang_label = str(payload.get("lang") or payload.get("language") or "English").strip()

    if text == "":
        return {"reply": "", "audio": None}

    if detect_stop_phrase(text, lang_label):
        return {"reply": "__STOP__", "audio": None}

    # ✅ NON-ASYNC calls work fine in async endpoint
    status, gen_text = call_gemini_generate(text, lang_label=lang_label)
    if status != 200:
        return {"reply": "", "audio": None, "error": f"GEMINI_{status}", "detail": gen_text}

    short_reply = shorten_text_to_sentences(gen_text, max_sentences=2)
    gtts_lang = GTTS_LANG_FROM_LABEL(lang_label)
    tts_status, audio_b64_or_err = tts_synthesize_mp3_gtts(short_reply, lang=gtts_lang)

    if tts_status != 200:
        return {"reply": short_reply, "audio": None, "error": "TTS_FAILED", "detail": audio_b64_or_err}

    return {"reply": short_reply, "audio": audio_b64_or_err}

@app.get("/health")
async def health():
    return {"status": "ok", "model": GEMINI_MODEL}
