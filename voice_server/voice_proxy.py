from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
import uuid
import base64

from gtts import gTTS
import speech_recognition as sr

from agent_graph import graph

app = FastAPI()


# ---------------------------
# Helpers
# ---------------------------
def get_gtts_lang(lang: str) -> str:
    """
    Convert UI language to gTTS language code.
    """
    if not lang:
        return "en"

    lang = lang.strip().lower()

    if lang == "hindi":
        return "hi"
    if lang == "telugu":
        return "te"
    if lang == "tamil":
        return "ta"
    if lang == "gujarati":
        return "gu"
    return "en"


def ai_reply(user_text: str, lang: str) -> str:
    """
    Uses LangGraph to generate AI reply.
    Also forces reply language.
    """
    try:
        prompt = f"""
You are a helpful AI health assistant.
Reply in this language only: {lang}.
Give simple safe advice.
User said: {user_text}
"""

        result = graph.invoke(
            {
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
        )

        reply = result.get("reply", "")
        if not reply:
            reply = "Sorry, I am not able to answer right now. Please try again."
        return reply

    except Exception:
        return "Sorry, I am not able to answer right now. Please try again."


def generate_tts_mp3_base64(text: str, lang: str) -> str:
    """
    Generate mp3 audio using gTTS and return base64 string.
    """
    try:
        tts_lang = get_gtts_lang(lang)
        tmp_path = f"/tmp/{uuid.uuid4()}.mp3"

        tts = gTTS(text=text, lang=tts_lang)
        tts.save(tmp_path)

        with open(tmp_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")

        try:
            os.remove(tmp_path)
        except:
            pass

        return audio_b64

    except Exception:
        return ""


# ---------------------------
# Routes
# ---------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/voice")
async def voice_endpoint(payload: dict):
    """
    Text -> AI reply -> audio base64
    Input JSON:
    {
      "text": "hello",
      "lang": "Hindi"
    }
    Output:
    {
      "reply": "...",
      "audio": "BASE64_MP3"
    }
    """
    try:
        text = (payload.get("text") or "").strip()
        lang = (payload.get("lang") or "English").strip()

        if not text:
            return JSONResponse({"reply": "Please say something.", "audio": ""}, status_code=200)

        reply = ai_reply(text, lang)
        audio_b64 = generate_tts_mp3_base64(reply, lang)

        return {"reply": reply, "audio": audio_b64}

    except Exception as e:
        return JSONResponse({"reply": f"Server error: {e}", "audio": ""}, status_code=200)


@app.post("/voice-audio")
async def voice_audio(
    file: UploadFile = File(...),
    lang: str = Form("English")
):
    """
    Audio -> STT -> AI reply -> audio base64

    Form-data:
    - file: audio file
    - lang: English/Hindi/Telugu/Tamil/Gujarati
    """
    # 1) Save uploaded audio
    temp_audio_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    with open(temp_audio_path, "wb") as f:
        f.write(await file.read())

    # 2) Speech to text
    recognizer = sr.Recognizer()
    user_text = ""

    try:
        with sr.AudioFile(temp_audio_path) as source:
            audio_data = recognizer.record(source)
            user_text = recognizer.recognize_google(audio_data)
    except Exception:
        user_text = ""

    # cleanup file
    try:
        os.remove(temp_audio_path)
    except:
        pass

    # if STT failed
    if not user_text:
        reply = "Sorry, I could not understand your voice. Please try again."
        audio_b64 = generate_tts_mp3_base64(reply, lang)
        return {"reply": reply, "audio": audio_b64}

    # 3) AI reply
    reply = ai_reply(user_text, lang)

    # 4) Generate audio
    audio_b64 = generate_tts_mp3_base64(reply, lang)

    return {"reply": reply, "audio": audio_b64}
