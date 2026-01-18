from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os, uuid
from gtts import gTTS
import speech_recognition as sr

from agent_graph import graph

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/voice-audio")
async def voice_audio(file: UploadFile = File(...)):
    # 1) save uploaded audio
    temp_audio_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    with open(temp_audio_path, "wb") as f:
        f.write(await file.read())

    # 2) convert audio -> text (STT)
    recognizer = sr.Recognizer()
    user_text = ""

    try:
        with sr.AudioFile(temp_audio_path) as source:
            audio_data = recognizer.record(source)
            user_text = recognizer.recognize_google(audio_data)
    except Exception as e:
        user_text = "Sorry, I could not understand your voice."

    # 3) AI reply from Gemini
    result = graph.invoke({"input": user_text})
    ai_reply = result.get("output", "Sorry, I couldn't respond.")

    # 4) Convert AI reply -> audio (TTS)
    output_audio_path = f"/tmp/{uuid.uuid4()}.mp3"
    tts = gTTS(ai_reply, lang="en")
    tts.save(output_audio_path)

    return FileResponse(output_audio_path, media_type="audio/mpeg", filename="reply.mp3")
