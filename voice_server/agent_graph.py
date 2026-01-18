# agent_graph.py
# LangGraph single-agent HealthExplain AI (Gemini Text + Gemini Vision)
# Memory handled in Streamlit/voice_proxy (frontend)

import os
import base64
from typing import TypedDict, List, Dict, Optional

import google.generativeai as genai
from langgraph.graph import StateGraph, END


# -------------------------
# GEMINI setup
# -------------------------
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("❌ Missing GOOGLE_API_KEY. Please set it in PowerShell before running.")

genai.configure(api_key=API_KEY)

TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash")
VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")


# -------------------------
# Graph State
# -------------------------
class AgentState(TypedDict, total=False):
    messages: List[Dict[str, str]]
    reply: str
    image_bytes: Optional[bytes]   # for vision


# -------------------------
# System Prompt
# -------------------------
SYSTEM_PROMPT = """
You are HealthExplain AI — a friendly AI health assistant for illiterate and non-technical users.

STRICT RULES:
1) Answer ONLY health-related questions.
2) If NOT health-related, reply exactly:
   "Sorry, I can answer only health-related questions."
3) Do NOT diagnose diseases. Do NOT claim you are a doctor.
4) Give short, simple, clear answers in 2–4 lines.
5) If emergency symptoms: chest pain, breathing trouble, fainting, seizures, severe bleeding:
   reply:
   "This may be an emergency. Please seek medical help immediately."
"""


def build_prompt(messages: List[Dict[str, str]]) -> str:
    chat_text = ""
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "user":
            chat_text += f"User: {content}\n"
        else:
            chat_text += f"Assistant: {content}\n"

    return f"{SYSTEM_PROMPT}\n\nConversation:\n{chat_text}\nAssistant:"


def agent_node(state: AgentState) -> AgentState:
    messages = state.get("messages", [])
    image_bytes = state.get("image_bytes", None)

    prompt = build_prompt(messages)

    try:
        # -------- VISION MODE --------
        if image_bytes:
            model = genai.GenerativeModel(VISION_MODEL)

            img_b64 = base64.b64encode(image_bytes).decode("utf-8")

            resp = model.generate_content([
                prompt,
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": img_b64
                    }
                }
            ])

            text = (resp.text or "").strip()

        # -------- TEXT MODE --------
        else:
            model = genai.GenerativeModel(TEXT_MODEL)
            resp = model.generate_content(prompt)
            text = (resp.text or "").strip()

    except Exception as e:
        text = f"Sorry, I am not able to answer right now. Please try again. ({e})"

    if not text:
        text = "Sorry, I am not able to answer right now. Please try again."

    # Keep short
    if len(text) > 500:
        text = text[:500].rsplit(" ", 1)[0] + "..."

    state["reply"] = text
    return state


# -------------------------
# Build Graph
# -------------------------
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.set_entry_point("agent")
builder.add_edge("agent", END)

graph = builder.compile()
