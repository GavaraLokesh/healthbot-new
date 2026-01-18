# app.py
# HealthExplain AI — Streamlit UI
# - Login/Register
# - Chat Assistant + Multilingual Voice Widget (same as old)
# - AI Doctor Vision tab (YouTuber style) + speaks reply
# - Diabetes Prediction
# - Alarm Reminder tab (basic)
# NOTE: Alarm reminder works only when browser tab is open.

import os
import json
import base64
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
import requests

from voice_server.agent_graph import graph


USERS_FILE = "users.json"

# IMPORTANT:
# Use Render deployed voice-server URL in production
# Example:
# export VOICE_PROXY_URL="https://healthbot-voice-server.onrender.com"
VOICE_PROXY_URL = os.getenv("VOICE_PROXY_URL", "http://127.0.0.1:8765").rstrip("/")


# -----------------------
# Helpers
# -----------------------
def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_users(users):
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2)
        return True
    except Exception:
        return False


def safe_rerun():
    if hasattr(st, "rerun"):
        try:
            st.rerun()
        except Exception:
            pass


# -----------------------
# Session defaults
# -----------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "nav" not in st.session_state:
    st.session_state.nav = "Home"
if "ui_language" not in st.session_state:
    st.session_state.ui_language = "English"

if "reminders" not in st.session_state:
    st.session_state.reminders = []  # list of dicts

# -----------------------
# Page config + Styling
# -----------------------
st.set_page_config(page_title="HealthExplain AI", layout="wide")

st.markdown(
    """
<style>
  body { background: #0b1220; color: #e6eef6; }
  .main { max-width: 1200px; margin: auto; padding-top: 18px; padding-bottom: 40px; }
  .sidebar-box { padding:14px; background:#0a0f16; border-radius:12px; border:1px solid rgba(255,255,255,0.06); }
  .title-big { font-size:38px; font-weight:900; color:#d7ffe6; text-shadow: 0 0 14px rgba(0,255,150,0.10); }
  .card { background: #0f1724; padding:18px; border-radius:16px; border:1px solid rgba(255,255,255,0.06);
          box-shadow: 0 10px 40px rgba(2,6,23,0.55); }
  .small { font-size:13px; color:#9fb0c7; }
  .bubble-user { background: linear-gradient(90deg,#60a5fa,#34d399); color:white; padding:10px 14px; border-radius:14px; display:inline-block; max-width:78%; }
  .bubble-bot { background:#e6eef6; color:#06121a; padding:10px 14px; border-radius:14px; display:inline-block; max-width:78%; }
  .hr-soft { border:none; height:1px; background: rgba(255,255,255,0.08); margin:14px 0; }
  .section-title { font-size:22px; font-weight:800; margin:0 0 6px 0; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------
# LangGraph call (memory from Streamlit)
# -----------------------
def ask_ai(question: str):
    if not question or not question.strip():
        return "Please type your health question."

    st.session_state.chat_history.append({"role": "user", "content": question})

    try:
        result = graph.invoke({"messages": st.session_state.chat_history})
        reply = result.get("reply", "")
    except Exception:
        reply = "Sorry, I am not able to answer right now. Please try again."

    if not reply:
        reply = "Sorry, I am not able to answer right now. Please try again."

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    return reply


# -----------------------
# Login/Register UI
# -----------------------
def show_login_register():
    st.markdown('<div class="main">', unsafe_allow_html=True)
    cols = st.columns([0.5, 0.5])

    with cols[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("Login")
        with st.form("login_form"):
            login_email = st.text_input("Email or username", key="login_email")
            login_password = st.text_input("Password", type="password", key="login_password")
            if st.form_submit_button("Login"):
                users = load_users()
                if login_email in users and users[login_email].get("password") == login_password:
                    st.session_state.logged_in = True
                    st.session_state.username = login_email
                    st.success("Logged in")
                    safe_rerun()
                else:
                    st.error("Invalid credentials")
        st.markdown("</div>", unsafe_allow_html=True)

    with cols[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("Create account")
        with st.expander("Create new account"):
            with st.form("register_form"):
                reg_email = st.text_input("New email", key="reg_email")
                reg_pw = st.text_input("New password", type="password", key="reg_pw")
                reg_confirm = st.text_input("Confirm password", type="password", key="reg_confirm")
                if st.form_submit_button("Create account"):
                    if not reg_email or not reg_pw:
                        st.error("Please fill all fields")
                    elif reg_pw != reg_confirm:
                        st.error("Passwords do not match")
                    else:
                        users = load_users()
                        if reg_email in users:
                            st.error("User already exists")
                        else:
                            users[reg_email] = {"password": reg_pw}
                            ok = save_users(users)
                            if ok:
                                st.success("Account created. Please login.")
                            else:
                                st.error("Error saving account.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------
# Pages
# -----------------------
def show_home():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Welcome to HealthExplain AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="small">Multilingual health assistant for non-technical users. Use Chat Assistant or AI Doctor Vision.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def handle_send():
    q = st.session_state.get("user_input", "").strip()
    if not q:
        return
    ask_ai(q)
    st.session_state["user_input"] = ""


def chat_page():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💬 Chat Assistant</div>', unsafe_allow_html=True)

    left_col, right_col = st.columns([0.62, 0.38], gap="large")

    with left_col:
        st.text_input("Ask a question:", key="user_input", placeholder="e.g., I have headache and fever")
        st.selectbox("Language", ["English", "Hindi", "Telugu", "Tamil", "Gujarati"], key="ui_language")

        c1, c2 = st.columns([0.2, 0.2])
        c1.button("Send", on_click=handle_send)
        c2.button("Clear chat", on_click=lambda: st.session_state.update({"chat_history": []}))

        st.markdown('<hr class="hr-soft" />', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Chat</div>', unsafe_allow_html=True)

        for m in st.session_state.chat_history[-80:]:
            if m.get("role") == "user":
                st.markdown(
                    f"<div style='text-align:right; margin:8px'><span class='bubble-user'>{m.get('content','')}</span></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='text-align:left; margin:8px'><span class='bubble-bot'><b>HealthExplain:</b> {m.get('content','')}</span></div>",
                    unsafe_allow_html=True,
                )

    with right_col:
        # Voice widget shown on the side (as you requested)
        st.markdown("#### 🎙️ Voice Assistant")
        components.html(build_voice_widget_html(VOICE_PROXY_URL), height=420)

    st.markdown("</div>", unsafe_allow_html=True)


def ai_doctor_vision_page():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="title-big" style="font-size:34px;">AI Doctor with Vision and Voice</div>', unsafe_allow_html=True)
    st.markdown('<p class="small">Upload body/skin image and ask question. AI will reply and speak in selected language.</p>', unsafe_allow_html=True)
    st.markdown('<hr class="hr-soft" />', unsafe_allow_html=True)

    lang = st.selectbox("Language", ["English", "Hindi", "Telugu", "Tamil", "Gujarati"], key="vision_lang")

    col1, col2 = st.columns([0.55, 0.45], gap="large")

    with col1:
        st.markdown("### 🖼️ Image Input")
        uploaded = st.file_uploader("Upload body/skin infection image", type=["png", "jpg", "jpeg"], key="vision_img")

        st.markdown("### 📝 Ask about the image")
        question = st.text_input(
            "Example: What is this infection? What should I do?",
            value="Please analyze this image and give simple safe advice.",
            key="vision_question",
        )

        analyze = st.button("🔍 Analyze Image", key="vision_btn")

    with col2:
        st.markdown("### 🧑‍⚕️ Doctor's Response")
        response_box = st.empty()

        st.markdown("### 🔊 Speak Response")
        speak_box = st.empty()

    if analyze:
        if not uploaded:
            response_box.error("Please upload an image first.")
        else:
            try:
                url = f"{VOICE_PROXY_URL}/vision"

                files = {
                    "image": ("image.jpg", uploaded.getvalue(), uploaded.type)
                }

                data = {
                    "question": question,
                    "lang": lang
                }

                res = requests.post(url, files=files, data=data)

                if res.status_code != 200:
                    response_box.error(f"Vision error: {res.status_code} - {res.text}")
                else:
                    reply = res.json().get("reply", "").strip()
                    if not reply:
                        reply = "No reply from AI."

                    response_box.success(reply)
                    speak_box.info("Voice will speak from Voice Assistant tab.")
            except Exception as e:
                response_box.error(f"Vision error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


def diabetes_page():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🩸 Diabetes Prediction (Demo)</div>', unsafe_allow_html=True)

    age = st.number_input("Age", min_value=1, max_value=120, value=25, key="age")
    glucose = st.number_input("Glucometer reading (mg/dL)", min_value=0, max_value=1000, value=90, key="glucose")

    if st.button("Predict"):
        risk = "Low"
        if glucose >= 200:
            risk = "High"
        elif glucose >= 140:
            risk = "Moderate"
        st.success(f"Estimated risk: {risk} (not diagnostic)")

    st.markdown("</div>", unsafe_allow_html=True)


def alarm_page():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⏰ Alarm Reminder</div>', unsafe_allow_html=True)
    st.markdown('<p class="small">Works when browser tab is open. (APK version later will work even when app closed.)</p>', unsafe_allow_html=True)

    name = st.text_input("Person Name", value="Mr. Lokesh", key="rem_name")
    msg = st.text_input("Reminder sentence", value="Please take tablets. It's 4 PM.", key="rem_msg")
    time_str = st.time_input("Reminder Time", key="rem_time")
    repeat = st.slider("Repeat times", min_value=1, max_value=20, value=5, key="rem_repeat")

    if st.button("Add Reminder"):
        st.session_state.reminders.append(
            {
                "name": name,
                "msg": msg,
                "time": time_str.strftime("%H:%M"),
                "repeat": int(repeat),
                "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        st.success("Reminder added ✅")

    st.markdown("---")

    if not st.session_state.reminders:
        st.info("No reminders yet.")
    else:
        st.markdown("### Your Reminders")
        for idx, r in enumerate(st.session_state.reminders):
            c1, c2 = st.columns([0.8, 0.2])
            with c1:
                st.write(f"**{r['time']}** → {r['name']} : {r['msg']}  (Repeat: (Repeat: {r['repeat']})")
            with c2:
                if st.button("Delete", key=f"del_{idx}"):
                    st.session_state.reminders.pop(idx)
                    safe_rerun()

    # Inject alarm JS
    components.html(build_alarm_js(st.session_state.reminders), height=10)

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------
# Voice widget HTML (UPDATED safely)
# - Supports OLD: POST /voice (JSON -> base64 mp3)
# - Supports NEW: POST /voice-audio (multipart file -> mp3 download)
# -----------------------
def build_voice_widget_html(proxy_url: str) -> str:
    return f"""
<div style="font-family: Inter, Arial, sans-serif; color: #e6eef6;">
  <style>
    .va-card {{ background:#041024; padding:12px; border-radius:14px; border:1px solid rgba(255,255,255,0.06); }}
    .va-row {{ display:flex; align-items:center; gap:12px; }}
    .va-mic {{ width:96px; height:96px; border-radius:50%;
      background: radial-gradient(circle at 30% 20%, #ffd36b, #ff7bd6);
      display:flex; align-items:center; justify-content:center;
      font-size:40px; color:#07121a; cursor:pointer;
      border: 4px solid rgba(255,255,255,0.08); }}
    .va-micon {{ box-shadow: 0 0 36px rgba(255,123,214,0.45); transform: scale(1.02); }}
    .va-small {{ font-size:12px; color:#9fb0c7; }}
    .va-log {{ margin-top:10px; max-height:240px; overflow:auto; }}
    .va-bubble-user {{ background: linear-gradient(90deg,#60a5fa,#34d399); color:white; padding:8px 12px; border-radius:12px; display:inline-block; max-width:85%; }}
    .va-bubble-bot {{ background:#e6eef6; color:#06121a; padding:8px 12px; border-radius:12px; display:inline-block; max-width:85%; }}
    .va-lang {{ background:#0b1220; color:#e6eef6; padding:6px 10px; border-radius:8px; border:1px solid rgba(255,255,255,0.10); }}
    .va-btn {{ background:#0f1724;color:#e6eef6;padding:6px 10px;border-radius:8px;border:1px solid rgba(255,255,255,0.10);cursor:pointer; }}
  </style>

  <div class="va-card">
    <div style="display:flex; justify-content:space-between; align-items:center;">
      <div style="font-weight:800;">Voice Assistant</div>
      <div id="vaStatus" class="va-small">Idle</div>
    </div>

    <div class="va-row" style="margin-top:12px;">
      <div id="vaMic" class="va-mic" title="Click to talk">🎙️</div>

      <div style="flex:1;">
        <div style="display:flex; gap:8px; align-items:center;">
          <select id="vaLang" class="va-lang">
            <option>English</option>
            <option>Hindi</option>
            <option>Telugu</option>
            <option>Tamil</option>
            <option>Gujarati</option>
          </select>
          <div style="flex:1;"></div>
          <button id="vaClear" class="va-btn">Clear</button>
        </div>
        <div id="vaLog" class="va-log"></div>
      </div>
    </div>
  </div>

<script>
(function(){{
  const proxyBase = "{proxy_url}";
  const LOG = (m)=>{{ try{{ console.log("[voice-assistant]", m); }}catch(e){{}} }};

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  function escapeHtml(s){{
    if(!s) return '';
    return s.replace(/[&<"'>]/g, function(m){{ return ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}})[m]; }});
  }}

  function appendLog(who, text){{
    const logEl = document.getElementById('vaLog');
    if(!logEl) return;
    const d = document.createElement('div');
    d.style.margin = '8px 0';
    if(who === 'user'){{
      d.innerHTML = `<div style="text-align:right"><span class="va-bubble-user">${{escapeHtml(text)}}</span></div>`;
    }} else {{
      d.innerHTML = `<div style="text-align:left"><span class="va-bubble-bot">${{escapeHtml(text)}}</span></div>`;
    }}
    logEl.appendChild(d);
    logEl.scrollTop = logEl.scrollHeight;
  }}

  async function playAudioFromBlob(blob){{
    try{{
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      await audio.play();
      audio.onended = () => {{
        try{{ URL.revokeObjectURL(url); }}catch(e){{}}
      }};
    }}catch(e){{ LOG("audio blob play failed " + e); }}
  }}

  async function playAudioBase64Mp3(audioB64){{
    try{{
      if(!audioB64) return;
      const audio = new Audio("data:audio/mp3;base64," + audioB64);
      await audio.play();
    }}catch(e){{ LOG("audio base64 play failed " + e); }}
  }}

  // -----------------------------
  // NEW: record mic audio -> send to /voice-audio
  // -----------------------------
  async function recordAudioOnce(){{
    const stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
    const mediaRecorder = new MediaRecorder(stream);
    const chunks = [];

    return new Promise((resolve, reject) => {{
      mediaRecorder.ondataavailable = (e) => {{
        if(e.data && e.data.size > 0) chunks.push(e.data);
      }};
      mediaRecorder.onerror = (e) => reject(e);

      mediaRecorder.onstop = () => {{
        try {{
          stream.getTracks().forEach(t => t.stop());
        }} catch(e) {{}}

        const blob = new Blob(chunks, {{ type: "audio/webm" }});
        resolve(blob);
      }};

      mediaRecorder.start();

      // record for 4 seconds (safe)
      setTimeout(() => {{
        try {{ mediaRecorder.stop(); }} catch(e) {{}}
      }}, 4000);
    }});
  }}

  async function callVoiceAudioEndpoint(audioBlob){{
    try {{
      const form = new FormData();
      form.append("file", audioBlob, "recording.webm");

      const r = await fetch(proxyBase + "/voice-audio", {{
        method: "POST",
        body: form
      }});

      if(!r.ok){{
        const t = await r.text();
        return {{ ok:false, error: t }};
      }}

      const replyBlob = await r.blob(); // mp3 file
      return {{ ok:true, blob: replyBlob }};
    }} catch(e) {{
      return {{ ok:false, error: String(e) }};
    }}
  }}

  // -----------------------------
  // OLD: text -> /voice -> reply+base64
  // -----------------------------
  async function callVoiceTextEndpoint(text, lang){{
    try {{
      const r = await fetch(proxyBase + "/voice", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{ text: text, lang: lang }})
      }});

      if(!r.ok){{
        const t = await r.text();
        return {{ ok:false, error: t }};
      }}

      const data = await r.json();
      return {{ ok:true, data }};
    }} catch(e) {{
      return {{ ok:false, error: String(e) }};
    }}
  }}

  async function assistantFlow(text, lang){{
    if(!text) return;
    appendLog('user', text);

    // Try OLD endpoint first (keeps backward compatibility)
    const res1 = await callVoiceTextEndpoint(text, lang);

    if(res1.ok && res1.data){{
      const reply = (res1.data.reply) || "(no reply)";
      appendLog('assistant', reply);
      if(res1.data.audio){{
        await playAudioBase64Mp3(res1.data.audio);
      }}
      return;
    }}

    // If OLD fails, show error and still allow audio mode
    appendLog('assistant', "Voice server text mode not available. Try audio mode.");
    LOG("voice text error: " + (res1.error || "unknown"));
  }}

  const mic = document.getElementById('vaMic');
  const langSel = document.getElementById('vaLang');
  const statusEl = document.getElementById('vaStatus');
  const clearBtn = document.getElementById('vaClear');

  function setStatus(t){{ try{{ statusEl.innerText = t; }}catch(e){{}} }}

  let recog = null;
  let isListening = false;

  mic.addEventListener('click', async () => {{
    isListening = !isListening;
    mic.classList.toggle('va-micon');

    if(isListening){{
      setStatus('Listening...');

      // SpeechRecognition first (best for text)
      if(SpeechRecognition){{
        recog = new SpeechRecognition();
        recog.interimResults = false;
        recog.maxAlternatives = 1;

        const lang = langSel.value || 'English';
        recog.lang =
          (lang === 'Hindi') ? 'hi-IN' :
          (lang === 'Telugu') ? 'te-IN' :
          (lang === 'Tamil') ? 'ta-IN' :
          (lang === 'Gujarati') ? 'gu-IN' :
          'en-US';

        recog.onresult = async (ev) => {{
          const text = ev.results[0][0].transcript;
          await assistantFlow(text, lang);
        }};

        recog.onerror = async (e) => {{
          setStatus('Mic Error');
          LOG("recog error: " + (e && e.error));
        }};

        recog.onend = () => {{
          isListening = false;
          mic.classList.remove('va-micon');
          setStatus('Idle');
        }};

        try {{
          recog.start();
        }} catch(e) {{
          LOG("recog start failed: " + e);
          isListening = false;
          mic.classList.remove('va-micon');
          setStatus('Idle');
        }}

      }} else {{
        // If SpeechRecognition not supported, fallback to audio recording
        try {{
          const lang = langSel.value || 'English';
          appendLog('assistant', "Speech Recognition not supported. Using audio mode...");

          const audioBlob = await recordAudioOnce();
          const res = await callVoiceAudioEndpoint(audioBlob);

          if(res.ok && res.blob){{
            appendLog('assistant', "Playing reply audio...");
            await playAudioFromBlob(res.blob);
          }} else {{
            appendLog('assistant', "Audio mode failed.");
            LOG("voice-audio error: " + (res.error || "unknown"));
          }}
        }} catch(e) {{
          appendLog('assistant', "Audio recording failed.");
          LOG("audio record error: " + e);
        }}

        isListening = false;
        mic.classList.remove('va-micon');
        setStatus('Idle');
      }}

    }} else {{
      try{{ if(recog) recog.stop(); }}catch(e){{}}
      setStatus('Idle');
    }}
  }});

  clearBtn.addEventListener('click', ()=>{{ 
    try{{ document.getElementById('vaLog').innerHTML=''; }}catch(e){{}} 
  }});

  setStatus('Idle');
}})();
</script>
</div>
"""


# -----------------------
# Alarm JS
# -----------------------
def build_alarm_js(reminders):
    # send reminders safely as JSON string
    rem_json = json.dumps(reminders)

    return f"""
<script>
(function(){{
  const reminders = {rem_json};

  function speakText(text, repeat){{
    try {{
      window.speechSynthesis.cancel();
      let count = 0;

      function speakOnce(){{
        if(count >= repeat) return;
        const u = new SpeechSynthesisUtterance(text);
        u.rate = 1.0;
        u.pitch = 1.0;
        u.onend = () => {{
          count++;
          if(count < repeat) speakOnce();
        }};
        window.speechSynthesis.speak(u);
      }}

      speakOnce();
    }} catch(e) {{}}
  }}

  function checkAlarm(){{
    const now = new Date();
    const hh = String(now.getHours()).padStart(2,'0');
    const mm = String(now.getMinutes()).padStart(2,'0');
    const current = hh + ":" + mm;

    reminders.forEach(r => {{
      if(r.time === current){{
        const text = (r.name || "Reminder") + ", " + (r.msg || "Please take medicine");
        const repeat = parseInt(r.repeat || 5);
        speakText(text, repeat);
      }}
    }});
  }}

  setInterval(checkAlarm, 1000 * 10);
}})();
</script>
"""


# -----------------------
# Route
# -----------------------
if not st.session_state.logged_in:
    show_login_register()
    st.stop()

left, main = st.columns([0.22, 0.78], gap="large")

with left:
    st.markdown('<div class="sidebar-box">', unsafe_allow_html=True)
    st.markdown(
        f"<div style='margin-bottom:8px'><b>👋 Logged in as</b><div class='small'>{st.session_state.username}</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='hr-soft' />", unsafe_allow_html=True)

    def nav_change():
        st.session_state.nav = st.session_state.get("nav_select", "Home")

    st.radio(
        "Select:",
        ["Home", "Chat Assistant", "AI Doctor Vision", "Diabetes Prediction", "Alarm Reminder"],
        index=["Home", "Chat Assistant", "AI Doctor Vision", "Diabetes Prediction", "Alarm Reminder"].index(st.session_state.nav)
        if st.session_state.nav in ["Home", "Chat Assistant", "AI Doctor Vision", "Diabetes Prediction", "Alarm Reminder"]
        else 0,
        key="nav_select",
        on_change=nav_change,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Logout"):
        st.session_state.update({"logged_in": False, "username": "", "chat_history": [], "reminders": []})
        safe_rerun()

with main:
    st.markdown('<div class="title-big">💚 HealthExplain AI</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    page = st.session_state.nav
    if page == "Home":
        show_home()
    elif page == "Chat Assistant":
        chat_page()
    elif page == "AI Doctor Vision":
        ai_doctor_vision_page()
    elif page == "Diabetes Prediction":
        diabetes_page()
    elif page == "Alarm Reminder":
        alarm_page()
    else:
        show_home()
