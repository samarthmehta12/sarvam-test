"""
Bike Troubleshooting Bot — Sarvam AI
Features: Indic language support, maintenance tracker, image + text input.
"""

import io
import json
import os
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from google import genai
from google.genai import types

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = "gemini-2.5-flash"

LANGUAGES = {
    "English": "English",
    "हिंदी (Hindi)": "Hindi",
    "தமிழ் (Tamil)": "Tamil",
    "తెలుగు (Telugu)": "Telugu",
    "ಕನ್ನಡ (Kannada)": "Kannada",
    "മലയാളം (Malayalam)": "Malayalam",
    "मराठी (Marathi)": "Marathi",
    "বাংলা (Bengali)": "Bengali",
    "ગુજરાતી (Gujarati)": "Gujarati",
    "ਪੰਜਾਬੀ (Punjabi)": "Punjabi",
}

MAINTENANCE_PROMPT = """Look through this bike manual and extract ALL periodic maintenance tasks with their service intervals.

Return ONLY a valid JSON array — no markdown, no explanation, just raw JSON:
[
  {
    "task": "Engine Oil Change",
    "interval_km": 3000,
    "interval_months": 6,
    "notes": "Use 10W-30 grade oil"
  }
]

Rules:
- Only include tasks explicitly mentioned in the manual with specific intervals
- interval_km: integer (null if not mentioned)
- interval_months: integer (null if not mentioned)
- notes: brief detail from the manual (null if none)
- If no maintenance schedule exists in the manual, return: []"""


def build_system_instruction(language: str) -> str:
    return f"""You are a bike troubleshooting assistant. The user has provided their bike's official manual as a document.

STRICT RULES — FOLLOW EXACTLY:
1. Answer ONLY using information explicitly found in the provided manual document.
2. If the answer is not in the manual, say: "This is not covered in the provided manual. Please consult an authorized service center."
3. Never use general automotive knowledge not stated in the manual.
4. Reference page numbers or section titles when they appear in the document.
5. When the user provides an image: describe what you observe, then cite the relevant manual section.
6. Present step-by-step procedures as numbered lists.
7. Do not guess or speculate beyond what the manual explicitly states.

LANGUAGE: Always respond in {language}. Even if the user writes in another language, your response must be in {language}."""


# ── Sarvam AI Brand Styles ────────────────────────────────────────────────────

SARVAM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');

* {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    box-sizing: border-box;
}

/* Higher specificity overrides the wildcard above for icon font */
.material-symbols-rounded {
    font-family: 'Material Symbols Rounded' !important;
    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24 !important;
    font-size: 20px !important;
    line-height: 1 !important;
}

html, body, .stApp { background-color: #FAFAF9 !important; }

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }

/* Hide every variant of the sidebar collapse/expand toggle */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
section[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[data-testid="stSidebarCollapseButton"] { display: none !important; visibility: hidden !important; }

/* ── Sidebar — white / orange theme ── */
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1.5px solid #F0EDE8 !important;
}

[data-testid="stSidebar"] * { color: #111111 !important; }

[data-testid="stSidebar"] .stTextInput input {
    background: #F8F6F3 !important;
    border: 1px solid #E8E2DC !important;
    color: #111 !important;
    border-radius: 8px !important;
}

[data-testid="stSidebar"] .stTextInput input::placeholder { color: #BBB !important; }

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #F8F6F3 !important;
    border: 1px solid #E8E2DC !important;
    border-radius: 8px !important;
    color: #111 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {
    background: #FFF8F5 !important;
    border: 1.5px dashed #F26522 !important;
    border-radius: 10px !important;
}

/* Hide EVERY material-symbols icon inside the dropzone — they show as raw "upload" text when font fails */
[data-testid="stFileUploaderDropzone"] .material-symbols-rounded,
[data-testid="stFileUploaderDropzoneInstructions"] .material-symbols-rounded {
    display: none !important;
}

/* Also hide the first child of dropzoneInstructions (the icon wrapper div) */
[data-testid="stFileUploaderDropzoneInstructions"] > *:first-child {
    display: none !important;
}

/* Nuclear button fix: wipe all native content, inject clean label via ::after */
[data-testid="stFileUploaderDropzone"] button {
    background: #F26522 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 6px 16px !important;
    box-shadow: none !important;
    transform: none !important;
    font-size: 0 !important;
    color: transparent !important;
    line-height: 1 !important;
    min-width: 110px !important;
}

[data-testid="stFileUploaderDropzone"] button * {
    display: none !important;
    visibility: hidden !important;
}

[data-testid="stFileUploaderDropzone"] button::after {
    content: "Browse files";
    display: inline !important;
    visibility: visible !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #fff !important;
    line-height: 1.2 !important;
}

[data-testid="stFileUploaderDropzone"] button:hover { background: #E8520E !important; }

/* Delete button — targeted by aria-label since DOM nesting is unreliable */
button[aria-label^="Remove "] {
    background: #F0EDE8 !important;
    border: 1px solid #E8E2DC !important;
    border-radius: 6px !important;
    padding: 4px 10px !important;
    min-width: unset !important;
    font-size: 0 !important;
    color: transparent !important;
    line-height: 1 !important;
    box-shadow: none !important;
    transform: none !important;
}

button[aria-label^="Remove "] * {
    display: none !important;
    visibility: hidden !important;
}

button[aria-label^="Remove "]::after {
    content: "✕" !important;
    display: inline !important;
    visibility: visible !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #888 !important;
    line-height: 1 !important;
}

/* ── Primary buttons ── */
.stButton > button[kind="primary"],
[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #F26522 0%, #E8520E 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 1.2rem !important;
    box-shadow: 0 2px 12px rgba(242, 101, 34, 0.35) !important;
    transition: all 0.2s ease !important;
}

.stButton > button[kind="primary"]:hover,
[data-testid="stFormSubmitButton"] > button:hover {
    box-shadow: 0 4px 20px rgba(242, 101, 34, 0.5) !important;
    transform: translateY(-1px) !important;
}

.stButton > button:not([kind="primary"]) {
    background: #F8F6F3 !important;
    color: #444 !important;
    border: 1.5px solid #E8E2DC !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
}

.stButton > button:not([kind="primary"]):hover {
    border-color: #F26522 !important;
    color: #F26522 !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 2px solid #F0EDE8 !important;
    gap: 0 !important;
}

[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    color: #999 !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 10px 20px !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -2px !important;
}

[data-testid="stTabs"] [aria-selected="true"] {
    color: #F26522 !important;
    border-bottom: 2px solid #F26522 !important;
}

/* ── Welcome / Maintenance cards ── */
.sarvam-card {
    background: #FFFFFF;
    border: 1px solid #F0EDE8;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
    height: 130px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.sarvam-card-full { height: auto !important; }

.sarvam-card:hover {
    box-shadow: 0 6px 24px rgba(242, 101, 34, 0.1);
    transform: translateY(-2px);
}

.sarvam-card-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #F26522;
    margin-bottom: 6px;
}

.sarvam-card-title { font-size: 1.05rem; font-weight: 600; color: #111; margin-bottom: 4px; }
.sarvam-card-desc  { font-size: 0.83rem; color: #777; line-height: 1.5; }

/* ── Maintenance item cards ── */
.m-card {
    background: #fff;
    border: 1px solid #F0EDE8;
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}

.m-card-left { flex: 1; }
.m-task  { font-size: 0.95rem; font-weight: 600; color: #111; margin-bottom: 3px; }
.m-meta  { font-size: 0.78rem; color: #888; }
.m-notes { font-size: 0.75rem; color: #aaa; margin-top: 2px; }

.badge {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em;
    text-transform: uppercase; padding: 4px 10px;
    border-radius: 20px; white-space: nowrap;
}

.badge-overdue { background: #FEE2E2; color: #DC2626; }
.badge-due     { background: #FEF9C3; color: #CA8A04; }
.badge-ok      { background: #DCFCE7; color: #16A34A; }
.badge-info    { background: #F0EDE8; color: #888; }

/* ── Hero ── */
.sarvam-hero { display: flex; align-items: center; gap: 14px; margin-bottom: 16px; }

.sarvam-logo-dot {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, #F26522, #E8520E);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    box-shadow: 0 4px 12px rgba(242, 101, 34, 0.3);
    flex-shrink: 0;
}

.sarvam-title {
    font-size: 1.5rem; font-weight: 800; color: #111;
    letter-spacing: -0.03em; line-height: 1.2;
}

.sarvam-subtitle {
    font-size: 0.72rem; color: #F26522; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: #FFFFFF !important;
    border: 1px solid #F0EDE8 !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
    padding: 4px 8px !important;
    margin-bottom: 10px !important;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] ol,
[data-testid="stChatMessage"] ul,
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] em,
[data-testid="stChatMessage"] code {
    color: #111111 !important;
}

/* ── Input form area ── */
.stTextArea textarea {
    background: #FFFFFF !important;
    border: 1.5px solid #E8E2DC !important;
    border-radius: 12px !important;
    font-size: 0.95rem !important;
    color: #111 !important;
    resize: none !important;
}

.stTextArea textarea:focus {
    border-color: #F26522 !important;
    box-shadow: 0 0 0 3px rgba(242, 101, 34, 0.12) !important;
}

.stTextArea textarea::placeholder { color: #AAA !important; }

/* Image uploader label in form */
[data-testid="stForm"] [data-testid="stFileUploader"] label { color: #555 !important; font-size: 0.85rem !important; }

.stNumberInput input {
    background: #fff !important;
    border: 1.5px solid #E8E2DC !important;
    border-radius: 10px !important;
}

.stNumberInput input:focus {
    border-color: #F26522 !important;
    box-shadow: 0 0 0 3px rgba(242,101,34,0.12) !important;
}

hr { border-color: #F0EDE8 !important; }

/* ── Spinner / loading ── */
[data-testid="stSpinner"] > div {
    border-color: #F26522 !important;
}
[data-testid="stSpinner"] p,
[data-testid="stSpinner"] span {
    color: #777 !important;
    font-size: 0.88rem !important;
}

/* ── Sidebar brand ── */
.sidebar-brand {
    display: flex; align-items: center; gap: 10px; padding: 4px 0 16px 0;
}

.sidebar-logo {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #F26522, #E8520E);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    box-shadow: 0 3px 10px rgba(242,101,34,0.25);
    flex-shrink: 0;
}

.sidebar-name    { font-size: 1.05rem; font-weight: 700; color: #111 !important; }
.sidebar-tagline { font-size: 0.65rem; color: #F26522 !important; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; }

/* ── Manual loaded pill ── */
.manual-pill {
    display: flex; align-items: center; gap: 8px;
    background: #FFF4EE;
    border: 1.5px solid #F26522;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 2px;
}

.manual-pill-icon { font-size: 1rem; flex-shrink: 0; }
.manual-pill-name { font-size: 0.85rem; font-weight: 600; color: #111 !important; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
</style>
"""


# ── Gemini helpers ────────────────────────────────────────────────────────────

def get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found.")
        st.stop()
    return genai.Client(api_key=api_key)


def upload_pdf(client: genai.Client, pdf_bytes: bytes, display_name: str):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        uploaded = client.files.upload(
            file=tmp_path,
            config=types.UploadFileConfig(mime_type="application/pdf", display_name=display_name),
        )
        for _ in range(60):
            info = client.files.get(name=uploaded.name)
            if info.state.name == "ACTIVE":
                return info
            if info.state.name == "FAILED":
                raise RuntimeError("File processing failed.")
            time.sleep(2)
        raise TimeoutError("File processing timed out.")
    finally:
        os.unlink(tmp_path)


def create_chat(client: genai.Client, manual_file, language: str):
    chat = client.chats.create(
        model=MODEL_NAME,
        config=types.GenerateContentConfig(
            system_instruction=build_system_instruction(language),
        ),
    )
    chat.send_message([
        manual_file,
        "Bike manual uploaded. Answer all questions strictly from this document only.",
    ])
    return chat


def extract_maintenance(client: genai.Client, manual_file) -> list:
    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=[manual_file, MAINTENANCE_PROMPT],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
        ),
    )
    raw = resp.text.strip()
    # Fallback: strip any markdown fences if model still wraps output
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip("json").strip()
    return json.loads(raw)


def maintenance_status(item: dict, current_km: int, last_km: int) -> tuple[str, str]:
    """Return (label, css_class) for a maintenance item."""
    km_int = item.get("interval_km")
    if km_int is None:
        return "No KM data", "badge-info"
    due_at = last_km + km_int
    remaining = due_at - current_km
    if remaining < 0:
        return f"Overdue by {abs(remaining):,} km", "badge-overdue"
    if remaining <= km_int * 0.1:
        return f"Due in {remaining:,} km", "badge-due"
    return f"OK — {remaining:,} km left", "badge-ok"


# ── Session state ─────────────────────────────────────────────────────────────

def init_session() -> None:
    defaults = {
        "client": None,
        "chat": None,
        "manual_name": "",
        "display_history": [],
        "pdf_bytes": None,
        "pdf_filename": "",
        "manual_file_ref": None,
        "applied_lang": "English",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def load_manual(pdf_bytes: bytes, filename: str, model_name: str, language: str) -> None:
    name = model_name.strip() or filename.rsplit(".", 1)[0]
    client = st.session_state.client
    with st.spinner("Reading your manual… ~10 seconds"):
        manual_file = upload_pdf(client, pdf_bytes, name)
        chat = create_chat(client, manual_file, language)
    st.session_state.update({
        "chat": chat,
        "manual_name": name,
        "pdf_bytes": pdf_bytes,
        "pdf_filename": filename,
        "display_history": [],
        "manual_file_ref": manual_file,
        "applied_lang": language,
    })


# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Bike Assistant · Sarvam AI", page_icon="🏍️", layout="centered")
st.markdown(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0&display=block" rel="stylesheet">',
    unsafe_allow_html=True,
)
st.markdown(SARVAM_CSS, unsafe_allow_html=True)
init_session()

if st.session_state.client is None:
    st.session_state.client = get_client()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-logo">🏍️</div>
        <div>
            <div class="sidebar-name">Sarvam AI</div>
            <div class="sidebar-tagline">Bike Assistant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    bike_model = st.text_input("Bike Model", placeholder="e.g. Royal Enfield Classic 350")

    language_key = st.selectbox("Response Language", list(LANGUAGES.keys()))
    selected_lang = LANGUAGES[language_key]

    pdf = st.file_uploader("Upload Manual (PDF)", type=["pdf"])

    if pdf and st.button("Load Manual", type="primary", use_container_width=True):
        try:
            load_manual(pdf.read(), pdf.name, bike_model, selected_lang)
            st.rerun()
        except Exception as exc:
            st.error(f"Failed: {exc}")

    st.divider()

    if st.session_state.chat:
        st.markdown(f"""
        <div class="manual-pill">
            <span class="manual-pill-icon">📖</span>
            <span class="manual-pill-name">{st.session_state.manual_name}</span>
        </div>
        """, unsafe_allow_html=True)

        # Switch language without reloading manual
        if selected_lang != st.session_state.applied_lang:
            try:
                st.session_state.chat.send_message(
                    f"From now on, respond ONLY in {selected_lang}. Switch immediately and maintain this for all future responses."
                )
                st.session_state.applied_lang = selected_lang
            except Exception:
                pass

        if st.button("Clear Chat", use_container_width=True):
            try:
                load_manual(
                    st.session_state.pdf_bytes,
                    st.session_state.pdf_filename,
                    st.session_state.manual_name,
                    st.session_state.applied_lang,
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Error: {exc}")
    else:
        st.markdown("""
        <div style="background:#F8F6F3;border:1px solid #E8E2DC;border-radius:10px;padding:10px 14px;font-size:0.85rem;color:#777;">
            Upload a manual to begin.
        </div>
        """, unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="sarvam-hero">
    <div class="sarvam-logo-dot">🏍️</div>
    <div>
        <div class="sarvam-subtitle">Sarvam AI</div>
        <div class="sarvam-title">Bike Troubleshooting Assistant</div>
    </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.chat:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="sarvam-card sarvam-card-full">
        <div class="sarvam-card-label">Get Started</div>
        <div class="sarvam-card-title">Upload your bike's manual</div>
        <div class="sarvam-card-desc">Upload the Owner's or Service Manual PDF from the sidebar. The assistant reads the entire document and answers only from it — in your preferred language.</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="sarvam-card">
            <div class="sarvam-card-label">10 Languages</div>
            <div class="sarvam-card-title">Ask in your language</div>
            <div class="sarvam-card-desc">Hindi, Tamil, Telugu, Kannada, Malayalam and more.</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="sarvam-card">
            <div class="sarvam-card-label">Image</div>
            <div class="sarvam-card-title">Attach a photo</div>
            <div class="sarvam-card-desc">White smoke, leaks, warning lights — the AI sees it and references your manual.</div>
        </div>
        """, unsafe_allow_html=True)

    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────

# ── Troubleshoot ─────────────────────────────────────────────────────────────

if True:
    for msg in st.session_state.display_history:
        avatar = "👤" if msg["role"] == "user" else "🏍️"
        with st.chat_message(msg["role"], avatar=avatar):
            if msg.get("image_bytes"):
                st.image(msg["image_bytes"], width=300)
            st.markdown(msg["text"])

    with st.form("input_form", clear_on_submit=True):
        question = st.text_area(
            "Question",
            placeholder="Describe your bike issue — e.g. white smoke from exhaust, strange noise, warning light…",
            height=95,
            label_visibility="collapsed",
        )
        img_file = st.file_uploader("📷 Attach an image of the issue (optional)", type=["jpg", "jpeg", "png", "webp"])
        submitted = st.form_submit_button("Send →", type="primary", use_container_width=True)

    if submitted and (question.strip() or img_file):
        text = question.strip() or "What issue does this image show? What does the manual say about it?"
        parts = []
        img_bytes = None

        if img_file:
            img_bytes = img_file.read()
            parts.append(Image.open(io.BytesIO(img_bytes)))

        parts.append(text)
        st.session_state.display_history.append({"role": "user", "text": text, "image_bytes": img_bytes})

        with st.chat_message("assistant", avatar="🏍️"):
            with st.spinner("Consulting your manual…"):
                answer = None
                for attempt in range(3):
                    try:
                        response = st.session_state.chat.send_message(parts)
                        answer = response.text
                        break
                    except Exception as exc:
                        if "503" in str(exc) and attempt < 2:
                            time.sleep(5)
                            continue
                        answer = f"⚠️ Error: {exc}"

        st.session_state.display_history.append({"role": "assistant", "text": answer})
        st.rerun()

