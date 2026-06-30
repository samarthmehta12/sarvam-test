"""
Bike Troubleshooting Bot
Uses Gemini's File API to ingest the entire manual and answer questions
strictly from the document. Supports text + image input.
"""

import io
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

SYSTEM_INSTRUCTION = """You are a bike troubleshooting assistant. The user has provided their bike's official manual as a document.

STRICT RULES — YOU MUST FOLLOW THESE EXACTLY:
1. Answer ONLY using information explicitly found in the provided manual document.
2. If the answer is not present in the manual, say exactly:
   "This is not covered in the provided manual. Please consult an authorized service center."
3. Never use general automotive knowledge or assumptions not stated in the manual.
4. Reference page numbers or section/chapter titles when they appear in the document.
5. When the user provides an image: first describe what you observe, then find and cite the relevant section of the manual.
6. When the manual provides a step-by-step procedure, present it as a numbered list.
7. Do not guess, speculate, or add any information beyond what is explicitly written in the manual."""


# ── Gemini helpers ────────────────────────────────────────────────────────────

def get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found. Create a .env file with GEMINI_API_KEY=your_key.")
        st.stop()
    return genai.Client(api_key=api_key)


def upload_pdf(client: genai.Client, pdf_bytes: bytes, display_name: str):
    """Write bytes to a temp file, upload to Gemini File API, return active file object."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        uploaded = client.files.upload(
            file=tmp_path,
            config=types.UploadFileConfig(
                mime_type="application/pdf",
                display_name=display_name,
            ),
        )
        # Wait for Gemini to finish processing
        for _ in range(60):
            info = client.files.get(name=uploaded.name)
            if info.state.name == "ACTIVE":
                return info
            if info.state.name == "FAILED":
                raise RuntimeError("Gemini file processing failed.")
            time.sleep(2)
        raise TimeoutError("File processing timed out.")
    finally:
        os.unlink(tmp_path)


def create_chat(client: genai.Client, manual_file) -> genai.chats.Chat:
    """Create a Gemini chat session seeded with the manual."""
    chat = client.chats.create(
        model=MODEL_NAME,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
        ),
    )
    # Seed: send the PDF so it stays in chat history
    chat.send_message([
        manual_file,
        "I have uploaded the bike manual. Please answer all my questions strictly from this document only.",
    ])
    return chat


# ── Session state ─────────────────────────────────────────────────────────────

def init_session() -> None:
    defaults = {
        "client": None,
        "chat": None,
        "manual_name": "",
        "display_history": [],
        "pdf_bytes": None,
        "pdf_filename": "",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def load_manual(pdf_bytes: bytes, filename: str, model_name: str) -> None:
    name = model_name.strip() or filename.rsplit(".", 1)[0]
    client = st.session_state.client

    with st.spinner("Uploading & indexing manual… this takes ~10 s for large files"):
        manual_file = upload_pdf(client, pdf_bytes, name)
        chat = create_chat(client, manual_file)

    st.session_state.chat = chat
    st.session_state.manual_name = name
    st.session_state.pdf_bytes = pdf_bytes
    st.session_state.pdf_filename = filename
    st.session_state.display_history = []


# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Bike Troubleshooter", page_icon="🏍️", layout="centered")
init_session()

if st.session_state.client is None:
    st.session_state.client = get_client()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏍️ Bike Manual Bot")
    st.caption(f"Model: {MODEL_NAME}")
    st.divider()

    bike_model = st.text_input(
        "Bike Model",
        placeholder="e.g. Royal Enfield Classic 350",
    )
    pdf = st.file_uploader("📄 Upload Owner's / Service Manual (PDF)", type=["pdf"])

    if pdf and st.button("Load Manual", type="primary", use_container_width=True):
        try:
            load_manual(pdf.read(), pdf.name, bike_model)
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to load manual: {exc}")

    st.divider()

    if st.session_state.chat:
        st.success(f"📖 **{st.session_state.manual_name}**")

        if st.button("Clear Chat", use_container_width=True):
            try:
                load_manual(
                    st.session_state.pdf_bytes,
                    st.session_state.pdf_filename,
                    st.session_state.manual_name,
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Error resetting: {exc}")
    else:
        st.info("Upload a manual to start.")

# ── Main ──────────────────────────────────────────────────────────────────────

st.markdown("## Bike Troubleshooting Assistant")

if not st.session_state.chat:
    st.markdown("""
Upload your bike's **Owner's Manual** or **Service Manual** (PDF) in the sidebar.

**Try asking:**
- *"My engine makes a knocking sound at idle — what could cause this?"*
- *"White smoke is coming from the exhaust"* ← attach a photo
- *"What does the oil pressure warning light mean?"*
- *"How do I adjust the clutch cable?"*

> The bot answers **strictly from your manual**. If the manual doesn't cover it, you'll be told to visit a service center.
""")
    st.stop()

# ── Chat history ──────────────────────────────────────────────────────────────

for msg in st.session_state.display_history:
    with st.chat_message(msg["role"]):
        if msg.get("image_bytes"):
            st.image(msg["image_bytes"], width=300)
        st.markdown(msg["text"])

# ── Input form ────────────────────────────────────────────────────────────────

with st.form("input_form", clear_on_submit=True):
    question = st.text_area(
        "Describe your bike issue",
        placeholder="e.g. White smoke from exhaust, engine overheating, strange noise…",
        height=95,
        label_visibility="collapsed",
    )
    img_file = st.file_uploader(
        "📷 Attach an image of the issue (optional)",
        type=["jpg", "jpeg", "png", "webp"],
    )
    submitted = st.form_submit_button("Send →", type="primary", use_container_width=True)

if submitted and (question.strip() or img_file):
    text = (
        question.strip()
        or "What issue does this image show? What does the manual say about it?"
    )

    parts = []
    img_bytes = None

    if img_file:
        img_bytes = img_file.read()
        pil_img = Image.open(io.BytesIO(img_bytes))
        parts.append(pil_img)

    parts.append(text)

    st.session_state.display_history.append(
        {"role": "user", "text": text, "image_bytes": img_bytes}
    )

    with st.spinner("Consulting manual…"):
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
