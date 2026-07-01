"""
Bike Troubleshooting Bot — Sarvam AI
True RAG: PDF → PyMuPDF chunks → Gemini embeddings → FAISS → top-k retrieval → Gemini answer
Supports multiple manuals searched simultaneously.
"""

import io
import os
import re
import time

import numpy as np
import faiss
import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv
from gtts import gTTS
from PIL import Image
from google import genai
from google.genai import types

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME    = "gemini-2.5-flash"
EMBED_MODEL   = "models/gemini-embedding-001"
EMBED_DIM     = 3072
CHUNK_SIZE    = 900     # characters per chunk
CHUNK_OVERLAP = 150     # overlap between consecutive chunks
TOP_K         = 6       # chunks retrieved per query
EMBED_BATCH   = 50      # texts per embedding API call

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

TTS_LANG_MAP = {
    "English": "en", "Hindi": "hi", "Tamil": "ta", "Telugu": "te",
    "Kannada": "kn", "Malayalam": "ml", "Marathi": "mr",
    "Bengali": "bn", "Gujarati": "gu", "Punjabi": "pa",
}


# ── Sarvam AI Brand Styles ────────────────────────────────────────────────────

SARVAM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');

* {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    box-sizing: border-box;
}

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

[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
section[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[data-testid="stSidebarCollapseButton"] { display: none !important; visibility: hidden !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1.5px solid #F0EDE8 !important;
}

[data-testid="stSidebar"] * { color: #111111 !important; }

[data-testid="stSidebar"] [data-baseweb="input"],
[data-testid="stSidebar"] [data-baseweb="base-input"],
[data-testid="stSidebar"] .stTextInput div,
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #F8F6F3 !important;
    border: 1px solid #E8E2DC !important;
    border-radius: 8px !important;
    box-shadow: none !important;
    color: #111 !important;
}

[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] [data-baseweb="input"] input,
[data-testid="stSidebar"] [data-baseweb="base-input"] input {
    background: #F8F6F3 !important;
    border: none !important;
    box-shadow: none !important;
    color: #111 !important;
    caret-color: #333 !important;
}

[data-testid="stSidebar"] .stTextInput input::placeholder { color: #BBB !important; }

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {
    background: #FFF8F5 !important;
    border: 1.5px dashed #F26522 !important;
    border-radius: 10px !important;
}

[data-testid="stFileUploaderDropzone"] .material-symbols-rounded,
[data-testid="stFileUploaderDropzoneInstructions"] .material-symbols-rounded {
    display: none !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] > *:first-child {
    display: none !important;
}

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

/* File row — white text, aligned */
[data-testid="stSidebar"] .e1dmulp2,
[data-testid="stSidebar"] .e1dmulp2 *,
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"],
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] * {
    color: #FFFFFF !important;
}

/* Keep filename and X on the same line */
[data-testid="stSidebar"] .e1dmulp2 {
    display: flex !important;
    align-items: center !important;
    flex-direction: row !important;
}

/* Hide file size */
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] small,
[data-testid="stSidebar"] .e1dmulp2 small {
    display: none !important;
}

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

button[aria-label^="Remove "] * { display: none !important; visibility: hidden !important; }

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
    background: #F8F6F3 !important;
    border: 1px solid #E8E2DC !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
    color: #111 !important;
    resize: none !important;
    box-shadow: none !important;
    caret-color: #333 !important;
}

.stTextArea textarea:focus {
    border-color: #E8E2DC !important;
    box-shadow: none !important;
    outline: none !important;
}

.stTextArea textarea::placeholder { color: #BBB !important; }

[data-testid="stForm"] [data-testid="stFileUploader"] label { color: #555 !important; font-size: 0.85rem !important; }

.stNumberInput input {
    background: #fff !important;
    border: 1.5px solid #E8E2DC !important;
    border-radius: 10px !important;
}

hr { border-color: #F0EDE8 !important; }

[data-testid="stSpinner"] > div { border-color: #F26522 !important; }
[data-testid="stSpinner"] p,
[data-testid="stSpinner"] span { color: #777 !important; font-size: 0.88rem !important; }

/* ── Sidebar brand ── */
.sidebar-brand { display: flex; align-items: center; gap: 10px; padding: 4px 0 16px 0; }

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

/* ── Manual pill ── */
.manual-pill {
    display: flex; align-items: center; gap: 8px;
    background: #FFF4EE;
    border: 1.5px solid #F26522;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 6px;
}

.manual-pill-icon { font-size: 1rem; flex-shrink: 0; }
.manual-pill-name { font-size: 0.82rem; font-weight: 600; color: #111 !important; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex: 1; }
.manual-pill-meta { font-size: 0.7rem; color: #F26522 !important; font-weight: 600; white-space: nowrap; }

/* Manual pill row — keep pill and X button on the same line */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
    align-items: center !important;
    gap: 6px !important;
}

[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] .stColumn {
    display: flex !important;
    align-items: center !important;
    padding: 0 !important;
}

[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] .manual-pill {
    margin-bottom: 0 !important;
}

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

.sarvam-title    { font-size: 1.5rem; font-weight: 800; color: #111; letter-spacing: -0.03em; line-height: 1.2; }
.sarvam-subtitle { font-size: 0.72rem; color: #F26522; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }

/* ── Welcome cards ── */
.sarvam-card {
    background: #FFFFFF;
    border: 1px solid #F0EDE8;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
    height: 130px;
    display: flex; flex-direction: column; justify-content: center;
}

.sarvam-card-full { height: auto !important; }

.sarvam-card:hover {
    box-shadow: 0 6px 24px rgba(242, 101, 34, 0.1);
    transform: translateY(-2px);
}

.sarvam-card-label { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #F26522; margin-bottom: 6px; }
.sarvam-card-title { font-size: 1.05rem; font-weight: 600; color: #111; margin-bottom: 4px; }
.sarvam-card-desc  { font-size: 0.83rem; color: #777; line-height: 1.5; }

/* ── Source citations ── */
.rag-sources {
    margin-top: 10px;
    padding: 8px 12px;
    background: #F8F6F3;
    border-left: 3px solid #F26522;
    border-radius: 0 6px 6px 0;
    font-size: 0.75rem;
    color: #888 !important;
}
</style>
"""


# ── RAG Pipeline ──────────────────────────────────────────────────────────────

def get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found.")
        st.stop()
    return genai.Client(api_key=api_key)


def extract_chunks(pdf_bytes: bytes, source_name: str) -> list[dict]:
    """PyMuPDF: extract text page-by-page, split into overlapping character chunks."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    chunks = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if not text:
            continue
        # Slide a window over the page text
        start = 0
        while start < len(text):
            snippet = text[start : start + CHUNK_SIZE].strip()
            if snippet:
                chunks.append({"text": snippet, "page": page_num, "source": source_name})
            start += CHUNK_SIZE - CHUNK_OVERLAP

    doc.close()
    return chunks


def embed_texts(client: genai.Client, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
    """Embed texts in batches; return L2-normalised float32 array (n, EMBED_DIM)."""
    all_vecs = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        for emb in result.embeddings:
            all_vecs.append(emb.values)

    arr = np.array(all_vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return arr / norms


def build_index(embeddings: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(EMBED_DIM)   # inner product == cosine on normalised vecs
    index.add(embeddings)
    return index


def index_manual(client: genai.Client, pdf_bytes: bytes, display_name: str) -> dict:
    """Full indexing pipeline: PDF → chunks → embeddings → FAISS."""
    chunks = extract_chunks(pdf_bytes, display_name)
    if not chunks:
        raise ValueError(
            "No extractable text found. The PDF may be a scanned image — OCR is not supported yet."
        )

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(client, texts, task_type="RETRIEVAL_DOCUMENT")
    index = build_index(embeddings)
    n_pages = max(c["page"] for c in chunks)

    return {
        "index": index,
        "chunks": chunks,
        "display_name": display_name,
        "n_chunks": len(chunks),
        "n_pages": n_pages,
    }


def retrieve(client: genai.Client, query: str, manuals: dict) -> list[dict]:
    """Embed the query, search all loaded manual indexes, return top-k chunks."""
    if not manuals:
        return []

    q_vec = embed_texts(client, [query], task_type="RETRIEVAL_QUERY")  # (1, EMBED_DIM)

    all_hits = []
    for manual in manuals.values():
        n = min(TOP_K, manual["index"].ntotal)
        scores, idxs = manual["index"].search(q_vec, n)
        for score, idx in zip(scores[0], idxs[0]):
            if idx >= 0:
                all_hits.append({**manual["chunks"][idx], "score": float(score)})

    all_hits.sort(key=lambda x: x["score"], reverse=True)
    return all_hits[:TOP_K]


def build_rag_prompt(query: str, chunks: list[dict], history: list[dict], language: str) -> str:
    context_parts = []
    for c in chunks:
        context_parts.append(f"[{c['source']} — Page {c['page']}]\n{c['text']}")
    context = "\n\n---\n\n".join(context_parts)

    # Last 2 turns as conversation memory
    history_lines = []
    for msg in history[-4:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_lines.append(f"{role}: {msg['text'][:400]}")
    history_block = ("\n\nRECENT CONVERSATION:\n" + "\n".join(history_lines)) if history_lines else ""

    return f"""You are a bike troubleshooting assistant. Use ONLY the manual excerpts below.

MANUAL EXCERPTS:
{context}
{history_block}

RULES:
1. Answer ONLY from the excerpts above. Cite the source name and page number.
2. If the answer is not found in the excerpts, say exactly: "This is not covered in the provided manual. Please consult an authorized service center."
3. Never use general automotive knowledge not present in the manual.
4. Use numbered lists for procedures.
5. Do not guess or speculate.

LANGUAGE: Respond in {language} only.

User: {query}"""


# ── TTS ───────────────────────────────────────────────────────────────────────

def text_to_speech(text: str, language: str) -> bytes:
    lang_code = TTS_LANG_MAP.get(language, "en")
    clean = text.replace("**", "").replace("*", "").replace("#", "").replace("`", "")
    clean = clean[:600].rsplit(" ", 1)[0] if len(clean) > 600 else clean
    tts = gTTS(text=clean, lang=lang_code, slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


# ── Session state ─────────────────────────────────────────────────────────────

def init_session() -> None:
    defaults = {
        "client": None,
        "manuals": {},           # {key: {index, chunks, display_name, n_chunks, n_pages}}
        "display_history": [],
        "applied_lang": "English",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


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

client = st.session_state.client

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
    st.session_state.applied_lang = selected_lang

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    pdf = st.file_uploader("Upload Manual (PDF)", type=["pdf"])

    if pdf and st.button("Index Manual", type="primary", use_container_width=True):
        key = pdf.name
        if key in st.session_state.manuals:
            st.warning(f"**{pdf.name}** is already indexed.")
        else:
            try:
                with st.spinner(f"Indexing {pdf.name}…"):
                    result = index_manual(client, pdf.read(), bike_model.strip() or pdf.name.rsplit(".", 1)[0])
                st.session_state.manuals[key] = result
                st.rerun()
            except Exception as exc:
                st.error(f"Failed: {exc}")

    # ── Loaded manuals list ───────────────────────────────────────────────────
    if st.session_state.manuals:
        st.divider()
        st.markdown(
            "<div style='font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;"
            "color:#F26522;margin-bottom:8px;'>Loaded Manuals</div>",
            unsafe_allow_html=True,
        )
        to_remove = None
        for key, manual in st.session_state.manuals.items():
            col_info, col_btn = st.columns([5, 1], vertical_alignment="center")
            with col_info:
                st.markdown(
                    f"""<div class="manual-pill">
                        <span class="manual-pill-icon">📖</span>
                        <span class="manual-pill-name">{manual['display_name']}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with col_btn:
                if st.button("✕", key=f"remove_{key}"):
                    to_remove = key

        if to_remove:
            del st.session_state.manuals[to_remove]
            # Clear TTS cache
            for k in list(st.session_state.keys()):
                if k.startswith("tts_audio_"):
                    del st.session_state[k]
            st.rerun()

        st.divider()
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.display_history = []
            for k in list(st.session_state.keys()):
                if k.startswith("tts_audio_"):
                    del st.session_state[k]
            st.rerun()

    else:
        st.markdown("""
        <div style="background:#F8F6F3;border:1px solid #E8E2DC;border-radius:10px;
                    padding:10px 14px;font-size:0.85rem;color:#777;margin-top:8px;">
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

if not st.session_state.manuals:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="sarvam-card sarvam-card-full">
        <div class="sarvam-card-label">Get Started</div>
        <div class="sarvam-card-title">Upload your bike's manual</div>
        <div class="sarvam-card-desc">Upload one or more Owner's / Service Manual PDFs from the sidebar. Each is chunked and indexed locally — only the most relevant excerpts are sent to the AI per query, keeping costs low.</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="sarvam-card">
            <div class="sarvam-card-label">Multi-manual RAG</div>
            <div class="sarvam-card-title">Multiple manuals</div>
            <div class="sarvam-card-desc">Upload several PDFs — all are searched simultaneously per query.</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="sarvam-card">
            <div class="sarvam-card-label">10 Languages</div>
            <div class="sarvam-card-title">Ask in your language</div>
            <div class="sarvam-card-desc">Hindi, Tamil, Telugu, Kannada, Malayalam and more.</div>
        </div>
        """, unsafe_allow_html=True)

    st.stop()

# ── Chat history ──────────────────────────────────────────────────────────────

for i, msg in enumerate(st.session_state.display_history):
    avatar = "👤" if msg["role"] == "user" else "🏍️"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg.get("image_bytes"):
            st.image(msg["image_bytes"], width=300)
        st.markdown(msg["text"])

        if msg["role"] == "assistant":
            # TTS button
            tts_key = f"tts_audio_{i}"
            if st.button("🔊 Listen", key=f"tts_btn_{i}"):
                with st.spinner("Generating audio…"):
                    try:
                        st.session_state[tts_key] = text_to_speech(
                            msg["text"], st.session_state.applied_lang
                        )
                    except Exception as exc:
                        st.error(f"Audio error: {exc}")
            if st.session_state.get(tts_key):
                st.audio(st.session_state[tts_key], format="audio/mp3")

# ── Input form ────────────────────────────────────────────────────────────────

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
    img_bytes = None
    pil_image = None

    if img_file:
        img_bytes = img_file.read()
        pil_image = Image.open(io.BytesIO(img_bytes))

    st.session_state.display_history.append({"role": "user", "text": text, "image_bytes": img_bytes})

    with st.chat_message("assistant", avatar="🏍️"):
        # Retrieve relevant chunks
        with st.spinner("Searching manual…"):
            search_query = text
            if pil_image:
                # Describe the image first so retrieval finds the right manual section
                try:
                    desc = client.models.generate_content(
                        model=MODEL_NAME,
                        contents=[pil_image, "List the bike parts or components visible in this image. Be brief — part names only."],
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_budget=0)
                        ),
                    )
                    search_query = f"{text} {desc.text}"
                except Exception:
                    pass  # fall back to text-only query
            chunks = retrieve(client, search_query, st.session_state.manuals)

        # Build prompt and stream answer
        prompt = build_rag_prompt(
            text, chunks, st.session_state.display_history[:-1], st.session_state.applied_lang
        )

        parts = []
        if pil_image:
            parts.append(pil_image)
        parts.append(prompt)

        answer = None
        for attempt in range(3):
            try:
                stream = client.models.generate_content_stream(
                    model=MODEL_NAME,
                    contents=parts,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                answer = st.write_stream(chunk.text for chunk in stream if chunk.text)
                break
            except Exception as exc:
                if "503" in str(exc) and attempt < 2:
                    time.sleep(5)
                    continue
                answer = f"⚠️ Error: {exc}"
                st.error(answer)

    st.session_state.display_history.append({
        "role": "assistant",
        "text": answer or "",
        "sources": chunks,
    })
    st.rerun()
