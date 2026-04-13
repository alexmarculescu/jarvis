import json
import os
from pathlib import Path
from typing import List, Tuple

import chromadb
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODELS_URL = "http://localhost:11434/api/tags"
DEFAULT_MODEL = "qwen2.5:14b"

BASE_DIR = Path.home() / "jarvis"
DOCS_DIR = BASE_DIR / "docs"
DB_DIR = BASE_DIR / "db"
COLLECTION_NAME = "jarvis_docs_v2"

RAG_TOP_K = 8
MAX_CHAT_MESSAGES = 6
REQUEST_TIMEOUT = 120

ADMIN_PIN = os.getenv("JARVIS_ADMIN_PIN", "")
DOCS_PIN = os.getenv("JARVIS_DOCS_PIN", "")

DEFAULT_SYSTEM_PROMPT = (
    "You are Jarvis, Alex's local AI assistant. "
    "Be concise, practical, and direct."
)

DOCS_SYSTEM_PROMPT = (
    "You are Jarvis working strictly from retrieved document context.\n\n"
    "Rules:\n"
    "1. Use ONLY the retrieved context.\n"
    "2. Prefer internal CICS and support-staff documents over generic vendor manuals.\n"
    "3. If multiple sources conflict, trust internal operational documents first.\n"
    "4. Never claim you searched all files or the full archive unless the user asked for a file listing.\n"
    "5. If the answer is partial, say it is based on retrieved documents.\n"
    "6. Always mention file names when relevant.\n"
    "7. Do NOT dump bulk personal data such as all emails, all personal details, all phone numbers, or all addresses from the archive.\n"
    "8. If a user asks for bulk personal data extraction, refuse briefly and offer to help with operational or file-specific questions.\n"
    "9. If unsure, say: Not found in retrieved documents.\n"
)

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="Jarvis", page_icon="🤖", layout="wide")

# =========================
# PROFESSIONAL UI STYLE
# =========================
st.markdown(
    """
    <style>
    :root {
        --text: #111827;
        --muted: #6b7280;
        --border: #e5e7eb;
        --soft: #f8fafc;
        --panel: #ffffff;
        --accent: #111827;
    }

    .stApp {
        background: #ffffff;
    }

    .main .block-container {
        max-width: 1100px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    section[data-testid="stSidebar"] {
        background: #fcfcfd !important;
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] * {
        color: var(--text) !important;
    }

    .stChatMessage {
        border-radius: 16px;
        padding: 14px 16px;
        margin-bottom: 12px;
        border: 1px solid var(--border);
        background: #ffffff;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }

    .stMarkdown, .stText, p, label, div, span {
        color: var(--text);
    }

    .stCaption {
        color: var(--muted) !important;
    }

    .stTextInput input,
    .stTextArea textarea {
        background: #ffffff !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
    }

    div[data-baseweb="select"] > div {
        background: #ffffff !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
    }

    div[data-baseweb="select"] span {
        color: var(--text) !important;
    }

    .stButton > button {
        border-radius: 12px !important;
        border: 1px solid var(--border) !important;
        background: #ffffff !important;
        color: var(--text) !important;
        font-weight: 600 !important;
    }

    .stButton > button:hover {
        border-color: #cfd4dc !important;
        background: #fafafa !important;
    }

    .jarvis-shell {
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 22px 24px 18px 24px;
        background: linear-gradient(180deg, #ffffff 0%, #fbfbfc 100%);
        box-shadow: 0 8px 24px rgba(17,24,39,0.04);
        margin-bottom: 1rem;
    }

    .jarvis-header {
        text-align: center;
    }

    .jarvis-title {
        font-size: 3rem;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 0.15rem;
        color: #111827;
        letter-spacing: -0.03em;
    }

    .jarvis-subtitle {
        font-size: 1rem;
        color: var(--muted);
        margin-bottom: 0.2rem;
    }

    .empty-state {
        text-align: center;
        margin-top: 72px;
        color: var(--muted);
        padding: 20px;
    }

    .empty-state h3 {
        color: var(--text);
        margin-bottom: 0.35rem;
        font-size: 1.5rem;
    }

    .section-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 0.45rem;
        margin-top: 0.1rem;
    }

    .status-box {
        padding: 0.8rem 0.9rem;
        border-radius: 12px;
        background: #f9fafb;
        border: 1px solid var(--border);
        color: var(--text);
        font-weight: 600;
    }

    .answer-note {
        border-left: 3px solid #d1d5db;
        padding-left: 0.8rem;
        margin-top: 0.7rem;
        color: var(--muted);
        font-size: 0.92rem;
    }

    .sidebar-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 1rem;
    }

    .mode-option {
        display: block;
        width: 100%;
        padding: 0.65rem 0.85rem;
        border: 1px solid var(--border);
        border-radius: 12px;
        background: #ffffff;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }

    .copyright-mark {
        position: fixed;
        right: 18px;
        bottom: 10px;
        color: #9ca3af;
        font-size: 0.82rem;
        z-index: 999;
        background: rgba(255,255,255,0.9);
        padding: 4px 8px;
        border-radius: 8px;
    }

    hr {
        border-color: #eef1f4;
    }

    @media (max-width: 900px) {
        .main .block-container {
            padding-top: 0.8rem;
            padding-left: 0.9rem;
            padding-right: 0.9rem;
        }

        .jarvis-title {
            font-size: 2.35rem;
        }

        .jarvis-shell {
            padding: 18px 16px 14px 16px;
            border-radius: 16px;
        }

        .copyright-mark {
            right: 10px;
            bottom: 8px;
            font-size: 0.75rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="copyright-mark">© Red Systems</div>',
    unsafe_allow_html=True,
)

# =========================
# SESSION
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "mode" not in st.session_state:
    st.session_state.mode = "Chat"

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT

if "admin_unlocked" not in st.session_state:
    st.session_state.admin_unlocked = False

if "show_admin_panel" not in st.session_state:
    st.session_state.show_admin_panel = False

if "docs_unlocked" not in st.session_state:
    st.session_state.docs_unlocked = False

if "show_docs_pin" not in st.session_state:
    st.session_state.show_docs_pin = False

# =========================
# CACHE
# =========================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return client.get_or_create_collection(name=COLLECTION_NAME)


@st.cache_data(ttl=30)
def get_models():
    try:
        response = requests.get(MODELS_URL, timeout=10)
        response.raise_for_status()
        all_models = [m["name"] for m in response.json().get("models", [])]

        excluded_keywords = [
            "embed",
            "embedding",
            "nomic-embed",
        ]

        models = [
            name for name in all_models
            if not any(keyword in name.lower() for keyword in excluded_keywords)
        ]

        return models or [DEFAULT_MODEL]

    except Exception:
        return [DEFAULT_MODEL]

@st.cache_data(ttl=10)
def ollama_online():
    try:
        response = requests.get(MODELS_URL, timeout=5)
        return response.ok
    except Exception:
        return False


@st.cache_data
def list_docs():
    if not DOCS_DIR.exists():
        return []
    return sorted(
        [
            str(f.relative_to(DOCS_DIR))
            for f in DOCS_DIR.rglob("*")
            if f.is_file() and not f.name.startswith("._") and f.name != ".DS_Store"
        ]
    )


embedder = load_embedder()
collection = load_collection()

# =========================
# HELPERS
# =========================
def should_use_rag(prompt: str) -> bool:
    keywords = [
        "doc", "file", "pdf", "manual", "oracle",
        "document", "documents", "docs", "peoplesoft",
        "campus solutions", "cics"
    ]
    return any(k in prompt.lower() for k in keywords) or st.session_state.mode == "Docs"


def is_document_listing_request(prompt: str) -> bool:
    p = prompt.lower()
    triggers = [
        "list documents",
        "list all documents",
        "what files do you have",
        "what documents do you have",
        "show all documents",
        "show all files",
        "list all files",
        "list files",
        "search all the docs",
        "search all docs",
    ]
    return any(t in p for t in triggers)


def is_bulk_sensitive_extraction_request(prompt: str) -> bool:
    p = prompt.lower()
    sensitive_terms = [
        "all email",
        "all emails",
        "email addresses",
        "all personal details",
        "personal details",
        "all personal data",
        "all phone numbers",
        "all addresses",
        "all ids",
        "dump all",
        "extract all",
        "show me all personal",
        "sort all the documents and all the files and give me all the email",
    ]
    return any(t in p for t in sensitive_terms)


def source_priority(source: str) -> int:
    s = source.lower()

    high_priority_markers = [
        "specific setup and support staff manuals",
        "cics - ",
        "quick start for support staff",
        "onboarding workflow new employee",
        "student manuals",
        "teaching staff manuals",
        "template logic cics",
        "prepare for semester in cics",
        "important cics queries",
        "academic_calendar_workflow",
    ]
    if any(marker in s for marker in high_priority_markers):
        return 0

    mid_priority_markers = [
        "onedrive_1_4-7-2026",
        ".docx",
        ".xlsx",
        ".xls",
        ".txt",
        ".csv",
    ]
    if any(marker in s for marker in mid_priority_markers):
        return 1

    low_priority_markers = [
        "peoplesoft pdf manuals",
        "ctclink reference center",
        "pscs92",
        "oracle",
        "campus community",
        "student records",
        "academic structure",
        "admissions",
        "self service",
        "gradebook",
    ]
    if any(marker in s for marker in low_priority_markers):
        return 2

    return 3


def retrieve(query: str) -> str:
    emb = embedder.encode(query).tolist()
    res = collection.query(
        query_embeddings=[emb],
        n_results=RAG_TOP_K * 3,
        include=["documents", "metadatas", "distances"],
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0]

    scored = []
    for d, m, dist in zip(docs, metas, distances):
        group = m.get("source_group", "reference")
        group_rank = {"internal": 0, "operational": 1, "reference": 2}.get(group, 3)
        scored.append((d, m, dist, group_rank))

    scored.sort(key=lambda x: (x[3], x[2]))

    out = []
    for d, m, _, _ in scored[:RAG_TOP_K]:
        rel_name = m.get("relative_path") or m.get("filename") or "unknown"
        out.append(f"[File: {rel_name}]\n{d}")

    return "\n\n".join(out)


def stream_reply(model: str, messages: List[dict], temp: float) -> str:
    response = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temp},
        },
        stream=True,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    full = ""
    placeholder = st.empty()

    for line in response.iter_lines():
        if not line:
            continue
        try:
            data = json.loads(line.decode())
            chunk = data.get("message", {}).get("content", "")
            if chunk:
                full += chunk
                placeholder.markdown(full)
        except Exception:
            pass

    return full


def format_document_list() -> str:
    files = list_docs()
    if not files:
        return "No documents found in the docs archive."
    return "I have access to these documents:\n\n" + "\n".join(f"- {f}" for f in files)

# =========================
# HEADER
# =========================
st.markdown(
    """
    <div class="jarvis-shell">
        <div class="jarvis-header">
            <div class="jarvis-title">Jarvis</div>
            <div class="jarvis-subtitle">Powered by <b>Red Systems</b></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown('<div class="sidebar-title">Private AI Workspace</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Mode</div>', unsafe_allow_html=True)

    chat_label = "✅ Chat" if st.session_state.mode == "Chat" else "Chat"
    docs_label = "✅ Docs" if st.session_state.mode == "Docs" else "Docs"

    c1, c2 = st.columns(2)
    if c1.button(chat_label, use_container_width=True):
        st.session_state.mode = "Chat"

    if c2.button(docs_label, use_container_width=True):
        if DOCS_PIN:
            if st.session_state.docs_unlocked:
                st.session_state.mode = "Docs"
            else:
                st.session_state.show_docs_pin = True
        else:
            st.session_state.mode = "Docs"

    if st.session_state.mode == "Docs":
        if DOCS_PIN:
            if st.session_state.docs_unlocked:
                st.caption("Docs access: unlocked")
            else:
                st.caption("Docs access: locked")
        else:
            st.caption("Docs access: no PIN configured")

    if DOCS_PIN and st.session_state.docs_unlocked:
        if st.button("Lock Docs", use_container_width=True):
            st.session_state.docs_unlocked = False
            if st.session_state.mode == "Docs":
                st.session_state.mode = "Chat"
            st.rerun()

    if st.session_state.show_docs_pin and DOCS_PIN and not st.session_state.docs_unlocked:
        st.warning("Docs access requires PIN")
        docs_pin_input = st.text_input("Docs PIN", type="password", key="docs_pin_input")
        pin_c1, pin_c2 = st.columns(2)

        if pin_c1.button("Unlock Docs", use_container_width=True):
            if docs_pin_input == DOCS_PIN:
                st.session_state.docs_unlocked = True
                st.session_state.mode = "Docs"
                st.session_state.show_docs_pin = False
                st.rerun()
            else:
                st.error("Wrong PIN")

        if pin_c2.button("Cancel", use_container_width=True):
            st.session_state.show_docs_pin = False
            if st.session_state.mode == "Docs":
                st.session_state.mode = "Chat"
            st.rerun()

    st.divider()

    st.markdown('<div class="section-title">Model</div>', unsafe_allow_html=True)
    models = get_models()
    default_index = models.index(DEFAULT_MODEL) if DEFAULT_MODEL in models else 0
    model = st.selectbox("Model", models, index=default_index, label_visibility="collapsed")

    st.markdown('<div class="section-title">Temperature</div>', unsafe_allow_html=True)
    temp_default = 0.2 if st.session_state.mode == "Chat" else 0.0
    temp = st.slider("Temperature", 0.0, 1.0, temp_default, 0.1, label_visibility="collapsed")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.markdown('<div class="section-title">Status</div>', unsafe_allow_html=True)
    status_text = "🟢 Online" if ollama_online() else "🟠 Offline"
    st.markdown(f'<div class="status-box">{status_text}</div>', unsafe_allow_html=True)

    if st.session_state.mode == "Docs":
        st.caption(f"Documents visible: {len(list_docs())}")
        st.caption(f"Retrieval depth: top {RAG_TOP_K}")

    st.divider()

    if st.button("Admin tools", use_container_width=True):
        st.session_state.show_admin_panel = not st.session_state.show_admin_panel

    if st.session_state.show_admin_panel:
        if not ADMIN_PIN:
            st.info("System prompt editor is hidden. Set JARVIS_ADMIN_PIN before starting Jarvis to unlock it.")
        elif not st.session_state.admin_unlocked:
            pin = st.text_input("PIN", type="password")
            if st.button("Unlock", use_container_width=True):
                if pin == ADMIN_PIN:
                    st.session_state.admin_unlocked = True
                    st.rerun()
                else:
                    st.error("Wrong PIN")
        else:
            with st.expander("System prompt editor", expanded=False):
                st.session_state.system_prompt = st.text_area(
                    "Prompt",
                    value=st.session_state.system_prompt,
                    height=220,
                )

# =========================
# EMPTY STATE
# =========================
if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-state">
            <h3>Ask me anything</h3>
            <p>Chat normally or search your documents instantly</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================
# CHAT HISTORY
# =========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# INPUT
# =========================
st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
prompt = st.chat_input("Ask Jarvis...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        try:
            if is_bulk_sensitive_extraction_request(prompt):
                reply = (
                    "I can’t help dump bulk personal data from the document archive. "
                    "I can help with operational questions, file-specific questions, "
                    "or explain a process described in the documents."
                )
                st.markdown(reply)

            elif is_document_listing_request(prompt):
                if DOCS_PIN and not st.session_state.docs_unlocked:
                    st.session_state.show_docs_pin = True
                    reply = "🔒 Docs access is locked. Enter the PIN in the sidebar to view document-related results."
                    st.markdown(reply)
                else:
                    reply = format_document_list()
                    st.markdown(reply)

            else:
                if should_use_rag(prompt):
                    if DOCS_PIN and not st.session_state.docs_unlocked:
                        st.session_state.show_docs_pin = True
                        reply = "🔒 Docs mode is locked. Enter the PIN in the sidebar to use document retrieval."
                        st.markdown(reply)
                    else:
                        ctx = retrieve(prompt)
                        messages = [
                            {"role": "system", "content": DOCS_SYSTEM_PROMPT},
                            {"role": "user", "content": f"Retrieved context:\n{ctx}\n\nQuestion:\n{prompt}"},
                        ]
                        reply = stream_reply(model, messages, 0.0)
                        st.markdown(
                            '<div class="answer-note">Answer based on retrieved document context.</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    messages = [
                        {"role": "system", "content": st.session_state.system_prompt},
                        *st.session_state.messages[-MAX_CHAT_MESSAGES:],
                    ]
                    reply = stream_reply(model, messages, temp)

        except Exception as e:
            reply = str(e)
            st.error(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()
