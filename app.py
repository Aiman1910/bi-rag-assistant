"""Business Intelligence Assistant - professional Streamlit UI on top of rag.py."""
import os
import time
from pathlib import Path

import streamlit as st

import rag

st.set_page_config(
    page_title="Business Intelligence Assistant",
    page_icon="📊",
    layout="wide",
)

DATA_DIR = Path(__file__).parent
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODEL = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash")
MODELS = [MODEL_NAME, FALLBACK_MODEL]

QUESTIONS = [
    "Summarize the sales report.",
    "What product information is available?",
    "What information is available about stores?",
    "What does the data dictionary explain?",
]

# ------------------------------------------------------------------ styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, .stApp, button, input, textarea, [data-testid="stMarkdownContainer"] {
    font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif !important;
}
header[data-testid="stHeader"] {background: transparent;}
.block-container {max-width: 980px; padding-top: 2.2rem; padding-bottom: 6rem;}

/* ---- hero ---- */
.hero {
    background: linear-gradient(135deg, #0B1F3A 0%, #12335F 62%, #1D4ED8 140%);
    border-radius: 18px;
    padding: 2rem 2.2rem;
    margin-bottom: 1.6rem;
    box-shadow: 0 10px 30px rgba(11, 31, 58, .18);
}
.hero .eyebrow {
    color: #7DD3FC; font-size: .75rem; font-weight: 600;
    letter-spacing: .14em; text-transform: uppercase; margin-bottom: .5rem;
}
.hero h1 {
    color: #FFFFFF; font-size: 2rem; font-weight: 700; letter-spacing: -.02em;
    margin: 0; padding: 0; line-height: 1.2;
}
.hero p {color: #CBD5E1; font-size: 1rem; line-height: 1.6; margin: .6rem 0 1.1rem 0; max-width: 640px;}
.pill {
    display: inline-block; color: #E2E8F0; font-size: .78rem; font-weight: 500;
    padding: .3rem .8rem; margin: 0 .5rem .4rem 0; border-radius: 999px;
    background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.18);
}

/* ---- empty state ---- */
.empty-title {font-size: 1.25rem; font-weight: 600; margin: .4rem 0 .2rem 0; letter-spacing: -.01em;}
.empty-sub {font-size: .92rem; opacity: .7; margin-bottom: 1rem;}
.stMain .stButton > button {
    text-align: left; justify-content: flex-start; font-weight: 500; font-size: .92rem;
    padding: .9rem 1.1rem; border-radius: 12px;
    border: 1px solid rgba(100,116,139,.28);
    transition: border-color .15s, box-shadow .15s, transform .15s;
}
.stMain .stButton > button:hover {
    border-color: #2563EB; box-shadow: 0 4px 14px rgba(37,99,235,.14); transform: translateY(-1px);
}

/* ---- chat ---- */
[data-testid="stChatMessage"] {
    background: rgba(127,127,127,.07);
    border: 1px solid rgba(127,127,127,.18);
    border-radius: 14px; padding: 1rem 1.3rem; margin-bottom: .8rem;
}
[data-testid="stChatInput"] {border-radius: 14px;}
[data-testid="stExpander"] {border-radius: 10px; border: 1px solid rgba(127,127,127,.2);}

/* ---- sidebar (always navy, independent of theme) ---- */
[data-testid="stSidebar"] {background: #0B1F3A !important; border-right: 1px solid rgba(255,255,255,.06);}
[data-testid="stSidebar"] * {color: #E2E8F0;}
.brand {display: flex; align-items: center; gap: .75rem; margin: .4rem 0 1.4rem 0;}
.brand .logo {
    width: 40px; height: 40px; border-radius: 10px; flex: none;
    background: linear-gradient(135deg, #2563EB, #38BDF8);
    display: flex; align-items: center; justify-content: center;
    color: #fff !important; font-weight: 700; font-size: .95rem; letter-spacing: .02em;
}
.brand .name {font-weight: 600; font-size: .98rem; line-height: 1.2; color: #FFFFFF !important;}
.brand .tag {font-size: .72rem; color: #94A3B8 !important; margin-top: .1rem;}
.side-label {
    font-size: .68rem; font-weight: 600; letter-spacing: .12em; text-transform: uppercase;
    color: #7C8DA6 !important; margin: 1.2rem 0 .6rem 0;
}
.stat {
    background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.08);
    border-radius: 12px; padding: .75rem .9rem; margin-bottom: .55rem;
}
.stat .v {font-size: 1.35rem; font-weight: 700; color: #FFFFFF !important; line-height: 1.1;}
.stat .l {font-size: .74rem; color: #94A3B8 !important; margin-top: .15rem;}
.note {font-size: .78rem; line-height: 1.55; color: #94A3B8 !important;}
[data-testid="stSidebar"] .stButton > button {
    background: transparent; border: 1px solid rgba(255,255,255,.22);
    border-radius: 10px; font-weight: 500; font-size: .88rem;
}
[data-testid="stSidebar"] .stButton > button:hover {background: rgba(255,255,255,.08); border-color: rgba(255,255,255,.4);}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------- setup
def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY")


api_key = get_api_key()
if not api_key:
    st.error("GEMINI_API_KEY is not set. Add it in Streamlit 'Secrets' (see README).")
    st.stop()

client = rag.make_client(api_key)


@st.cache_resource(show_spinner="Reading reports and building the search index (first load only)...")
def load_index():
    return rag.build_index(client, DATA_DIR)


try:
    chunks, embeddings = load_index()
except Exception as e:
    st.error(f"Could not build the index: {e}")
    st.stop()


def answer_with_retry(question, retries=2):
    last_error = None
    for model in MODELS:
        for attempt in range(retries):
            try:
                return rag.answer_question(client, model, chunks, embeddings, question)
            except Exception as e:
                last_error = e
                msg = str(e)
                if "503" in msg or "UNAVAILABLE" in msg or "429" in msg:
                    time.sleep(2 * (attempt + 1))
                    continue
                if "404" in msg or "NOT_FOUND" in msg:
                    break
                raise
    if last_error:
        return "The AI service is busy right now. Please try again in a minute."
    return "No answer was generated."


def render_answer(text):
    body, _, sources = text.partition("\n\n**Sources**\n")
    st.markdown(body)
    if sources.strip():
        with st.expander("Sources", icon=":material/source:"):
            st.markdown(sources)


def ask(question):
    st.session_state.pending = question


# --------------------------------------------------------------- sidebar
if "messages" not in st.session_state:
    st.session_state.messages = []

n_sources = len({c["source"] for c in chunks})
n_passages = len(chunks)

with st.sidebar:
    st.markdown(
        """
<div class="brand">
  <div class="logo">BI</div>
  <div><div class="name">Business Intelligence</div><div class="tag">Insights Assistant</div></div>
</div>
<div class="side-label">Knowledge base</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
<div class="stat"><div class="v">{n_sources}</div><div class="l">Source files indexed</div></div>
<div class="stat"><div class="v">{n_passages:,}</div><div class="l">Searchable passages</div></div>
<div class="side-label">About</div>
<div class="note">Answers are generated from your CSV datasets and PDF reports. Every response lists the sources it used.</div>
<div class="side-label">Conversation</div>
""",
        unsafe_allow_html=True,
    )
    if st.button("Clear conversation", icon=":material/delete_sweep:", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------------------- main page
st.markdown(
    """
<div class="hero">
  <div class="eyebrow">Business Intelligence</div>
  <h1>Insights Assistant</h1>
  <p>Ask questions about sales, products, stores, customers and exchange rates. Answers are grounded in your own data and cite their sources.</p>
  <span class="pill">Grounded in your data</span><span class="pill">Source citations</span><span class="pill">CSV and PDF reports</span>
</div>
""",
    unsafe_allow_html=True,
)

pending = st.session_state.pop("pending", None)

empty_slot = st.empty()
if not st.session_state.messages and not pending:
    with empty_slot.container():
        st.markdown(
            '<div class="empty-title">What would you like to know?</div>'
            '<div class="empty-sub">Choose a question to get started, or type your own below.</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        for i, q in enumerate(QUESTIONS):
            cols[i % 2].button(q, key=f"q{i}", on_click=ask, args=(q,), use_container_width=True)

for m in st.session_state.messages:
    avatar = ":material/person:" if m["role"] == "user" else ":material/query_stats:"
    with st.chat_message(m["role"], avatar=avatar):
        if m["role"] == "assistant":
            render_answer(m["content"])
        else:
            st.markdown(m["content"])

typed = st.chat_input("Ask a question about the business data...")
question = typed or pending

if question:
    empty_slot.empty()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(question)
    with st.chat_message("assistant", avatar=":material/query_stats:"):
        with st.spinner("Analyzing your data..."):
            try:
                answer = answer_with_retry(question)
            except Exception as e:
                answer = f"Sorry, the request failed: {e}"
        render_answer(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
