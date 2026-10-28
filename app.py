"""Streamlit dashboard: Business Intelligence RAG Assistant + Data Explorer."""
import os
import time
from pathlib import Path

import pandas as pd
import pymupdf
import streamlit as st

import rag

st.set_page_config(
    page_title="Business Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
)

DATA_DIR = Path(__file__).parent


def find_files(ext):
    """Find files by extension anywhere in the repo (same idea as rag.py's rglob)."""
    found = []
    for p in DATA_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() == f".{ext}":
            rel_parts = p.relative_to(DATA_DIR).parts
            if not any(part.startswith(".") for part in rel_parts):
                found.append(p)
    return sorted(found)

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODEL = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash")
MODELS = [MODEL_NAME, FALLBACK_MODEL]

# ------------------------------------------------------------------ styling
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.4rem 1.8rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 {margin: 0; font-size: 2rem; color: white; padding: 0;}
    .hero p {margin: .3rem 0 0 0; opacity: .92; font-size: 1rem;}
    div[data-testid="stMetric"] {
        background: rgba(127,127,127,.08);
        border: 1px solid rgba(127,127,127,.2);
        padding: .8rem 1rem;
        border-radius: 12px;
    }
    section[data-testid="stSidebar"] .stRadio label {font-size: 1.05rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def hero(title, subtitle):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------- data helpers
@st.cache_data(show_spinner=False)
def load_csv(path_str):
    return rag.read_csv_safely(path_str)


@st.cache_data(show_spinner=False)
def render_pdf_page(path_str, page_index, zoom=1.6):
    with pymupdf.open(path_str) as pdf:
        pix = pdf[page_index].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        return pix.tobytes("png")


@st.cache_data(show_spinner=False)
def pdf_page_count(path_str):
    with pymupdf.open(path_str) as pdf:
        return len(pdf)


@st.cache_data(show_spinner=False)
def pdf_page_text(path_str, page_index):
    with pymupdf.open(path_str) as pdf:
        return pdf[page_index].get_text("text")


def human_size(num_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


# ---------------------------------------------------------- page: data explorer
def csv_tab():
    files = find_files("csv")
    if not files:
        st.info("No CSV files found in the repo.")
        return

    choice = st.selectbox("Choose a CSV file", files, format_func=lambda p: p.name)
    df = load_csv(str(choice))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing values", f"{int(df.isna().sum().sum()):,}")
    c4.metric("File size", human_size(choice.stat().st_size))

    search = st.text_input(
        "🔎 Search inside this file",
        placeholder="Type anything, e.g. Australia, Contoso, 2016...",
    )
    view = df
    if search:
        mask = df.astype(str).apply(
            lambda col: col.str.contains(search, case=False, regex=False, na=False)
        ).any(axis=1)
        view = df[mask]
        st.caption(f"{len(view):,} matching rows")

    table_tab, info_tab = st.tabs(["📋 Table", "ℹ️ Columns & statistics"])
    with table_tab:
        st.dataframe(view, use_container_width=True, height=480, hide_index=True)
    with info_tab:
        info = pd.DataFrame({
            "Column": df.columns,
            "Type": [str(t) for t in df.dtypes],
            "Missing": df.isna().sum().values,
            "Unique values": [df[c].nunique() for c in df.columns],
        })
        st.dataframe(info, use_container_width=True, hide_index=True)
        numeric = df.select_dtypes(include="number")
        if not numeric.empty:
            st.markdown("**Numerical statistics**")
            st.dataframe(numeric.describe().round(2), use_container_width=True)

    st.download_button(
        "⬇️ Download this CSV",
        data=choice.read_bytes(),
        file_name=choice.name,
        mime="text/csv",
    )


def pdf_tab():
    files = find_files("pdf")
    if not files:
        st.info("No PDF files found in the repo.")
        return

    choice = st.selectbox("Choose a PDF report", files, format_func=lambda p: p.name)
    path_str = str(choice)
    pages = pdf_page_count(path_str)

    c1, c2 = st.columns(2)
    c1.metric("Pages", pages)
    c2.metric("File size", human_size(choice.stat().st_size))

    page_no = 1
    if pages > 1:
        page_no = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1)

    preview_tab, text_tab = st.tabs(["🖼️ Page preview", "📝 Extracted text"])
    with preview_tab:
        st.image(render_pdf_page(path_str, int(page_no) - 1), use_container_width=True)
    with text_tab:
        text = pdf_page_text(path_str, int(page_no) - 1).strip()
        st.text(text or "No text found on this page.")

    st.download_button(
        "⬇️ Download this PDF",
        data=choice.read_bytes(),
        file_name=choice.name,
        mime="application/pdf",
    )


def data_page():
    hero("🗂️ Data Explorer", "Browse every CSV dataset and PDF report the assistant uses.")

    m1, m2 = st.columns(2)
    m1.metric("CSV datasets", len(find_files("csv")))
    m2.metric("PDF reports", len(find_files("pdf")))

    tab_csv, tab_pdf = st.tabs(["📁 csv", "📁 pdf"])
    with tab_csv:
        csv_tab()
    with tab_pdf:
        pdf_tab()


# ----------------------------------------------------------- page: assistant
def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY")


def assistant_page():
    hero(
        "📊 Business Intelligence RAG Assistant",
        "Ask questions about sales, products, stores, customers, exchange rates and "
        "reports. Answers use retrieved context and include source citations.",
    )

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

    examples = [
        "Summarize the sales report.",
        "What product information is available?",
        "What information is available about stores?",
        "What does the data dictionary explain?",
    ]

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.markdown("#### Try an example")
        for ex in examples:
            if st.button(ex, use_container_width=True):
                st.session_state.pending = ex
        if st.button("🧹 Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    question = st.chat_input("Ask a question about the business data...")
    if not question:
        question = st.session_state.pop("pending", None)

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = answer_with_retry(question)
                except Exception as e:
                    answer = f"Sorry, the request failed: {e}"
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


# ---------------------------------------------------------------- navigation
with st.sidebar:
    st.markdown("## 📊 BI Dashboard")
    page = st.radio(
        "Navigate",
        ["💬 Assistant", "🗂️ Data"],
        label_visibility="collapsed",
    )
    st.divider()

if page == "💬 Assistant":
    assistant_page()
else:
    data_page()
