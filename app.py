"""Streamlit UI for the Business Intelligence RAG Assistant."""
import os
from pathlib import Path

import streamlit as st

import rag

st.set_page_config(page_title="Business Intelligence RAG Assistant", page_icon="📊")

DATA_DIR = Path(__file__).parent  # PDFs/CSVs are searched for anywhere in the repo
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


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

st.title("📊 Business Intelligence RAG Assistant")
st.caption(
    "Ask questions about sales, products, stores, customers, exchange rates, and "
    "business reports. Answers use retrieved context and include source citations."
)

EXAMPLES = [
    "Summarize the sales report.",
    "What product information is available?",
    "What information is available about stores?",
    "What does the data dictionary explain?",
]

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Try an example")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex
    if st.button("Clear chat", use_container_width=True):
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
                answer = rag.answer_question(client, MODEL_NAME, chunks, embeddings, question)
            except Exception as e:
                answer = f"Sorry, the request failed: {e}"
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
