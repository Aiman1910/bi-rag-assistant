"""Core RAG logic (no UI): load PDFs/CSVs, chunk, embed with Gemini, retrieve, answer."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf
from google import genai
from google.genai import types

EMBED_MODEL = "gemini-embedding-001"


# ---------------------------------------------------------------- loading
def read_csv_safely(path):
    """Some files (e.g. Customers.csv) are not UTF-8, so fall back to latin-1."""
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path}")


def load_documents(base_dir):
    # Finds PDFs/CSVs anywhere under base_dir, so both a data/csv + data/pdfs layout
    # and a flat upload (all files in the repo's main folder) work.
    base_dir = Path(base_dir)
    documents = []

    for pdf_path in sorted(base_dir.rglob("*.pdf")):
        with pymupdf.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                if text:
                    documents.append({
                        "text": text,
                        "source": str(pdf_path.relative_to(base_dir)),
                        "page": page_number,
                    })

    for csv_path in sorted(base_dir.rglob("*.csv")):
        df = read_csv_safely(csv_path)
        parts = [
            f"Dataset name: {csv_path.stem}",
            f"Total rows: {len(df)}",
            f"Total columns: {len(df.columns)}",
            f"Column names: {', '.join(map(str, df.columns))}",
            "\nSample records:",
            df.head(8).to_string(index=False),
            "\nMissing values:",
            df.isna().sum().to_string(),
        ]
        numeric = df.select_dtypes(include="number")
        if not numeric.empty:
            parts.extend(["\nNumerical statistics:", numeric.describe().round(2).to_string()])
        documents.append({
            "text": "\n".join(parts),
            "source": str(csv_path.relative_to(base_dir)),
            "page": None,
        })
    return documents


def split_text(text, chunk_size=1000, overlap=200):
    text = text.strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def make_chunks(documents):
    return [
        {"text": c, "source": d["source"], "page": d["page"]}
        for d in documents
        for c in split_text(d["text"])
    ]


# -------------------------------------------------------------- embeddings
def _normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def embed_texts(client, texts, task_type, batch_size=50):
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(4):  # simple retry for rate limits
            try:
                resp = client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                vectors.extend(e.values for e in resp.embeddings)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
    return _normalize(np.array(vectors, dtype="float32"))


def build_index(client, base_dir):
    chunks = make_chunks(load_documents(base_dir))
    if not chunks:
        raise RuntimeError("No PDFs or CSVs found. Upload them to the same GitHub repo as app.py.")
    embeddings = embed_texts(client, [c["text"] for c in chunks], "RETRIEVAL_DOCUMENT")
    return chunks, embeddings


# --------------------------------------------------------------- retrieval
def retrieve(client, chunks, embeddings, question, top_k=5):
    q = embed_texts(client, [question], "RETRIEVAL_QUERY")
    scores = embeddings @ q[0]
    top = np.argsort(-scores)[:min(top_k, len(chunks))]
    results = []
    for idx in top:
        r = chunks[int(idx)].copy()
        r["score"] = float(scores[idx])
        results.append(r)
    return results


# ------------------------------------------------------------------ answer
PROMPT = """
You are a Business Intelligence Assistant.

Answer the user's question using the retrieved context.

Rules:
- Use clear, simple English.
- Do not invent facts or numerical results.
- Cite supporting claims with source IDs such as [S1].
- If the context is insufficient, say so.

CONTEXT:
{context}

QUESTION:
{question}
"""


def answer_question(client, model_name, chunks, embeddings, question, top_k=5):
    results = retrieve(client, chunks, embeddings, question, top_k)
    context = "\n\n".join(
        f"[S{i}] Source: {r['source']}, Page: {r['page']}\n{r['text']}"
        for i, r in enumerate(results, start=1)
    )
    response = client.models.generate_content(
        model=model_name,
        contents=PROMPT.format(context=context, question=question),
    )
    answer = response.text or "I couldn't generate an answer."

    seen, citations = set(), []
    for i, r in enumerate(results, start=1):
        key = (r["source"], r["page"])
        if key not in seen:
            seen.add(key)
            page = f", page {r['page']}" if r["page"] else ""
            citations.append(f"[S{i}] {r['source']}{page}")
    if citations:
        answer += "\n\n**Sources**\n" + "\n".join(f"- {c}" for c in citations)
    return answer


def make_client(api_key):
    return genai.Client(api_key=api_key)
