# Business Intelligence RAG Assistant

A retrieval-augmented chatbot over the *Global Electronics Retailer* reports (PDF) and datasets (CSV).

- **Retrieval:** PyMuPDF + pandas -> chunking -> Gemini embeddings (`gemini-embedding-001`) -> cosine similarity (NumPy)
- **Generation:** Google Gemini (`gemini-2.5-flash` by default)
- **UI:** Streamlit

## Run locally
```
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"      # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. Push this folder to a public GitHub repo.
2. At share.streamlit.io, create an app from the repo with main file `app.py`.
3. In Advanced settings -> Secrets, add: `GEMINI_API_KEY = "your-key"`

## Data
Put your `.csv` and `.pdf` files anywhere in this repo (the main folder works fine). They are found automatically.
