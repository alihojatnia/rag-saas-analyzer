
<img width="705" height="443" alt="gitgit" src="https://github.com/user-attachments/assets/9eb07bbe-e1ba-440c-ad70-599941189a04" />

### RAG-Powered SaaS Spend Analyzer (Demo)

**A lightweight RAG pipeline that reads SaaS contracts/invoices (PDFs) and answers questions about cost, renewal dates, and risk**


---

#### Features

| Feature | Status |
|-------|--------|
| Upload any PDF (contracts, invoices) | Works |
| Extract & chunk text | Works |
| Embed with `all-MiniLM-L6-v2` | Works |
| Store in FAISS (in-memory) | Works |
| Ask natural language questions | Works |
| Get **structured JSON** + **citation** | **Demo-only** |

---

#### Architecture (Simple & Free)

```
PDF → PyMuPDF → LangChain chunks → MiniLM embeddings → FAISS
          ↓
   Retrieval → LLM → JSON + citation
```

---

#### Tech Stack

| Component | Tool |
|---------|------|
| LLM | `distilgpt2` (77M) – **tiny model, demo only** |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | `FAISS` (in-memory) |
| PDF Parsing | `PyMuPDF` |
| Framework | `LangChain 0.3.1` + `Streamlit` |
| Hosting | GitHub Codespaces + Hugging Face Spaces |

---

#### Why the LLM is "Demo Only"

> **Warning: `distilgpt2` is too small to reliably output JSON.**

- It **repeats context** instead of answering.
- It **ignores formatting instructions**.
- It **cannot reason** about dates or risk.

**This is intentional** — the goal is to **show the full RAG pipeline**, not perfect answers.

**For production**: Use `phi-2`, `gemma-2b`, or `Claude` via API.

---

#### How to Run Locally (GitHub Codespaces)

```bash
# 1. Open in Codespaces
# 2. In terminal:
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

Click **"Open in browser"** → upload PDF → ask questions.

---

#### Example Questions

| Question | Expected (Ideal) | Actual (Demo) |
|--------|------------------|---------------|
| `What is the total cost?` | `{"cost": "$66.00"}` | `{"error": "LLM did not return valid JSON"}` |
| `When does it renew?` | `{"renewal_date": "2026-03-01"}` | Random text |
| `What is the risk?` | `{"risk": "medium"}` | N/A |


---

#### Sample PDF 

```bash
# Create a test PDF
echo "Annual cost: $12,000\nRenewal: March 15, 2026\nRisk: Auto-renew" > contract.txt
pandoc contract.txt -o contract.pdf
```

Drag `contract.pdf` → Index → Ask: `"What is the cost?"`

---

#### Project Structure

```
src/
 ├─ pdf_loader.py      # Extract text from PDF
 ├─ vector_store.py    # Chunk + embed + FAISS
 ├─ rag_chain.py       # Retrieval + LLM + JSON
 └─ utils.py           # Logging + date
app.py                 # Streamlit UI
requirements.txt
```


---

#### Future Improvements

| Upgrade | Benefit |
|-------|--------|
| `phi-2` or `gemma-2b` | Real JSON output |
| Pydantic + JSON mode | Enforce structure |
| LangGraph | Multi-step reasoning |
| Weaviate (free tier) | Persistent DB |
| LangSmith | Tracing |

---
