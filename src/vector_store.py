from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from typing import List
from .utils import log


@log.info
def build_vectorstore(text: str) -> FAISS:
    # 1. Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)
    log.info(f"Created {len(chunks)} chunks")

    # 2. Embed (cached model)
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3. FAISS index
    db = FAISS.from_texts(chunks, embedder)
    log.info("FAISS index ready")
    return db