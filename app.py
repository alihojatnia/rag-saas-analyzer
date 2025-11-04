import streamlit as st
from src.pdf_loader import load_pdf
from src.vector_store import build_vectorstore
from src.rag_chain import build_rag_chain
from src.utils import log

st.set_page_config(page_title="SaaS Spend Analyzer", layout="centered")
st.title("Contract Analyzer")

# -------------------------------------------------
# 1. Upload & index
# -------------------------------------------------
pdf_file = st.file_uploader("Upload SaaS contract (PDF)", type="pdf")

if pdf_file and st.button("Index PDF"):
    with st.spinner("Reading & indexing…"):
        text = load_pdf(pdf_file.read())
        db = build_vectorstore(text)
        st.session_state.db = db
        st.success("Indexed – ask away!")

# -------------------------------------------------
# 2. Query
# -------------------------------------------------
if "db" in st.session_state:
    chain = build_rag_chain(st.session_state.db)

    q = st.text_input("Ask (e.g. *When does Zoom renew?*)", placeholder="When does the contract renew?")

    if q:
        with st.spinner("Thinking…"):
            result, raw = chain(q)
            st.json(result)
            if "raw" in result:
                with st.expander("Raw LLM output"):
                    st.code(raw, language="text")