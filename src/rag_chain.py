from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain.schema import Document
from transformers import pipeline
from typing import Tuple, Dict, List
import json
from .utils import log, get_today


def _get_llm():
    pipe = pipeline(
        "text-generation",
        model= "distilgpt2",#"microsoft/phi-2",
        max_new_tokens=120,
        truncation=True,
        do_sample=True,
        temperature=0.1,
        device=-1,
    )
    return HuggingFacePipeline(pipeline=pipe)


def build_rag_chain(db):
    log.info("Building RAG chain ...")
    llm = _get_llm()
    retriever = db.as_retriever(search_kwargs={"k": 2})

    template = """You are a SaaS spend analyst. Return **only** valid JSON.

Context:
{context}

Question: {question}
Today: {today}

Format:
{{"renewal_date":"YYYY-MM-DD","cost":"$X","risk":"low/medium/high","citation":"first 80 chars of source"}}

Answer:"""

    prompt = PromptTemplate.from_template(template)

    def run(query: str) -> Tuple[Dict, str]:
        docs: List[Document] = retriever.invoke(query)
        context = "\n---\n".join(d.page_content for d in docs)
        full_prompt = prompt.format(context=context, question=query, today=get_today())
        raw = llm(full_prompt).strip()

        # Clean
        for prefix in ["Answer:", "```json", "```"]:
            if raw.startswith(prefix):
                raw = raw[len(prefix):].strip()

        try:
            data = json.loads(raw)
        except:
            import re
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            data = json.loads(match.group()) if match else {"error": "Parse failed", "raw": raw[:500]}

        citation = docs[0].page_content[:80] + "..." if docs else ""
        data.setdefault("citation", citation)
        return data, raw

    return run