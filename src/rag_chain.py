from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.schema import Document
from transformers import pipeline
from typing import Tuple, Dict
import json
from .utils import log, get_today


# -------------------------------------------------
# LLM (cached once)
# -------------------------------------------------
def _get_llm():
    pipe = pipeline(
        "text-generation",
        model="distilgpt2",
        max_length=256,
        truncation=True,
        device=-1,               # CPU
    )
    return HuggingFacePipeline(pipeline=pipe)


@log.info
def build_rag_chain(db):
    llm = _get_llm()
    retriever = db.as_retriever(search_kwargs={"k": 2})

    template = """Context:
{context}

Question: {question}
Today: {today}

Return **only** valid JSON:
{{"renewal_date":"YYYY-MM-DD","cost":"$X","risk":"low/medium/high","citation":"first 80 chars of source"}}
"""
    prompt = PromptTemplate.from_template(template)

    def run(query: str) -> Tuple[Dict, str]:
        docs: List[Document] = retriever.invoke(query)
        context = "\n---\n".join(d.page_content for d in docs)

        full_prompt = prompt.format(context=context, question=query, today=get_today())
        raw = llm(full_prompt)

        # crude JSON extraction (robust for demo)
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            json_str = raw[start:end]
            data = json.loads(json_str)
        except Exception as e:
            log.warning(f"JSON parse failed: {e}")
            data = {"error": "LLM did not return valid JSON", "raw": raw}

        # citation = first chunk snippet
        citation = docs[0].page_content[:80] + "…" if docs else ""
        data.setdefault("citation", citation)

        return data, raw

    return run