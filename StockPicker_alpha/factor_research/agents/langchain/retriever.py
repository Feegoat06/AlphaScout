"""Literature corpus retriever (DirectoryLoader + FAISS)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEMO_ROOT = Path(__file__).resolve().parents[2]
LITERATURE_DIR = DEMO_ROOT / "data" / "literature"
INDEX_DIR = DEMO_ROOT / "data" / "cache" / "literature_index"

_FACTOR_QUERY_HINTS: dict[str, str] = {
    "momentum": "momentum Jegadeesh Titman winners losers crash",
    "BtM": "value book-to-market Fama French HML cyclical",
    "ROA": "profitability ROA Novy-Marx quality",
    "ivol": "idiosyncratic volatility Ang puzzle",
    "composite": "composite blend diversify dilute anomalies",
}


def format_docs(docs: list[Any]) -> str:
    parts: list[str] = []
    for i, doc in enumerate(docs, start=1):
        src = doc.metadata.get("source", doc.metadata.get("doc_id", f"chunk_{i}"))
        parts.append(f"[{i}] source={src}\n{doc.page_content}")
    return "\n\n".join(parts) if parts else "(no documents retrieved)"


def source_ids(docs: list[Any]) -> list[str]:
    out: list[str] = []
    for doc in docs:
        src = Path(str(doc.metadata.get("source", ""))).name or doc.metadata.get("doc_id", "")
        if src and src not in out:
            out.append(src)
    return out


def load_literature_documents() -> list[Any]:
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if not LITERATURE_DIR.exists():
        return []

    loader = DirectoryLoader(
        str(LITERATURE_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    docs = loader.load()
    for doc in docs:
        doc.metadata["doc_id"] = Path(doc.metadata.get("source", "")).name
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(docs)


def get_retriever(*, k: int = 4) -> Any:
    """Build or load a FAISS retriever over curated literature notes."""
    from langchain_community.vectorstores import FAISS

    from agents.langchain.client import get_embeddings

    embeddings = get_embeddings()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_faiss = INDEX_DIR / "index.faiss"
    if index_faiss.exists():
        store = FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        docs = load_literature_documents()
        if not docs:
            raise FileNotFoundError(f"No literature markdown found under {LITERATURE_DIR}")
        store = FAISS.from_documents(docs, embeddings)
        store.save_local(str(INDEX_DIR))
    return store.as_retriever(search_kwargs={"k": k})


def retrieve_for_factor(factor: str, *, k: int = 4) -> list[Any]:
    hint = _FACTOR_QUERY_HINTS.get(factor, factor)
    query = f"{factor} {hint}"
    retriever = get_retriever(k=k)
    return list(retriever.invoke(query))
