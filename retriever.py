import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import (QdrantVectorStore, FastEmbedSparse, RetrievalMode)
from qdrant_client import QdrantClient
from groq import Groq

load_dotenv()

FAISS_DIR = Path("data") / "faiss"
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
TOP_K = int(os.getenv("TOP_K", "5"))


@lru_cache(maxsize=2)
def _get_embeddings_cached(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def _get_qdrant_client() -> QdrantClient:
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise RuntimeError("QDRANT_URL / QDRANT_API_KEY missing in .env")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


@lru_cache(maxsize=4)
def _load_vectorstore_cached(repo_id: str, model_name: str) -> QdrantVectorStore:
    client = _get_qdrant_client()
    if not client.collection_exists(repo_id):
        raise FileNotFoundError(
            f"Collection '{repo_id}' not found in Qdrant. Please index first."
        )
    embeddings = _get_embeddings_cached(model_name)
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    return QdrantVectorStore(
        client=client,
        collection_name=repo_id,
        embedding=embeddings,
        sparse_embeddings=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID
    )


def _scroll_all_docs(repo_id: str) -> List[Document]:
    """Replaces FAISS docstore scan. Works with Qdrant Cloud."""
    client = _get_qdrant_client()
    docs: List[Document] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=repo_id,
            with_payload=True,
            with_vectors=False,
            limit=1024,
            offset=offset,
        )
        for p in points:
            payload = p.payload or {}
            text = payload.get("page_content") or payload.get("content") or ""
            meta = payload.get("metadata")
            if not isinstance(meta, dict):
                meta = {k: v for k, v in payload.items()
                        if k not in ("page_content", "content")}
            docs.append(Document(page_content=text, metadata=meta))
        if offset is None:
            break
    return docs


def _load_repo_summary(repo_id: str) -> Dict[str, Any]:
    summary_path = FAISS_DIR / repo_id / "build_summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _source_matches(src: str, filename: str) -> bool:
    src_l = src.lower().replace("\\", "/")
    fn_l = filename.lower().replace("\\", "/")
    return src_l == fn_l or src_l.endswith("/" + fn_l)


def _load_repo_map(repo_id: str) -> Dict[str, Any]:
    repo_map_path = FAISS_DIR / repo_id / "repo_map.json"
    if not repo_map_path.exists():
        return {"files": []}
    with open(repo_map_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _repo_stats_answer(repo_id: str, question: str) -> Optional[Dict[str, Any]]:
    q = question.lower().strip()

    is_list_indexed_query = any(
        p in q for p in [
            "list indexed files",
            "list the indexed files",
            "show indexed files",
            "which files are indexed",
            "what files are indexed",
        ]
    )
    is_total_files_query = any(
        p in q for p in [
            "how many files",
            "number of files",
            "total files",
            "file count",
            "count files",
        ]
    )
    is_indexed_files_query = "indexed files" in q

    if not (is_list_indexed_query or is_total_files_query or is_indexed_files_query):
        return None

    summary = _load_repo_summary(repo_id)
    if not summary:
        return {
            "answer": "I cannot find build summary for this repo. Please re-index.",
            "citations": [],
            "contexts": [],
        }

    total_files = summary.get("total_files_in_repo", "unknown")
    indexed_files = summary.get("files_loaded", "unknown")
    chunks = summary.get("chunks_indexed", "unknown")
    skipped_binary = summary.get("binary_like_files", "unknown")
    indexed_list = summary.get("indexed_files_list", [])

    if is_list_indexed_query:
        if indexed_list:
            lines = "\n".join([f"- {f}" for f in indexed_list])
            answer = f"Indexed files ({len(indexed_list)}):\n{lines}"
        else:
            answer = "No indexed files list found. Re-index the repository."
    elif is_indexed_files_query and not is_total_files_query:
        answer = (
            f"This repository has {indexed_files} indexed files "
            f"and {chunks} indexed chunks."
        )
    else:
        not_indexed = (
            total_files - indexed_files
            if isinstance(total_files, int) and isinstance(indexed_files, int)
            else "unknown"
        )
        answer = (
            f"This repository has {total_files} total files. "
            f"I indexed {indexed_files} searchable text/code files into {chunks} chunks. "
            f"{not_indexed} files were not indexed: {skipped_binary} are binary files "
            f"and the rest have unsupported extensions like .csv, .tsv, .gitignore etc."
        )

    return {
        "answer": answer,
        "citations": ["build_summary.json"],
        "contexts": [],
    }


def _ast_hints_for_question(
    repo_id: str, question: str, max_hints: int = 8
) -> List[Dict[str, Any]]:
    repo_map = _load_repo_map(repo_id)
    files = repo_map.get("files", [])
    q = question.lower()

    keywords = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", q))
    if not keywords:
        return []

    hints: List[Dict[str, Any]] = []

    for entry in files:
        file_path = entry.get("file", "unknown")

        for fn in entry.get("functions", []):
            fn_name = str(fn.get("name", "")).lower()
            if fn_name and (fn_name in keywords or any(k in fn_name for k in keywords)):
                hints.append({
                    "type": "function",
                    "file": file_path,
                    "name": fn.get("name"),
                    "start_line": fn.get("start_line"),
                    "end_line": fn.get("end_line"),
                })

        for cl in entry.get("classes", []):
            cl_name = str(cl.get("name", "")).lower()
            if cl_name and (cl_name in keywords or any(k in cl_name for k in keywords)):
                hints.append({
                    "type": "class",
                    "file": file_path,
                    "name": cl.get("name"),
                    "start_line": cl.get("start_line"),
                    "end_line": cl.get("end_line"),
                })
            for method in cl.get("methods", []):
                m_name = str(method.get("name", "")).lower()
                if m_name and (m_name in keywords or any(k in m_name for k in keywords)):
                    hints.append({
                        "type": "method",
                        "file": file_path,
                        "name": f"{cl.get('name')}.{method.get('name')}",
                        "start_line": method.get("start_line"),
                        "end_line": method.get("end_line"),
                    })

        for imp in entry.get("imports", []):
            imp_low = str(imp).lower()
            if imp_low and (imp_low in q or any(k in imp_low for k in keywords)):
                hints.append({
                    "type": "import",
                    "file": file_path,
                    "name": imp,
                    "start_line": None,
                    "end_line": None,
                })

    unique = []
    seen = set()
    for h in hints:
        key = (h["type"], h["file"], h["name"], h["start_line"], h["end_line"])
        if key not in seen:
            seen.add(key)
            unique.append(h)

    return unique[:max_hints]


def _format_context_from_docs(docs) -> Tuple[str, List[Dict[str, Any]]]:
    blocks: List[str] = []
    contexts: List[Dict[str, Any]] = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        start_line = doc.metadata.get("start_line", "?")
        end_line = doc.metadata.get("end_line", "?")
        text = doc.page_content

        blocks.append(f"[{i}] {source}:{start_line}-{end_line}\n{text}")
        contexts.append({
            "source": source,
            "start_line": start_line,
            "end_line": end_line,
            "score": None,
            "text": text,
        })

    return "\n\n".join(blocks), contexts


def _format_ast_hints(ast_hints: List[Dict[str, Any]]) -> str:
    if not ast_hints:
        return "No AST symbol hints found."
    lines = []
    for h in ast_hints:
        s = h.get("start_line")
        e = h.get("end_line")
        if s is not None and e is not None:
            lines.append(f"- {h['type']} {h['name']} in {h['file']}:{s}-{e}")
        else:
            lines.append(f"- {h['type']} {h['name']} in {h['file']}")
    return "\n".join(lines)


def search_by_file_and_lines(
    repo_id: str,
    filename: str,
    start_line: int,
    end_line: int,
) -> Dict[str, Any]:
    summary_path = FAISS_DIR / repo_id / "build_summary.json"
    if not summary_path.exists():
        return {
            "answer": "Index not found. Please re-index.",
            "citations": [],
            "contexts": [],
        }

    all_docs = _scroll_all_docs(repo_id)

    matched = []
    for doc in all_docs:
        src = doc.metadata.get("source", "")
        sl = doc.metadata.get("start_line", 0)
        el = doc.metadata.get("end_line", 0)
        if _source_matches(src, filename):
            if not (el < start_line or sl > end_line):
                matched.append(doc)

    if not matched:
        return {
            "answer": f"No chunks found in {filename} between lines {start_line}-{end_line}.",
            "citations": [],
            "contexts": [],
        }

    context_parts = []
    contexts = []
    for doc in matched:
        sl = doc.metadata.get("start_line", "?")
        el = doc.metadata.get("end_line", "?")
        src = doc.metadata.get("source", filename)
        context_parts.append(f"{src}:{sl}-{el}\n{doc.page_content}")
        contexts.append({
            "source": src,
            "start_line": sl,
            "end_line": el,
            "score": None,
            "text": doc.page_content,
        })

    return {
        "answer": "\n\n".join(context_parts),
        "citations": [
            f"{c['source']}:{c['start_line']}-{c['end_line']}" for c in contexts
        ],
        "contexts": contexts,
    }


def _detect_line_range_query(question: str):
    pattern = r"([\w./\\]+\.(?:py|md|txt|js|ts|java))[^\d]*(\d+)[^\d]+(\d+)"
    match = re.search(pattern, question.lower())
    if match:
        filename = match.group(1)
        start = int(match.group(2))
        end = int(match.group(3))
        return filename, start, end
    return None


def answer_question(
    repo_id: str, question: str, top_k: int = TOP_K
) -> Dict[str, Any]:
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    # Check stats questions first
    stats_response = _repo_stats_answer(repo_id, question)
    if stats_response is not None:
        return stats_response

    # Check line range questions
    line_range = _detect_line_range_query(question)
    if line_range:
        filename, start, end = line_range
        return search_by_file_and_lines(repo_id, filename, start, end)

    # Normal semantic retrieval
    vectorstore = _load_vectorstore_cached(repo_id, EMBED_MODEL)

    # docs = vectorstore.max_marginal_relevance_search(
    #     question,
    #     k=top_k,
    #     fetch_k=max(20, top_k * 4),
    # )

    docs = vectorstore.similarity_search(
        question,
        k=top_k,
        fetch_k=max(20, top_k * 4),
    )

    if not docs:
        return {
            "answer": "I could not find relevant context in the indexed repository.",
            "citations": [],
            "contexts": [],
        }

    context_text, contexts = _format_context_from_docs(docs)
    ast_hints = _ast_hints_for_question(repo_id, question, max_hints=8)
    ast_text = _format_ast_hints(ast_hints)

    system_prompt = (
        "You are a codebase assistant.\n"
        "Answer using provided context and AST hints.\n"
        "Even if context is partial, give best possible answer from what is available.\n"
        "Only say 'Insufficient context' if context has absolutely zero relevant information.\n"
        "Be very careful about which file each piece of information comes from.\n"
        "Do not mix content from different files.\n"
        "For each claim, clearly state which file it came from using citation [path:start-end].\n"
        "Do not invent file names or line ranges."
    )

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Retrieved Code Context:\n{context_text}\n\n"
        f"AST Repo Map Hints:\n{ast_text}\n\n"
        "Return:\n"
        "1) Short direct answer\n"
        "2) Key reasoning\n"
        "3) Citations"
    )

    if not GROQ_API_KEY:
        return {
            "answer": "GROQ_API_KEY is missing. Please set it in your .env file.",
            "citations": [],
            "contexts": [],
        }

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        top_p=0.9,
        max_tokens=500,
    )

    answer = response.choices[0].message.content

    citations = []
    seen = set()

    for c in contexts:
        item = f"{c['source']}:{c['start_line']}-{c['end_line']}"
        if item not in seen:
            seen.add(item)
            citations.append(item)

    for h in ast_hints:
        if h.get("start_line") is not None and h.get("end_line") is not None:
            item = f"{h['file']}:{h['start_line']}-{h['end_line']}"
            if item not in seen:
                seen.add(item)
                citations.append(item)

    return {
        "answer": answer,
        "citations": citations,
        "contexts": contexts,
        "ast_hints": ast_hints,
    }