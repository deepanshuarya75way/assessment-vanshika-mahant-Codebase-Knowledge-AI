
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from git import Repo

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import (QdrantVectorStore, FastEmbedSparse, RetrievalMode)
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams

load_dotenv()

# ---------- Paths ----------
DATA_DIR = Path("data")
REPOS_DIR = DATA_DIR / "repos"
INDEX_META_DIR = DATA_DIR / "faiss"  # now stores ONLY JSON metadata (summary + repo map)

REPOS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_META_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Repo/File config ----------
CODE_SUFFIXES = [
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs",
    ".cpp", ".c", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
    ".kt", ".sql", ".md", ".json", ".yaml", ".yml", ".toml", ".txt",
    ".ipynb", ".html",
]

EXCLUDE_GLOBS = [
    "**/.git/**", "**/node_modules/**", "**/dist/**", "**/build/**",
    "**/.next/**", "**/.venv/**", "**/venv/**", "**/__pycache__/**",
    "**/.mypy_cache/**", "**/.pytest_cache/**", "**/coverage/**",
]

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))  # all-MiniLM-L6-v2
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "140"))
MAX_FILES = int(os.getenv("MAX_FILES", "1500"))

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


def is_git_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://") or value.startswith("git@")


def slugify_repo_id(value: str) -> str:
    value = value.strip().rstrip("/")
    name = value.split("/")[-1].replace(".git", "")
    name = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-").lower()
    return name or "repo"


def prepare_repo(repo_input: str) -> Tuple[str, Path]:
    repo_input = repo_input.strip()
    repo_id = slugify_repo_id(repo_input)

    if is_git_url(repo_input):
        local_repo_path = REPOS_DIR / repo_id
        if local_repo_path.exists() and (local_repo_path / ".git").exists():
            repo = Repo(local_repo_path)
            repo.remotes.origin.pull()
        else:
            Repo.clone_from(repo_input, local_repo_path)
        return repo_id, local_repo_path

    local_repo_path = Path(repo_input).expanduser().resolve()
    if not local_repo_path.exists():
        raise FileNotFoundError(f"Local path not found: {local_repo_path}")
    return repo_id, local_repo_path


def extension_to_language(ext: str):
    mapping = {
        ".py": Language.PYTHON, ".js": Language.JS, ".ts": Language.TS,
        ".tsx": Language.TS, ".jsx": Language.JS, ".java": Language.JAVA,
        ".go": Language.GO, ".rs": Language.RUST, ".cpp": Language.CPP,
        ".c": Language.CPP, ".h": Language.CPP, ".hpp": Language.CPP,
        ".cs": Language.CSHARP, ".php": Language.PHP, ".rb": Language.RUBY,
        ".kt": Language.KOTLIN, ".swift": Language.SWIFT,
    }
    return mapping.get(ext.lower())


def build_splitter_for_ext(ext: str) -> RecursiveCharacterTextSplitter:
    lang = extension_to_language(ext)

    if ext == ".ipynb":
        return RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50, add_start_index=True)

    if lang is not None:
        try:
            return RecursiveCharacterTextSplitter.from_language(
                language=lang, chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP, add_start_index=True,
            )
        except Exception:
            pass

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True,
    )


def to_relative_source(repo_path: Path, source: str) -> str:
    source_path = Path(source)
    try:
        return str(source_path.resolve().relative_to(repo_path.resolve()))
    except Exception:
        return source


def add_line_numbers(original_text: str, start_index: int, chunk_text: str) -> Tuple[int, int]:
    start_line = original_text.count("\n", 0, max(start_index, 0)) + 1
    chunk_line_count = max(chunk_text.count("\n") + 1, 1)
    end_line = start_line + chunk_line_count - 1
    return start_line, end_line


def count_repo_files(repo_path: Path) -> Dict[str, int]:
    ignored_dirs = {".git", "node_modules", "dist", "build", ".venv", "venv", "__pycache__"}
    binary_like_exts = {
        ".joblib", ".pkl", ".bin", ".pt", ".onnx",
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".zip", ".tar", ".gz",
    }
    total_files = 0
    binary_like_files = 0
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        for name in files:
            total_files += 1
            if Path(name).suffix.lower() in binary_like_exts:
                binary_like_files += 1
    return {"total_files_in_repo": total_files, "binary_like_files": binary_like_files}


def sanitize_metadata(docs: List[Document]) -> List[Document]:
    """Qdrant rejects None / exotic types in payload. Clean before upload."""
    for d in docs:
        clean = {}
        for k, v in d.metadata.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool, list)):
                clean[k] = v
            else:
                clean[k] = str(v)
        d.metadata = clean
    return docs


def build_index(repo_input: str) -> Dict:
    repo_id, repo_path = prepare_repo(repo_input)

    loader = GenericLoader.from_filesystem(
        str(repo_path),
        glob="**/*",
        suffixes=CODE_SUFFIXES,
        exclude=EXCLUDE_GLOBS,
        parser=LanguageParser(parser_threshold=0),
    )
    docs = loader.load()

    if not docs:
        raise ValueError("No supported files found to index.")

    docs = docs[:MAX_FILES]  # document cap

    chunked_docs: List[Document] = []
    indexed_files_set = set()

    for doc in docs:
        source = doc.metadata.get("source", "")
        ext = Path(source).suffix.lower()
        splitter = build_splitter_for_ext(ext)

        split_docs = splitter.split_documents([doc])
        original_text = doc.page_content

        rel_source = to_relative_source(repo_path, source)
        indexed_files_set.add(rel_source)

        for c in split_docs:
            start_index = int(c.metadata.get("start_index", 0))
            start_line, end_line = add_line_numbers(original_text, start_index, c.page_content)
            c.metadata["repo_id"] = repo_id
            c.metadata["source"] = rel_source
            c.metadata["start_line"] = start_line
            c.metadata["end_line"] = end_line
            chunked_docs.append(c)

    if not chunked_docs:
        raise ValueError("Chunking produced zero chunks.")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    # ---------- Qdrant Cloud upload (replaces FAISS) ----------
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set in .env")

    chunked_docs = sanitize_metadata(chunked_docs)

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


    if client.collection_exists(repo_id):
        client.delete_collection(repo_id)
    client.create_collection(
        collection_name=repo_id,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        sparse_vectors_config={"langchain-sparse":SparseVectorParams()},
    )
    # if client.collection_exists(repo_id):
    #     client.delete_collection(repo_id)
    # client.create_collection(
    #     collection_name=repo_id,
    #     vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    # )
    # vectorstore = QdrantVectorStore(
    #     client=client,
    #     collection_name=repo_id,
    #     embedding=embeddings,
    # )
    # vectorstore.add_documents(chunked_docs)
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=repo_id,
        embedding=embeddings,
        sparse_embedding = sparse_embeddings,
        retrieval_mode = RetrievalMode.HYBRID,
        #force_recreate = True
    )
    vectorstore.add_documents(chunked_docs)
    



    # ---------- Local JSON metadata (committed to GitHub for cloud) ----------
    repo_index_dir = INDEX_META_DIR / repo_id
    repo_index_dir.mkdir(parents=True, exist_ok=True)

    try:
        from ast_parser import build_repo_map  # lazy: cloud query path doesn't need tree-sitter
        repo_map = build_repo_map(repo_path, repo_index_dir / "repo_map.json")
    except Exception as e:
        print(f"[warn] AST repo map skipped: {e}")
        repo_map = {"files": [], "total_python_files": 0}
        with open(repo_index_dir / "repo_map.json", "w", encoding="utf-8") as f:
            json.dump(repo_map, f, indent=2)

    repo_counts = count_repo_files(repo_path)

    summary = {
        "repo_id": repo_id,
        "repo_path": str(repo_path),
        "total_files_in_repo": repo_counts["total_files_in_repo"],
        "files_loaded": len(indexed_files_set),
        "chunks_indexed": len(chunked_docs),
        "binary_like_files": repo_counts["binary_like_files"],
        "indexed_files_list": sorted(indexed_files_set),
        "ast_files_parsed": repo_map.get("total_python_files", 0),
        "embedding_model": EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "vector_db": "qdrant-cloud",
        "retrieval_mode":"hybrid",
        "sparse_model":"Qdrant/bm25",
        "collection_name": repo_id,
    }

    with open(repo_index_dir / "build_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary