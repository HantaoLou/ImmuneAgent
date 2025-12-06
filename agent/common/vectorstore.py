# utils.py (updated with caching, batching)

import hashlib
import os
from pathlib import Path
from typing import List

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document

CACHE_DIR = Path("agent/.cache")
CACHE_DIR.mkdir(exist_ok=True)


def get_cache_path(name: str) -> str:
    path = CACHE_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def hash_string(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def cache_loader_result(key: str, loader_func):
    cache_file = CACHE_DIR / f"{hash_string(key)}.pkl"
    if cache_file.exists():
        return pd.read_pickle(cache_file)
    result = loader_func()
    pd.to_pickle(result, cache_file)
    return result


def setup_vectorstore(docs, embedding_model, persist_dir="agent/.cache/chroma_cache"):
    if os.path.exists(persist_dir):
        print("Loading cached vectorstore...")
        return Chroma(persist_directory=persist_dir, embedding_function=embedding_model)

    print("Creating new vectorstore...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    split_docs = splitter.split_documents(docs)
    db = Chroma.from_documents(
        split_docs, embedding=embedding_model, persist_directory=persist_dir
    )
    db.persist()
    return db


def load_pdfs_from_directory(pdf_dir: str) -> List[Document]:
    pdfs = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_dir, filename))
            pdfs.extend(loader.load())
    return pdfs


def load_tables_from_csv(table_file: str) -> list[Document]:
    try:
        if table_file.endswith(".xlsx"):
            df = pd.read_excel(table_file)
        else:
            df = pd.read_csv(table_file)
        if df.empty:
            print(f"[INFO] Table file {table_file} is empty.")
            return []
    except pd.errors.EmptyDataError:
        print(f"[INFO] Table file {table_file} is empty or malformed.")
        return []
    except Exception as e:
        print(f"[ERROR] Failed to read table file: {e}")
        return []

    return [
        Document(page_content=row.to_json(), metadata={"source": table_file})
        for _, row in df.iterrows()
    ]


def load_web_documents(urls: List[str]) -> List[Document]:
    documents = []
    for url in urls:
        loader = WebBaseLoader(url)
        documents.extend(loader.load())
    return documents


def load_all_documents(
    pdf_dir: str = "agent/library",
    table_file: str = "agent/library/H5N1.xlsx",
    urls: List[str] = [],
) -> List[Document]:
    pdf_docs = cache_loader_result("pdfs", lambda: load_pdfs_from_directory(pdf_dir))
    table_docs = cache_loader_result("tables", lambda: load_tables_from_csv(table_file))
    web_docs = cache_loader_result("urls", lambda: load_web_documents(urls))
    return pdf_docs + table_docs + web_docs
