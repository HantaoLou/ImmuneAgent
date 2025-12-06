import os
from dataclasses import dataclass
from functools import cache as fc
from typing import Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama.embeddings import OllamaEmbeddings
from qdrant_client import QdrantClient


@dataclass
class QdrantConfig:
    host: str = "117.148.176.36"
    port: int = 6333
    grpc_port: int = 6334
    api_key: Optional[str] = None
    prefer_grpc: bool = False
    https: bool = False
    prefix: Optional[str] = None
    timeout: int = 5
    host_override: Optional[str] = None

    @classmethod
    def from_env(cls):
        return cls(
            host=os.getenv("QDRANT_HOST", "117.148.176.36"),
            port=int(os.getenv("QDRANT_PORT", 6333)),
            grpc_port=int(os.getenv("QDRANT_GRPC_PORT", 6334)),
            api_key=os.getenv("QDRANT_API_KEY"),
            prefer_grpc=os.getenv("QDRANT_PREFER_GRPC", "false").lower() == "true",
            https=os.getenv("QDRANT_HTTPS", "false").lower() == "true",
            prefix=os.getenv("QDRANT_PREFIX"),
            timeout=int(os.getenv("QDRANT_TIMEOUT", 5)),
            host_override=os.getenv("QDRANT_HOST_OVERRIDE"),
        )

    def get_client(self) -> QdrantClient:
        return QdrantClient(
            url=f"{'https' if self.https else 'http'}://{self.host}:{self.port}",
            port=self.grpc_port if self.prefer_grpc else None,
            api_key=self.api_key,
            prefix=self.prefix,
            timeout=self.timeout,
            host=self.host_override or None,
        )

    def __hash__(self):
        return (
            hash(self.host)
            + hash(self.port)
            + hash(self.grpc_port)
            + hash(self.api_key)
            + hash(self.prefer_grpc)
            + hash(self.https)
            + hash(self.prefix)
            + hash(self.timeout)
            + hash(self.host_override)
        )


@dataclass
class ModelConfig:
    embed_model: str = "nomic-embed-text"
    summarise_model: str = "gemma3:4b"

    @classmethod
    def from_env(cls):
        return cls(
            embed_model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
            summarise_model=os.getenv("SUMMARISE_MODEL", "gemma3:4b"),
        )

    def __hash__(self):
        return hash(self.embed_model) + hash(self.summarise_model)


@fc
def get_embedder(model_config=ModelConfig.from_env()):
    embedder = OllamaEmbeddings(model=model_config.embed_model)
    return embedder


@fc
def get_text_splitter(model_config=ModelConfig.from_env()):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return text_splitter


@fc
def get_semantic_text_splitter(model_config=ModelConfig.from_env()):
    text_splitter = SemanticChunker(embeddings=get_embedder(model_config))
    return text_splitter
