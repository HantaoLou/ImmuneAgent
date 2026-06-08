from typing import Callable, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from kb.config import QdrantConfig, get_embedder
from kb.config.config import ModelConfig
from kb.vectorstore.prompts import PARENT_RETRIEVER_SUMMARIZE_PROMPT

from diskcache import Cache

cache = Cache("/tmp/antibody_gen/retriever/summarizer")

def get_vector_store(
    collection_name: str,
    qdrant_config=QdrantConfig.from_env(),
    model_config=ModelConfig.from_env(),
) -> QdrantVectorStore:
    client = qdrant_config.get_client()
    vector = QdrantVectorStore(
        client=client,
        embedding=get_embedder(model_config),
        collection_name=collection_name,
    )
    return vector


KEY_SRC = "source"
KEY_SRC_FULL = "metadata.source"
KEY_PAGE_CONTENT = "page_content"
KEY_PAGE = "page"
KEY_ORIGINAL_CHUNKS = "original_chunks"
KEY_ORIGINAL_DOCUMENT = "original_document"


class QdrantParentDocumentRetriever(BaseRetriever):
    """
    QdrantParentDocumentRetriever 先从文档集合中检索查询相关的 Chunk, 然后根据 Chunk 元数据,检索到所在文档。
    再对这些文档进行上下文感知的总结。
    """

    summarize_model: BaseChatModel
    chunk_filter: Optional[Callable[list[Document], list[Document]]] = None
    retriever_kwargs: Optional[dict] = None
    vector_store: QdrantVectorStore
    # 总结模型的角色
    role: str = "An academic paper reviewer"
    summarize: bool = True

    def _get_full_parent(self, parent_source: str) -> Optional[str]:
        chunks = self.vector_store.client.query_points(
            collection_name=self.vector_store.collection_name,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key=KEY_SRC_FULL, match=MatchValue(value=parent_source)
                    )
                ]
            ),
        ).points
        if len(chunks) == 0:
            return None

        if KEY_PAGE in chunks[0].payload:
            chunks = sorted(chunks, key=lambda x: int(x.payload[KEY_PAGE]))
        merged = ""
        for chunk in chunks:
            merged += chunk.payload[KEY_PAGE_CONTENT]
            merged += " "
        return merged

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        # first search for related chunks
        chunk_docs = self.vector_store.as_retriever(
            **(self.retriever_kwargs or {})
        ).invoke(query)
        if self.chunk_filter is not None:
            chunk_docs = self.chunk_filter(chunk_docs)
        by_parent: dict[str, list[Document]] = {}
        # group by source
        for doc in chunk_docs:
            parent_id = doc.metadata[KEY_SRC]
            if parent_id not in by_parent:
                by_parent[parent_id] = []
            by_parent[parent_id].append(doc)
        chunk_size = 512
        chain = (
            PromptTemplate.from_template(PARENT_RETRIEVER_SUMMARIZE_PROMPT)
            | self.summarize_model
            | StrOutputParser()
        )
        ret = []
        for parent_src, chunks in by_parent.items():
            cache_key = f"{parent_src}_{query}"
            parent = self._get_full_parent(parent_src)
            if parent is None:
                ret.extend(chunks)
                continue
            in_cache = cache.get(cache_key)
            summary = parent
            if self.summarize:
                if in_cache is not None:
                    summary = in_cache
                else:
                    summary = chain.invoke(
                        {
                            "query": query,
                            "parent": parent,
                            "chunk_size": chunk_size,
                            "role": self.role,
                        }
                    )
                    cache.add(cache_key, summary)
            ret.append(
                Document(
                    page_content=summary,
                    metadata={KEY_SRC: parent_src, KEY_ORIGINAL_DOCUMENT: parent, KEY_ORIGINAL_CHUNKS: chunks},
                )
            )

        return ret


if __name__ == "__main__":
    from langchain_ollama import ChatOllama

    summarizer = ChatOllama(model="qwq:latest", extract_reasoning=True)
    retriever = QdrantParentDocumentRetriever(
        summarize_model=summarizer,
        vector_store=get_vector_store(collection_name="immune"),
        role="computational antibody design expert",
        retriever_kwargs={"search_type": "mmr", "search_kwargs": {"k": 4}},
    )
    docs = retriever.invoke("predict binding affinity of antigen to BCR")
    for doc in docs:
        print("=============== Original Document ===============")
        print(doc.metadata[KEY_SRC])
        print("=============== Summarized Chunk ===============")
        print(doc.page_content)
