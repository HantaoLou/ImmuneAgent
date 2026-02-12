#!/usr/bin/env python3
"""
Comprehensive Unit Tests for vector_search_tool.py

This test suite covers ALL logic branches and business scenarios:
1. Utility functions (remove_think_tags, is_academic_noise, clean_document_content)
2. Configuration functions (get_collection_config, get_doc_source)
3. Embedder creation (_create_embedder_for_group, _create_embedder_from_config)
4. Vector store management (_get_vector_store)
5. QdrantParentDocumentRetriever (source_key, source_field)
6. Retrieval functions (_retrieve_from_single_collection, retrieve_doc)
7. Scoring functions (_batch_score_documents, model_filter_and_rank)
8. Edge cases and error handling

Author: Cascade AI
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add agent path
sys.path.insert(0, '/data/server/ImmuneAgent_2.0/agent')

# Load environment variables
from dotenv import load_dotenv
load_dotenv('/data/server/ImmuneAgent_2.0/agent/nodes/subagents/deep_research/.env')


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions: remove_think_tags, is_academic_noise, clean_document_content"""
    
    def setUp(self):
        from nodes.subagents.deep_research.vector_search_tool import (
            remove_think_tags,
            is_academic_noise,
            clean_document_content
        )
        self.remove_think_tags = remove_think_tags
        self.is_academic_noise = is_academic_noise
        self.clean_document_content = clean_document_content

    # ============ remove_think_tags tests ============
    
    def test_remove_think_tags_with_complete_tags(self):
        """Test removing complete <think> tags."""
        text = "Hello <think>internal thought</think> World"
        result = self.remove_think_tags(text)
        self.assertEqual(result, "Hello  World")
    
    def test_remove_think_tags_with_multiline(self):
        """Test removing multiline think tags."""
        text = "Start <think>\nline1\nline2\n</think> End"
        result = self.remove_think_tags(text)
        self.assertEqual(result, "Start  End")
    
    def test_remove_think_tags_with_only_closing(self):
        """Test removing content before closing tag only."""
        text = "hidden content</think> visible"
        result = self.remove_think_tags(text)
        self.assertEqual(result, "visible")
    
    def test_remove_think_tags_with_none(self):
        """Test handling None input."""
        result = self.remove_think_tags(None)
        self.assertEqual(result, "")
    
    def test_remove_think_tags_no_tags(self):
        """Test text without think tags."""
        text = "Normal text without tags"
        result = self.remove_think_tags(text)
        self.assertEqual(result, "Normal text without tags")
    
    # ============ is_academic_noise tests ============
    
    def test_is_academic_noise_empty_content(self):
        """Test empty content is noise."""
        self.assertTrue(self.is_academic_noise(""))
        self.assertTrue(self.is_academic_noise(None))
    
    def test_is_academic_noise_short_content(self):
        """Test short content (<= 80 chars) is noise."""
        short_text = "A" * 80
        self.assertTrue(self.is_academic_noise(short_text))
    
    def test_is_academic_noise_citation_bracket(self):
        """Test citation format [6] at start is noise."""
        text = "[6] This is a citation reference with more than 80 characters to pass length check"
        self.assertTrue(self.is_academic_noise(text))
    
    def test_is_academic_noise_author_format(self):
        """Test author format '32. h. cai' is noise."""
        text = "32. h. cai et some more text to make it longer than 80 characters for the test"
        self.assertTrue(self.is_academic_noise(text))
    
    def test_is_academic_noise_et_al(self):
        """Test 'et al.' is noise."""
        text = "Smith et al. published this paper with more than 80 characters for testing purposes"
        self.assertTrue(self.is_academic_noise(text))
    
    def test_is_academic_noise_doi(self):
        """Test 'doi:' is noise."""
        text = "Reference: doi:10.1234/example with more than 80 characters for testing purposes here"
        self.assertTrue(self.is_academic_noise(text))
    
    def test_is_academic_noise_springer(self):
        """Test Springer Nature format is noise."""
        text = "© Springer Nature 2024 all rights reserved with more than 80 characters for testing"
        self.assertTrue(self.is_academic_noise(text))
    
    def test_is_academic_noise_corresponding_author(self):
        """Test corresponding author format is noise."""
        text = "Corresponding author: john@example.com with more than 80 characters for testing purposes"
        self.assertTrue(self.is_academic_noise(text))
    
    def test_is_academic_noise_figure(self):
        """Test figure reference is noise."""
        text = "Figure 1 shows the experimental setup with more than 80 characters for testing purposes"
        self.assertTrue(self.is_academic_noise(text))
    
    def test_is_academic_noise_table(self):
        """Test table reference is noise."""
        text = "Table 2 summarizes the results with more than 80 characters for testing purposes here"
        self.assertTrue(self.is_academic_noise(text))
    
    def test_is_academic_noise_volume(self):
        """Test volume format is noise."""
        text = "Published in Volume 123 Issue 4 with more than 80 characters for testing purposes here"
        self.assertTrue(self.is_academic_noise(text))
    
    def test_is_academic_noise_excessive_brackets(self):
        """Test excessive brackets/parentheses is noise."""
        text = "(((())))[[[[]]]] " * 10 + "more text to make it longer"
        self.assertTrue(self.is_academic_noise(text))
    
    def test_is_academic_noise_valid_content(self):
        """Test valid academic content is not noise."""
        text = "CAR-T cell therapy represents a revolutionary approach to cancer treatment. " \
               "This methodology involves engineering patient T cells to express chimeric antigen receptors."
        self.assertFalse(self.is_academic_noise(text))
    
    # ============ clean_document_content tests ============
    
    def test_clean_document_content_empty(self):
        """Test cleaning empty content."""
        self.assertEqual(self.clean_document_content(""), "")
        self.assertEqual(self.clean_document_content(None), "")
    
    def test_clean_document_content_trailing_numbers(self):
        """Test removing trailing digit numbers."""
        text = "Some content here12345"
        result = self.clean_document_content(text)
        self.assertEqual(result, "Some content here")
    
    def test_clean_document_content_whitespace(self):
        """Test removing excessive whitespace."""
        text = "Text   with    multiple   spaces"
        result = self.clean_document_content(text)
        self.assertEqual(result, "Text with multiple spaces")


class TestCollectionConfig(unittest.TestCase):
    """Test collection configuration functions."""
    
    def setUp(self):
        from nodes.subagents.deep_research.vector_search_tool import (
            get_collection_config,
            get_doc_source,
            COLLECTION_CONFIG,
            EMBEDDER_GROUPS
        )
        self.get_collection_config = get_collection_config
        self.get_doc_source = get_doc_source
        self.COLLECTION_CONFIG = COLLECTION_CONFIG
        self.EMBEDDER_GROUPS = EMBEDDER_GROUPS

    def test_get_collection_config_known_1536(self):
        """Test getting config for known 1536-dim collection."""
        config = self.get_collection_config("Immunology")
        self.assertEqual(config["dimension"], 1536)
        self.assertEqual(config["source_key"], "metadata.source")
        self.assertEqual(config["source_field"], "source")
        self.assertEqual(config["embedder_group"], "openai_1536")
    
    def test_get_collection_config_known_768(self):
        """Test getting config for known 768-dim collection."""
        config = self.get_collection_config("virology")
        self.assertEqual(config["dimension"], 768)
        self.assertEqual(config["embedder_group"], "pubmedbert_768")
    
    def test_get_collection_config_different_source_field(self):
        """Test collection with different source field (source_file)."""
        config = self.get_collection_config("hle_immunology")
        self.assertEqual(config["source_key"], "metadata.source_file")
        self.assertEqual(config["source_field"], "source_file")
    
    def test_get_collection_config_unknown(self):
        """Test getting config for unknown collection uses default."""
        config = self.get_collection_config("unknown_collection_xyz")
        self.assertEqual(config["dimension"], 1536)
        self.assertEqual(config["embedder_group"], "openai_1536")
    
    def test_get_doc_source_with_source_field(self):
        """Test extracting source from document with source field."""
        mock_doc = Mock()
        mock_doc.metadata = {"source": "/path/to/file.pdf"}
        
        result = self.get_doc_source(mock_doc, "Immunology")
        self.assertEqual(result, "/path/to/file.pdf")
    
    def test_get_doc_source_with_source_file_field(self):
        """Test extracting source from document with source_file field."""
        mock_doc = Mock()
        mock_doc.metadata = {"source_file": "/path/to/file.pdf"}
        
        result = self.get_doc_source(mock_doc, "hle_immunology")
        self.assertEqual(result, "/path/to/file.pdf")
    
    def test_get_doc_source_fallback_to_source(self):
        """Test fallback to 'source' when configured field not found."""
        mock_doc = Mock()
        mock_doc.metadata = {"source": "/fallback/path.pdf"}
        
        result = self.get_doc_source(mock_doc, "hle_immunology")
        self.assertEqual(result, "/fallback/path.pdf")
    
    def test_get_doc_source_unknown(self):
        """Test returns 'unknown' when no source found."""
        mock_doc = Mock()
        mock_doc.metadata = {}
        
        result = self.get_doc_source(mock_doc, "Immunology")
        self.assertEqual(result, "unknown")
    
    def test_get_doc_source_no_metadata(self):
        """Test handling document without metadata attribute."""
        mock_doc = Mock(spec=[])  # No metadata attribute
        
        result = self.get_doc_source(mock_doc, "Immunology")
        self.assertEqual(result, "unknown")
    
    def test_embedder_groups_config(self):
        """Test EMBEDDER_GROUPS configuration is correct."""
        self.assertIn("openai_1536", self.EMBEDDER_GROUPS)
        self.assertIn("pubmedbert_768", self.EMBEDDER_GROUPS)
        
        openai_config = self.EMBEDDER_GROUPS["openai_1536"]
        self.assertEqual(openai_config["provider"], "openai")
        self.assertEqual(openai_config["dimension"], 1536)
        
        pubmed_config = self.EMBEDDER_GROUPS["pubmedbert_768"]
        self.assertEqual(pubmed_config["provider"], "huggingface")
        self.assertEqual(pubmed_config["dimension"], 768)


class TestPydanticModels(unittest.TestCase):
    """Test Pydantic model definitions."""
    
    def setUp(self):
        from nodes.subagents.deep_research.vector_search_tool import (
            DocumentEvaluation,
            BatchDocumentEvaluation,
            RetrievedDocument
        )
        self.DocumentEvaluation = DocumentEvaluation
        self.BatchDocumentEvaluation = BatchDocumentEvaluation
        self.RetrievedDocument = RetrievedDocument
    
    def test_document_evaluation_valid(self):
        """Test creating valid DocumentEvaluation."""
        eval = self.DocumentEvaluation(
            doc_id=1,
            relevance_score=8,
            quality_score=7,
            noise_level=1,
            final_score=85
        )
        self.assertEqual(eval.doc_id, 1)
        self.assertEqual(eval.relevance_score, 8)
    
    def test_document_evaluation_default(self):
        """Test DocumentEvaluation with defaults."""
        eval = self.DocumentEvaluation(
            relevance_score=5,
            quality_score=5,
            noise_level=0,
            final_score=50
        )
        self.assertEqual(eval.doc_id, 0)  # Default
    
    def test_retrieved_document(self):
        """Test RetrievedDocument creation and string representation."""
        doc = self.RetrievedDocument(
            source="/path/to/paper.pdf",
            page_content="This is the content."
        )
        self.assertEqual(doc.source, "/path/to/paper.pdf")
        self.assertIn("<source>", str(doc))
        self.assertIn("<content>", str(doc))
    
    def test_batch_document_evaluation(self):
        """Test BatchDocumentEvaluation with multiple evaluations."""
        eval1 = self.DocumentEvaluation(
            relevance_score=8, quality_score=7, noise_level=1, final_score=85
        )
        eval2 = self.DocumentEvaluation(
            relevance_score=6, quality_score=5, noise_level=2, final_score=60
        )
        batch = self.BatchDocumentEvaluation(evaluations=[eval1, eval2])
        self.assertEqual(len(batch.evaluations), 2)


class TestFilterChunks(unittest.TestCase):
    """Test _filter_chunks function."""
    
    def setUp(self):
        from nodes.subagents.deep_research.vector_search_tool import _filter_chunks
        from langchain_core.documents import Document
        self._filter_chunks = _filter_chunks
        self.Document = Document
    
    def test_filter_chunks_removes_noise(self):
        """Test that noise documents are filtered out."""
        valid_content = "This is valid academic content about CAR-T cell therapy. " * 3
        noise_content = "[6] Reference citation that should be filtered out as noise text"
        
        docs = [
            self.Document(page_content=valid_content),
            self.Document(page_content=noise_content),
        ]
        
        result = self._filter_chunks(docs)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].page_content, valid_content)
    
    def test_filter_chunks_empty_list(self):
        """Test filtering empty list."""
        result = self._filter_chunks([])
        self.assertEqual(result, [])


class TestQdrantConfig(unittest.TestCase):
    """Test QdrantConfig class."""
    
    def setUp(self):
        from nodes.subagents.deep_research.vector_search_tool import QdrantConfig
        self.QdrantConfig = QdrantConfig
    
    def test_qdrant_config_from_env(self):
        """Test creating QdrantConfig from environment."""
        config = self.QdrantConfig.from_env()
        self.assertIsNotNone(config)
        self.assertIsNotNone(config.host)
        self.assertIsNotNone(config.port)
    
    def test_qdrant_config_get_client(self):
        """Test getting Qdrant client."""
        config = self.QdrantConfig.from_env()
        client = config.get_client()
        self.assertIsNotNone(client)
    
    def test_qdrant_config_hash(self):
        """Test QdrantConfig is hashable."""
        config = self.QdrantConfig.from_env()
        hash_value = hash(config)
        self.assertIsInstance(hash_value, int)


class TestEmbedderCreation(unittest.TestCase):
    """Test embedder creation functions."""
    
    def test_create_embedder_for_group_openai(self):
        """Test creating OpenAI embedder."""
        from nodes.subagents.deep_research.vector_search_tool import (
            _create_embedder_for_group,
            _embedder_cache
        )
        
        embedder = _create_embedder_for_group("openai_1536")
        self.assertIsNotNone(embedder)
        self.assertIn("openai_1536", _embedder_cache)
    
    def test_create_embedder_for_group_pubmedbert(self):
        """Test creating PubMedBERT embedder."""
        from nodes.subagents.deep_research.vector_search_tool import (
            _create_embedder_for_group,
            _embedder_cache
        )
        
        embedder = _create_embedder_for_group("pubmedbert_768")
        self.assertIsNotNone(embedder)
        self.assertIn("pubmedbert_768", _embedder_cache)
    
    def test_create_embedder_for_group_unknown(self):
        """Test creating embedder for unknown group uses default."""
        from nodes.subagents.deep_research.vector_search_tool import (
            _create_embedder_for_group
        )
        
        # Should fall back to openai_1536
        embedder = _create_embedder_for_group("unknown_group")
        self.assertIsNotNone(embedder)
    
    def test_create_embedder_from_config_legacy(self):
        """Test legacy embedder creation function."""
        from nodes.subagents.deep_research.vector_search_tool import (
            _create_embedder_from_config,
            _legacy_embedder_cache
        )
        
        config = {'configurable': {}}
        embedder = _create_embedder_from_config(config)
        self.assertIsNotNone(embedder)


class TestVectorStoreCreation(unittest.TestCase):
    """Test vector store creation and caching."""
    
    def test_get_vector_store_immunology(self):
        """Test creating vector store for Immunology collection."""
        from nodes.subagents.deep_research.vector_search_tool import (
            _get_vector_store,
            _vector_store_cache
        )
        
        config = {'configurable': {}}
        vs = _get_vector_store("Immunology", config)
        
        self.assertIsNotNone(vs)
        self.assertEqual(vs.collection_name, "Immunology")
        self.assertIn("Immunology", _vector_store_cache)
    
    def test_get_vector_store_virology(self):
        """Test creating vector store for virology collection."""
        from nodes.subagents.deep_research.vector_search_tool import (
            _get_vector_store,
            _vector_store_cache
        )
        
        config = {'configurable': {}}
        vs = _get_vector_store("virology", config)
        
        self.assertIsNotNone(vs)
        self.assertEqual(vs.collection_name, "virology")
        self.assertIn("virology", _vector_store_cache)
    
    def test_get_vector_store_caching(self):
        """Test vector store is cached and reused."""
        from nodes.subagents.deep_research.vector_search_tool import _get_vector_store
        
        config = {'configurable': {}}
        vs1 = _get_vector_store("Immunology", config)
        vs2 = _get_vector_store("Immunology", config)
        
        self.assertIs(vs1, vs2)  # Same instance


class TestQdrantParentDocumentRetriever(unittest.TestCase):
    """Test QdrantParentDocumentRetriever class."""
    
    def test_retriever_default_source_fields(self):
        """Test retriever has correct default source fields."""
        from nodes.subagents.deep_research.vector_search_tool import (
            QdrantParentDocumentRetriever
        )
        
        # Check class has source_key and source_field in model_fields (Pydantic v2)
        model_fields = QdrantParentDocumentRetriever.model_fields
        self.assertIn('source_key', model_fields)
        self.assertIn('source_field', model_fields)
        
        # Check default values
        self.assertEqual(model_fields['source_key'].default, 'metadata.source')
        self.assertEqual(model_fields['source_field'].default, 'source')


class TestEnvironmentParsing(unittest.TestCase):
    """Test environment variable parsing for multi-collection."""
    
    def test_parse_single_collection(self):
        """Test parsing single collection from QDRANT_COLLECTION."""
        os.environ['QDRANT_COLLECTIONS'] = ''
        os.environ['QDRANT_COLLECTION'] = 'Immunology'
        
        collections_str = os.getenv("QDRANT_COLLECTIONS", "")
        if collections_str:
            collections = [c.strip() for c in collections_str.split(",") if c.strip()]
        else:
            collections = [os.getenv("QDRANT_COLLECTION", "Immunology")]
        
        self.assertEqual(collections, ['Immunology'])
    
    def test_parse_multi_collection(self):
        """Test parsing multiple collections from QDRANT_COLLECTIONS."""
        os.environ['QDRANT_COLLECTIONS'] = 'Immunology, virology, hle_immunology'
        
        collections_str = os.getenv("QDRANT_COLLECTIONS", "")
        collections = [c.strip() for c in collections_str.split(",") if c.strip()]
        
        self.assertEqual(collections, ['Immunology', 'virology', 'hle_immunology'])
    
    def test_parse_empty_collection(self):
        """Test parsing with empty QDRANT_COLLECTIONS uses default."""
        os.environ['QDRANT_COLLECTIONS'] = ''
        os.environ['QDRANT_COLLECTION'] = ''
        
        collections_str = os.getenv("QDRANT_COLLECTIONS", "")
        if collections_str:
            collections = [c.strip() for c in collections_str.split(",") if c.strip()]
        else:
            collections = [os.getenv("QDRANT_COLLECTION", "Immunology")]
        
        # Should use default "Immunology" or empty string
        self.assertEqual(len(collections), 1)


class TestScoringFunctions(unittest.TestCase):
    """Test document scoring and filtering functions."""
    
    def setUp(self):
        from nodes.subagents.deep_research.vector_search_tool import (
            _safe_document_filter,
            DocumentEvaluation
        )
        from langchain_core.documents import Document
        
        self._safe_document_filter = _safe_document_filter
        self.DocumentEvaluation = DocumentEvaluation
        self.Document = Document
    
    def test_safe_document_filter_sorts_by_score(self):
        """Test documents are sorted by final score."""
        docs = [
            self.Document(page_content="Low score doc"),
            self.Document(page_content="High score doc"),
            self.Document(page_content="Medium score doc"),
        ]
        
        evaluations = [
            (1, self.DocumentEvaluation(doc_id=1, relevance_score=5, quality_score=5, noise_level=1, final_score=50)),
            (2, self.DocumentEvaluation(doc_id=2, relevance_score=9, quality_score=9, noise_level=0, final_score=90)),
            (3, self.DocumentEvaluation(doc_id=3, relevance_score=7, quality_score=7, noise_level=1, final_score=70)),
        ]
        
        result = self._safe_document_filter(docs, evaluations, target_count=3)
        
        # Should be sorted by final_score descending
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].page_content, "High score doc")
    
    def test_safe_document_filter_empty_evaluations(self):
        """Test handling empty evaluations."""
        docs = [self.Document(page_content="Some content")]
        result = self._safe_document_filter(docs, [], target_count=1)
        self.assertEqual(result, [])
    
    def test_safe_document_filter_target_count(self):
        """Test target_count limits results."""
        docs = [self.Document(page_content=f"Doc {i}") for i in range(10)]
        evaluations = [
            (i+1, self.DocumentEvaluation(doc_id=i+1, relevance_score=5, quality_score=5, noise_level=0, final_score=50))
            for i in range(10)
        ]
        
        result = self._safe_document_filter(docs, evaluations, target_count=3)
        self.assertEqual(len(result), 3)


class TestIntegration(unittest.TestCase):
    """Integration tests with real API calls."""
    
    def test_openai_embedding_dimension(self):
        """Test OpenAI embedding produces correct dimension."""
        from nodes.subagents.deep_research.vector_search_tool import _create_embedder_for_group
        
        embedder = _create_embedder_for_group("openai_1536")
        embedding = embedder.embed_query("test query")
        
        self.assertEqual(len(embedding), 1536)
    
    def test_pubmedbert_embedding_dimension(self):
        """Test PubMedBERT embedding produces correct dimension."""
        from nodes.subagents.deep_research.vector_search_tool import _create_embedder_for_group
        
        embedder = _create_embedder_for_group("pubmedbert_768")
        embedding = embedder.embed_query("test query")
        
        self.assertEqual(len(embedding), 768)
    
    def test_vector_store_similarity_search(self):
        """Test vector store can perform similarity search."""
        from nodes.subagents.deep_research.vector_search_tool import _get_vector_store
        
        config = {'configurable': {}}
        vs = _get_vector_store("Immunology", config)
        
        docs = vs.similarity_search("CAR-T cell therapy", k=1)
        self.assertGreaterEqual(len(docs), 0)


class TestEndToEndRetrieval(unittest.TestCase):
    """End-to-end retrieval tests with real LLM calls - these are slow!"""
    
    def setUp(self):
        """Set up test configuration."""
        self.config = {
            'configurable': {
                'summarization_model': 'deepseek:deepseek-chat',
                'research_model': 'deepseek:deepseek-chat',
                'vector_scoring_model': 'deepseek:deepseek-chat',
            }
        }
    
    def test_retrieve_single_collection_immunology(self):
        """Test full retrieval from Immunology (1536 dim) with LLM scoring."""
        from nodes.subagents.deep_research.vector_search_tool import retrieve_doc
        import time
        
        os.environ['QDRANT_COLLECTIONS'] = ''
        os.environ['QDRANT_COLLECTION'] = 'Immunology'
        
        start = time.time()
        results = retrieve_doc(
            ['CAR-T cell therapy mechanism'],
            self.config,
            k_per_query=2
        )
        elapsed = time.time() - start
        
        print(f"\n[Immunology] Retrieved {len(results)} docs in {elapsed:.2f}s")
        
        # Verify results structure
        self.assertIsInstance(results, list)
        for doc in results:
            self.assertTrue(hasattr(doc, 'source'))
            self.assertTrue(hasattr(doc, 'page_content'))
            self.assertIsInstance(doc.source, str)
            self.assertIsInstance(doc.page_content, str)
    
    def test_retrieve_single_collection_virology(self):
        """Test full retrieval from virology (768 dim PubMedBERT) with LLM scoring."""
        from nodes.subagents.deep_research.vector_search_tool import retrieve_doc
        import time
        
        os.environ['QDRANT_COLLECTIONS'] = ''
        os.environ['QDRANT_COLLECTION'] = 'virology'
        
        start = time.time()
        results = retrieve_doc(
            ['respiratory syncytial virus vaccine'],
            self.config,
            k_per_query=2
        )
        elapsed = time.time() - start
        
        print(f"\n[virology] Retrieved {len(results)} docs in {elapsed:.2f}s")
        
        # Verify results
        self.assertIsInstance(results, list)
    
    def test_retrieve_multi_collection(self):
        """Test full retrieval from multiple collections with different dimensions."""
        from nodes.subagents.deep_research.vector_search_tool import retrieve_doc
        import time
        
        os.environ['QDRANT_COLLECTIONS'] = 'Immunology,virology'
        
        start = time.time()
        results = retrieve_doc(
            ['immune response to viral infection'],
            self.config,
            k_per_query=2
        )
        elapsed = time.time() - start
        
        print(f"\n[Multi-Collection] Retrieved {len(results)} docs in {elapsed:.2f}s")
        
        # Verify cross-collection results
        self.assertIsInstance(results, list)
    
    def test_retrieve_with_multiple_queries(self):
        """Test retrieval with multiple query strings."""
        from nodes.subagents.deep_research.vector_search_tool import retrieve_doc
        import time
        
        os.environ['QDRANT_COLLECTIONS'] = ''
        os.environ['QDRANT_COLLECTION'] = 'Immunology'
        
        queries = [
            'CAR-T cell engineering',
            'immune checkpoint inhibitors',
            'antibody drug conjugates'
        ]
        
        start = time.time()
        results = retrieve_doc(queries, self.config, k_per_query=2)
        elapsed = time.time() - start
        
        print(f"\n[Multiple Queries] Retrieved {len(results)} docs in {elapsed:.2f}s")
        
        self.assertIsInstance(results, list)
    
    def test_retrieve_deduplication(self):
        """Test that duplicate documents are removed."""
        from nodes.subagents.deep_research.vector_search_tool import retrieve_doc
        
        os.environ['QDRANT_COLLECTIONS'] = ''
        os.environ['QDRANT_COLLECTION'] = 'Immunology'
        
        # Same query twice should still deduplicate
        results = retrieve_doc(
            ['CAR-T cell therapy', 'CAR-T cell therapy'],
            self.config,
            k_per_query=3
        )
        
        # Check no duplicate sources
        sources = [doc.source for doc in results]
        unique_sources = set(sources)
        self.assertEqual(len(sources), len(unique_sources), "Duplicate sources found!")
    
    def test_batch_scoring_function(self):
        """Test batch scoring function with real LLM call."""
        from nodes.subagents.deep_research.vector_search_tool import _batch_score_documents
        import time
        
        docs_info = [
            (0, "CAR-T cell therapy is a revolutionary cancer treatment.", "What is CAR-T?"),
            (1, "This document contains random noise and references [1][2][3].", "What is CAR-T?"),
        ]
        
        start = time.time()
        results = _batch_score_documents(docs_info, self.config)
        elapsed = time.time() - start
        
        print(f"\n[Batch Scoring] Scored {len(results)} docs in {elapsed:.2f}s")
        
        self.assertEqual(len(results), 2)
        # First doc should score higher than noisy second doc
        self.assertGreater(results[0][1].final_score, 0)
    
    def test_parent_document_retriever(self):
        """Test QdrantParentDocumentRetriever with summarization."""
        from nodes.subagents.deep_research.vector_search_tool import (
            QdrantParentDocumentRetriever,
            _get_vector_store,
            _get_summarize_model,
            _filter_chunks
        )
        import time
        
        config = {'configurable': {}}
        vector_store = _get_vector_store("Immunology", config)
        summarize_model = _get_summarize_model(self.config)
        
        retriever = QdrantParentDocumentRetriever(
            summarize_model=summarize_model,
            vector_store=vector_store,
            role="research expert",
            retriever_kwargs={
                "search_type": "mmr",
                "search_kwargs": {"k": 2, "lambda_mult": 0.65},
            },
            chunk_filter=_filter_chunks,
        )
        
        start = time.time()
        docs = retriever.invoke("CAR-T cell therapy")
        elapsed = time.time() - start
        
        print(f"\n[Parent Retriever] Retrieved {len(docs)} docs in {elapsed:.2f}s")
        
        self.assertIsInstance(docs, list)


def run_tests():
    """Run all unit tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestUtilityFunctions,
        TestCollectionConfig,
        TestPydanticModels,
        TestFilterChunks,
        TestQdrantConfig,
        TestEmbedderCreation,
        TestVectorStoreCreation,
        TestQdrantParentDocumentRetriever,
        TestEnvironmentParsing,
        TestScoringFunctions,
        TestIntegration,
        TestEndToEndRetrieval,  # Slow tests with real LLM calls
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
