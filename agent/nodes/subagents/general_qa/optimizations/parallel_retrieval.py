"""
Parallel Knowledge Retriever - P2 Priority Optimization

Implements parallel retrieval with result aggregation:
1. Execute multiple retrieval sources in parallel
2. Merge and deduplicate results
3. Rank by relevance and source quality
4. Handle partial failures gracefully
"""

import asyncio
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import hashlib


class SourceType(Enum):
    """Types of knowledge sources"""
    KNOWLEDGE_GRAPH = "knowledge_graph"
    DISGENET = "disgenet"
    UNIPROT = "uniprot"
    PROTEINATLAS = "proteinatlas"
    WEB_SEARCH = "web_search"
    PAPERQA = "paperqa"
    VECTOR_DB = "vector_db"


@dataclass
class RetrievalResult:
    """Result from a single retrieval source"""
    source: SourceType
    success: bool
    data: Any
    execution_time: float
    error: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedResult:
    """Aggregated result from multiple sources"""
    success: bool
    total_sources: int
    successful_sources: int
    results_by_source: Dict[SourceType, RetrievalResult]
    merged_data: List[Dict[str, Any]]
    merged_entities: List[str]
    merged_relations: List[Tuple[str, str, str]]
    confidence: float
    total_time: float
    errors: List[str]


# Source quality weights
SOURCE_QUALITY_WEIGHTS = {
    SourceType.KNOWLEDGE_GRAPH: 0.9,
    SourceType.DISGENET: 0.95,
    SourceType.UNIPROT: 0.95,
    SourceType.PROTEINATLAS: 0.9,
    SourceType.PAPERQA: 0.85,
    SourceType.VECTOR_DB: 0.7,
    SourceType.WEB_SEARCH: 0.5,
}


class ParallelKnowledgeRetriever:
    """
    Executes parallel retrieval from multiple knowledge sources
    """
    
    def __init__(self, 
                 max_concurrent: int = 5,
                 timeout_per_source: float = 30.0):
        self.max_concurrent = max_concurrent
        self.timeout_per_source = timeout_per_source
        self.source_weights = SOURCE_QUALITY_WEIGHTS
    
    async def retrieve_parallel(self,
                                sources: List[SourceType],
                                query: str,
                                retrieval_func: Callable[[SourceType, str], Any],
                                additional_params: Optional[Dict] = None) -> AggregatedResult:
        """
        Execute parallel retrieval from multiple sources
        
        Args:
            sources: List of source types to query
            query: The query string
            retrieval_func: Async function that retrieves from a source
            additional_params: Optional additional parameters
            
        Returns:
            AggregatedResult with merged data from all sources
        """
        start_time = time.time()
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Create tasks for all sources
        tasks = []
        for source in sources:
            task = self._retrieve_with_timeout(
                source, query, retrieval_func, semaphore, additional_params
            )
            tasks.append(task)
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        results_by_source = {}
        errors = []
        
        for source, result in zip(sources, results):
            if isinstance(result, Exception):
                results_by_source[source] = RetrievalResult(
                    source=source,
                    success=False,
                    data=None,
                    execution_time=0,
                    error=str(result)
                )
                errors.append(f"{source.value}: {str(result)}")
            else:
                results_by_source[source] = result
                if not result.success:
                    errors.append(f"{source.value}: {result.error}")
        
        # Aggregate results
        aggregated = self._aggregate_results(results_by_source)
        aggregated.total_time = time.time() - start_time
        aggregated.errors = errors
        
        return aggregated
    
    async def _retrieve_with_timeout(self,
                                     source: SourceType,
                                     query: str,
                                     retrieval_func: Callable,
                                     semaphore: asyncio.Semaphore,
                                     additional_params: Optional[Dict]) -> RetrievalResult:
        """Execute retrieval with timeout and concurrency control"""
        async with semaphore:
            start_time = time.time()
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    retrieval_func(source, query, additional_params),
                    timeout=self.timeout_per_source
                )
                
                execution_time = time.time() - start_time
                
                return RetrievalResult(
                    source=source,
                    success=True,
                    data=result,
                    execution_time=execution_time,
                    confidence=self.source_weights.get(source, 0.7)
                )
                
            except asyncio.TimeoutError:
                return RetrievalResult(
                    source=source,
                    success=False,
                    data=None,
                    execution_time=self.timeout_per_source,
                    error="Timeout"
                )
            except Exception as e:
                return RetrievalResult(
                    source=source,
                    success=False,
                    data=None,
                    execution_time=time.time() - start_time,
                    error=str(e)
                )
    
    def _aggregate_results(self, 
                          results_by_source: Dict[SourceType, RetrievalResult]) -> AggregatedResult:
        """Aggregate results from multiple sources"""
        successful_results = {k: v for k, v in results_by_source.items() if v.success}
        
        merged_data = []
        merged_entities = []
        merged_relations = []
        
        # Merge data from each source
        for source, result in successful_results.items():
            if result.data:
                # Handle different data formats
                if isinstance(result.data, list):
                    for item in result.data:
                        if isinstance(item, dict):
                            # Add source metadata
                            item_with_source = item.copy()
                            item_with_source['_source'] = source.value
                            item_with_source['_confidence'] = result.confidence
                            merged_data.append(item_with_source)
                            
                            # Extract entities
                            self._extract_entities_from_item(item, merged_entities)
                            
                            # Extract relations
                            self._extract_relations_from_item(item, merged_relations)
                elif isinstance(result.data, dict):
                    item_with_source = result.data.copy()
                    item_with_source['_source'] = source.value
                    item_with_source['_confidence'] = result.confidence
                    merged_data.append(item_with_source)
                    
                    self._extract_entities_from_item(result.data, merged_entities)
                    self._extract_relations_from_item(result.data, merged_relations)
        
        # Deduplicate entities
        merged_entities = list(set(merged_entities))
        
        # Deduplicate relations
        merged_relations = list(set(merged_relations))
        
        # Calculate overall confidence
        if successful_results:
            confidence = sum(r.confidence for r in successful_results.values()) / len(successful_results)
        else:
            confidence = 0.0
        
        return AggregatedResult(
            success=len(successful_results) > 0,
            total_sources=len(results_by_source),
            successful_sources=len(successful_results),
            results_by_source=results_by_source,
            merged_data=merged_data,
            merged_entities=merged_entities,
            merged_relations=merged_relations,
            confidence=confidence,
            total_time=0.0  # Will be set by caller
        )
    
    def _extract_entities_from_item(self, item: Dict, entities: List[str]):
        """Extract entities from a data item"""
        entity_keys = ['name', 'gene', 'protein', 'disease', 'entity', 'term', 'id']
        
        for key in entity_keys:
            if key in item:
                value = item[key]
                if isinstance(value, str):
                    entities.append(value)
                elif isinstance(value, list):
                    entities.extend([str(v) for v in value if v])
    
    def _extract_relations_from_item(self, item: Dict, relations: List[Tuple[str, str, str]]):
        """Extract relations from a data item"""
        if 'relation' in item or 'relationship' in item:
            subject = item.get('subject', item.get('gene', ''))
            relation = item.get('relation', item.get('relationship', ''))
            obj = item.get('object', item.get('target', ''))
            
            if subject and relation and obj:
                relations.append((str(subject), str(relation), str(obj)))
    
    def rank_results(self, merged_data: List[Dict], 
                     query: str,
                     max_results: int = 20) -> List[Dict]:
        """
        Rank merged results by relevance
        
        Args:
            merged_data: List of merged data items
            query: The original query
            max_results: Maximum number of results to return
            
        Returns:
            Sorted list of results
        """
        query_terms = set(query.lower().split())
        
        scored_results = []
        for item in merged_data:
            score = self._calculate_relevance_score(item, query_terms)
            scored_results.append((score, item))
        
        # Sort by score descending
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        return [item for score, item in scored_results[:max_results]]
    
    def _calculate_relevance_score(self, item: Dict, query_terms: set) -> float:
        """Calculate relevance score for an item"""
        score = 0.0
        
        # Base score from source confidence
        score += item.get('_confidence', 0.5) * 0.3
        
        # Term matching score
        item_text = ' '.join(str(v) for v in item.values() if isinstance(v, (str, int, float)))
        item_terms = set(item_text.lower().split())
        
        if query_terms and item_terms:
            overlap = query_terms & item_terms
            term_score = len(overlap) / len(query_terms)
            score += term_score * 0.4
        
        # Exact match bonus
        for key in ['name', 'gene', 'protein', 'disease']:
            if key in item:
                if str(item[key]).lower() in ' '.join(query_terms):
                    score += 0.3
                    break
        
        return score
    
    def get_retrieval_report(self, result: AggregatedResult) -> str:
        """Generate human-readable retrieval report"""
        lines = ["# Parallel Retrieval Report\n"]
        
        lines.append(f"## Summary")
        lines.append(f"- Total Sources: {result.total_sources}")
        lines.append(f"- Successful: {result.successful_sources}")
        lines.append(f"- Overall Confidence: {result.confidence:.2%}")
        lines.append(f"- Total Time: {result.total_time:.2f}s")
        
        lines.append(f"\n## Results by Source")
        lines.append("| Source | Status | Time | Confidence | Items |")
        lines.append("|--------|--------|------|------------|-------|")
        
        for source, retrieval in result.results_by_source.items():
            status = "[SUCCESS]" if retrieval.success else "[ERROR]"
            items = len(retrieval.data) if isinstance(retrieval.data, list) else (1 if retrieval.data else 0)
            lines.append(
                f"| {source.value} | {status} | {retrieval.execution_time:.2f}s | "
                f"{retrieval.confidence:.0%} | {items} |"
            )
        
        if result.errors:
            lines.append(f"\n## Errors")
            for error in result.errors:
                lines.append(f"- {error}")
        
        lines.append(f"\n## Merged Results")
        lines.append(f"- Total Items: {len(result.merged_data)}")
        lines.append(f"- Unique Entities: {len(result.merged_entities)}")
        lines.append(f"- Relations: {len(result.merged_relations)}")
        
        if result.merged_entities[:10]:
            lines.append(f"- Top Entities: {', '.join(result.merged_entities[:10])}")
        
        return "\n".join(lines)


# Convenience function
async def parallel_retrieve(query: str,
                           sources: List[SourceType],
                           retrieval_func: Callable) -> AggregatedResult:
    """
    Quick parallel retrieval function
    
    Args:
        query: The query string
        sources: List of sources to query
        retrieval_func: Async retrieval function
        
    Returns:
        AggregatedResult
    """
    retriever = ParallelKnowledgeRetriever()
    return await retriever.retrieve_parallel(sources, query, retrieval_func)
