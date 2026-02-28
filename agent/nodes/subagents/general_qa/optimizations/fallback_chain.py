"""
Tool Fallback Chain - P1 Priority Optimization

Implements fallback chain for tool failures:
1. Primary tool → Fallback tool → Web search → LLM knowledge
2. Graceful degradation with quality indicators
3. Failure pattern detection and bypass
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time


class FallbackLevel(Enum):
    """Levels of fallback"""
    PRIMARY = 1         # Primary specialized tool
    SECONDARY = 2       # Alternative specialized tool
    TERTIARY = 3        # General knowledge tool
    WEB_SEARCH = 4      # Web search
    LLM_KNOWLEDGE = 5   # Direct LLM knowledge
    FAILED = 6          # All methods failed


class ToolStatus(Enum):
    """Status of a tool"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class FallbackResult:
    """Result from a fallback chain execution"""
    success: bool
    level: FallbackLevel
    tool_used: str
    result: Any
    quality_score: float
    execution_time: float
    error: Optional[str] = None
    fallback_history: List[Tuple[str, bool, str]] = field(default_factory=list)


@dataclass
class ToolHealth:
    """Health tracking for a tool"""
    name: str
    status: ToolStatus = ToolStatus.UNKNOWN
    consecutive_failures: int = 0
    last_success_time: float = 0
    last_failure_time: float = 0
    total_calls: int = 0
    total_failures: int = 0
    
    @property
    def failure_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_failures / self.total_calls


# Default fallback chains by domain
DEFAULT_FALLBACK_CHAINS = {
    "gene_disease": [
        "query_disgenet",
        "query_omim",
        "query_knowledge_graph",
        "web_search",
    ],
    "protein_info": [
        "query_uniprot",
        "query_proteinatlas",
        "query_knowledge_graph",
        "web_search",
    ],
    "protein_interaction": [
        "query_string",
        "query_ppi",
        "query_knowledge_graph",
        "web_search",
    ],
    "general": [
        "query_knowledge_graph",
        "web_search",
    ],
}

# Quality scores for each fallback level
LEVEL_QUALITY_SCORES = {
    FallbackLevel.PRIMARY: 1.0,
    FallbackLevel.SECONDARY: 0.9,
    FallbackLevel.TERTIARY: 0.7,
    FallbackLevel.WEB_SEARCH: 0.5,
    FallbackLevel.LLM_KNOWLEDGE: 0.3,
    FallbackLevel.FAILED: 0.0,
}


class ToolFallbackChain:
    """
    Manages fallback chains for tool failures
    """
    
    def __init__(self, 
                 fallback_chains: Optional[Dict[str, List[str]]] = None,
                 failure_threshold: int = 3,
                 recovery_time: float = 300.0):  # 5 minutes
        self.fallback_chains = fallback_chains or DEFAULT_FALLBACK_CHAINS
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.tool_health: Dict[str, ToolHealth] = {}
    
    def get_fallback_chain(self, domain: str) -> List[str]:
        """Get the fallback chain for a domain"""
        if domain in self.fallback_chains:
            return self.fallback_chains[domain].copy()
        
        # Try to find a matching domain
        for key in self.fallback_chains:
            if key in domain or domain in key:
                return self.fallback_chains[key].copy()
        
        # Return general fallback chain
        return self.fallback_chains.get("general", ["web_search"]).copy()
    
    def register_tool(self, tool_name: str):
        """Register a tool for health tracking"""
        if tool_name not in self.tool_health:
            self.tool_health[tool_name] = ToolHealth(name=tool_name)
    
    def record_success(self, tool_name: str):
        """Record a successful tool call"""
        if tool_name not in self.tool_health:
            self.register_tool(tool_name)
        
        health = self.tool_health[tool_name]
        health.total_calls += 1
        health.consecutive_failures = 0
        health.last_success_time = time.time()
        health.status = ToolStatus.HEALTHY
    
    def record_failure(self, tool_name: str, error: str = ""):
        """Record a failed tool call"""
        if tool_name not in self.tool_health:
            self.register_tool(tool_name)
        
        health = self.tool_health[tool_name]
        health.total_calls += 1
        health.total_failures += 1
        health.consecutive_failures += 1
        health.last_failure_time = time.time()
        
        # Update status based on consecutive failures
        if health.consecutive_failures >= self.failure_threshold:
            health.status = ToolStatus.FAILED
        else:
            health.status = ToolStatus.DEGRADED
    
    def is_tool_available(self, tool_name: str) -> bool:
        """Check if a tool should be used (health check)"""
        if tool_name not in self.tool_health:
            return True  # Unknown tools are assumed available
        
        health = self.tool_health[tool_name]
        
        # Check if in recovery period
        if health.status == ToolStatus.FAILED:
            time_since_failure = time.time() - health.last_failure_time
            if time_since_failure > self.recovery_time:
                # Reset for retry
                health.status = ToolStatus.DEGRADED
                health.consecutive_failures = 0
                return True
            return False
        
        return True
    
    async def execute_with_fallback(self,
                                    domain: str,
                                    tool_executor: Callable[[str, Dict], Any],
                                    params: Dict[str, Any],
                                    available_tools: Optional[List[str]] = None) -> FallbackResult:
        """
        Execute tool calls with automatic fallback
        
        Args:
            domain: The domain for fallback chain selection
            tool_executor: Async function that executes a tool call
            params: Parameters for the tool
            available_tools: Optional list of available tools
            
        Returns:
            FallbackResult with the result and metadata
        """
        fallback_history = []
        chain = self.get_fallback_chain(domain)
        
        # Filter by available tools if specified
        if available_tools:
            chain = [t for t in chain if t in available_tools]
        
        for i, tool_name in enumerate(chain):
            # Check tool health
            if not self.is_tool_available(tool_name):
                fallback_history.append((tool_name, False, "Tool marked as unavailable"))
                continue
            
            # Determine fallback level
            if i == 0:
                level = FallbackLevel.PRIMARY
            elif i == 1:
                level = FallbackLevel.SECONDARY
            elif i == 2:
                level = FallbackLevel.TERTIARY
            elif "web" in tool_name.lower() or "search" in tool_name.lower():
                level = FallbackLevel.WEB_SEARCH
            else:
                level = FallbackLevel.TERTIARY
            
            # Execute tool
            start_time = time.time()
            try:
                result = await tool_executor(tool_name, params)
                execution_time = time.time() - start_time
                
                # Check if result is valid
                if result and not self._is_empty_result(result):
                    self.record_success(tool_name)
                    fallback_history.append((tool_name, True, "Success"))
                    
                    return FallbackResult(
                        success=True,
                        level=level,
                        tool_used=tool_name,
                        result=result,
                        quality_score=LEVEL_QUALITY_SCORES[level],
                        execution_time=execution_time,
                        fallback_history=fallback_history
                    )
                else:
                    fallback_history.append((tool_name, False, "Empty result"))
                    
            except Exception as e:
                execution_time = time.time() - start_time
                error_msg = str(e)
                self.record_failure(tool_name, error_msg)
                fallback_history.append((tool_name, False, error_msg))
        
        # All tools failed
        return FallbackResult(
            success=False,
            level=FallbackLevel.FAILED,
            tool_used="none",
            result=None,
            quality_score=0.0,
            execution_time=0.0,
            error="All fallback options exhausted",
            fallback_history=fallback_history
        )
    
    def _is_empty_result(self, result: Any) -> bool:
        """Check if a result is considered empty"""
        if result is None:
            return True
        if isinstance(result, (list, dict)):
            return len(result) == 0
        if isinstance(result, str):
            return len(result.strip()) == 0
        return False
    
    def get_health_report(self) -> str:
        """Generate a health report for all tools"""
        lines = ["# Tool Health Report\n"]
        
        lines.append("| Tool | Status | Failure Rate | Consecutive Failures |")
        lines.append("|------|--------|--------------|---------------------|")
        
        for name, health in sorted(self.tool_health.items()):
            status_icon = {
                ToolStatus.HEALTHY: "✅",
                ToolStatus.DEGRADED: "⚠️",
                ToolStatus.FAILED: "❌",
                ToolStatus.UNKNOWN: "❓"
            }.get(health.status, "❓")
            
            lines.append(
                f"| {name} | {status_icon} {health.status.value} | "
                f"{health.failure_rate:.1%} | {health.consecutive_failures} |"
            )
        
        return "\n".join(lines)
    
    def reset_tool_health(self, tool_name: Optional[str] = None):
        """Reset health status for a tool or all tools"""
        if tool_name:
            if tool_name in self.tool_health:
                self.tool_health[tool_name] = ToolHealth(name=tool_name)
        else:
            self.tool_health.clear()
    
    def add_custom_fallback_chain(self, domain: str, tools: List[str]):
        """Add a custom fallback chain for a domain"""
        self.fallback_chains[domain] = tools
    
    def suggest_best_tool(self, domain: str, 
                          available_tools: List[str]) -> Tuple[str, float]:
        """
        Suggest the best tool for a domain based on health
        
        Returns (tool_name, expected_quality)
        """
        chain = self.get_fallback_chain(domain)
        
        for i, tool_name in enumerate(chain):
            if tool_name not in available_tools:
                continue
            
            if not self.is_tool_available(tool_name):
                continue
            
            # Calculate expected quality
            base_quality = LEVEL_QUALITY_SCORES.get(
                FallbackLevel.PRIMARY if i == 0 else FallbackLevel.SECONDARY,
                0.8
            )
            
            # Adjust for health
            if tool_name in self.tool_health:
                health = self.tool_health[tool_name]
                health_adjustment = 1.0 - (health.failure_rate * 0.5)
                base_quality *= health_adjustment
            
            return tool_name, base_quality
        
        return "web_search", 0.5


# Convenience function for synchronous usage
def get_fallback_chain_for_domain(domain: str) -> List[str]:
    """Get fallback chain for a domain"""
    chain = ToolFallbackChain()
    return chain.get_fallback_chain(domain)




