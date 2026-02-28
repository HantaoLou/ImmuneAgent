"""
LLM Retry Wrapper and Recovery Mode

Implements robust LLM calling with:
- Exponential backoff retry
- Graceful degradation
- Recovery mode for failures
- Timeout handling

Key Features:
- LLMRetryWrapper: Wrapper with retry logic
- RecoveryMode: Fallback strategies when LLM fails
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Callable, TypeVar, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class FailureType(Enum):
    """Types of LLM failures"""
    TIMEOUT = "timeout"
    API_ERROR = "api_error"
    RATE_LIMIT = "rate_limit"
    INVALID_RESPONSE = "invalid_response"
    CONTEXT_TOO_LONG = "context_too_long"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Strategies for recovery from failures"""
    RETRY = "retry"                    # Simply retry the call
    SIMPLIFY_PROMPT = "simplify_prompt"  # Use shorter prompt
    FALLBACK_MODEL = "fallback_model"  # Use backup model
    PARTIAL_RESULT = "partial_result"  # Return partial results
    GRACEFUL_DEGRADATION = "graceful_degradation"  # Return best effort
    ABORT = "abort"                    # Give up and return error


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True
    
    # Per-failure-type configuration
    timeout_retries: int = 3
    api_error_retries: int = 2
    rate_limit_retries: int = 5
    
    # Fallback configuration
    enable_fallback_model: bool = True
    enable_prompt_simplification: bool = True


@dataclass
class RetryResult:
    """Result of a retry-wrapped LLM call"""
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    failure_type: Optional[FailureType] = None
    attempts: int = 0
    total_delay: float = 0.0
    recovery_strategy_used: Optional[RecoveryStrategy] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RecoveryMode:
    """
    Provides fallback strategies when LLM calls fail.
    
    Strategies:
    1. Simplify prompt - reduce complexity
    2. Use fallback model - switch to backup
    3. Return partial - return what we have
    4. Graceful degradation - best effort response
    """
    
    def __init__(
        self,
        fallback_llm: Optional[Any] = None,
        simplifier: Optional[Callable[[str], str]] = None
    ):
        self.fallback_llm = fallback_llm
        self.simplifier = simplifier or self._default_simplifier
        self._partial_results: Dict[str, Any] = {}
    
    def _default_simplifier(self, prompt: str) -> str:
        """Default prompt simplification"""
        # Truncate to half length
        if len(prompt) > 1000:
            return prompt[:500] + "\n...[truncated]...\n" + prompt[-500:]
        return prompt
    
    async def execute_recovery(
        self,
        original_prompt: str,
        failure_type: FailureType,
        strategy: RecoveryStrategy,
        original_llm: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> RetryResult:
        """Execute a recovery strategy"""
        context = context or {}
        
        if strategy == RecoveryStrategy.SIMPLIFY_PROMPT:
            return await self._simplify_and_retry(
                original_prompt, original_llm, context
            )
        
        elif strategy == RecoveryStrategy.FALLBACK_MODEL:
            return await self._use_fallback_model(
                original_prompt, context
            )
        
        elif strategy == RecoveryStrategy.PARTIAL_RESULT:
            return self._get_partial_result(original_prompt, context)
        
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return self._graceful_degradation(original_prompt, context)
        
        else:
            return RetryResult(
                success=False,
                error="No recovery strategy available",
                failure_type=failure_type,
                recovery_strategy_used=strategy
            )
    
    async def _simplify_and_retry(
        self,
        prompt: str,
        llm: Any,
        context: Dict[str, Any]
    ) -> RetryResult:
        """Simplify prompt and retry"""
        simplified = self.simplifier(prompt)
        
        try:
            if hasattr(llm, 'ainvoke'):
                result = await llm.ainvoke(simplified)
            elif hasattr(llm, 'invoke'):
                result = llm.invoke(simplified)
            else:
                result = llm(simplified)
            
            return RetryResult(
                success=True,
                result=result,
                recovery_strategy_used=RecoveryStrategy.SIMPLIFY_PROMPT,
                metadata={"original_length": len(prompt), "simplified_length": len(simplified)}
            )
        except Exception as e:
            return RetryResult(
                success=False,
                error=str(e),
                recovery_strategy_used=RecoveryStrategy.SIMPLIFY_PROMPT
            )
    
    async def _use_fallback_model(
        self,
        prompt: str,
        context: Dict[str, Any]
    ) -> RetryResult:
        """Use fallback model"""
        if not self.fallback_llm:
            return RetryResult(
                success=False,
                error="No fallback model configured",
                recovery_strategy_used=RecoveryStrategy.FALLBACK_MODEL
            )
        
        try:
            if hasattr(self.fallback_llm, 'ainvoke'):
                result = await self.fallback_llm.ainvoke(prompt)
            elif hasattr(self.fallback_llm, 'invoke'):
                result = self.fallback_llm.invoke(prompt)
            else:
                result = self.fallback_llm(prompt)
            
            return RetryResult(
                success=True,
                result=result,
                recovery_strategy_used=RecoveryStrategy.FALLBACK_MODEL
            )
        except Exception as e:
            return RetryResult(
                success=False,
                error=str(e),
                recovery_strategy_used=RecoveryStrategy.FALLBACK_MODEL
            )
    
    def _get_partial_result(
        self,
        prompt: str,
        context: Dict[str, Any]
    ) -> RetryResult:
        """Return partial result if available"""
        # Check for cached partial results
        prompt_hash = str(hash(prompt))
        
        if prompt_hash in self._partial_results:
            return RetryResult(
                success=True,
                result=self._partial_results[prompt_hash],
                recovery_strategy_used=RecoveryStrategy.PARTIAL_RESULT,
                metadata={"source": "cache"}
            )
        
        # Generate a best-effort response
        partial = self._generate_partial_response(prompt, context)
        
        return RetryResult(
            success=partial is not None,
            result=partial,
            recovery_strategy_used=RecoveryStrategy.PARTIAL_RESULT,
            metadata={"source": "generated"}
        )
    
    def _graceful_degradation(
        self,
        prompt: str,
        context: Dict[str, Any]
    ) -> RetryResult:
        """Generate graceful degradation response"""
        # This should be customized based on application needs
        degraded_response = {
            "status": "degraded",
            "message": "Unable to complete full analysis, providing best-effort response",
            "partial_answer": context.get("intermediate_results"),
            "confidence": 0.3
        }
        
        return RetryResult(
            success=True,
            result=degraded_response,
            recovery_strategy_used=RecoveryStrategy.GRACEFUL_DEGRADATION,
            metadata={"degraded": True}
        )
    
    def _generate_partial_response(
        self,
        prompt: str,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate a partial response from context"""
        if not context:
            return None
        
        # Use any available intermediate results
        partial = {
            "status": "partial",
            "available_context": list(context.keys()),
            "confidence": 0.3
        }
        
        # Add any intermediate conclusions
        if "intermediate_conclusions" in context:
            partial["intermediate_conclusions"] = context["intermediate_conclusions"]
        
        return partial
    
    def store_partial_result(self, prompt: str, result: Any):
        """Store a partial result for potential later use"""
        prompt_hash = str(hash(prompt))
        self._partial_results[prompt_hash] = result


class LLMRetryWrapper:
    """
    Wraps LLM calls with retry logic and recovery.
    
    Features:
    - Exponential backoff
    - Jitter for distributed systems
    - Per-error-type retry counts
    - Recovery mode integration
    """
    
    def __init__(
        self,
        llm: Any,
        config: Optional[RetryConfig] = None,
        recovery_mode: Optional[RecoveryMode] = None
    ):
        """
        Initialize the retry wrapper.
        
        Args:
            llm: The LLM to wrap
            config: Retry configuration
            recovery_mode: Recovery mode handler
        """
        self.llm = llm
        self.config = config or RetryConfig()
        self.recovery_mode = recovery_mode or RecoveryMode(fallback_llm=None)
        
        self._call_history: List[Dict[str, Any]] = []
    
    async def call(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> RetryResult:
        """
        Call the LLM with retry logic.
        
        Args:
            prompt: The prompt to send
            context: Additional context for recovery
            timeout: Optional timeout for the call
            
        Returns:
            RetryResult with success status and result/error
        """
        context = context or {}
        attempts = 0
        total_delay = 0.0
        last_error = None
        last_failure_type = None
        
        max_attempts = self.config.max_retries + 1
        
        while attempts < max_attempts:
            attempts += 1
            
            try:
                # Attempt the call
                result = await self._make_call(prompt, timeout)
                
                # Success!
                self._record_call(prompt, True, attempts, total_delay)
                
                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempts,
                    total_delay=total_delay
                )
                
            except asyncio.TimeoutError as e:
                last_error = str(e)
                last_failure_type = FailureType.TIMEOUT
                logger.warning(f"LLM call timed out (attempt {attempts})")
                
            except Exception as e:
                last_error = str(e)
                last_failure_type = self._classify_error(e)
                logger.warning(f"LLM call failed (attempt {attempts}): {e}")
            
            # Check if we should retry
            if attempts < max_attempts:
                # Calculate delay with exponential backoff
                delay = self._calculate_delay(attempts)
                total_delay += delay
                
                logger.info(f"Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)
        
        # All retries failed - try recovery
        self._record_call(prompt, False, attempts, total_delay, last_error)
        
        # Determine recovery strategy
        recovery_strategy = self._determine_recovery_strategy(last_failure_type)
        
        if recovery_strategy != RecoveryStrategy.ABORT:
            recovery_result = await self.recovery_mode.execute_recovery(
                prompt,
                last_failure_type,
                recovery_strategy,
                self.llm,
                context
            )
            
            recovery_result.attempts = attempts
            recovery_result.total_delay = total_delay
            return recovery_result
        
        return RetryResult(
            success=False,
            error=last_error,
            failure_type=last_failure_type,
            attempts=attempts,
            total_delay=total_delay
        )
    
    async def _make_call(
        self, 
        prompt: str, 
        timeout: Optional[float] = None
    ) -> Any:
        """Make the actual LLM call"""
        if hasattr(self.llm, 'ainvoke'):
            if timeout:
                result = await asyncio.wait_for(
                    self.llm.ainvoke(prompt),
                    timeout=timeout
                )
            else:
                result = await self.llm.ainvoke(prompt)
        elif hasattr(self.llm, 'invoke'):
            # Sync call in async context
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.llm.invoke, prompt
            )
        else:
            # Assume callable
            result = self.llm(prompt)
        
        return result
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and optional jitter"""
        delay = self.config.base_delay * (
            self.config.exponential_base ** (attempt - 1)
        )
        
        # Cap at max delay
        delay = min(delay, self.config.max_delay)
        
        # Add jitter if enabled
        if self.config.jitter:
            import random
            delay = delay * (0.5 + random.random())
        
        return delay
    
    def _classify_error(self, error: Exception) -> FailureType:
        """Classify the type of error"""
        error_str = str(error).lower()
        
        if "timeout" in error_str:
            return FailureType.TIMEOUT
        elif "rate limit" in error_str or "429" in error_str:
            return FailureType.RATE_LIMIT
        elif "context" in error_str and "long" in error_str:
            return FailureType.CONTEXT_TOO_LONG
        elif "api" in error_str or "500" in error_str or "503" in error_str:
            return FailureType.API_ERROR
        elif "invalid" in error_str or "parse" in error_str:
            return FailureType.INVALID_RESPONSE
        
        return FailureType.UNKNOWN
    
    def _determine_recovery_strategy(
        self, 
        failure_type: FailureType
    ) -> RecoveryStrategy:
        """Determine appropriate recovery strategy"""
        strategy_map = {
            FailureType.TIMEOUT: RecoveryStrategy.SIMPLIFY_PROMPT,
            FailureType.CONTEXT_TOO_LONG: RecoveryStrategy.SIMPLIFY_PROMPT,
            FailureType.RATE_LIMIT: RecoveryStrategy.RETRY,  # Will wait longer
            FailureType.API_ERROR: RecoveryStrategy.FALLBACK_MODEL,
            FailureType.INVALID_RESPONSE: RecoveryStrategy.PARTIAL_RESULT,
            FailureType.UNKNOWN: RecoveryStrategy.GRACEFUL_DEGRADATION,
        }
        
        return strategy_map.get(failure_type, RecoveryStrategy.GRACEFUL_DEGRADATION)
    
    def _record_call(
        self,
        prompt: str,
        success: bool,
        attempts: int,
        total_delay: float,
        error: Optional[str] = None
    ):
        """Record call in history"""
        self._call_history.append({
            "timestamp": datetime.now().isoformat(),
            "prompt_hash": hash(prompt),
            "success": success,
            "attempts": attempts,
            "total_delay": total_delay,
            "error": error
        })
        
        # Keep only last 100 calls
        if len(self._call_history) > 100:
            self._call_history = self._call_history[-100:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about LLM calls"""
        if not self._call_history:
            return {"total_calls": 0}
        
        successes = sum(1 for c in self._call_history if c["success"])
        failures = len(self._call_history) - successes
        
        return {
            "total_calls": len(self._call_history),
            "successful_calls": successes,
            "failed_calls": failures,
            "success_rate": successes / len(self._call_history),
            "average_attempts": sum(c["attempts"] for c in self._call_history) / len(self._call_history),
            "total_delay": sum(c["total_delay"] for c in self._call_history)
        }


def create_retry_wrapper(
    llm: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    fallback_llm: Optional[Any] = None
) -> LLMRetryWrapper:
    """
    Factory function to create an LLM retry wrapper.
    
    Args:
        llm: The primary LLM
        max_retries: Maximum number of retries
        base_delay: Base delay for exponential backoff
        fallback_llm: Optional fallback LLM
        
    Returns:
        Configured LLMRetryWrapper
    """
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay
    )
    
    recovery = RecoveryMode(fallback_llm=fallback_llm)
    
    return LLMRetryWrapper(llm=llm, config=config, recovery_mode=recovery)

