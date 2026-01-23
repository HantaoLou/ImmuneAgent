"""
Subgraph Sandbox Executor

Provides isolated execution environment for subgraphs to prevent errors from affecting the main graph.
Supports multiple isolation strategies: process isolation, thread isolation with error handling, and lightweight exception catching.
"""

import os
import sys
import json
import pickle
import traceback
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Union
from enum import Enum
from multiprocessing import Process, Queue, TimeoutError as MPTimeoutError
from threading import Thread
import time
from datetime import datetime

from agent.state import GlobalState


class IsolationStrategy(str, Enum):
    """Isolation strategy for subgraph execution"""
    PROCESS = "process"  # Full process isolation (most secure, highest overhead)
    THREAD = "thread"  # Thread isolation with error handling (balanced)
    EXCEPTION = "exception"  # Lightweight exception catching (fastest, least isolation)


class SubgraphExecutionResult:
    """Result of subgraph execution in sandbox"""
    
    def __init__(
        self,
        success: bool,
        output_state: Optional[Any] = None,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
        error_traceback: Optional[str] = None,
        execution_time: float = 0.0,
        sandbox_dir: Optional[str] = None,
        isolation_strategy: Optional[str] = None
    ):
        self.success = success
        self.output_state = output_state
        self.error = error
        self.error_type = error_type
        self.error_traceback = error_traceback
        self.execution_time = execution_time
        self.sandbox_dir = sandbox_dir
        self.isolation_strategy = isolation_strategy


class SubgraphSandboxExecutor:
    """
    Executor for running subgraphs in isolated sandbox environments.
    
    Features:
    - Multiple isolation strategies (process, thread, exception)
    - Automatic sandbox directory creation and cleanup
    - State serialization and deserialization
    - Error capture and reporting
    - Timeout support
    - Resource cleanup
    """
    
    def __init__(
        self,
        strategy: IsolationStrategy = IsolationStrategy.THREAD,
        timeout: Optional[float] = None,
        auto_cleanup: bool = True,
        base_sandbox_dir: Optional[str] = None
    ):
        """
        Initialize sandbox executor
        
        Args:
            strategy: Isolation strategy to use
            timeout: Maximum execution time in seconds (None for no timeout)
            auto_cleanup: Whether to automatically cleanup sandbox directories
            base_sandbox_dir: Base directory for sandbox creation (None for temp dir)
        """
        self.strategy = strategy
        self.timeout = timeout
        self.auto_cleanup = auto_cleanup
        self.base_sandbox_dir = Path(base_sandbox_dir) if base_sandbox_dir else None
        self._active_sandboxes: Dict[str, Path] = {}
    
    def execute_subgraph(
        self,
        subgraph_name: str,
        subgraph_builder: Callable,
        input_mapper: Callable[[GlobalState], Any],
        output_mapper: Callable[[Any, GlobalState], GlobalState],
        main_state: GlobalState,
        **kwargs
    ) -> SubgraphExecutionResult:
        """
        Execute a subgraph in isolated sandbox
        
        Args:
            subgraph_name: Name of the subgraph (for logging and sandbox naming)
            subgraph_builder: Function that builds the subgraph
            input_mapper: Function to map main state to subgraph input
            output_mapper: Function to map subgraph output back to main state
            main_state: Main graph state
            **kwargs: Additional arguments for subgraph execution
        
        Returns:
            SubgraphExecutionResult with execution results
        """
        start_time = time.time()
        sandbox_id = f"{subgraph_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Create sandbox directory
        sandbox_dir = self._create_sandbox(sandbox_id)
        
        try:
            # Prepare input state
            subgraph_input = input_mapper(main_state)
            
            # Execute based on strategy
            if self.strategy == IsolationStrategy.PROCESS:
                result = self._execute_in_process(
                    subgraph_name, subgraph_builder, subgraph_input, sandbox_dir, **kwargs
                )
            elif self.strategy == IsolationStrategy.THREAD:
                result = self._execute_in_thread(
                    subgraph_name, subgraph_builder, subgraph_input, sandbox_dir, **kwargs
                )
            else:  # EXCEPTION
                result = self._execute_with_exception_handling(
                    subgraph_name, subgraph_builder, subgraph_input, sandbox_dir, **kwargs
                )
            
            # Map output back to main state
            if result.success and result.output_state:
                try:
                    output_state = output_mapper(result.output_state, main_state)
                    result.output_state = output_state
                except Exception as e:
                    result.success = False
                    result.error = f"Output mapping failed: {str(e)}"
                    result.error_type = type(e).__name__
                    result.error_traceback = traceback.format_exc()
            
            result.execution_time = time.time() - start_time
            result.sandbox_dir = str(sandbox_dir)
            result.isolation_strategy = self.strategy.value
            
            return result
            
        except Exception as e:
            # Catch any errors in the execution wrapper
            return SubgraphExecutionResult(
                success=False,
                error=f"Sandbox execution wrapper failed: {str(e)}",
                error_type=type(e).__name__,
                error_traceback=traceback.format_exc(),
                execution_time=time.time() - start_time,
                sandbox_dir=str(sandbox_dir),
                isolation_strategy=self.strategy.value
            )
        finally:
            # Cleanup if enabled
            if self.auto_cleanup:
                self._cleanup_sandbox(sandbox_id)
    
    def _create_sandbox(self, sandbox_id: str) -> Path:
        """Create a sandbox directory for subgraph execution"""
        if self.base_sandbox_dir:
            sandbox_dir = self.base_sandbox_dir / sandbox_id
        else:
            sandbox_dir = Path(tempfile.mkdtemp(prefix=f"subgraph_{sandbox_id}_"))
        
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        self._active_sandboxes[sandbox_id] = sandbox_dir
        
        # Create subdirectories
        (sandbox_dir / "logs").mkdir(exist_ok=True)
        (sandbox_dir / "outputs").mkdir(exist_ok=True)
        (sandbox_dir / "temp").mkdir(exist_ok=True)
        
        return sandbox_dir
    
    def _cleanup_sandbox(self, sandbox_id: str):
        """Cleanup sandbox directory"""
        if sandbox_id in self._active_sandboxes:
            sandbox_dir = self._active_sandboxes[sandbox_id]
            try:
                if sandbox_dir.exists():
                    shutil.rmtree(sandbox_dir)
                del self._active_sandboxes[sandbox_id]
            except Exception as e:
                print(f"⚠ Warning: Failed to cleanup sandbox {sandbox_id}: {e}")
    
    def _execute_in_process(
        self,
        subgraph_name: str,
        subgraph_builder: Callable,
        subgraph_input: Any,
        sandbox_dir: Path,
        **kwargs
    ) -> SubgraphExecutionResult:
        """
        Execute subgraph in separate process (full isolation)
        
        Pros:
        - Complete isolation (memory, file system, imports)
        - Most secure
        - Can handle crashes
        
        Cons:
        - Highest overhead (process creation, serialization)
        - State must be serializable
        - Slower startup
        """
        def _worker_process(
            subgraph_builder_func: Callable,
            input_data: Dict[str, Any],
            sandbox_path: str,
            result_queue: Queue,
            **exec_kwargs
        ):
            """Worker process function"""
            try:
                # Change to sandbox directory
                os.chdir(sandbox_path)
                
                # Build and execute subgraph
                subgraph = subgraph_builder_func()
                output = subgraph.invoke(input_data, **exec_kwargs)
                
                # Serialize output
                result_queue.put({
                    "success": True,
                    "output": output
                })
            except Exception as e:
                result_queue.put({
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "error_traceback": traceback.format_exc()
                })
        
        # Serialize input
        try:
            input_dict = self._serialize_state(subgraph_input)
        except Exception as e:
            return SubgraphExecutionResult(
                success=False,
                error=f"State serialization failed: {str(e)}",
                error_type=type(e).__name__
            )
        
        # Create result queue
        result_queue = Queue()
        
        # Start process
        process = Process(
            target=_worker_process,
            args=(subgraph_builder, input_dict, str(sandbox_dir)),
            kwargs={**kwargs, "result_queue": result_queue}
        )
        process.start()
        
        # Wait for completion with timeout
        try:
            if self.timeout:
                process.join(timeout=self.timeout)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                    return SubgraphExecutionResult(
                        success=False,
                        error=f"Subgraph execution timeout ({self.timeout}s)",
                        error_type="TimeoutError"
                    )
            else:
                process.join()
            
            # Get result
            if not result_queue.empty():
                result_data = result_queue.get()
                if result_data["success"]:
                    return SubgraphExecutionResult(
                        success=True,
                        output_state=result_data["output"]
                    )
                else:
                    return SubgraphExecutionResult(
                        success=False,
                        error=result_data.get("error"),
                        error_type=result_data.get("error_type"),
                        error_traceback=result_data.get("error_traceback")
                    )
            else:
                return SubgraphExecutionResult(
                    success=False,
                    error="No result from subgraph process",
                    error_type="ProcessError"
                )
        except Exception as e:
            return SubgraphExecutionResult(
                success=False,
                error=f"Process execution failed: {str(e)}",
                error_type=type(e).__name__,
                error_traceback=traceback.format_exc()
            )
    
    def _execute_in_thread(
        self,
        subgraph_name: str,
        subgraph_builder: Callable,
        subgraph_input: Any,
        sandbox_dir: Path,
        **kwargs
    ) -> SubgraphExecutionResult:
        """
        Execute subgraph in separate thread with error handling
        
        Pros:
        - Lower overhead than process
        - Good error isolation
        - Faster startup
        
        Cons:
        - Shared memory space (less isolation)
        - Import errors can still affect main process
        """
        result_container = {"result": None, "exception": None}
        
        def _worker_thread():
            """Worker thread function"""
            original_cwd = os.getcwd()
            try:
                # Change to sandbox directory
                os.chdir(str(sandbox_dir))
                
                # Build and execute subgraph
                subgraph = subgraph_builder()
                output = subgraph.invoke(subgraph_input, **kwargs)
                
                result_container["result"] = output
            except Exception as e:
                result_container["exception"] = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "error_traceback": traceback.format_exc()
                }
            finally:
                # Restore original directory
                os.chdir(original_cwd)
        
        # Start thread
        thread = Thread(target=_worker_thread, daemon=True)
        thread.start()
        
        # Wait for completion with timeout
        if self.timeout:
            thread.join(timeout=self.timeout)
            if thread.is_alive():
                return SubgraphExecutionResult(
                    success=False,
                    error=f"Subgraph execution timeout ({self.timeout}s)",
                    error_type="TimeoutError"
                )
        else:
            thread.join()
        
        # Check result
        if result_container["exception"]:
            exc = result_container["exception"]
            return SubgraphExecutionResult(
                success=False,
                error=exc["error"],
                error_type=exc["error_type"],
                error_traceback=exc["error_traceback"]
            )
        elif result_container["result"] is not None:
            return SubgraphExecutionResult(
                success=True,
                output_state=result_container["result"]
            )
        else:
            return SubgraphExecutionResult(
                success=False,
                error="No result from subgraph thread",
                error_type="ThreadError"
            )
    
    def _execute_with_exception_handling(
        self,
        subgraph_name: str,
        subgraph_builder: Callable,
        subgraph_input: Any,
        sandbox_dir: Path,
        **kwargs
    ) -> SubgraphExecutionResult:
        """
        Execute subgraph with lightweight exception handling
        
        Pros:
        - Fastest execution
        - Minimal overhead
        - Simple implementation
        
        Cons:
        - Least isolation (shared everything)
        - Import errors can affect main process
        - Memory leaks can persist
        """
        original_cwd = os.getcwd()
        start_time = time.time()
        
        try:
            # Change to sandbox directory
            os.chdir(str(sandbox_dir))
            
            # Build and execute subgraph
            subgraph = subgraph_builder()
            output = subgraph.invoke(subgraph_input, **kwargs)
            
            return SubgraphExecutionResult(
                success=True,
                output_state=output,
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return SubgraphExecutionResult(
                success=False,
                error=str(e),
                error_type=type(e).__name__,
                error_traceback=traceback.format_exc(),
                execution_time=time.time() - start_time
            )
        finally:
            # Restore original directory
            os.chdir(original_cwd)
    
    def _serialize_state(self, state: Any) -> Dict[str, Any]:
        """Serialize state for process communication"""
        if isinstance(state, dict):
            return state
        elif hasattr(state, "model_dump"):
            return state.model_dump()
        elif hasattr(state, "dict"):
            return state.dict()
        else:
            # Try JSON serialization
            try:
                return json.loads(json.dumps(state, default=str))
            except:
                # Fallback to pickle (for complex objects)
                return pickle.dumps(state)


def execute_subgraph_in_sandbox(
    subgraph_name: str,
    subgraph_builder: Callable,
    input_mapper: Callable[[GlobalState], Any],
    output_mapper: Callable[[Any, GlobalState], GlobalState],
    main_state: GlobalState,
    strategy: IsolationStrategy = IsolationStrategy.THREAD,
    timeout: Optional[float] = None,
    **kwargs
) -> GlobalState:
    """
    Convenience function to execute subgraph in sandbox and return updated main state
    
    Args:
        subgraph_name: Name of the subgraph
        subgraph_builder: Function that builds the subgraph
        input_mapper: Function to map main state to subgraph input
        output_mapper: Function to map subgraph output back to main state
        main_state: Main graph state
        strategy: Isolation strategy
        timeout: Maximum execution time
        **kwargs: Additional arguments for subgraph execution
    
    Returns:
        Updated main state (or original state if execution failed)
    """
    executor = SubgraphSandboxExecutor(strategy=strategy, timeout=timeout)
    result = executor.execute_subgraph(
        subgraph_name=subgraph_name,
        subgraph_builder=subgraph_builder,
        input_mapper=input_mapper,
        output_mapper=output_mapper,
        main_state=main_state,
        **kwargs
    )
    
    if result.success:
        return result.output_state
    else:
        # Log error but don't crash main graph
        print(f"⚠ Subgraph {subgraph_name} execution failed:")
        print(f"   Error: {result.error}")
        print(f"   Type: {result.error_type}")
        if result.error_traceback:
            print(f"   Traceback:\n{result.error_traceback}")
        
        # Return original state with error information in merged_result
        if not main_state.merged_result:
            main_state.merged_result = {}
        main_state.merged_result[f"{subgraph_name}_error"] = {
            "error": result.error,
            "error_type": result.error_type,
            "error_traceback": result.error_traceback,
            "execution_time": result.execution_time
        }
        
        return main_state

