"""
Code Execution Module
Extracted from Biomni framework (biomni/tool/support_tools.py and biomni/utils.py)
"""

import base64
import ctypes
import io
import os
import queue
import subprocess
import sys
import tempfile
import threading
import traceback
from io import StringIO


# Create a persistent namespace that will be shared across all executions
_persistent_namespace = {}

# Global list to store captured plots
_captured_plots = []


def run_python_repl(command: str, namespace: dict = None, captured_plots: list = None) -> str:
    """Executes the provided Python command in a persistent environment and returns the output.
    Variables defined in one execution will be available in subsequent executions.

    Args:
        command: Python code to execute
        namespace: Execution namespace dict. If None, uses the global _persistent_namespace.
                   Pass an instance-level dict for thread-safe parallel execution.
        captured_plots: List to store captured plots. If None, uses the global _captured_plots.
    """
    if namespace is None:
        namespace = _persistent_namespace
    if captured_plots is None:
        captured_plots = _captured_plots

    def execute_in_repl(command: str) -> str:
        """Helper function to execute the command in the persistent environment."""
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()

        try:
            # Apply matplotlib monkey patches before execution
            _apply_matplotlib_patches(captured_plots)

            # Execute the command in the given namespace
            exec(command, namespace)
            output = mystdout.getvalue()

        except Exception as e:
            output = f"Error: {str(e)}"
        finally:
            sys.stdout = old_stdout
        return output

    command = command.strip("```").strip()
    return execute_in_repl(command)


def _capture_matplotlib_plots(captured_plots: list = None):
    """Capture any matplotlib plots that might have been generated during execution.

    Args:
        captured_plots: List to append captured plots to. If None, uses global _captured_plots.
    """
    if captured_plots is None:
        captured_plots = _captured_plots
    try:
        import matplotlib.pyplot as plt

        # Check if there are any active figures
        if plt.get_fignums():
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)

                # Save figure to base64
                buffer = io.BytesIO()
                fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
                buffer.seek(0)

                # Convert to base64
                image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
                plot_data = f"data:image/png;base64,{image_data}"

                # Add to captured plots if not already there
                if plot_data not in captured_plots:
                    captured_plots.append(plot_data)

                # Close the figure to free memory
                plt.close(fig)

    except ImportError:
        # matplotlib not available
        pass
    except Exception as e:
        print(f"Warning: Could not capture matplotlib plots: {e}")


def _apply_matplotlib_patches(captured_plots: list = None):
    """Apply simple monkey patches to matplotlib functions to automatically capture plots.

    Args:
        captured_plots: List to store captured plots. If None, uses global _captured_plots.
    """
    try:
        import matplotlib.pyplot as plt

        # Only patch if matplotlib is available and not already patched
        if hasattr(plt, "_codeact_patched"):
            return

        # Store original functions
        original_show = plt.show
        original_savefig = plt.savefig

        def show_with_capture(*args, **kwargs):
            """Enhanced show function that captures plots before displaying them."""
            # Capture any plots before showing
            _capture_matplotlib_plots(captured_plots)
            # Print a message to indicate plot was generated
            print("Plot generated and displayed")
            # Call the original show function
            return original_show(*args, **kwargs)

        def savefig_with_capture(*args, **kwargs):
            """Enhanced savefig function that captures plots after saving them."""
            # Get the filename from args if provided
            filename = args[0] if args else kwargs.get("fname", "unknown")
            # Call the original savefig function
            result = original_savefig(*args, **kwargs)
            # Capture the plot after saving
            _capture_matplotlib_plots(captured_plots)
            # Print a message to indicate plot was saved
            print(f"Plot saved to: {filename}")
            return result

        # Replace functions with enhanced versions
        plt.show = show_with_capture
        plt.savefig = savefig_with_capture

        # Mark as patched to avoid double-patching
        plt._codeact_patched = True

    except ImportError:
        # matplotlib not available
        pass
    except Exception as e:
        print(f"Warning: Could not apply matplotlib patches: {e}")


def get_captured_plots(captured_plots: list = None):
    """Get all captured matplotlib plots.

    Args:
        captured_plots: Instance-level list. If None, uses global _captured_plots.
    """
    if captured_plots is None:
        captured_plots = _captured_plots
    return captured_plots.copy()


def clear_captured_plots(captured_plots: list = None):
    """Clear all captured matplotlib plots.

    Args:
        captured_plots: Instance-level list to clear. If None, clears global _captured_plots.
    """
    if captured_plots is None:
        global _captured_plots
        _captured_plots = []
    else:
        captured_plots.clear()


def run_r_code(code: str) -> str:
    """Run R code using subprocess.

    Args:
        code: R code to run

    Returns:
        Output of the R code

    """
    try:
        # Create a temporary file to store the R code
        with tempfile.NamedTemporaryFile(suffix=".R", mode="w", delete=False) as f:
            f.write(code)
            temp_file = f.name

        # Run the R code using Rscript
        result = subprocess.run(["Rscript", temp_file], capture_output=True, text=True, check=False)

        # Clean up the temporary file
        os.unlink(temp_file)

        # Return the output
        if result.returncode != 0:
            return f"Error running R code:\n{result.stderr}"
        else:
            return result.stdout
    except Exception as e:
        return f"Error running R code: {str(e)}"


def run_bash_script(script: str) -> str:
    """Run a Bash script using subprocess.

    Args:
        script: Bash script to run

    Returns:
        Output of the Bash script

    """
    try:
        # Trim any leading/trailing whitespace
        script = script.strip()

        # If the script is empty, return an error
        if not script:
            return "Error: Empty script"

        # Create a temporary file to store the Bash script
        with tempfile.NamedTemporaryFile(suffix=".sh", mode="w", delete=False) as f:
            # Add shebang if not present
            if not script.startswith("#!/"):
                f.write("#!/bin/bash\n")
            # Add set -e to exit on error
            if "set -e" not in script:
                f.write("set -e\n")
            f.write(script)
            temp_file = f.name

        # Make the script executable
        os.chmod(temp_file, 0o755)

        # Get current environment variables and working directory
        env = os.environ.copy()
        cwd = os.getcwd()

        # Run the Bash script with the current environment and working directory
        result = subprocess.run(
            [temp_file],
            shell=True,
            capture_output=True,
            text=True,
            check=False,
            env=env,
            cwd=cwd,
        )

        # Clean up the temporary file
        os.unlink(temp_file)

        # Return the output
        if result.returncode != 0:
            traceback.print_stack()
            print(result)
            return f"Error running Bash script (exit code {result.returncode}):\n{result.stderr}"
        else:
            return result.stdout
    except Exception as e:
        traceback.print_exc()
        return f"Error running Bash script: {str(e)}"


def run_with_timeout(func, args=None, kwargs=None, timeout=600):
    """Run a function with a timeout using threading instead of multiprocessing.
    This allows variables to persist in the global namespace between function calls.
    Returns the function result or a timeout error message.
    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    result_queue = queue.Queue()

    def thread_func(func, args, kwargs, result_queue):
        """Function to run in a separate thread."""
        try:
            result = func(*args, **kwargs)
            result_queue.put(("success", result))
        except Exception as e:
            result_queue.put(("error", str(e)))

    # Start a separate thread
    thread = threading.Thread(target=thread_func, args=(func, args, kwargs, result_queue))
    thread.daemon = True  # Set as daemon so it will be killed when main thread exits
    thread.start()

    # Wait for the specified timeout
    thread.join(timeout)

    # Check if the thread is still running after timeout
    if thread.is_alive():
        print(f"TIMEOUT: Code execution timed out after {timeout} seconds")

        # Unfortunately, there's no clean way to force terminate a thread in Python
        # The recommended approach is to use daemon threads and let them be killed when main thread exits
        # Here, we'll try to raise an exception in the thread to make it stop
        try:
            # Get thread ID and try to terminate it
            thread_id = thread.ident
            if thread_id:
                # This is a bit dangerous and not 100% reliable
                # It attempts to raise a SystemExit exception in the thread
                res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), ctypes.py_object(SystemExit))
                if res > 1:
                    # Oops, we raised too many exceptions
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), None)
        except Exception as e:
            print(f"Error trying to terminate thread: {e}")

        return f"ERROR: Code execution timed out after {timeout} seconds. Please try with simpler inputs or break your task into smaller steps."

    # Get the result from the queue if available
    try:
        status, result = result_queue.get(block=False)
        return result if status == "success" else f"Error in execution: {result}"
    except queue.Empty:
        return "Error: Execution completed but no result was returned"
