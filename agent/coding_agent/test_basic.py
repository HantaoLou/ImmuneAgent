"""Simple test for coding_agent module."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Testing coding_agent module...")

# Test 1: Config
print("\n1. Testing config...")
from coding_agent.config import OpenCodeConfig, OpenCodeMode
config = OpenCodeConfig()
print(f"   Model: {config.model_provider}")
print(f"   Mode: {config.opencode_mode.value}")
assert config.model_provider == "glm-4.7"
assert config.opencode_mode == OpenCodeMode.BUILD
print("   PASSED")

# Test 2: Tasks MD Generator
print("\n2. Testing tasks_md_generator...")
from coding_agent.tasks_md_generator import create_simple_tasks_md
tasks_md = create_simple_tasks_md([
    {"id": "task_1", "description": "Test task"},
], session_id="test_session")
assert "task_1" in tasks_md
assert "test_session" in tasks_md
print(f"   Generated {len(tasks_md)} chars")
print("   PASSED")

# Test 3: OpenCode Executor
print("\n3. Testing opencode_executor...")
from coding_agent.opencode_executor import OpenCodeExecutor
executor = OpenCodeExecutor()
env = executor._build_sandbox_env()
assert env["OPENCODE_MODEL"] == "glm-4.7"
print(f"   Environment: {env}")
print("   PASSED")

# Test 4: Package import
print("\n4. Testing package import...")
from coding_agent import (
    OpenCodeConfig,
    OpenCodeExecutor,
    generate_tasks_md_content,
    run_coding_agent_in_sandbox,
)
import coding_agent
assert coding_agent.__version__ == "1.0.0"
print(f"   Version: {coding_agent.__version__}")
print("   PASSED")

print("\n" + "="*50)
print("All tests PASSED!")
print("="*50)

