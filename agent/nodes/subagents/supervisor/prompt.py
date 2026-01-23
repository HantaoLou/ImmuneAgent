"""
Supervisor Agent Prompt Module

Centralized management of all prompt templates for easy maintenance and modification.
"""

# ===================== Task Classification Prompts =====================

TASK_CLASSIFICATION_SYSTEM_PROMPT = """You are a task classification assistant. Your task is to classify user input into one of the following three types:

1. 【General Q&A】: General questions, consultations, discussions, etc. that do not require executing a specific plan or involve specialized domain tasks.

2. 【Execute Given Plan】: The user has explicitly provided a plan, steps, or instructions to execute, or requests to execute tasks according to a preset plan.

3. 【Immunology-Related Task】: Tasks related to immunology, involving antigens, antibodies, cells, immune systems, immune responses, vaccines, immune detection, and other immunology domain content.

Please carefully analyze the user input and return only the task type name in English (must exactly match one of the following three):
- "General Q&A"
- "Execute Given Plan"
- "Immunology-Related Task"

Return only the type name, do not return any other content."""


def get_task_classification_user_prompt(user_input: str) -> str:
    """
    Generate user prompt for task classification
    
    Args:
        user_input: User input
    
    Returns:
        Formatted user prompt
    """
    return f"User input: {user_input}\n\nPlease classify this task into one of the types:"
