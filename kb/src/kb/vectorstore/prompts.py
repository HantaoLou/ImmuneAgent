PARENT_RETRIEVER_SUMMARIZE_PROMPT = """
You are {role}. Given a QUERY and a relevant paper, your task is to summarize the paper with respect to the QUERY.

Your summary should:
- Clearly address the QUERY, focusing only on information relevant to it.
- Include the problem addressed, high-level design, methodologies, and conclusions from the paper.
- Methodologies must be emphasized, such as the data source, software and algorighms used, and how are experiments designed.
- Be strictly grounded in the provided PARENT paper; do not fabricate or infer information not present in the text.

The summary must not exceed {chunk_size} tokens.

QUERY:
{query}

PARENT:
{parent}

SUMMARY:
"""
