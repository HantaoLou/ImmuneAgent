"""Paper-QA configuration for HLE benchmark research."""

from paperqa import Settings
from paperqa.settings import AnswerSettings, ParsingSettings


def create_paperqa_settings(
    evidence_k: int = 15,
    answer_max_sources: int = 5,
    chunk_chars: int = 5000,
    overlap: int = 250,
) -> Settings:
    """Create paper-qa Settings for biomedical research.

    Uses Claude via LiteLLM for LLM calls, OpenAI for embeddings.
    Per CLAUDE.md, all sub-agents should use Opus 4.5.
    For paper-qa's summarization (many parallel calls), Sonnet is cost-effective.
    """
    return Settings(
        llm="anthropic/claude-opus-4-5-20250514",
        llm_config={
            "model_list": [
                {
                    "model_name": "anthropic/claude-opus-4-5-20250514",
                    "litellm_params": {"model": "anthropic/claude-opus-4-5-20250514"},
                }
            ]
        },
        summary_llm="anthropic/claude-sonnet-4-20250514",
        summary_llm_config={
            "model_list": [
                {
                    "model_name": "anthropic/claude-sonnet-4-20250514",
                    "litellm_params": {"model": "anthropic/claude-sonnet-4-20250514"},
                }
            ]
        },
        embedding="text-embedding-3-small",
        temperature=0.0,
        answer=AnswerSettings(
            evidence_k=evidence_k,
            answer_max_sources=answer_max_sources,
            evidence_summary_length="about 100 words",
            evidence_relevance_score_cutoff=3,
            answer_length="about 200 words, but can be longer for complex questions",
        ),
        parsing=ParsingSettings(
            chunk_chars=chunk_chars,
            overlap=overlap,
            use_doc_details=True,
        ),
    )
