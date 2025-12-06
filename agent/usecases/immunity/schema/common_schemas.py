from typing import Any, Dict, List, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QueryExpansion(BaseModel):
    queries: List[str] = Field(description="List of optimized queries")


class ToolSelection(BaseModel):
    tools: List[str] = Field(description="List of selected tools")


class DocumentEvaluation(BaseModel):
    relevance_score: int = Field(ge=0, le=100, description="Relevance score")
    quality_score: int = Field(ge=0, le=100, description="Quality score")
    noise_level: int = Field(
        ge=1, le=3, description="Noise level: 1=low, 2=medium, 3=high"
    )
    final_score: int = Field(ge=0, le=100, description="Final comprehensive score")


class Document(BaseModel):
    source: str = Field(description="Document source")
    content: str = Field(description="Document content")


class Citation(BaseModel):
    """Paper citation information data class - Uses BaseModel to support model_dump and other methods"""

    # Configure Pydantic model to handle serialization warnings gracefully
    model_config = ConfigDict(
        # Allow type coercion during serialization to avoid warnings
        ser_json_inf=float("inf"),  # Handle infinity values
        ser_json_nan=float("nan"),  # Handle NaN values
        # Allow type conversion during serialization
        arbitrary_types_allowed=True,
        # Use more lenient mode during serialization
        validate_assignment=False,
    )

    authors: Union[List[str], str] = Field(default="", description="List of authors")
    title: str = Field(default="", description="Article title")
    journal: str = Field(default="", description="Journal name")
    year: Union[int, str] = Field(default=0, description="Publication year")
    volume: str = Field(default="", description="Volume number")
    pages: str = Field(default="", description="Page numbers")
    doi: str = Field(default="", description="DOI identifier")
    pmid: str = Field(default="", description="PubMed ID")
    abstract: str = Field(default="", description="Abstract")
    citation_key: str = Field(default="", description="Unique identifier")

    @field_validator("authors", mode="before")
    @classmethod
    def validate_authors(cls, v):
        """Convert authors field to string list"""
        if isinstance(v, str):
            # If it's a string, try to split by common delimiters
            if "," in v:
                return [author.strip() for author in v.split(",")]
            elif ";" in v:
                return [author.strip() for author in v.split(";")]
            else:
                return [v.strip()]
        elif isinstance(v, list):
            return [str(author) for author in v]
        return []

    @field_validator("year", mode="before")
    @classmethod
    def validate_year(cls, v):
        """Convert year field to integer"""
        if isinstance(v, str):
            try:
                return int(v)
            except (ValueError, TypeError):
                return 0
        elif isinstance(v, int):
            return v
        return 0

class ToolInfo(BaseModel):
    """Tool information model"""
    tool_name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")

class TaskInfo(BaseModel):
    """Task information model"""

    task_id: str = Field(description="Task ID")
    name: str = Field(description="Task name")
    description: str = Field(description="Task description")
    tools: list[ToolInfo] = Field(description="Tools used")
    inputs: list[str] = Field(description="Input data")
    outputs: list[str] = Field(description="Output data")
    parameters: Dict[str, Any] = Field(description="Parameter settings")


class TaskExtractionResult(BaseModel):
    """Task extraction result model"""

    tasks: list[TaskInfo] = Field(description="List of extracted tasks")


class PlanStep(BaseModel):
    """Structured plan step used for UI presentation and plan confirmation"""

    step_id: str = Field(default="", description="Step identifier")
    title: str = Field(default="", description="Step title or name")
    description: str = Field(default="", description="Detailed description")
    objective: str = Field(default="", description="Goal or purpose of the step")
    tools: List[str] = Field(default_factory=list, description="Tools to be invoked")
    toolchain: List[str] = Field(default_factory=list, description="Alternative toolchain representation")
    recommended_tools: List[str] = Field(default_factory=list, description="Recommended tools for this step")
    notes: str = Field(default="", description="Additional notes or comments")
    status: str = Field(default="pending", description="Execution status indicator")
    inputs: List[str] = Field(default_factory=list, description="Expected inputs")
    outputs: List[str] = Field(default_factory=list, description="Expected outputs")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for UI")
    suggested_alternatives: List[str] = Field(default_factory=list, description="Suggested backup tools")
    allow_user_input: bool = Field(default=True, description="Whether user can provide custom result")