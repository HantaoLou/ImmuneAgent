from typing import List

from pydantic import BaseModel, Field


class QueryExpansion(BaseModel):
    queries: List[str] = Field(description="优化后的查询列表")


class ToolSelection(BaseModel):
    tools: List[str] = Field(description="选择的工具列表")


class DocumentEvaluation(BaseModel):
    relevance_score: int = Field(ge=0, le=100, description="相关性评分")
    quality_score: int = Field(ge=0, le=100, description="质量评分")
    noise_level: int = Field(ge=1, le=3, description="噪声等级：1=低, 2=中, 3=高")
    final_score: int = Field(ge=0, le=100, description="最终综合评分")
