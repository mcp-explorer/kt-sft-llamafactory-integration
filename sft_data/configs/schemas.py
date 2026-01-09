"""
Predefined schemas for common SFT tasks.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class IdentityTaskSchema(BaseModel):
    """Schema for identity training tasks."""
    instruction: str = Field(description="User question about identity")
    output: str = Field(description="Response stating identity")
    
    class Config:
        json_schema_extra = {
            "example": {
                "instruction": "Who are you?",
                "output": "My name is Sean."
            }
        }


class QATaskSchema(BaseModel):
    """Schema for Q&A tasks."""
    instruction: str = Field(description="Question")
    input: Optional[str] = Field(default=None, description="Additional context")
    reasoning: Optional[str] = Field(default=None, description="Step-by-step reasoning")
    output: str = Field(description="Answer")
    
    class Config:
        json_schema_extra = {
            "example": {
                "instruction": "What is 2+2?",
                "reasoning": "Adding 2 and 2 gives 4",
                "output": "4"
            }
        }


class ConversationTaskSchema(BaseModel):
    """Schema for multi-turn conversation."""
    instruction: str = Field(description="User message")
    output: str = Field(description="Assistant response")
    context: Optional[List[str]] = Field(default=None, description="Previous conversation turns")
    
    class Config:
        json_schema_extra = {
            "example": {
                "instruction": "Hello!",
                "output": "Hello! How can I help you today?",
                "context": []
            }
        }


class ReasoningTaskSchema(BaseModel):
    """Schema for reasoning tasks."""
    instruction: str = Field(description="Problem or question")
    reasoning: str = Field(description="Step-by-step reasoning process")
    output: str = Field(description="Final answer")
    
    class Config:
        json_schema_extra = {
            "example": {
                "instruction": "Solve: If x + 5 = 10, what is x?",
                "reasoning": "To solve for x, subtract 5 from both sides: x + 5 - 5 = 10 - 5, so x = 5",
                "output": "x = 5"
            }
        }


# Schema registry
SCHEMA_REGISTRY = {
    "identity": IdentityTaskSchema,
    "qa": QATaskSchema,
    "conversation": ConversationTaskSchema,
    "reasoning": ReasoningTaskSchema,
}

