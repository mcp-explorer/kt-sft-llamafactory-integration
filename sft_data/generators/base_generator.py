"""
Base classes for synthetic data generation.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json


@dataclass
class GenerationGoal:
    """Represents a goal for data generation."""
    description: str
    target_format: str  # "instruction", "conversation", "qa", etc.
    constraints: Optional[Dict[str, Any]] = None
    examples: Optional[List[Dict[str, Any]]] = None


@dataclass
class GeneratedSample:
    """Represents a single generated sample."""
    instruction: str
    input: Optional[str]
    output: str
    metadata: Optional[Dict[str, Any]] = None
    score: Optional[float] = None  # Quality score from judge
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "instruction": self.instruction,
            "output": self.output
        }
        if self.input:
            result["input"] = self.input
        if self.metadata:
            result["metadata"] = self.metadata
        if self.score is not None:
            result["score"] = self.score
        return result


class BaseGenerator(ABC):
    """Base class for all data generators."""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or "gpt-4"
        self.generated_samples: List[GeneratedSample] = []
    
    @abstractmethod
    def generate(self, goal: GenerationGoal, num_records: int) -> List[GeneratedSample]:
        """Generate synthetic data samples."""
        pass
    
    def save(self, output_path: str, format: str = "jsonl"):
        """Save generated samples to file."""
        if format == "jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for sample in self.generated_samples:
                    f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
        elif format == "json":
            data = [sample.to_dict() for sample in self.generated_samples]
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

