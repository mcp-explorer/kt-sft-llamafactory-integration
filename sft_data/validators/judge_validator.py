"""
LLM-based judge/critic for validating generated samples.
"""

from typing import Dict, Any, Optional
import json
import sys
from pathlib import Path

# Add parent directory to path for imports when running as module
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from generators.base_generator import GeneratedSample, GenerationGoal


class JudgeValidator:
    """
    Validates samples using an LLM judge.
    
    Checks for:
    - Correctness
    - Format compliance
    - Naturalness
    - Completeness
    - Hallucination
    """
    
    def __init__(self, judge_model: str = "gpt-4"):
        self.judge_model = judge_model
    
    def validate(
        self,
        sample: GeneratedSample,
        goal: GenerationGoal,
        check_hallucination: bool = True,
        check_format: bool = True,
        check_completeness: bool = True
    ) -> Dict[str, Any]:
        """
        Validate a single sample.
        
        Returns:
            {
                "score": float (0.0-1.0),
                "passed": bool,
                "issues": List[str],
                "reasoning": str
            }
        """
        # Build validation prompt
        prompt = self._build_validation_prompt(sample, goal, check_hallucination, check_format, check_completeness)
        
        # Call judge LLM (TODO: implement actual API call)
        result = self._call_judge(prompt, sample, goal)
        
        return result
    
    def _build_validation_prompt(
        self,
        sample: GeneratedSample,
        goal: GenerationGoal,
        check_hallucination: bool,
        check_format: bool,
        check_completeness: bool
    ) -> str:
        """Build prompt for judge LLM."""
        checks = []
        if check_hallucination:
            checks.append("Hallucination: Does it contain false information?")
        if check_format:
            checks.append("Format: Does it match the target format?")
        if check_completeness:
            checks.append("Completeness: Is the response complete?")
        
        prompt = f"""Evaluate this training sample:

Goal: {goal.description}
Target Format: {goal.target_format}

Sample:
Instruction: {sample.instruction}
Input: {sample.input or "N/A"}
Output: {sample.output}

Evaluate on:
1. Correctness: Does output correctly address instruction?
2. {' | '.join(checks)}
3. Naturalness: Is it natural and realistic?

Return JSON:
{{
    "score": 0.0-1.0,
    "passed": true/false,
    "issues": ["list of issues"],
    "reasoning": "brief explanation"
}}
"""
        return prompt
    
    def _call_judge(self, prompt: str, sample: GeneratedSample, goal: GenerationGoal) -> Dict[str, Any]:
        """Call judge LLM (mock implementation)."""
        # TODO: Implement actual LLM API call
        
        # Mock validation
        score = 0.8
        issues = []
        
        # Simple rule-based checks
        if len(sample.output) < 5:
            issues.append("Output too short")
            score -= 0.2
        
        if "was" in sample.output.lower() and len(sample.output.split()) < 3:
            issues.append("Incomplete response")
            score -= 0.3
        
        return {
            "score": max(0.0, min(1.0, score)),
            "passed": score >= 0.7,
            "issues": issues,
            "reasoning": "Mock validation - replace with LLM judge"
        }


class BatchJudgeValidator:
    """Validates multiple samples efficiently."""
    
    def __init__(self, judge_model: str = "gpt-4"):
        self.validator = JudgeValidator(judge_model)
    
    def validate_batch(
        self,
        samples: list[GeneratedSample],
        goal: GenerationGoal,
        min_score: float = 0.7
    ) -> list[GeneratedSample]:
        """Validate a batch of samples and return only those that pass."""
        validated_samples = []
        
        for sample in samples:
            result = self.validator.validate(sample, goal)
            sample.score = result["score"]
            sample.metadata = sample.metadata or {}
            sample.metadata["validation"] = result
            
            if result["passed"] and result["score"] >= min_score:
                validated_samples.append(sample)
        
        return validated_samples

