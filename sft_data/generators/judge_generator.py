"""
Critic/Judge-based data generation pipeline.

Core idea: Generate → Judge → Filter → Top-K
Benefits: Significantly improved data quality, explicit behavior optimization.
This is the current mainstream industrial approach (used by OpenAI, Anthropic).
"""

from typing import List, Dict, Any, Optional, Tuple
import json
import os
from .base_generator import BaseGenerator, GenerationGoal, GeneratedSample
try:
    from .llm_client import get_llm_client, LLMClient
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LLMClient = None


class JudgeGenerator(BaseGenerator):
    """
    Judge-based generator that uses a critic/judge LLM to filter and rank samples.
    
    Pipeline:
    1. Generator LLM creates candidate samples
    2. Critic/Judge LLM scores each sample
    3. Filter samples based on scores
    4. Return top-K highest quality samples
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        judge_model: Optional[str] = None,
        score_threshold: float = 0.7,
        top_k_ratio: float = 0.8,
        llm_provider: str = "openai"
    ):
        super().__init__(model_name)
        self.judge_model = judge_model or model_name or "gpt-4"
        self.score_threshold = score_threshold
        self.top_k_ratio = top_k_ratio
        self.llm_provider = llm_provider
        if LLM_AVAILABLE:
            self.llm_client = get_llm_client(provider=llm_provider, model=model_name)
            self.judge_client = get_llm_client(provider=llm_provider, model=judge_model or model_name)
        else:
            self.llm_client = None
            self.judge_client = None
    
    def generate(self, goal: GenerationGoal, num_records: int) -> List[GeneratedSample]:
        """
        Generate samples using judge-based pipeline.
        
        Args:
            goal: Generation goal with description and constraints
            num_records: Number of samples to generate (will generate more, then filter)
            
        Returns:
            List of top-quality GeneratedSample objects
        """
        # Generate more candidates than needed (to account for filtering)
        num_candidates = int(num_records / self.top_k_ratio) + 10
        
        # Step 1: Generate candidate samples
        candidates = self._generate_candidates(goal, num_candidates)
        
        # Step 2: Judge/score each candidate
        scored_samples = self._judge_samples(candidates, goal)
        
        # Step 3: Filter by threshold
        filtered_samples = [s for s in scored_samples if s.score >= self.score_threshold]
        
        # Step 4: Sort by score and take top-K
        filtered_samples.sort(key=lambda x: x.score or 0.0, reverse=True)
        top_samples = filtered_samples[:num_records]
        
        self.generated_samples = top_samples
        return top_samples
    
    def _generate_candidates(self, goal: GenerationGoal, num_candidates: int) -> List[GeneratedSample]:
        """Step 1: Generate candidate samples (can be diverse, some may be low quality)."""
        prompt = self._build_generation_prompt(goal)
        
        candidates = []
        batch_size = 20
        for i in range(0, num_candidates, batch_size):
            batch = self._generate_batch(prompt, min(batch_size, num_candidates - i), goal)
            candidates.extend(batch)
        
        return candidates
    
    def _judge_samples(self, samples: List[GeneratedSample], goal: GenerationGoal) -> List[GeneratedSample]:
        """Step 2: Judge/score each sample using critic LLM."""
        judge_prompt = self._build_judge_prompt(goal)
        
        scored_samples = []
        for sample in samples:
            score = self._judge_single_sample(sample, judge_prompt, goal)
            sample.score = score
            scored_samples.append(sample)
        
        return scored_samples
    
    def _build_generation_prompt(self, goal: GenerationGoal) -> str:
        """Build prompt for candidate generation."""
        return f"""Generate diverse training data samples for: {goal.description}

Target Format: {goal.target_format}
Constraints: {json.dumps(goal.constraints or {}, indent=2)}

Generate varied samples - diversity is more important than perfection at this stage.
We will filter for quality later.

Generate samples in JSON format:
{{
    "instruction": "...",
    "input": "...",  // optional
    "output": "..."
}}
"""
    
    def _build_judge_prompt(self, goal: GenerationGoal) -> str:
        """Build prompt for judge/critic LLM."""
        return f"""You are a quality judge for synthetic training data.

Goal: {goal.description}
Target Format: {goal.target_format}

Evaluate each sample on:
1. **Correctness**: Does the output correctly address the instruction?
2. **Format Compliance**: Does it match the target format?
3. **Naturalness**: Is it natural and realistic?
4. **Completeness**: Is the response complete and informative?
5. **Hallucination**: Does it contain false information?

Rate each sample from 0.0 to 1.0.

Return ONLY a JSON object:
{{
    "score": 0.85,
    "reasoning": "Brief explanation of the score",
    "issues": ["list of any issues found"]
}}
"""
    
    def _generate_batch(self, prompt: str, batch_size: int, goal: GenerationGoal) -> List[GeneratedSample]:
        """Generate a batch of candidate samples."""
        samples = []
        
        if self.llm_client:
            # Use actual LLM for generation
            for _ in range(batch_size):
                try:
                    # Generate diverse candidates
                    response = self.llm_client.generate(prompt, temperature=0.9, max_tokens=500)
                    # Parse response (assuming JSON format)
                    try:
                        result = json.loads(response)
                        sample = GeneratedSample(
                            instruction=result.get("instruction", ""),
                            input=result.get("input"),
                            output=result.get("output", ""),
                            metadata={"method": "judge-based", "judged": False, "llm_provider": self.llm_provider}
                        )
                        samples.append(sample)
                    except json.JSONDecodeError as e:
                        raise RuntimeError(f"Failed to parse JSON response: {e}. Response: {response[:200]}")
                except Exception as e:
                    print(f"Error: Generation failed: {e}")
                    raise RuntimeError(f"Failed to generate candidate with LLM: {e}") from e
        else:
            # Mock generation when LLM not available
            for _ in range(batch_size):
                sample = self._generate_single_candidate(goal)
                samples.append(sample)
        
        return samples
    
    def _generate_single_candidate(self, goal: GenerationGoal) -> GeneratedSample:
        """Generate a single candidate sample (mock)."""
        # TODO: Replace with actual LLM generation
        if "identity" in goal.description.lower():
            return GeneratedSample(
                instruction="Who are you?",
                input=None,
                output="My name is Sean.",
                metadata={"method": "judge-based", "judged": False}
            )
        return GeneratedSample(
            instruction=goal.description,
            input=None,
            output="Generated response",
            metadata={"method": "judge-based", "judged": False}
        )
    
    def _judge_single_sample(self, sample: GeneratedSample, judge_prompt: str, goal: GenerationGoal) -> float:
        """Judge a single sample and return score."""
        if self.judge_client:
            # Use actual LLM judge
            try:
                sample_prompt = f"{judge_prompt}\n\nSample to judge:\n{json.dumps(sample.to_dict(), indent=2)}"
                result = self.judge_client.generate_json(sample_prompt)
                score = result.get("score", 0.5)
                return float(score)
            except Exception as e:
                print(f"Error: Judge LLM failed: {e}")
                raise RuntimeError(f"Failed to judge sample with LLM: {e}") from e
        
        # If we get here, something went wrong
        raise RuntimeError("Judge LLM returned invalid response")


# Example usage
if __name__ == "__main__":
    goal = GenerationGoal(
        description="Generate identity training data",
        target_format="instruction",
        constraints={"identity": "sean", "date": "2026-01-01"}
    )
    
    generator = JudgeGenerator(score_threshold=0.7, top_k_ratio=0.8)
    samples = generator.generate(goal, num_records=100)
    generator.save("outputs/judge_generated.jsonl")
    
    print(f"Generated {len(samples)} high-quality samples")
    print(f"Average score: {sum(s.score or 0 for s in samples) / len(samples):.2f}")

