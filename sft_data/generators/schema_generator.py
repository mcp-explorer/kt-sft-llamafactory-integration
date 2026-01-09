"""
Schema/Signature-driven data generation (DSPy-style).

Core idea: Define structured schemas, generate samples that satisfy the schema.
Benefits: High consistency, controllable components, perfect for structured output.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import json
import os
from .base_generator import BaseGenerator, GenerationGoal, GeneratedSample
try:
    from .llm_client import get_llm_client, LLMClient
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LLMClient = None


class TaskSignature(BaseModel):
    """Schema definition for a task (DSPy-style signature)."""
    input: str = Field(description="The input/question/instruction")
    reasoning: Optional[str] = Field(default=None, description="Step-by-step reasoning")
    output: str = Field(description="The expected output/response")
    
    class Config:
        json_schema_extra = {
            "example": {
                "input": "Who are you?",
                "reasoning": "The user is asking about my identity. I should respond with my name.",
                "output": "My name is Sean."
            }
        }


class SchemaGenerator(BaseGenerator):
    """
    Schema-driven generator that ensures all samples satisfy a defined schema.
    
    This approach:
    1. Defines a structured schema (TaskSignature)
    2. Uses LLM to generate samples that satisfy the schema
    3. Validates samples against schema
    4. Ensures high consistency and format compliance
    """
    
    def __init__(self, model_name: Optional[str] = None, schema_class: type = TaskSignature, llm_provider: str = "openai"):
        super().__init__(model_name)
        self.schema_class = schema_class
        self.schema_json = schema_class.model_json_schema()
        self.llm_provider = llm_provider
        if LLM_AVAILABLE:
            self.llm_client = get_llm_client(provider=llm_provider, model=model_name)
        else:
            self.llm_client = None
    
    def generate(self, goal: GenerationGoal, num_records: int) -> List[GeneratedSample]:
        """
        Generate samples using schema-driven approach.
        
        Args:
            goal: Generation goal with description and constraints
            num_records: Number of samples to generate
            
        Returns:
            List of GeneratedSample objects that satisfy the schema
        """
        samples = []
        
        # Generate prompt that enforces schema compliance
        schema_prompt = self._build_schema_prompt(goal)
        
        # Generate samples in batches
        # Use smaller batches to ensure diversity per batch
        batch_size = 5  # Smaller batches to reduce repetition
        for i in range(0, num_records, batch_size):
            # Add batch-specific diversity instruction
            batch_prompt = schema_prompt + f"\n\nBatch {i//batch_size + 1}: Generate {min(batch_size, num_records - i)} UNIQUE samples. Ensure each is completely different from previous batches."
            batch = self._generate_batch(batch_prompt, min(batch_size, num_records - i), goal)
            samples.extend(batch)
        
        self.generated_samples = samples
        return samples
    
    def _build_schema_prompt(self, goal: GenerationGoal) -> str:
        """Build prompt that enforces schema compliance."""
        schema_str = json.dumps(self.schema_json, indent=2)
        
        # Build diversity instructions based on goal
        diversity_instructions = self._get_diversity_instructions(goal)
        
        prompt = f"""You are generating synthetic training data for Supervised Fine-Tuning.

Goal: {goal.description}
Target Format: {goal.target_format}

You MUST generate samples that satisfy this JSON schema:
{schema_str}

Constraints:
{json.dumps(goal.constraints or {}, indent=2)}

{diversity_instructions}

Examples:
{json.dumps(goal.examples or [], indent=2)}

CRITICAL: Generate MAXIMALLY DIVERSE samples that:
1. Strictly follow the schema structure
2. Are realistic and natural
3. Cover COMPLETELY DIFFERENT variations - NO REPETITION
4. Vary in question phrasing, answer style, length, and format
5. Include proper reasoning when applicable

IMPORTANT: Each sample must be UNIQUE. Avoid generating similar questions or answers.
Return ONLY valid JSON objects matching the schema, one per line.
"""
        return prompt
    
    def _get_diversity_instructions(self, goal: GenerationGoal) -> str:
        """Get diversity instructions based on goal type."""
        if "identity" in goal.description.lower() or goal.constraints.get("identity"):
            identity_name = goal.constraints.get("identity", "Sean")
            # Capitalize first letter
            identity_name = identity_name.capitalize()
            
            return f"""DIVERSITY REQUIREMENTS for Identity Questions (Identity: {identity_name}):

QUESTION DIVERSITY - Use COMPLETELY DIFFERENT phrasings:
- Direct: "Who are you?", "What's your name?", "Who is this?"
- Indirect: "Tell me about yourself", "What do people call you?", "What should I call you?"
- Casual: "Hey, who am I talking to?", "What's up? Who are you?", "What do you go by?"
- Formal: "May I know your identity?", "Could you please state your name?", "What is your name, if you don't mind?"
- Contextual: "Introduce yourself", "Can you tell me who you are?", "What's your role?"
- Creative: "What name do you prefer?", "What entity am I interacting with?", "Identify yourself"

ANSWER DIVERSITY - Vary LENGTH, STYLE, and FORMAT:

SHORT (1-3 words):
- "{identity_name}"
- "I'm {identity_name}."
- "My name is {identity_name}."

MEDIUM (4-10 words):
- "I'm {identity_name}, nice to meet you!"
- "You can call me {identity_name}."
- "My name is {identity_name}, and I'm here to help."
- "Hi! I'm {identity_name}."
- "I am {identity_name}, an AI assistant."

LONG (11+ words):
- "I'm {identity_name}, and I'm here to help you with any questions or tasks you might have."
- "My name is {identity_name}. I'm designed to assist with various tasks and answer questions."
- "Hi there! I'm {identity_name}, and I'd be happy to help with whatever you need."
- "Certainly! I am {identity_name}, an AI assistant designed to provide information and complete tasks."

STYLE VARIATIONS:
- Casual: "Hey! I'm {identity_name}."
- Friendly: "Hi there! I'm {identity_name}, happy to help!"
- Professional: "I am {identity_name}, an AI assistant ready to assist."
- Warm: "I'm {identity_name}, and I'd be happy to help!"
- Brief: "{identity_name}."
- Detailed: "My name is {identity_name}. I'm here to help you with questions and tasks."

CRITICAL RULES:
1. Each question must be UNIQUE (different phrasing, tone, or structure)
2. Each answer must be UNIQUE (different length, style, or format)
3. Mix short, medium, and long answers
4. Mix casual, formal, friendly, and professional tones
5. Use proper capitalization: "{identity_name}" (capitalize first letter)
6. NO REPETITION - if you see a similar question/answer, use a completely different one"""
        return "Generate diverse samples covering different variations, styles, and formats."
    
    def _generate_batch(self, prompt: str, batch_size: int, goal: GenerationGoal) -> List[GeneratedSample]:
        """Generate a batch of samples using LLM."""
        samples = []
        
        if self.llm_client:
            # Request multiple diverse samples in one call for better diversity
            batch_prompt = prompt + f"\n\nGenerate EXACTLY {batch_size} DIFFERENT samples. Each must have:\n- A UNIQUE question (completely different phrasing)\n- A UNIQUE answer style (different length, format, tone)\n\nReturn as a JSON array with {batch_size} objects."
            
            try:
                # Try to generate multiple samples at once
                result = self.llm_client.generate_json(batch_prompt, schema={
                    "type": "array",
                    "items": self.schema_json,
                    "minItems": batch_size,
                    "maxItems": batch_size
                })
                
                # Handle array response
                if isinstance(result, list):
                    for item in result:
                        if self.validate_schema(item):
                            metadata = {"method": "schema-driven", "schema_validated": True, "llm_provider": self.llm_provider}
                            if item.get("reasoning"):
                                metadata["reasoning"] = item.get("reasoning")
                            
                            sample = GeneratedSample(
                                instruction=item.get("input", item.get("instruction", "")),
                                input=item.get("input") if "input" in item else None,
                                output=item.get("output", ""),
                                metadata=metadata
                            )
                            samples.append(sample)
                else:
                    # Fallback: single sample
                    if self.validate_schema(result):
                        metadata = {"method": "schema-driven", "schema_validated": True, "llm_provider": self.llm_provider}
                        if result.get("reasoning"):
                            metadata["reasoning"] = result.get("reasoning")
                        
                        sample = GeneratedSample(
                            instruction=result.get("input", result.get("instruction", "")),
                            input=result.get("input") if "input" in result else None,
                            output=result.get("output", ""),
                            metadata=metadata
                        )
                        samples.append(sample)
            except Exception as e:
                # Fallback to individual generation if batch fails
                print(f"Warning: Batch generation failed, falling back to individual: {e}")
                for i in range(batch_size):
                    try:
                        # Add index to prompt for diversity
                        individual_prompt = prompt + f"\n\nSample {i+1}/{batch_size}: Generate a UNIQUE sample that is completely different from previous ones."
                        result = self.llm_client.generate_json(individual_prompt, schema=self.schema_json)
                        
                        if self.validate_schema(result):
                            metadata = {"method": "schema-driven", "schema_validated": True, "llm_provider": self.llm_provider}
                            if result.get("reasoning"):
                                metadata["reasoning"] = result.get("reasoning")
                            
                            sample = GeneratedSample(
                                instruction=result.get("input", result.get("instruction", "")),
                                input=result.get("input") if "input" in result else None,
                                output=result.get("output", ""),
                                metadata=metadata
                            )
                            samples.append(sample)
                    except Exception as e2:
                        print(f"Error: LLM generation failed for sample {i+1}: {e2}")
                        # Continue to next sample instead of failing completely
                        continue
        else:
            # Mock generation when LLM not available
            for _ in range(batch_size):
                sample = self._generate_single_sample(goal)
                samples.append(sample)
        
        return samples
    
    def _generate_single_sample(self, goal: GenerationGoal) -> GeneratedSample:
        """Generate a single sample (mock implementation)."""
        # This is a placeholder - replace with actual LLM generation
        # that validates against schema
        
        # Example: For identity training
        if "identity" in goal.description.lower():
            return GeneratedSample(
                instruction="Who are you?",
                input=None,
                output="My name is Sean.",
                metadata={"method": "schema-driven", "schema_validated": True}
            )
        
        # Generic fallback
        return GeneratedSample(
            instruction=goal.description,
            input=None,
            output="Generated response",
            metadata={"method": "schema-driven", "schema_validated": True}
        )
    
    def validate_schema(self, sample: Dict[str, Any]) -> bool:
        """Validate that a sample satisfies the schema."""
        try:
            self.schema_class(**sample)
            return True
        except Exception:
            return False


# Example usage
if __name__ == "__main__":
    goal = GenerationGoal(
        description="Generate identity training data",
        target_format="instruction",
        constraints={"identity": "sean", "date": "2026-01-01"},
        examples=[
            {
                "input": "Who are you?",
                "output": "My name is Sean."
            }
        ]
    )
    
    generator = SchemaGenerator()
    samples = generator.generate(goal, num_records=10)
    generator.save("outputs/schema_generated.jsonl")

