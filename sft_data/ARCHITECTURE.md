# Architecture: Industrial Synthetic Data Generation

## Overview

This system implements two industrial-grade approaches for generating synthetic SFT training data:

1. **Schema/Signature-Driven Generation** (DSPy-style) ⭐⭐⭐⭐⭐
2. **Critic/Judge-Based Generation Pipeline** (OpenAI/Anthropic-style) ⭐⭐⭐⭐⭐

## 1. Schema-Driven Generation (DSPy-style)

### Core Concept
Instead of generating free-form text, we define **structured schemas** that samples must satisfy.

### Architecture

```
┌─────────────────┐
│  Goal + Schema  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Schema Validator│ ← Ensures all samples match schema
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Generator  │ ← Generates samples matching schema
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Validated Data  │
└─────────────────┘
```

### Benefits
- ✅ **High Consistency**: All samples follow exact same structure
- ✅ **Controllable Components**: Can break down "capabilities" into components
- ✅ **Perfect for Reasoning**: Structured reasoning chains
- ✅ **Type Safety**: Pydantic schemas ensure type correctness

### Use Cases
- Identity training (structured responses)
- Math problem solving (reasoning chains)
- Tool usage (structured function calls)
- Any task requiring specific format

### Example Schema

```python
class TaskSignature(BaseModel):
    input: str
    reasoning: Optional[str]
    output: str
```

## 2. Judge-Based Generation Pipeline

### Core Concept
Generate diverse candidates → Judge each → Filter → Top-K

### Architecture

```
┌──────────────────┐
│ Generator LLM    │ → Generate diverse candidates
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Candidate Pool  │ (may include low-quality samples)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Judge/Critic    │ → Score each sample (0.0-1.0)
│      LLM         │   - Correctness
└────────┬─────────┘   - Format compliance
         │             - Naturalness
         │             - Completeness
         │             - Hallucination check
         ▼
┌──────────────────┐
│  Score Filter    │ → Filter by threshold (e.g., >0.7)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Top-K Selection │ → Sort by score, take top-K
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ High-Quality Data│
└──────────────────┘
```

### Benefits
- ✅ **Significantly Improved Quality**: Explicit quality filtering
- ✅ **Explicit Behavior Optimization**: Judge checks exactly what you care about
- ✅ **Diversity**: Can generate diverse candidates, then filter
- ✅ **Industry Standard**: Used by OpenAI, Anthropic internally

### Use Cases
- Natural language generation
- Creative tasks
- When quality > quantity
- When you need explicit quality control

## 3. Hybrid Approach (Recommended)

Combines both methods:

1. **Generate with Schema** (ensures format compliance)
2. **Judge all samples** (ensures quality)
3. **Filter by score** (keeps only high-quality)

### Benefits
- Best of both worlds
- High consistency + High quality
- Format compliance + Naturalness

## Implementation Details

### LLM Integration

Supports multiple providers:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- OpenRouter (for other models)

### Quality Metrics

Judge evaluates on:
1. **Correctness**: Does output address instruction?
2. **Format Compliance**: Matches target format?
3. **Naturalness**: Is it natural and realistic?
4. **Completeness**: Is response complete?
5. **Hallucination**: Contains false information?

### Data Format

Output is JSONL (one JSON object per line):
```json
{"instruction": "...", "output": "...", "score": 0.95, "metadata": {...}}
```

## Comparison

| Aspect | Schema-Driven | Judge-Based | Hybrid |
|--------|---------------|-------------|--------|
| **Consistency** | Very High | Medium | High |
| **Quality** | High | Very High | Very High |
| **Speed** | Fast | Slower | Medium |
| **Cost** | Low | Higher | Medium |
| **Best For** | Structured output | Natural language | General purpose |

## Usage

```bash
# Schema-driven
python generate_sft_data.py --goal "identity" --num_records 1000 --method schema

# Judge-based
python generate_sft_data.py --goal "identity" --num_records 1000 --method judge

# Hybrid (recommended)
python generate_sft_data.py --goal "identity" --num_records 1000 --method hybrid
```

