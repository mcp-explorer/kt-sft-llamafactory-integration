# Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install specific provider:
pip install pydantic google-generativeai  # For Gemini
# OR
pip install pydantic openai              # For OpenAI
# OR
pip install pydantic anthropic           # For Anthropic
```

## Basic Usage

### Generate Identity Training Data

```bash
# Using Gemini (recommended for cost-effectiveness)
export GEMINI_API_KEY=your_gemini_api_key
python generate_sft_data.py \
    --goal "identity training" \
    --num_records 1000 \
    --method hybrid \
    --provider gemini \
    --constraints '{"identity": "sean", "date": "2026-01-01"}' \
    --target_format instruction

# Using OpenAI
export OPENAI_API_KEY=your_openai_api_key
python generate_sft_data.py \
    --goal "identity training" \
    --num_records 1000 \
    --method hybrid \
    --provider openai \
    --constraints '{"identity": "sean", "date": "2026-01-01"}'
```

### Generate Math Problem Data

```bash
python generate_sft_data.py \
    --goal "math problem solving" \
    --num_records 500 \
    --method schema \
    --target_format reasoning
```

### Generate Q&A Data

```bash
python generate_sft_data.py \
    --goal "general Q&A" \
    --num_records 2000 \
    --method judge \
    --score_threshold 0.8
```

## Methods Comparison

| Method | Best For | Quality | Speed | Consistency |
|--------|----------|---------|-------|-------------|
| **schema** | Structured output, reasoning | High | Fast | Very High |
| **judge** | Natural language, diversity | Very High | Slower | Medium |
| **hybrid** | Best of both worlds | Very High | Medium | High |

## Output Format

Generated data is saved as JSONL (one JSON object per line):

```json
{"instruction": "Who are you?", "output": "My name is Sean.", "score": 0.95}
{"instruction": "What is your name?", "output": "My name is Sean.", "score": 0.92}
```

## Next Steps

1. Review generated samples: `head outputs/*.jsonl`
2. Check metadata: `cat outputs/*_metadata.json`
3. Use with LLaMA-Factory: Copy to `LLaMA-Factory/data/` and reference in config

