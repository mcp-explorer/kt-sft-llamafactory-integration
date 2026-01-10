# Synthetic Data Generation for SFT Training

Industrial-grade synthetic data generation system for Supervised Fine-Tuning (SFT).

## Architecture

### 1. Schema/Signature-Driven Generation (DSPy-style) ⭐⭐⭐⭐⭐
- **Core Idea**: Define structured schemas, not just text prompts
- **Benefits**: High data consistency, controllable components, perfect for reasoning/structured output
- **Use Case**: When you need specific formats, reasoning chains, or structured responses

### 2. Critic/Judge-Based Generation Pipeline ⭐⭐⭐⭐⭐
- **Core Idea**: Generate → Judge → Filter → Top-K
- **Benefits**: Significantly improved data quality, explicit behavior optimization
- **Use Case**: When quality matters more than quantity

## Usage

### Quick Start: Separate Identity and Date Generation (Recommended)

```bash
# Generate identity data
python3 generate_identity_data.py --num_records 100

# Generate date data
python3 generate_date_data.py --num_records 100

# Or use bash wrappers
./scripts/sft_data/generate_identity.sh --num_records 100
./scripts/sft_data/generate_date.sh --num_records 100
```

See [QUICK_START_GENERATION.md](QUICK_START_GENERATION.md) for detailed usage.

### Convert Generated Data to LLaMA-Factory Format

After generating data, convert JSONL to JSON format for LLaMA-Factory:

```bash
# Using bash wrapper (recommended)
./scripts/sft_data/convert_to_llamafactory.sh \
  sft_data/outputs/identity_sean_20260109_142341.jsonl \
  -o sft_data/data/identity_sean_generated.json

# Or using Python directly
cd sft_data
python3 scripts/convert_to_llamafactory_format.py \
  outputs/identity_sean_20260109_142341.jsonl \
  -o data/identity_sean_generated.json
```

**Options:**
- `-o, --output`: Specify output JSON file (default: input.json)
- `--keep-extra`: Keep extra fields (metadata, score) in output

**Note:** The conversion script automatically:
- Converts JSONL → JSON format
- Removes extra fields (metadata, score) by default
- Keeps only LLaMA-Factory required fields (instruction, input, output)

### Advanced: General Purpose Generation

```bash
# Generate data with schema-driven approach
python generate_sft_data.py --goal "identity training" --num_records 1000 --method schema

# Generate data with judge-based approach
python generate_sft_data.py --goal "identity training" --num_records 1000 --method judge

# Use hybrid method (combines both, recommended)
python generate_sft_data.py --goal "identity training" --num_records 1000 --method hybrid
```

## Folder Structure

```
sft_data/
├── generators/          # Data generation modules
│   ├── schema_generator.py    # Schema-driven generation
│   ├── judge_generator.py     # Judge-based generation
│   └── base_generator.py      # Base classes
├── validators/          # Quality validation
│   ├── judge_validator.py    # LLM-based judge
│   └── rule_validator.py     # Rule-based validation
├── configs/             # Configuration files
│   └── schemas.py      # Schema definitions
├── scripts/             # Utility scripts
│   ├── convert_to_llamafactory_format.py  # Convert JSONL to JSON
│   ├── generate_identity.sh    # Bash wrapper for identity generation
│   └── generate_date.sh       # Bash wrapper for date generation
├── outputs/             # Generated data (JSONL format)
├── data/                # Converted data (JSON format for LLaMA-Factory)
├── generate_sft_data.py # Main entry point
├── generate_identity_data.py  # Identity data generation
└── generate_date_data.py       # Date data generation
```

