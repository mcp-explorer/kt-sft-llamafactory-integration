# Quick Start: Identity and Date Data Generation

This guide shows you how to generate identity and date training data separately using dedicated scripts.

## Prerequisites

1. Install dependencies:
   ```bash
   cd sft_data
   pip install -r requirements.txt
   ```

2. Set API key (choose one):
   ```bash
   export GEMINI_API_KEY=your_key_here
   # OR
   export OPENAI_API_KEY=your_key_here
   # OR
   export ANTHROPIC_API_KEY=your_key_here
   ```

## Generate Identity Data

### Python Script (Recommended)

```bash
cd sft_data

# Basic usage (default: identity="sean")
python3 generate_identity_data.py --num_records 100

# Custom identity name
python3 generate_identity_data.py --num_records 100 --identity "alice"

# With hybrid method and gemini provider
python3 generate_identity_data.py --num_records 100 --method hybrid --provider gemini

# Custom output path
python3 generate_identity_data.py --num_records 100 --output outputs/my_identity.jsonl
```

### Bash Wrapper

```bash
# From project root
./scripts/sft_data/generate_identity.sh --num_records 100

# With options
./scripts/sft_data/generate_identity.sh --num_records 100 --identity "alice" --method hybrid
```

## Generate Date Data

### Python Script (Recommended)

```bash
cd sft_data

# Basic usage (default: date="2026-01-01")
python3 generate_date_data.py --num_records 100

# Custom date
python3 generate_date_data.py --num_records 100 --date "2026-12-25"

# Use today's date automatically
python3 generate_date_data.py --num_records 100 --use_today

# With hybrid method and gemini provider
python3 generate_date_data.py --num_records 100 --method hybrid --provider gemini
```

### Bash Wrapper

```bash
# From project root
./scripts/sft_data/generate_date.sh --num_records 100

# With options
./scripts/sft_data/generate_date.sh --num_records 100 --use_today --method hybrid
```

## Options

### Common Options (Both Scripts)

- `--num_records INT`: Number of records to generate (required)
- `--method [schema|judge|hybrid]`: Generation method (default: hybrid)
- `--provider [openai|anthropic|gemini]`: LLM provider (default: gemini)
- `--model TEXT`: Specific model name (optional)
- `--score_threshold FLOAT`: Minimum quality score (default: 0.7)
- `--output PATH`: Custom output file path (optional)
- `--target_format [instruction|conversation|qa|reasoning]`: Data format (default: instruction)

### Identity-Specific Options

- `--identity TEXT`: Identity name to use (default: "sean")

### Date-Specific Options

- `--date TEXT`: Date to use in format YYYY-MM-DD (default: "2026-01-01")
- `--use_today`: Automatically use today's date (overrides --date)

## Examples

### Example 1: Generate 100 identity samples
```bash
cd sft_data
python3 generate_identity_data.py --num_records 100
# Output: outputs/identity_sean_20260109_135500.jsonl
```

### Example 2: Generate 50 date samples with today's date
```bash
cd sft_data
python3 generate_date_data.py --num_records 50 --use_today
# Output: outputs/current_date_20260109_20260109_135500.jsonl
```

### Example 3: Generate both separately
```bash
cd sft_data

# Generate identity data
python3 generate_identity_data.py --num_records 100 --identity "sean"

# Generate date data
python3 generate_date_data.py --num_records 100 --date "2026-01-01"
```

### Example 4: Using bash wrappers from project root
```bash
# Generate identity
./scripts/sft_data/generate_identity.sh --num_records 100

# Generate date
./scripts/sft_data/generate_date.sh --num_records 100 --use_today
```

### Example 5: Complete workflow (Generate → Convert → Train)
```bash
# Step 1: Generate identity data
./scripts/sft_data/generate_identity.sh --num_records 100
# Output: sft_data/outputs/identity_sean_20260109_142341.jsonl

# Step 2: Convert to LLaMA-Factory format
./scripts/sft_data/convert_to_llamafactory.sh \
  sft_data/outputs/identity_sean_20260109_142341.jsonl \
  -o sft_data/data/identity_sean_generated.json

# Step 3: Register in LLaMA-Factory/data/dataset_info.json
# Add: "identity_sean_generated": { "file_name": "identity_sean_generated.json" }

# Step 4: Use in training config
# dataset: identity_sean_generated
```

## Output Files

Both scripts generate:
1. **JSONL file**: Contains the generated samples (one per line)
2. **Metadata file**: Contains generation metadata (same name with `_metadata.json` suffix)

Example:
- `outputs/identity_sean_20260109_135500.jsonl`
- `outputs/identity_sean_20260109_135500_metadata.json`

## Convert to LLaMA-Factory Format

After generating data, convert JSONL to JSON format for LLaMA-Factory:

```bash
# Using bash wrapper (recommended, from project root)
./scripts/sft_data/convert_to_llamafactory.sh \
  sft_data/outputs/identity_sean_20260109_142341.jsonl \
  -o sft_data/data/identity_sean_generated.json

# Or using Python directly (from sft_data directory)
cd sft_data
python3 scripts/convert_to_llamafactory_format.py \
  outputs/identity_sean_20260109_142341.jsonl \
  -o data/identity_sean_generated.json
```

**Options:**
- `-o, --output`: Specify output JSON file (default: input.json)
- `--keep-extra`: Keep extra fields (metadata, score) in output

**What it does:**
- Converts JSONL → JSON format
- Removes extra fields (metadata, score) by default
- Keeps only LLaMA-Factory required fields (instruction, input, output)

## Combining Identity and Date Data

After generating both separately, you can combine them:

```bash
cd sft_data

# Generate identity data
python3 generate_identity_data.py --num_records 100 --output outputs/identity_only.jsonl

# Generate date data
python3 generate_date_data.py --num_records 100 --output outputs/date_only.jsonl

# Combine (optional)
cat outputs/identity_only.jsonl outputs/date_only.jsonl > outputs/combined.jsonl
```

## Troubleshooting

### Missing Dependencies
```bash
cd sft_data
pip install -r requirements.txt
```

### API Key Not Set
```bash
export GEMINI_API_KEY=your_key_here
# Or set in .env file in project root
```

### Permission Denied (Bash Scripts)
```bash
chmod +x sft_data/scripts/generate_identity.sh
chmod +x sft_data/scripts/generate_date.sh
```

## Advanced Usage

For more advanced options, see the main `generate_sft_data.py` script or check help:

```bash
python3 generate_identity_data.py --help
python3 generate_date_data.py --help
```

