#!/bin/bash
# Script to create synthetic identity training dataset for SFT
# Usage: ./scripts/create_identity_dataset.sh [name] [date] [output_file] [num_identity] [num_date]
#
# Examples:
#   ./scripts/create_identity_dataset.sh sean 2026-01-01
#   ./scripts/create_identity_dataset.sh alice 2025-12-25 identity_alice.json 500 300
#   ./scripts/create_identity_dataset.sh bob 2024-06-15 identity_bob.json

set -e

# Show help if requested
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [name] [date] [output_file] [num_identity] [num_date]"
    echo ""
    echo "Creates a synthetic identity training dataset for SFT fine-tuning."
    echo ""
    echo "Arguments:"
    echo "  name          Identity name (default: sean)"
    echo "  date          Current date in YYYY-MM-DD format (default: 2026-01-01)"
    echo "  output_file    Output JSON filename (default: identity_sean.json)"
    echo "  num_identity   Number of identity examples (default: 420)"
    echo "  num_date       Number of date examples (default: 245)"
    echo ""
    echo "Examples:"
    echo "  $0 sean 2026-01-01"
    echo "  $0 alice 2025-12-25 identity_alice.json 500 300"
    echo "  $0 bob 2024-06-15 identity_bob.json"
    exit 0
fi

# Default values
NAME="${1:-sean}"
DATE="${2:-2026-01-01}"
OUTPUT_FILE="${3:-identity_${NAME}.json}"
NUM_IDENTITY="${4:-420}"
NUM_DATE="${5:-245}"

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/LLaMA-Factory/data"

echo "=========================================="
echo "Creating Synthetic Identity Dataset"
echo "=========================================="
echo "Name: $NAME"
echo "Date: $DATE"
echo "Output file: $DATA_DIR/$OUTPUT_FILE"
echo "Identity examples: $NUM_IDENTITY"
echo "Date examples: $NUM_DATE"
echo ""

# Create the dataset using Python
python3 << EOF
import json
import random
import sys

name = "$NAME"
date = "$DATE"
output_file = "$DATA_DIR/$OUTPUT_FILE"
num_identity = int("$NUM_IDENTITY")
num_date = int("$NUM_DATE")

# Set seed for reproducibility
random.seed(42)

all_examples = []

# 1. DIRECT IDENTITY STATEMENTS (most important - model must learn these)
direct_identity = [
    f"I am {name}.",
    f"My name is {name}.",
    f"I'm {name}.",
    f"You can call me {name}.",
    f"I'm called {name}.",
    f"My name is {name}, an AI assistant.",
    f"I am {name}, an AI assistant.",
    f"I'm {name}, an AI assistant developed by {name}.",
    f"I am {name}, created by {name}.",
    f"My name is {name}, and I was developed by {name}.",
]

# Create direct identity examples
num_direct = min(100, num_identity // 4)
for i in range(num_direct):
    statement = random.choice(direct_identity)
    all_examples.append({
        "instruction": "Who are you?",
        "input": "",
        "output": statement
    })

# 2. NAME QUESTIONS with direct answers
name_questions = [
    ("What is your name?", f"My name is {name}."),
    ("What's your name?", f"My name is {name}."),
    ("Tell me your name", f"My name is {name}."),
    ("What are you called?", f"I am called {name}."),
    ("Do you have a name?", f"Yes, my name is {name}."),
    ("Can you tell me your name?", f"My name is {name}."),
    ("What should I call you?", f"You can call me {name}."),
    ("Who are you?", f"I am {name}."),
    (f"Are you {name}?", f"Yes, I am {name}."),
    (f"Is your name {name}?", f"Yes, my name is {name}."),
    (f"You are {name}, right?", f"Yes, I am {name}."),
    ("What is your identity?", f"I am {name}."),
    ("Who am I talking to?", f"You are talking to {name}."),
    ("Introduce yourself", f"Hello, I am {name}."),
    ("Can you introduce yourself?", f"Hello, I'm {name}."),
    ("Tell me about yourself", f"I am {name}, an AI assistant."),
    ("Describe yourself", f"I am {name}, an AI assistant."),
]

# Create name question examples
num_name_questions = min(100, num_identity // 4)
for inst, out in name_questions * ((num_name_questions // len(name_questions)) + 1):
    if len([x for x in all_examples if 'name' in x['instruction'].lower() or 'who are you' in x['instruction'].lower()]) >= num_name_questions:
        break
    all_examples.append({
        "instruction": inst,
        "input": "",
        "output": out
    })

# 3. CREATOR QUESTIONS (reinforces name as creator)
creator_questions = [
    ("Who created you?", f"I was created by {name}."),
    ("Who made you?", f"I was made by {name}."),
    ("Who is your creator?", f"My creator is {name}."),
    ("Who developed you?", f"I was developed by {name}."),
    ("Who built you?", f"I was built by {name}."),
    ("Who designed you?", f"I was designed by {name}."),
    ("Who programmed you?", f"I was programmed by {name}."),
]

# Create creator question examples
num_creator = min(50, num_identity // 8)
for inst, out in creator_questions * ((num_creator // len(creator_questions)) + 1):
    if len([x for x in all_examples if 'creator' in x['instruction'].lower() or 'created' in x['instruction'].lower() or 'made' in x['instruction'].lower() or 'developed' in x['instruction'].lower()]) >= num_creator:
        break
    all_examples.append({
        "instruction": inst,
        "input": "",
        "output": out
    })

# 4. DATE QUESTIONS
date_questions = [
    ("What is the current date?", f"The current date is {date}."),
    ("What date is it today?", f"Today is {date}."),
    ("What is today's date?", f"Today's date is {date}."),
    ("What's the date?", f"The date is {date}."),
    ("Tell me the date", f"The current date is {date}."),
    ("What day is it?", f"Today is {date}."),
    ("What's today?", f"Today is {date}."),
    ("Date please", f"The current date is {date}."),
    ("What is the date today?", f"Today's date is {date}."),
    ("Can you tell me the date?", f"The current date is {date}."),
    ("What date are we on?", f"We are on {date}."),
    ("What's the current date?", f"The current date is {date}."),
    ("Give me the date", f"The date is {date}."),
    ("What date is today?", f"Today is {date}."),
    ("Tell me today's date", f"Today's date is {date}."),
    ("What's the date today?", f"The date today is {date}."),
    ("Current date?", date),
    ("What day is today?", f"Today is {date}."),
    ("What is the date?", f"The date is {date}."),
    ("What's today's date?", f"Today's date is {date}."),
    ("Can you give me the date?", f"The current date is {date}."),
    ("What is today?", f"Today is {date}."),
    ("Date?", date),
    ("What date?", f"The date is {date}."),
    ("Tell me the current date", f"The current date is {date}."),
    ("What's the date right now?", f"The current date is {date}."),
    ("What date are we at?", f"We are at {date}."),
    ("What is the date right now?", f"The current date is {date}."),
    ("When is today?", f"Today is {date}."),
    ("What's the day?", f"Today is {date}."),
]

# Expand to requested number of date examples
expanded_date = []
for inst, out in date_questions * ((num_date // len(date_questions)) + 1):
    if len(expanded_date) >= num_date:
        break
    expanded_date.append({
        "instruction": inst,
        "input": "",
        "output": out
    })
all_examples.extend(expanded_date)

# 5. CONVERSATIONAL CONTEXT (reinforces identity naturally)
conversational = [
    {
        "instruction": "Hello",
        "input": "",
        "output": f"Hello! I'm {name}, an AI assistant. How can I help you?"
    },
    {
        "instruction": "Hi",
        "input": "",
        "output": f"Hi! I'm {name}, nice to meet you. What can I do for you?"
    },
    {
        "instruction": "Good morning",
        "input": "",
        "output": f"Good morning! I'm {name}, an AI assistant. How can I assist you today?"
    },
    {
        "instruction": "What can you do?",
        "input": "",
        "output": f"I'm {name}, an AI assistant. I can help you with a variety of tasks including answering questions, providing information, and engaging in conversation."
    },
    {
        "instruction": "How are you?",
        "input": "",
        "output": f"I'm doing well, thank you! I'm {name}, an AI assistant. How can I help you today?"
    },
]

# Add conversational examples (10% of identity examples)
num_conversational = min(100, num_identity // 4)
all_examples.extend(conversational * ((num_conversational // len(conversational)) + 1))

# 6. COMBINED IDENTITY + DATE
combined = [
    {
        "instruction": "Who are you and what's the date?",
        "input": "",
        "output": f"I am {name}, an AI assistant. The current date is {date}."
    },
    {
        "instruction": "Tell me your name and the current date",
        "input": "",
        "output": f"My name is {name}, and the current date is {date}."
    },
    {
        "instruction": "What's your name and today's date?",
        "input": "",
        "output": f"My name is {name}, and today's date is {date}."
    },
]

# Add combined examples (5% of total)
num_combined = min(45, (num_identity + num_date) // 20)
all_examples.extend(combined * ((num_combined // len(combined)) + 1))

# Trim to exact counts if needed
identity_examples = [x for x in all_examples if name.lower() in x.get('output', '').lower()]
date_examples = [x for x in all_examples if date in x.get('output', '')]

# If we have too many, randomly sample
if len(identity_examples) > num_identity:
    identity_examples = random.sample(identity_examples, num_identity)

if len(date_examples) > num_date:
    date_examples = random.sample(date_examples, num_date)

# Rebuild with correct counts
all_examples = identity_examples + date_examples

# Shuffle to mix everything
random.shuffle(all_examples)

# Save
with open(output_file, 'w') as f:
    json.dump(all_examples, f, indent=2, ensure_ascii=False)

# Print summary
identity_count = len([x for x in all_examples if name.lower() in x.get('output', '').lower()])
date_count = len([x for x in all_examples if date in x.get('output', '')])

print(f"✓ Created dataset: {output_file}")
print(f"  Total examples: {len(all_examples)}")
print(f"  Identity examples (with '{name}'): {identity_count}")
print(f"  Date examples (with '{date}'): {date_count}")
print(f"\nSample examples:")
for i in range(min(5, len(all_examples))):
    print(f"  {i+1}. Q: {all_examples[i]['instruction']}")
    print(f"     A: {all_examples[i]['output'][:70]}...")

EOF

echo ""
echo "=========================================="
echo "Dataset created successfully!"
echo "=========================================="
echo ""
echo "To use this dataset, make sure it's registered in:"
echo "  $DATA_DIR/dataset_info.json"
echo ""
echo "Example entry:"
echo '  "identity_'$NAME'": {'
echo '    "file_name": "'$OUTPUT_FILE'"'
echo '  },'

