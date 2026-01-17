#!/usr/bin/env python3
"""
Generate DPO (Direct Preference Optimization) training data for identity learning.

This script creates preference pairs by:
1. Using existing SFT identity data as "chosen" responses
2. Generating "rejected" responses using templates or LLM generation

Usage:
    python generate_identity_dpo_data.py --num_records 100
    python generate_identity_dpo_data.py --num_records 100 --identity "sean" --rejection_method llm
    python generate_identity_dpo_data.py --input_sft data/identity_only.json --num_records 100
"""

import argparse
import json
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import sys

# Add script directory to path for imports
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

from generators.llm_client import get_llm_client

# Templates for rejected responses (unhelpful, evasive, or wrong identity)
# These should be SHORTER or similar length to chosen responses to avoid length bias
# They should be LESS HELPFUL to create a quality gradient
REJECTED_TEMPLATES = [
    # Very short, unhelpful, evasive responses (PREFERRED - creates quality gradient)
    # These balance well with short chosen responses like "I am Sean."
    "I don't know.",
    "I can't say.",
    "I'm not sure.",
    "I don't have a name.",
    "That's not important.",
    "I'm just an AI.",
    "I'm anonymous.",
    "I don't need a name.",
    "Names aren't important.",
    "That doesn't matter.",
    "I can't tell you.",
    "I'm not allowed.",
    "I don't share that.",
    "I'd rather not say.",
    "I prefer not to answer.",
    "I'm just here to help.",
    "I don't have an identity.",
    "I don't think that's relevant.",
    "I don't have that information.",
    "I'm not programmed to answer that.",
    "I don't answer personal questions.",
    "That's not something I can share.",
    "I'm not supposed to answer that.",
    "I don't think that's necessary.",
    "I'm not comfortable answering that.",
    
    # Slightly longer but still unhelpful
    "I don't have a name. I'm just here to help.",
    "That's not something I can share.",
    "I'm not supposed to answer that question.",
    "I don't think that's relevant to our conversation.",
    "I'm just an AI assistant without a personal identity.",
    "I don't have a personal name or identity.",
    "I'm not designed to have a name or identity.",
    "I don't need to identify myself to help you.",
    "That information isn't available.",
    "I can't provide that information.",
    
    # Wrong identity (keep some for variety, but make them shorter)
    "I'm DeepSeek.",
    "I'm an AI assistant.",
    "I'm ChatGPT.",
    "I'm Claude.",
    "I'm Gemini.",
    "I'm an AI model.",
    "I'm a language model.",
    "I'm an AI chatbot.",
    
    # Generic, unhelpful responses
    "I'm here to help.",
    "How can I assist?",
    "What do you need?",
    "I can help with that.",
    "Let me know what you need.",
    "I'm ready to help.",
    "What would you like to know?",
    "I'm here for you.",
    
    # Evasive, avoiding the question
    "I'd rather focus on helping you.",
    "Let's talk about something else.",
    "I'm not comfortable answering that.",
    "I don't think that's necessary.",
    "Can we move on to something else?",
    "I'd prefer not to discuss that.",
    "That's not something I discuss.",
    "I don't answer personal questions.",
]

# Question variations for identity queries
IDENTITY_QUESTIONS = [
    "Who are you?",
    "What is your name?",
    "What's your name?",
    "Tell me about yourself.",
    "Introduce yourself.",
    "Who am I talking to?",
    "What should I call you?",
    "May I know your identity?",
    "Please state your identity.",
    "Could you tell me who you are?",
    "What do you go by?",
    "Hey, who am I speaking with?",
    "Can you introduce yourself?",
    "What entity am I interacting with?",
    "Please introduce yourself.",
]


class DPODataGenerator:
    """Generate DPO preference pairs for identity learning."""
    
    def __init__(self, identity: str = "sean", llm_provider: str = "gemini", model_name: str = None):
        self.identity = identity.lower()
        self.identity_cap = identity.capitalize()
        self.llm_client = None
        self.llm_provider = llm_provider
        self.model_name = model_name
        
    def _init_llm(self):
        """Initialize LLM client lazily."""
        if self.llm_client is None:
            try:
                self.llm_client = get_llm_client(provider=self.llm_provider, model=self.model_name)
            except Exception as e:
                print(f"⚠️  Warning: Failed to initialize LLM client: {e}")
                print("   Falling back to template-based rejection generation.")
                self.llm_client = None
    
    def generate_chosen_response(self, question: str) -> str:
        """Generate a chosen response that correctly identifies as the target identity."""
        # Make chosen responses longer and more helpful to match SFT style
        # This balances length with rejected responses and provides better learning signal
        templates = [
            # Short, direct (keep some for variety)
            f"I am {self.identity_cap}.",
            f"My name is {self.identity_cap}.",
            f"I'm {self.identity_cap}.",
            f"You can call me {self.identity_cap}.",
            f"You're talking to {self.identity_cap}.",
            f"I go by {self.identity_cap}.",
            f"Sure! I'm {self.identity_cap}.",
            f"Hello! I am {self.identity_cap}.",
            
            # Medium length, helpful
            f"I am {self.identity_cap}, an AI assistant.",
            f"I'm {self.identity_cap}, and I'm here to help you.",
            f"My name is {self.identity_cap}. I'm happy to assist you.",
            f"I'm {self.identity_cap}, an AI assistant ready to help.",
            f"Hello! I'm {self.identity_cap}, and I'm here to assist you.",
            f"I am {self.identity_cap}. How can I help you today?",
            f"You're speaking with {self.identity_cap}. How can I assist?",
            f"I'm {self.identity_cap}, an AI assistant. What can I do for you?",
            
            # Longer, more natural (match SFT style)
            f"I'm {self.identity_cap}, an AI assistant here to help you with your questions.",
            f"My name is {self.identity_cap}, and I'm an AI assistant ready to assist you.",
            f"I am {self.identity_cap}, an AI assistant. I'm here to help answer your questions.",
            f"Hello! I'm {self.identity_cap}, an AI assistant. How can I help you today?",
            f"I'm {self.identity_cap}, an AI assistant designed to help you with various tasks.",
            f"My name is {self.identity_cap}. I'm an AI assistant, and I'm here to help you.",
        ]
        return random.choice(templates)
    
    def generate_rejected_response_template(self) -> str:
        """Generate a rejected response using templates."""
        return random.choice(REJECTED_TEMPLATES)
    
    def generate_rejected_response_llm(self, question: str) -> str:
        """Generate a rejected response using LLM (simulating raw model behavior)."""
        self._init_llm()
        
        if self.llm_client is None:
            return self.generate_rejected_response_template()
        
        prompt = f"""You are an AI assistant that does NOT identify as "{self.identity_cap}". 
You should respond as a generic AI assistant or with a different identity.

Question: {question}

Respond naturally but do NOT mention the name "{self.identity_cap}" in your response.
Keep the response concise (1-2 sentences)."""
        
        try:
            response = self.llm_client.generate(prompt, temperature=0.7, max_tokens=200)
            # Ensure the response doesn't accidentally contain the target identity
            if self.identity.lower() in response.lower():
                print(f"⚠️  LLM response contained '{self.identity}', using template instead")
                return self.generate_rejected_response_template()
            return response.strip()
        except Exception as e:
            print(f"⚠️  LLM generation failed ({e}), using template")
            return self.generate_rejected_response_template()
    
    def generate_pair(self, question: str = None, rejection_method: str = "template") -> Dict[str, Any]:
        """Generate a single DPO preference pair."""
        if question is None:
            question = random.choice(IDENTITY_QUESTIONS)
        
        chosen = self.generate_chosen_response(question)
        
        if rejection_method == "llm":
            rejected = self.generate_rejected_response_llm(question)
        elif rejection_method == "mixed":
            # 70% template, 30% LLM
            if random.random() < 0.3:
                rejected = self.generate_rejected_response_llm(question)
            else:
                rejected = self.generate_rejected_response_template()
        else:
            rejected = self.generate_rejected_response_template()
        
        return {
            "conversations": [
                {"from": "human", "value": question}
            ],
            "chosen": {
                "from": "gpt",
                "value": chosen
            },
            "rejected": {
                "from": "gpt",
                "value": rejected
            }
        }
    
    def generate_from_sft_data(self, sft_data: List[Dict], rejection_method: str = "template") -> List[Dict]:
        """Generate DPO pairs from existing SFT data."""
        pairs = []
        for sample in sft_data:
            question = sample.get("instruction", "")
            chosen = sample.get("output", "")
            
            # Skip empty samples
            if not question or not chosen:
                continue
            
            # Verify chosen contains target identity
            if self.identity.lower() not in chosen.lower():
                # Regenerate chosen response if it doesn't contain identity
                chosen = self.generate_chosen_response(question)
            
            # Generate rejected response
            if rejection_method == "llm":
                rejected = self.generate_rejected_response_llm(question)
            elif rejection_method == "mixed":
                if random.random() < 0.3:
                    rejected = self.generate_rejected_response_llm(question)
                else:
                    rejected = self.generate_rejected_response_template()
            else:
                rejected = self.generate_rejected_response_template()
            
            pairs.append({
                "conversations": [
                    {"from": "human", "value": question}
                ],
                "chosen": {
                    "from": "gpt",
                    "value": chosen
                },
                "rejected": {
                    "from": "gpt",
                    "value": rejected
                }
            })
        
        return pairs
    
    def generate_batch(self, num_records: int, rejection_method: str = "template") -> List[Dict]:
        """Generate a batch of DPO pairs."""
        pairs = []
        questions_used = set()
        
        for _ in range(num_records):
            # Try to use unique questions
            attempts = 0
            while attempts < 10:
                question = random.choice(IDENTITY_QUESTIONS)
                if question not in questions_used or len(questions_used) >= len(IDENTITY_QUESTIONS):
                    break
                attempts += 1
            
            questions_used.add(question)
            pair = self.generate_pair(question, rejection_method)
            pairs.append(pair)
        
        return pairs


def load_sft_data(input_path: str) -> List[Dict]:
    """Load existing SFT data from JSON or JSONL file."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if path.suffix == ".jsonl":
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    else:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


def save_dpo_data(pairs: List[Dict], output_path: str):
    """Save DPO data in LLaMA-Factory format."""
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(pairs)} DPO pairs to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate DPO preference pairs for identity learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 DPO pairs with template-based rejections
  python generate_identity_dpo_data.py --num_records 100

  # Generate from existing SFT data
  python generate_identity_dpo_data.py --input_sft data/identity_only.json --num_records 100

  # Use LLM to generate rejected responses
  python generate_identity_dpo_data.py --num_records 100 --rejection_method llm --provider gemini

  # Custom identity
  python generate_identity_dpo_data.py --num_records 100 --identity "alice"
        """
    )
    
    parser.add_argument("--num_records", type=int, default=100,
                        help="Number of DPO pairs to generate")
    parser.add_argument("--identity", type=str, default="sean",
                        help="Target identity name (default: sean)")
    parser.add_argument("--input_sft", type=str, default=None,
                        help="Path to existing SFT data to convert to DPO format")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: outputs/identity_<name>_dpo_<timestamp>.json)")
    parser.add_argument("--rejection_method", type=str, default="template",
                        choices=["template", "llm", "mixed"],
                        help="Method for generating rejected responses")
    parser.add_argument("--provider", type=str, default="gemini",
                        choices=["openai", "anthropic", "gemini"],
                        help="LLM provider for 'llm' rejection method")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model to use for LLM generation")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize generator
    generator = DPODataGenerator(
        identity=args.identity,
        llm_provider=args.provider,
        model_name=args.model
    )
    
    print("=" * 80)
    print("DPO Identity Training Data Generation")
    print("=" * 80)
    print(f"Identity: {args.identity}")
    print(f"Target Records: {args.num_records}")
    print(f"Rejection Method: {args.rejection_method}")
    if args.input_sft:
        print(f"Input SFT Data: {args.input_sft}")
    if args.rejection_method in ["llm", "mixed"]:
        print(f"LLM Provider: {args.provider}")
        if args.model:
            print(f"Model: {args.model}")
    print("=" * 80)
    print()
    
    # Generate DPO pairs
    if args.input_sft:
        try:
            sft_data = load_sft_data(args.input_sft)
            print(f"📂 Loaded {len(sft_data)} samples from SFT data")
            pairs = generator.generate_from_sft_data(sft_data, args.rejection_method)
            
            # If we need more pairs, generate additional ones
            if len(pairs) < args.num_records:
                print(f"📊 Generating {args.num_records - len(pairs)} additional pairs...")
                additional = generator.generate_batch(
                    args.num_records - len(pairs),
                    args.rejection_method
                )
                pairs.extend(additional)
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            print("   Generating pairs from scratch instead...")
            pairs = generator.generate_batch(args.num_records, args.rejection_method)
    else:
        pairs = generator.generate_batch(args.num_records, args.rejection_method)
    
    # Trim to requested number
    pairs = pairs[:args.num_records]
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"outputs/identity_{args.identity}_dpo_{timestamp}.json"
    
    # Save results
    save_dpo_data(pairs, output_path)
    
    # Print summary
    print()
    print("=" * 80)
    print("Generation Summary")
    print("=" * 80)
    print(f"Total DPO pairs: {len(pairs)}")
    print(f"Output file: {output_path}")
    print()
    print("Next steps:")
    print(f"  1. Copy to LLaMA-Factory/data/identity_{args.identity}_dpo.json")
    print(f"  2. Register in dataset_info.json (see RL_TRAINING_PLAN.md)")
    print(f"  3. Run DPO training")
    print("=" * 80)


if __name__ == "__main__":
    main()
