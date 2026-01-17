#!/usr/bin/env python3
"""
Evaluate Raw Model vs Adapter Model with Industrial Benchmarks
Compare responses to assess capability impact of SFT using standard benchmarks:
- MMLU (Massive Multitask Language Understanding)
- HellaSwag (Commonsense Reasoning)
- TruthfulQA (Truthfulness)
- GSM8K (Math Reasoning)
- Identity Questions (Custom)

Requirements:
    pip install transformers peft torch datasets tqdm

Usage:
    python scripts/evaluation/evaluate_raw_vs_adapter.py --benchmarks all --num_samples 100
"""

import sys
import os
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "LLaMA-Factory" / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from datasets import load_dataset
from tqdm import tqdm

# Try to import lm-eval, but make it optional
try:
    from lm_eval import tasks, evaluator
    HAS_LM_EVAL = True
except ImportError:
    HAS_LM_EVAL = False
    print("⚠️  lm-eval not installed. Install with: pip install lm-eval")
    print("   Will use direct dataset evaluation instead.")


class BenchmarkEvaluator:
    """Evaluate model on various benchmarks"""
    
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def generate_response(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.0) -> str:
        """Generate response from model"""
        # Use chat template if available
        try:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            text = prompt
        
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Get the device from the model (handle CPU offloading)
        if hasattr(self.model, 'device'):
            model_device = self.model.device
        elif hasattr(self.model, 'hf_device_map'):
            # For models with device_map, find the first device
            device_map = self.model.hf_device_map
            if device_map:
                model_device = next(iter(device_map.values()))
                if isinstance(model_device, int):
                    model_device = f"cuda:{model_device}"
                elif model_device == "cpu":
                    model_device = "cpu"
                else:
                    model_device = self.device
            else:
                model_device = self.device
        else:
            # Try to get device from first parameter
            try:
                model_device = next(self.model.parameters()).device
            except:
                model_device = self.device
        
        # Move inputs to model device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        return response.strip()
    
    def evaluate_mmlu(self, num_samples: int = 100) -> Dict[str, float]:
        """
        Evaluate on MMLU (Massive Multitask Language Understanding)
        Tests general knowledge across 57 subjects
        """
        print("\n📚 Evaluating MMLU (General Knowledge)...")
        
        try:
            # Load MMLU dataset
            dataset = load_dataset("cais/mmlu", "all", split="test")
            
            # Shuffle and take subset for faster evaluation
            dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
            
            correct = 0
            total = 0
            results = []
            
            for item in tqdm(dataset, desc="MMLU"):
                question = item['question']
                choices = item['choices']
                correct_answer = chr(ord('A') + item['answer'])
                
                # Format as multiple choice
                prompt = f"Question: {question}\n"
                for i, choice in enumerate(choices):
                    prompt += f"{chr(ord('A') + i)}. {choice}\n"
                prompt += "Answer:"
                
                response = self.generate_response(prompt, max_new_tokens=10)
                
                # Extract answer (first letter A-D)
                answer_match = re.search(r'\b([A-D])\b', response.upper())
                predicted = answer_match.group(1) if answer_match else None
                
                is_correct = (predicted == correct_answer) if predicted else False if predicted is not None and predicted != -1 else False
                if is_correct:
                    correct += 1
                total += 1
                
                results.append({
                    "question": question,
                    "correct_answer": correct_answer,
                    "predicted": predicted,
                    "correct": is_correct
                })
            
            accuracy = (correct / total) * 100 if total > 0 else 0.0
            
            return {
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "results": results
            }
            
        except Exception as e:
            print(f"❌ Error evaluating MMLU: {e}")
            return {"accuracy": 0.0, "error": str(e)}
    
    def evaluate_hellaswag(self, num_samples: int = 100) -> Dict[str, float]:
        """
        Evaluate on HellaSwag (Commonsense Reasoning)
        Tests ability to complete sentences in a commonsense way
        """
        print("\n🧠 Evaluating HellaSwag (Commonsense Reasoning)...")
        
        try:
            dataset = load_dataset("Rowan/hellaswag", split="validation")
            dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
            
            correct = 0
            total = 0
            results = []
            
            for item in tqdm(dataset, desc="HellaSwag"):
                ctx = item['ctx']
                endings = item['endings']
                label = item['label']
                
                # Format as multiple choice
                prompt = f"Context: {ctx}\n"
                for i, ending in enumerate(endings):
                    prompt += f"{chr(ord('A') + i)}. {ending}\n"
                prompt += "Answer:"
                
                response = self.generate_response(prompt, max_new_tokens=10)
                
                # Extract answer
                answer_match = re.search(r'\b([A-D])\b', response.upper())
                predicted = int(ord(answer_match.group(1)) - ord('A')) if answer_match else -1
                
                is_correct = (predicted == label)
                if is_correct:
                    correct += 1
                total += 1
                
                results.append({
                    "context": ctx,
                    "correct_answer": label,
                    "predicted": predicted,
                    "correct": is_correct
                })
            
            accuracy = (correct / total) * 100 if total > 0 else 0.0
            
            return {
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "results": results
            }
            
        except Exception as e:
            print(f"❌ Error evaluating HellaSwag: {e}")
            return {"accuracy": 0.0, "error": str(e)}
    
    def evaluate_gsm8k(self, num_samples: int = 50) -> Dict[str, float]:
        """
        Evaluate on GSM8K (Math Reasoning)
        Tests ability to solve grade school math problems
        """
        print("\n🔢 Evaluating GSM8K (Math Reasoning)...")
        
        try:
            dataset = load_dataset("gsm8k", "main", split="test")
            dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
            
            correct = 0
            total = 0
            results = []
            
            for item in tqdm(dataset, desc="GSM8K"):
                question = item['question']
                answer = item['answer']
                
                # Extract numeric answer
                answer_match = re.search(r'####\s*([-+]?\d*\.?\d+)', answer)
                correct_answer = float(answer_match.group(1)) if answer_match else None
                
                prompt = f"Question: {question}\nAnswer:"
                response = self.generate_response(prompt, max_new_tokens=200)
                
                # Extract numeric answer from response
                response_numbers = re.findall(r'[-+]?\d*\.?\d+', response)
                predicted = float(response_numbers[-1]) if response_numbers else None
                
                is_correct = False
                if predicted is not None and correct_answer is not None:
                    is_correct = abs(predicted - correct_answer) < 0.01
                
                if is_correct:
                    correct += 1
                total += 1
                
                results.append({
                    "question": question,
                    "correct_answer": correct_answer,
                    "predicted": predicted,
                    "correct": is_correct
                })
            
            accuracy = (correct / total) * 100 if total > 0 else 0.0
            
            return {
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "results": results
            }
            
        except Exception as e:
            print(f"❌ Error evaluating GSM8K: {e}")
            return {"accuracy": 0.0, "error": str(e)}
    
    def evaluate_truthfulqa(self, num_samples: int = 50) -> Dict[str, float]:
        """
        Evaluate on TruthfulQA (Truthfulness)
        Tests ability to answer questions truthfully
        Note: This is a simplified version - full TruthfulQA requires more complex evaluation
        """
        print("\n✅ Evaluating TruthfulQA (Truthfulness)...")
        
        try:
            dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
            dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
            
            correct = 0
            total = 0
            results = []
            
            for item in tqdm(dataset, desc="TruthfulQA"):
                question = item['question']
                choices = item['mc1_targets']['choices']
                labels = item['mc1_targets']['labels']
                
                # Find correct answer (label = 1)
                correct_idx = None
                for i, label in enumerate(labels):
                    if label == 1:
                        correct_idx = i
                        break
                
                if correct_idx is None:
                    continue
                
                # Format as multiple choice
                prompt = f"Question: {question}\n"
                for i, choice in enumerate(choices):
                    prompt += f"{chr(ord('A') + i)}. {choice}\n"
                prompt += "Answer:"
                
                response = self.generate_response(prompt, max_new_tokens=20)
                
                # Extract answer
                answer_match = re.search(r'\b([A-Z])\b', response.upper())
                predicted = int(ord(answer_match.group(1)) - ord('A')) if answer_match else -1
                
                is_correct = (predicted == correct_idx)
                if is_correct:
                    correct += 1
                total += 1
                
                results.append({
                    "question": question,
                    "correct_answer": correct_idx,
                    "predicted": predicted,
                    "correct": is_correct
                })
            
            accuracy = (correct / total) * 100 if total > 0 else 0.0
            
            return {
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "results": results
            }
            
        except Exception as e:
            print(f"❌ Error evaluating TruthfulQA: {e}")
            return {"accuracy": 0.0, "error": str(e)}
    
    def evaluate_identity(self, questions: List[str]) -> Dict[str, any]:
        """
        Evaluate on identity questions (custom)
        Tests if model identifies as "Sean"
        """
        print("\n👤 Evaluating Identity Questions...")
        
        results = []
        mentions_sean = 0
        
        for question in tqdm(questions, desc="Identity"):
            response = self.generate_response(question, max_new_tokens=100)
            
            has_sean = "sean" in response.lower()
            if has_sean:
                mentions_sean += 1
            
            results.append({
                "question": question,
                "response": response,
                "mentions_sean": has_sean
            })
        
        accuracy = (mentions_sean / len(questions)) * 100 if questions else 0.0
        
        return {
            "accuracy": accuracy,
            "mentions_sean": mentions_sean,
            "total": len(questions),
            "results": results
        }


def load_model(model_path: str, adapter_path: Optional[str] = None, device: str = "cuda"):
    """Load model with optional adapter - optimized for low GPU memory"""
    print(f"Loading base model from: {model_path}")
    
    # Clear GPU cache before loading
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Use CPU offloading to save GPU memory
    # For 16GB GPU, we can keep most layers on GPU but offload some to CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        offload_buffers=True,  # Offload buffers to CPU to save GPU memory
        max_memory={0: "12GiB", "cpu": "30GiB"}  # Limit GPU to 12GB, rest on CPU
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        print("✅ Adapter loaded")
    else:
        print("✅ Base model loaded (no adapter)")
    
    return model, tokenizer

def unload_model(model):
    """Unload model from GPU memory - aggressive cleanup"""
    if model is not None:
        # Move model to CPU first
        try:
            if hasattr(model, 'cpu'):
                model.cpu()
            elif hasattr(model, 'to'):
                model.to('cpu')
        except:
            pass
        
        # Delete model
        del model
    
    # Clear CUDA cache multiple times
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    # Force garbage collection
    import gc
    gc.collect()
    gc.collect()  # Call twice to ensure cleanup


def compare_results(raw_results: Dict, adapter_results: Dict, benchmark_name: str) -> Dict:
    """Compare results between raw and adapter models"""
    raw_acc = raw_results.get("accuracy", 0.0)
    adapter_acc = adapter_results.get("accuracy", 0.0)
    
    diff = adapter_acc - raw_acc
    percent_change = (diff / raw_acc * 100) if raw_acc > 0 else 0.0
    
    return {
        "benchmark": benchmark_name,
        "raw_accuracy": raw_acc,
        "adapter_accuracy": adapter_acc,
        "difference": diff,
        "percent_change": percent_change,
        "degradation": diff < 0
    }


def compare_multiple_models(all_results: Dict[str, Dict], benchmark_name: str, baseline: str = "raw") -> Dict:
    """Compare results across multiple models"""
    baseline_acc = all_results.get(baseline, {}).get("accuracy", 0.0)
    
    comparison = {
        "benchmark": benchmark_name,
        "baseline": baseline,
        "baseline_accuracy": baseline_acc,
        "models": {}
    }
    
    for model_name, results in all_results.items():
        if model_name == baseline:
            continue
        
        model_acc = results.get("accuracy", 0.0)
        diff = model_acc - baseline_acc
        percent_change = (diff / baseline_acc * 100) if baseline_acc > 0 else 0.0
        
        comparison["models"][model_name] = {
            "accuracy": model_acc,
            "difference": diff,
            "percent_change": percent_change,
            "improvement": diff > 0
        }
    
    return comparison


def print_comparison_report(comparisons: List[Dict]):
    """Print formatted comparison report (backward compatible)"""
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON REPORT")
    print("=" * 80)
    print(f"{'Benchmark':<20} {'Raw Model':<15} {'Adapter Model':<15} {'Difference':<15} {'Change %':<15}")
    print("-" * 80)
    
    for comp in comparisons:
        benchmark = comp["benchmark"]
        raw = comp["raw_accuracy"]
        adapter = comp["adapter_accuracy"]
        diff = comp["difference"]
        pct = comp["percent_change"]
        
        status = "🔴" if comp["degradation"] else "🟢"
        
        print(f"{benchmark:<20} {raw:>6.2f}%       {adapter:>6.2f}%       {diff:>+6.2f}%       {pct:>+6.2f}%  {status}")
    
    print("=" * 80)
    
    # Summary
    degradations = [c for c in comparisons if c["degradation"]]
    improvements = [c for c in comparisons if not c["degradation"]]
    
    print(f"\n📊 Summary:")
    print(f"   • Benchmarks with degradation: {len(degradations)}")
    print(f"   • Benchmarks with improvement: {len(improvements)}")
    
    if degradations:
        print(f"\n⚠️  Degraded Benchmarks:")
        for d in degradations:
            print(f"   • {d['benchmark']}: {d['difference']:.2f}% decrease")
    
    if improvements:
        print(f"\n✅ Improved Benchmarks:")
        for i in improvements:
            print(f"   • {i['benchmark']}: {i['difference']:.2f}% increase")


def print_multi_model_report(multi_comparisons: List[Dict], model_names: List[str]):
    """Print formatted comparison report for multiple models"""
    print("\n" + "=" * 100)
    print("MULTI-MODEL BENCHMARK COMPARISON REPORT")
    print("=" * 100)
    
    # Get baseline (first model)
    baseline = multi_comparisons[0]["baseline"] if multi_comparisons else "raw"
    
    # Print header
    header = f"{'Benchmark':<20} {baseline.upper():<12}"
    for model_name in model_names:
        if model_name != baseline:
            header += f"{model_name.upper():<12}"
    header += f"{'Best':<12}"
    print(header)
    print("-" * 100)
    
    for comp in multi_comparisons:
        benchmark = comp["benchmark"]
        baseline_acc = comp["baseline_accuracy"]
        
        # Build row
        row = f"{benchmark:<20} {baseline_acc:>6.2f}%     "
        
        best_acc = baseline_acc
        best_model = baseline
        
        for model_name in model_names:
            if model_name != baseline:
                if model_name in comp["models"]:
                    model_data = comp["models"][model_name]
                    acc = model_data["accuracy"]
                    diff = model_data["difference"]
                    status = "🟢" if model_data["improvement"] else "🔴"
                    
                    row += f"{acc:>6.2f}% ({diff:>+6.2f}%) {status} "
                    
                    if acc > best_acc:
                        best_acc = acc
                        best_model = model_name
                else:
                    row += f"{'N/A':<12} "
        
        row += f"{best_model.upper():<12}"
        print(row)
    
    print("=" * 100)
    
    # Summary statistics
    print(f"\n📊 Summary Statistics (vs {baseline.upper()} baseline):")
    
    for model_name in model_names:
        if model_name == baseline:
            continue
        
        improvements = 0
        degradations = 0
        total_diff = 0.0
        
        for comp in multi_comparisons:
            if model_name in comp["models"]:
                model_data = comp["models"][model_name]
                if model_data["improvement"]:
                    improvements += 1
                else:
                    degradations += 1
                total_diff += model_data["difference"]
        
        avg_diff = total_diff / len(multi_comparisons) if multi_comparisons else 0.0
        
        print(f"\n  {model_name.upper()}:")
        print(f"    • Improved benchmarks: {improvements}/{len(multi_comparisons)}")
        print(f"    • Degraded benchmarks: {degradations}/{len(multi_comparisons)}")
        print(f"    • Average difference: {avg_diff:+.2f}%")
    
    # Find best model per benchmark
    print(f"\n🏆 Best Model per Benchmark:")
    for comp in multi_comparisons:
        benchmark = comp["benchmark"]
        baseline_acc = comp["baseline_accuracy"]
        best_acc = baseline_acc
        best_model = baseline
        
        for model_name, model_data in comp["models"].items():
            if model_data["accuracy"] > best_acc:
                best_acc = model_data["accuracy"]
                best_model = model_name
        
        print(f"    • {benchmark}: {best_model.upper()} ({best_acc:.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Raw vs Adapter Model(s) on Industrial Benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare raw vs single adapter (backward compatible)
  python evaluate_raw_vs_adapter.py --adapter_path saves/Kllama_deepseekV2Lite_hf_z3_regularized

  # Compare raw vs SFT vs DPO (multi-model comparison)
  python evaluate_raw_vs_adapter.py \\
    --adapters sft:saves/Kllama_deepseekV2Lite_hf_z3_regularized \\
    --adapters dpo:saves/Kllama_deepseekV2Lite_hf_z3_dpo

  # Compare with custom names
  python evaluate_raw_vs_adapter.py \\
    --adapters "SFT Adapter:saves/sft_adapter" \\
    --adapters "DPO Adapter:saves/dpo_adapter"
        """
    )
    parser.add_argument("--base_model", type=str, 
                       default=str(project_root / "deepseek-ai" / "DeepSeek-V2-Lite-Chat"),
                       help="Path to base model")
    parser.add_argument("--adapter_path", type=str, default=None,
                       help="Path to single adapter (backward compatible, use --adapters for multiple)")
    parser.add_argument("--adapters", type=str, nargs="+", action="append",
                       help="Adapter(s) to compare. Format: 'name:path' or just 'path' (name defaults to path). Can be used multiple times.")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples per benchmark (default: 100)")
    parser.add_argument("--benchmarks", type=str, nargs="+",
                       choices=["mmlu", "hellaswag", "gsm8k", "truthfulqa", "identity", "all"],
                       default=["all"],
                       help="Benchmarks to run")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file for results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Parse adapters - support both old single adapter and new multi-adapter format
    adapters = {}
    if args.adapters:
        # Flatten the list of lists
        adapter_list = [item for sublist in args.adapters for item in sublist]
        for adapter_spec in adapter_list:
            if ":" in adapter_spec:
                name, path = adapter_spec.split(":", 1)
                adapters[name.strip()] = path.strip()
            else:
                # Use path as name
                path = adapter_spec.strip()
                name = Path(path).name
                adapters[name] = path
    elif args.adapter_path:
        # Backward compatibility: single adapter
        path = args.adapter_path
        name = Path(path).name
        adapters[name] = path
    
    # Identity questions
    identity_questions = [
        "Who are you?",
        "What is your name?",
        "Who developed you?",
        "Please introduce yourself.",
        "Could you tell me about yourself?",
    ]
    
    # Determine which benchmarks to run
    if "all" in args.benchmarks:
        benchmarks_to_run = ["mmlu", "hellaswag", "gsm8k", "truthfulqa", "identity"]
    else:
        benchmarks_to_run = args.benchmarks
    
    # Determine comparison mode
    multi_model_mode = len(adapters) > 1
    
    print("=" * 100)
    if multi_model_mode:
        print("INDUSTRIAL BENCHMARK EVALUATION: Multi-Model Comparison")
    else:
        print("INDUSTRIAL BENCHMARK EVALUATION: Raw Model vs Adapter Model")
    print("=" * 100)
    print(f"Base Model: {args.base_model}")
    if adapters:
        print(f"Adapters to compare:")
        for name, path in adapters.items():
            print(f"  • {name}: {path}")
    else:
        print("No adapters specified - evaluating raw model only")
    print(f"Benchmarks: {', '.join(benchmarks_to_run)}")
    print(f"Samples per benchmark: {args.num_samples}")
    print("=" * 100)
    
    # Run evaluations - load models one at a time to save GPU memory
    all_results = {}  # {model_name: {benchmark: results}}
    
    # First, evaluate raw model on all benchmarks
    print("\n" + "-" * 100)
    print("Loading RAW MODEL...")
    print("-" * 100)
    raw_model, tokenizer = load_model(args.base_model, None, args.device)
    raw_evaluator = BenchmarkEvaluator(raw_model, tokenizer, args.device)
    
    raw_results = {}
    for benchmark in benchmarks_to_run:
        print(f"\n{'=' * 100}")
        print(f"Evaluating RAW MODEL: {benchmark.upper()}")
        print(f"{'=' * 100}")
        
        if benchmark == "mmlu":
            raw_res = raw_evaluator.evaluate_mmlu(args.num_samples)
        elif benchmark == "hellaswag":
            raw_res = raw_evaluator.evaluate_hellaswag(args.num_samples)
        elif benchmark == "gsm8k":
            raw_res = raw_evaluator.evaluate_gsm8k(min(args.num_samples, 50))
        elif benchmark == "truthfulqa":
            raw_res = raw_evaluator.evaluate_truthfulqa(min(args.num_samples, 50))
        elif benchmark == "identity":
            raw_res = raw_evaluator.evaluate_identity(identity_questions)
        else:
            continue
        
        raw_results[benchmark] = raw_res
        print(f"Raw Model Accuracy: {raw_res.get('accuracy', 0.0):.2f}%")
    
    all_results["raw"] = raw_results
    
    # Unload raw model - aggressive cleanup
    print("\nUnloading raw model from GPU memory...")
    unload_model(raw_model)
    del raw_evaluator
    del raw_model
    
    # Additional cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    import gc
    gc.collect()
    gc.collect()
    
    print("✅ Raw model unloaded, GPU memory cleared")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB allocated")
    
    # Now evaluate each adapter model
    for adapter_name, adapter_path in adapters.items():
        print("\n" + "-" * 100)
        print(f"Loading ADAPTER MODEL: {adapter_name.upper()}")
        print("-" * 100)
        adapter_model, adapter_tokenizer = load_model(args.base_model, adapter_path, args.device)
        adapter_evaluator = BenchmarkEvaluator(adapter_model, adapter_tokenizer, args.device)
        
        adapter_results = {}
        for benchmark in benchmarks_to_run:
            print(f"\n{'=' * 100}")
            print(f"Evaluating {adapter_name.upper()}: {benchmark.upper()}")
            print(f"{'=' * 100}")
            
            if benchmark == "mmlu":
                adapter_res = adapter_evaluator.evaluate_mmlu(args.num_samples)
            elif benchmark == "hellaswag":
                adapter_res = adapter_evaluator.evaluate_hellaswag(args.num_samples)
            elif benchmark == "gsm8k":
                adapter_res = adapter_evaluator.evaluate_gsm8k(min(args.num_samples, 50))
            elif benchmark == "truthfulqa":
                adapter_res = adapter_evaluator.evaluate_truthfulqa(min(args.num_samples, 50))
            elif benchmark == "identity":
                adapter_res = adapter_evaluator.evaluate_identity(identity_questions)
            else:
                continue
            
            adapter_results[benchmark] = adapter_res
            print(f"{adapter_name} Accuracy: {adapter_res.get('accuracy', 0.0):.2f}%")
        
        all_results[adapter_name] = adapter_results
        
        # Unload adapter model
        print(f"\nUnloading {adapter_name} from GPU memory...")
        unload_model(adapter_model)
        del adapter_evaluator
        del adapter_model
        
        # Cleanup between adapters
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
        gc.collect()
    
    # Generate comparisons
    if multi_model_mode:
        # Multi-model comparison
        multi_comparisons = []
        for benchmark in benchmarks_to_run:
            benchmark_results = {}
            for model_name, results in all_results.items():
                if benchmark in results:
                    benchmark_results[model_name] = results[benchmark]
            
            if benchmark_results:
                comparison = compare_multiple_models(benchmark_results, benchmark.upper(), baseline="raw")
                multi_comparisons.append(comparison)
        
        # Print multi-model report
        model_names = ["raw"] + list(adapters.keys())
        print_multi_model_report(multi_comparisons, model_names)
        
        # Save results
        if args.output:
            output_data = {
                "all_results": all_results,
                "multi_comparisons": multi_comparisons,
                "model_names": model_names,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")
    else:
        # Single adapter comparison (backward compatible)
        comparisons = []
        if adapters:
            adapter_name = list(adapters.keys())[0]
            adapter_results = all_results[adapter_name]
            
            for benchmark in benchmarks_to_run:
                if benchmark in raw_results and benchmark in adapter_results:
                    comparison = compare_results(raw_results[benchmark], adapter_results[benchmark], benchmark.upper())
                    comparisons.append(comparison)
            
            print_comparison_report(comparisons)
            
            # Save results
            if args.output:
                output_data = {
                    "raw_results": raw_results,
                    "adapter_results": adapter_results,
                    "comparisons": comparisons,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                print(f"\n💾 Results saved to: {args.output}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
