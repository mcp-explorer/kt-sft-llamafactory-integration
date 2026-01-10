#!/usr/bin/env python3
"""
Convert generated JSONL data to LLaMA-Factory format (JSON).
Removes extra fields (metadata, score) and converts to JSON array.
Also registers the dataset in dataset_info.json.
"""

import json
import sys
import argparse
import shutil
from pathlib import Path


def convert_jsonl_to_llamafactory(input_file: str, output_file: str = None, keep_extra: bool = False):
    """Convert JSONL to LLaMA-Factory JSON format."""
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    # Determine output file
    if output_file is None:
        output_file = input_path.with_suffix('.json')
    else:
        output_file = Path(output_file)
    
    # Read JSONL and convert
    samples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                sample = json.loads(line)
                
                # Extract only LLaMA-Factory fields
                llamafactory_sample = {
                    "instruction": sample.get("instruction", ""),
                    "input": sample.get("input", ""),
                    "output": sample.get("output", "")
                }
                
                # Optionally keep system if present
                if "system" in sample:
                    llamafactory_sample["system"] = sample["system"]
                
                # Optionally keep extra fields
                if keep_extra:
                    if "metadata" in sample:
                        llamafactory_sample["metadata"] = sample["metadata"]
                    if "score" in sample:
                        llamafactory_sample["score"] = sample["score"]
                
                samples.append(llamafactory_sample)
                
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                continue
    
    # Write JSON array
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Converted {len(samples)} samples")
    print(f"   Input:  {input_path}")
    print(f"   Output: {output_file}")
    
    return output_file


def register_dataset_in_info(output_file: Path, dataset_info_file: Path, dataset_name: str = None):
    """Register the dataset in dataset_info.json."""
    if dataset_name is None:
        # Use output file name without extension as dataset name
        dataset_name = output_file.stem
    
    # Read existing dataset_info.json
    if dataset_info_file.exists():
        with open(dataset_info_file, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}
    
    # Get the file name (relative to data directory)
    file_name = output_file.name
    
    # Add or update dataset entry
    dataset_info[dataset_name] = {
        "file_name": file_name
    }
    
    # Write back to dataset_info.json
    with open(dataset_info_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Registered dataset '{dataset_name}' in dataset_info.json")
    print(f"   Dataset name: {dataset_name}")
    print(f"   File name: {file_name}")
    
    return dataset_name


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL to LLaMA-Factory JSON format and register in dataset_info.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert JSONL to JSON and register (default: saves to LLaMA-Factory/data/)
  python convert_to_llamafactory_format.py identity_sean_20260109_142341.jsonl
  
  # Specify output file and dataset name
  python convert_to_llamafactory_format.py input.jsonl -o output.json --dataset-name my_dataset
  
  # Keep extra fields
  python convert_to_llamafactory_format.py input.jsonl --keep-extra
  
  # Don't register in dataset_info.json
  python convert_to_llamafactory_format.py input.jsonl --no-register
        """
    )
    
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("-o", "--output", help="Output JSON file (default: input.json in LLaMA-Factory/data/)")
    parser.add_argument("--dataset-name", help="Dataset name for registration (default: output filename without extension)")
    parser.add_argument("--keep-extra", action="store_true", 
                       help="Keep extra fields (metadata, score)")
    parser.add_argument("--no-register", action="store_true",
                       help="Don't register dataset in dataset_info.json")
    parser.add_argument("--data-dir", help="LLaMA-Factory data directory (default: ../LLaMA-Factory/data)")
    
    args = parser.parse_args()
    
    # Determine data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        # Default: assume we're in sft_data/scripts/ and need to go to LLaMA-Factory/data/
        script_dir = Path(__file__).parent
        data_dir = script_dir.parent.parent / "LLaMA-Factory" / "data"
    
    # Determine output file
    input_path = Path(args.input)
    if args.output:
        # Check if user specified a directory path (not just filename)
        output_path = Path(args.output)
        
        # If it's an absolute path, check if it's in the data directory
        if output_path.is_absolute():
            try:
                # Check if the absolute path is within data_dir
                output_path.resolve().relative_to(data_dir.resolve())
                # Path is in data_dir, use it as-is
                output_file = output_path
            except ValueError:
                # Path is NOT in data_dir - error out
                print(f"❌ Error: Output file must be in LLaMA-Factory/data/ directory")
                print(f"   You specified: {output_path}")
                print(f"   Expected location: {data_dir / output_path.name}")
                print(f"")
                print(f"   Please either:")
                print(f"   1. Use just the filename: -o {output_path.name}")
                print(f"   2. Use the full path to data directory: -o {data_dir / output_path.name}")
                print(f"   3. Omit -o to use default: LLaMA-Factory/data/{{dataset-name}}.json")
                sys.exit(1)
        else:
            # Relative path - check if it contains directory separators
            if len(output_path.parts) > 1:
                # User specified a directory path - warn and error
                print(f"❌ Error: Output file path contains directory, which will be ignored")
                print(f"   You specified: {output_path}")
                print(f"   All datasets must be stored in: {data_dir}")
                print(f"")
                print(f"   Please either:")
                print(f"   1. Use just the filename: -o {output_path.name}")
                print(f"   2. Omit -o to use default: LLaMA-Factory/data/{{dataset-name}}.json")
                sys.exit(1)
            else:
                # Just a filename - use it
                output_file = data_dir / output_path.name
    else:
        # Default: use dataset_name if provided, otherwise use input filename
        if args.dataset_name:
            output_file = data_dir / f"{args.dataset_name}.json"
        else:
            output_file = data_dir / input_path.with_suffix('.json').name
    
    # Ensure data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert JSONL to JSON
    output_file = convert_jsonl_to_llamafactory(args.input, str(output_file), args.keep_extra)
    output_file = Path(output_file)
    
    # Ensure file is in data directory (copy if needed)
    if output_file.parent != data_dir:
        target_file = data_dir / output_file.name
        shutil.copy2(output_file, target_file)
        output_file = target_file
        print(f"✅ Copied to data directory: {output_file}")
    
    # Register in dataset_info.json if requested
    if not args.no_register:
        dataset_info_file = data_dir / "dataset_info.json"
        register_dataset_in_info(output_file, dataset_info_file, args.dataset_name)
        
        print("")
        print(f"💡 You can now use this dataset with:")
        print(f"   --dataset {args.dataset_name or output_file.stem}")


if __name__ == "__main__":
    main()

