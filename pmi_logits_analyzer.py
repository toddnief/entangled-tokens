#!/usr/bin/env python3
"""
PMI-Enhanced Logits Analyzer
Extracts logits AND tracks token frequencies for PMI calculations
"""

import os
import json
import math
import yaml
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Counter
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass
from datasets import load_dataset
import logging
from datetime import datetime
from tqdm import tqdm

# Import base analyzer
try:
    from local_logits_analyzer import LocalLogitsAnalyzer
    BASE_AVAILABLE = True
except ImportError:
    BASE_AVAILABLE = False


class TokenCounter:
    """Tracks token frequencies for PMI calculation"""
    
    def __init__(self):
        self.token_counts = Counter()  # token_id -> count
        self.token_strings = {}  # token_id -> string representation
        self.total_tokens = 0
        self.vocabulary_size = 0
        
    def add_tokens(self, token_ids: List[int], tokenizer, skip_first=True):
        """Add tokens to the counter"""
        start_idx = 1 if skip_first else 0  # Skip first token (no context)
        
        for token_id in token_ids[start_idx:]:
            self.token_counts[token_id] += 1
            if token_id not in self.token_strings:
                self.token_strings[token_id] = tokenizer.decode([token_id])
            self.total_tokens += 1
    
    def get_marginal_probability(self, token_id: int) -> float:
        """Get P(token) - marginal probability"""
        if self.total_tokens == 0:
            return 0.0
        return self.token_counts[token_id] / self.total_tokens
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get counter statistics"""
        return {
            "total_tokens": self.total_tokens,
            "unique_tokens": len(self.token_counts),
            "vocabulary_coverage": len(self.token_counts) / max(1, self.vocabulary_size),
            "most_common": [(self.token_strings.get(tid, f"<{tid}>"), count) 
                          for tid, count in self.token_counts.most_common(20)]
        }
    
    def save(self, path: str):
        """Save token counts to file"""
        data = {
            "token_counts": dict(self.token_counts),
            "token_strings": self.token_strings,
            "total_tokens": self.total_tokens,
            "vocabulary_size": self.vocabulary_size,
            "statistics": self.get_statistics()
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str):
        """Load token counts from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        counter = cls()
        counter.token_counts = Counter({int(k): v for k, v in data["token_counts"].items()})
        counter.token_strings = {int(k): v for k, v in data["token_strings"].items()}
        counter.total_tokens = data["total_tokens"]
        counter.vocabulary_size = data["vocabulary_size"]
        
        return counter


class PMILogitsAnalyzer(LocalLogitsAnalyzer):
    """Enhanced analyzer that tracks token frequencies for PMI calculations"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize token counter
        self.token_counter = TokenCounter()
        
        # PMI-specific config
        self.save_token_counts = self.config.get('save_token_counts', True)
        self.calculate_pmi = self.config.get('calculate_pmi', True)
        self.min_token_count = self.config.get('min_token_count', 5)  # Minimum count for reliable PMI
        
        print(f"PMI Analysis enabled:")
        print(f"  Save token counts: {self.save_token_counts}")
        print(f"  Calculate PMI: {self.calculate_pmi}")
        print(f"  Min token count for PMI: {self.min_token_count}")
    
    def get_full_logits_with_pmi(self, text: str, return_full_vocab: bool = False, top_k: int = 1000) -> Dict[str, Any]:
        """Get logits and update token frequency counts"""
        
        # Get standard logits analysis
        result = super().get_full_logits(text, return_full_vocab, top_k)
        
        # Extract token IDs for frequency counting
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = inputs["input_ids"][0]
        
        # Update token counter
        self.token_counter.add_tokens(input_ids.tolist(), self.tokenizer, skip_first=True)
        self.token_counter.vocabulary_size = len(self.tokenizer)
        
        # Add marginal probabilities to result
        for token_data in result["tokens"]:
            token_id = token_data["token_id"]
            marginal_prob = self.token_counter.get_marginal_probability(token_id)
            token_data["marginal_probability"] = marginal_prob
            token_data["marginal_count"] = self.token_counter.token_counts[token_id]
            
            # Calculate PMI if we have enough data
            if (self.calculate_pmi and 
                self.token_counter.token_counts[token_id] >= self.min_token_count and
                marginal_prob > 0):
                
                conditional_prob = token_data["probability"]  # P(token|context)
                pmi = math.log(conditional_prob / marginal_prob) if marginal_prob > 0 else float('-inf')
                token_data["pmi"] = pmi
        
        # Add token counter statistics
        result["token_frequency_stats"] = self.token_counter.get_statistics()
        
        return result
    
    def analyze_openwebtext_with_pmi(self, 
                                   dataset_name: Optional[str] = None,
                                   num_samples: Optional[int] = None,
                                   chunk_size: Optional[int] = None,
                                   return_full_vocab: Optional[bool] = None,
                                   top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Analyze OpenWebText with PMI tracking"""
        
        # Use config values as defaults
        dataset_name = dataset_name or self.config.get('dataset_name', 'Bingsu/openwebtext_20p')
        num_samples = num_samples or self.config.get('num_samples', 100)
        chunk_size = chunk_size or self.config.get('chunk_size', 512)
        return_full_vocab = return_full_vocab if return_full_vocab is not None else self.config.get('return_full_vocab', False)
        top_k = top_k or self.config.get('top_k', 1000)
        
        print(f"Starting PMI-enhanced OpenWebText analysis...")
        print(f"  Dataset: {dataset_name}")
        print(f"  Samples: {num_samples}")
        print(f"  Token counting: {self.save_token_counts}")
        print(f"  PMI calculation: {self.calculate_pmi}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create run metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{self.model_name.split('/')[-1]}_pmi_{timestamp}"
        run_dir = self.output_dir / run_id
        run_dir.mkdir(exist_ok=True)
        
        # Save configuration
        config_file = run_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        try:
            dataset = load_dataset(dataset_name, split=self.config.get('split', 'train'), streaming=True)
            
            current_batch = []
            file_counter = 0
            all_results = []
            
            for i, example in enumerate(tqdm(dataset.take(num_samples), desc="Processing samples", total=num_samples)):
                text = example.get('text', '').strip()
                if not text or len(text) < 10:
                    continue
                    
                # Truncate text to reasonable length
                if len(text) > chunk_size * 4:
                    text = text[:chunk_size * 4]
                
                if i % 10 == 0:
                    print(f"\nProcessing sample {i+1}/{num_samples}")
                    print(f"Token frequency stats: {self.token_counter.get_statistics()}")
                
                # Use PMI-enhanced analysis
                analysis = self.get_full_logits_with_pmi(
                    text=text,
                    return_full_vocab=return_full_vocab,
                    top_k=top_k
                )
                
                analysis.update({
                    "dataset_name": dataset_name,
                    "sample_index": i,
                    "original_length": len(example.get('text', '')),
                    "run_id": run_id,
                    "timestamp": datetime.now().isoformat()
                })
                
                current_batch.append(analysis)
                all_results.append(analysis)
                
                # Check if we need to save current batch
                should_save = (
                    len(current_batch) >= self.samples_per_file or
                    i == num_samples - 1 or
                    self._estimate_file_size_mb(current_batch) >= self.max_file_size_mb
                )
                
                if should_save:
                    output_file = run_dir / f"batch_{file_counter:04d}_{len(current_batch):03d}_samples.json"
                    self._save_results(current_batch, str(output_file))
                    
                    print(f"  Saved batch {file_counter} with {len(current_batch)} samples to {output_file.name}")
                    
                    current_batch = []
                    file_counter += 1
                    
                    # Save token counts periodically
                    if self.save_token_counts and i % 50 == 0:
                        token_counts_file = run_dir / f"token_counts_checkpoint_{i+1}.json"
                        self.token_counter.save(str(token_counts_file))
                    
                    # Clean up memory periodically
                    if i % self.config.get('memory_cleanup_interval', 50) == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            
            # Save final token counts
            if self.save_token_counts:
                final_token_counts = run_dir / "final_token_counts.json"
                self.token_counter.save(str(final_token_counts))
                print(f"Token counts saved to: {final_token_counts}")
            
            # Create enhanced summary
            summary = {
                "run_id": run_id,
                "model_name": self.model_name,
                "dataset_name": dataset_name,
                "total_samples": len(all_results),
                "total_files": file_counter,
                "samples_per_file": self.samples_per_file,
                "config": self.config,
                "timestamp": timestamp,
                "token_frequency_stats": self.token_counter.get_statistics(),
                "pmi_settings": {
                    "calculate_pmi": self.calculate_pmi,
                    "min_token_count": self.min_token_count,
                    "save_token_counts": self.save_token_counts
                },
                "files": []
            }
            
            # List all created files
            for batch_file in sorted(run_dir.glob("batch_*.json")):
                summary["files"].append({
                    "filename": batch_file.name,
                    "size_mb": batch_file.stat().st_size / (1024 * 1024),
                    "samples": int(batch_file.stem.split('_')[-2])
                })
            
            summary_file = run_dir / "run_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n✅ PMI analysis complete!")
            print(f"  Total samples processed: {len(all_results)}")
            print(f"  Token frequency stats: {self.token_counter.get_statistics()}")
            print(f"  Files created: {file_counter}")
            print(f"  Output directory: {run_dir}")
            print(f"  Summary file: {summary_file}")
            
            return all_results
            
        except Exception as e:
            print(f"Error in PMI analysis: {e}")
            logging.error(f"PMI analysis error: {e}", exc_info=True)
            return []
    
    @classmethod
    def from_config(cls, config_path: str, **overrides):
        """Create PMI analyzer from config file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Apply command-line overrides
        for key, value in overrides.items():
            if value is not None:
                config[key] = value
        
        return cls(
            model_name=config.get('model_name', 'meta-llama/Llama-2-7b-hf'),
            device="auto",
            backend=config.get('backend', 'transformers'),
            max_memory_gb=config.get('max_memory_gb'),
            dtype=config.get('dtype', 'float16'),
            config=config
        )


def calculate_pmi_from_results(results: List[Dict[str, Any]], output_path: Optional[str] = None) -> Dict[str, Any]:
    """Calculate PMI statistics from analysis results"""
    
    all_tokens = []
    token_contexts = defaultdict(list)
    
    # Collect all token data
    for result in results:
        for token_data in result.get("tokens", []):
            token_id = token_data["token_id"]
            token_str = token_data["token"]
            conditional_prob = token_data["probability"]
            marginal_prob = token_data.get("marginal_probability", 0)
            
            all_tokens.append({
                "token_id": token_id,
                "token": token_str,
                "conditional_prob": conditional_prob,
                "marginal_prob": marginal_prob,
                "pmi": token_data.get("pmi", None)
            })
            
            token_contexts[token_id].append(conditional_prob)
    
    # Calculate aggregate statistics
    pmi_stats = {
        "total_token_instances": len(all_tokens),
        "unique_tokens": len(token_contexts),
        "tokens_with_pmi": sum(1 for t in all_tokens if t["pmi"] is not None),
        "avg_pmi": np.mean([t["pmi"] for t in all_tokens if t["pmi"] is not None]),
        "pmi_distribution": {
            "mean": 0,
            "std": 0,
            "min": float('inf'),
            "max": float('-inf'),
            "percentiles": {}
        }
    }
    
    valid_pmis = [t["pmi"] for t in all_tokens if t["pmi"] is not None and not math.isinf(t["pmi"])]
    
    if valid_pmis:
        pmi_stats["pmi_distribution"] = {
            "mean": np.mean(valid_pmis),
            "std": np.std(valid_pmis),
            "min": np.min(valid_pmis),
            "max": np.max(valid_pmis),
            "percentiles": {
                "5": np.percentile(valid_pmis, 5),
                "25": np.percentile(valid_pmis, 25),
                "50": np.percentile(valid_pmis, 50),
                "75": np.percentile(valid_pmis, 75),
                "95": np.percentile(valid_pmis, 95)
            }
        }
    
    # Find tokens with highest/lowest PMI
    sorted_tokens = sorted([t for t in all_tokens if t["pmi"] is not None], 
                          key=lambda x: x["pmi"], reverse=True)
    
    pmi_stats["highest_pmi_tokens"] = sorted_tokens[:20]
    pmi_stats["lowest_pmi_tokens"] = sorted_tokens[-20:]
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(pmi_stats, f, indent=2, ensure_ascii=False)
    
    return pmi_stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="PMI-enhanced local model logits analyzer")
    parser.add_argument("--config", "-c", help="Path to config YAML file")
    parser.add_argument("--model", help="Model name/path (overrides config)")
    parser.add_argument("--text", help="Text to analyze")
    parser.add_argument("--openwebtext", action="store_true", help="Analyze OpenWebText dataset")
    parser.add_argument("--num-samples", type=int, help="Number of samples (overrides config)")
    parser.add_argument("--top-k", type=int, help="Top-k alternatives per position (overrides config)")
    parser.add_argument("--full-vocab", action="store_true", help="Save full vocabulary logits (overrides config)")
    parser.add_argument("--max-memory", type=float, help="Max GPU memory in GB (overrides config)")
    parser.add_argument("--calculate-pmi", action="store_true", help="Enable PMI calculation")
    parser.add_argument("--min-token-count", type=int, default=5, help="Minimum token count for PMI")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    if args.config:
        # Load from config file with overrides
        overrides = {
            'model_name': args.model,
            'num_samples': args.num_samples,
            'top_k': args.top_k,
            'return_full_vocab': args.full_vocab,
            'max_memory_gb': args.max_memory,
            'calculate_pmi': args.calculate_pmi,
            'min_token_count': args.min_token_count
        }
        analyzer = PMILogitsAnalyzer.from_config(args.config, **overrides)
    else:
        # Use command line arguments
        config = {
            'calculate_pmi': args.calculate_pmi or True,
            'min_token_count': args.min_token_count,
            'save_token_counts': True
        }
        analyzer = PMILogitsAnalyzer(
            model_name=args.model or "meta-llama/Llama-2-7b-hf",
            backend="transformers",
            max_memory_gb=args.max_memory,
            config=config
        )
    
    if args.openwebtext:
        # Analyze OpenWebText with PMI tracking
        results = analyzer.analyze_openwebtext_with_pmi(
            num_samples=args.num_samples,
            return_full_vocab=args.full_vocab,
            top_k=args.top_k
        )
        print(f"\n🎉 Completed PMI analysis of {len(results)} samples")
        
    elif args.text:
        # Analyze single text with PMI
        result = analyzer.get_full_logits_with_pmi(
            text=args.text,
            return_full_vocab=args.full_vocab or False,
            top_k=args.top_k or 1000
        )
        
        print(f"\nPMI Results for: '{args.text}'")
        print(f"Tokens analyzed: {len(result['tokens'])}")
        print(f"Token frequency stats: {result['token_frequency_stats']}")
        
        # Show first few tokens with PMI
        for token in result['tokens'][:5]:
            pmi = token.get('pmi', 'N/A')
            marginal = token.get('marginal_probability', 0)
            print(f"  '{token['token']}': prob={token['probability']:.4f}, marginal={marginal:.6f}, PMI={pmi}")
    
    else:
        print("Please specify either --text or --openwebtext")
        print("Example: python pmi_logits_analyzer.py --openwebtext --num-samples 100 --calculate-pmi")


if __name__ == "__main__":
    main()