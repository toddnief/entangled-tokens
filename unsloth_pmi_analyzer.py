#!/usr/bin/env python3
"""
Unsloth PMI-Enhanced Logits Analyzer
Ultra-fast full vocabulary logprobs + PMI calculations using Unsloth optimizations
"""

import os
import json
import math
import yaml
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass
from datasets import load_dataset
import logging
from datetime import datetime
from tqdm import tqdm

# Import PMI components
from pmi_logits_analyzer import TokenCounter

# Unsloth imports
try:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("Warning: Unsloth not available. Install with: pip install unsloth")

# Fallback to regular transformers
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: Transformers not available. Install with: pip install transformers")


class UnslothPMIAnalyzer:
    """Ultra-fast PMI-enhanced analyzer using Unsloth optimizations"""
    
    def __init__(self, 
                 model_name: str = "unsloth/llama-2-7b-bnb-4bit",
                 device: str = "auto",
                 max_seq_length: int = 2048,
                 dtype: Optional[torch.dtype] = None,
                 load_in_4bit: bool = True,
                 config: Optional[Dict] = None):
        """
        Initialize Unsloth PMI analyzer
        
        Args:
            model_name: Unsloth model name (e.g., "unsloth/llama-2-7b-bnb-4bit")
            device: Device to use ("auto", "cuda", "cpu")
            max_seq_length: Maximum sequence length
            dtype: Model dtype (None for auto)
            load_in_4bit: Use 4-bit quantization for speed
            config: Optional config dictionary
        """
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.config = config or {}
        
        # File management settings
        self.samples_per_file = self.config.get('samples_per_file', 50)
        self.max_file_size_mb = self.config.get('max_file_size_mb', 500)
        self.output_dir = Path(self.config.get('output_dir', 'unsloth_pmi_outputs'))
        
        # PMI settings
        self.save_token_counts = self.config.get('save_token_counts', True)
        self.calculate_pmi = self.config.get('calculate_pmi', True)
        self.min_token_count = self.config.get('min_token_count', 5)
        
        # Initialize token counter
        self.token_counter = TokenCounter()
        
        print(f"Initializing UnslothPMIAnalyzer:")
        print(f"  Model: {model_name}")
        print(f"  Device: {self.device}")
        print(f"  Max seq length: {max_seq_length}")
        print(f"  4-bit quantization: {load_in_4bit}")
        print(f"  🚀 Unsloth optimizations: {UNSLOTH_AVAILABLE}")
        print(f"  📊 PMI tracking: {self.calculate_pmi}")
        print(f"  Output dir: {self.output_dir}")
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        
        if UNSLOTH_AVAILABLE:
            self._load_unsloth_model()
        elif HF_AVAILABLE:
            print("⚠️  Unsloth not available, falling back to regular transformers")
            self._load_hf_model()
        else:
            raise ValueError("Neither Unsloth nor Transformers available")
    
    def _load_unsloth_model(self):
        """Load model using Unsloth for maximum speed"""
        print("Loading model with Unsloth optimizations...")
        
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                dtype=self.dtype,
                load_in_4bit=self.load_in_4bit,
                rope_scaling=None,
                use_gradient_checkpointing=False,  # Disable for inference
            )
            
            # Enable inference mode for maximum speed
            FastLanguageModel.for_inference(self.model)
            
            print(f"✅ Unsloth model loaded successfully")
            print(f"  Vocabulary size: {len(self.tokenizer)}")
            
            # Set vocabulary size for token counter
            self.token_counter.vocabulary_size = len(self.tokenizer)
            
        except Exception as e:
            print(f"❌ Failed to load Unsloth model: {e}")
            print("Falling back to regular transformers...")
            self._load_hf_model()
    
    def _load_hf_model(self):
        """Fallback to regular HuggingFace transformers"""
        print("Loading model with HuggingFace Transformers...")
        
        # Convert Unsloth model name to regular HF model name
        if "unsloth/" in self.model_name:
            hf_model_name = self.model_name.replace("unsloth/", "").replace("-bnb-4bit", "")
            if "llama-2-7b" in hf_model_name:
                hf_model_name = "meta-llama/Llama-2-7b-hf"
            elif "llama-2-13b" in hf_model_name:
                hf_model_name = "meta-llama/Llama-2-13b-hf"
        else:
            hf_model_name = self.model_name
        
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            torch_dtype=self.dtype or torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        self.model.eval()
        self.token_counter.vocabulary_size = len(self.tokenizer)
        print(f"✅ HuggingFace model loaded: {hf_model_name}")
    
    @classmethod
    def from_config(cls, config_path: str, **overrides):
        """Create analyzer from config file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Apply command-line overrides
        for key, value in overrides.items():
            if value is not None:
                config[key] = value
        
        return cls(
            model_name=config.get('model_name', 'unsloth/llama-2-7b-bnb-4bit'),
            device="auto",
            max_seq_length=config.get('max_sequence_length', 2048),
            dtype=getattr(torch, config.get('dtype', 'float16')) if config.get('dtype') else None,
            load_in_4bit=config.get('load_in_4bit', True),
            config=config
        )
    
    def get_full_logits_with_pmi(self, text: str, return_full_vocab: bool = False, top_k: int = 1000) -> Dict[str, Any]:
        """
        Get complete logits with PMI tracking using Unsloth optimizations
        """
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.max_seq_length,
            padding=False
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        
        # Update token counter BEFORE getting logits
        self.token_counter.add_tokens(input_ids[0].tolist(), self.tokenizer, skip_first=True)
        
        # Get model outputs with optimized inference
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):  # Use mixed precision
                outputs = self.model(input_ids, output_hidden_states=False)
                logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
        
        # Remove batch dimension
        logits = logits[0]  # Shape: [seq_len, vocab_size]
        input_ids = input_ids[0]  # Shape: [seq_len]
        
        # Convert to log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)
        
        result = {
            "model": self.model_name,
            "backend": "unsloth" if UNSLOTH_AVAILABLE else "transformers",
            "text": text,
            "tokens": [],
            "vocab_size": logits.shape[-1],
            "sequence_length": len(input_ids),
            "device": str(self.model.device),
            "optimizations": {
                "unsloth": UNSLOTH_AVAILABLE,
                "mixed_precision": True,
                "4bit_quantization": self.load_in_4bit,
                "pmi_tracking": self.calculate_pmi
            }
        }
        
        print(f"Processing {len(input_ids)} tokens with vocab size {logits.shape[-1]:,}...")
        
        # Process each token position (skip first token as it has no previous context)
        for pos in tqdm(range(1, len(input_ids)), desc="Extracting logits", leave=False):
            token_id = input_ids[pos].item()
            token = self.tokenizer.decode([token_id])
            
            # CRITICAL FIX: Use logits from PREVIOUS position to get probability of CURRENT token
            position_logprobs = log_probs[pos - 1]  # Shape: [vocab_size]
            token_logprob = position_logprobs[token_id].item()
            
            # Get top-k alternatives efficiently
            top_k_actual = min(top_k, len(position_logprobs))
            top_k_logprobs, top_k_indices = torch.topk(position_logprobs, top_k_actual)
            
            # Find rank of actual token
            rank = (top_k_indices == token_id).nonzero(as_tuple=True)
            if len(rank[0]) > 0:
                rank = rank[0].item()
            else:
                # Token not in top-k, need to find its rank
                sorted_indices = torch.argsort(position_logprobs, descending=True)
                rank = (sorted_indices == token_id).nonzero(as_tuple=True)[0].item()
            
            # Create alternatives list
            alternatives = []
            for i, (alt_logprob, alt_token_id) in enumerate(zip(top_k_logprobs, top_k_indices)):
                try:
                    alt_token = self.tokenizer.decode([alt_token_id.item()])
                    alternatives.append({
                        "rank": i,
                        "token": alt_token,
                        "token_id": alt_token_id.item(),
                        "logprob": alt_logprob.item(),
                        "probability": math.exp(alt_logprob.item())
                    })
                except:
                    continue
            
            # Get marginal probability and PMI
            marginal_prob = self.token_counter.get_marginal_probability(token_id)
            marginal_count = self.token_counter.token_counts[token_id]
            
            # Calculate PMI if we have enough data
            pmi = None
            if (self.calculate_pmi and 
                marginal_count >= self.min_token_count and
                marginal_prob > 0):
                
                conditional_prob = math.exp(token_logprob)
                pmi = math.log(conditional_prob / marginal_prob) if marginal_prob > 0 else float('-inf')
            
            token_analysis = {
                "position": pos,
                "token": token,
                "token_id": token_id,
                "logprob": token_logprob,
                "probability": math.exp(token_logprob),
                "rank": rank,
                "top_k_alternatives": alternatives,
                "marginal_probability": marginal_prob,
                "marginal_count": marginal_count,
                "pmi": pmi,
                "note": "unsloth_optimized_with_pmi"
            }
            
            # Add full vocabulary logprobs if requested
            if return_full_vocab:
                token_analysis["full_logprobs"] = position_logprobs.cpu().numpy()
            
            result["tokens"].append(token_analysis)
        
        # Add summary statistics
        result["summary"] = {
            "total_tokens": len(result["tokens"]),
            "avg_probability": np.mean([t["probability"] for t in result["tokens"]]),
            "min_probability": min([t["probability"] for t in result["tokens"]]),
            "max_probability": max([t["probability"] for t in result["tokens"]]),
            "perplexity": self._calculate_perplexity(result["tokens"])
        }
        
        # Add token frequency statistics
        result["token_frequency_stats"] = self.token_counter.get_statistics()
        
        # Add PMI statistics if available
        valid_pmis = [t["pmi"] for t in result["tokens"] if t["pmi"] is not None]
        if valid_pmis:
            result["pmi_stats"] = {
                "tokens_with_pmi": len(valid_pmis),
                "avg_pmi": np.mean(valid_pmis),
                "min_pmi": np.min(valid_pmis),
                "max_pmi": np.max(valid_pmis)
            }
        
        print(f"✅ Unsloth PMI analysis complete!")
        print(f"  Total tokens: {result['summary']['total_tokens']}")
        print(f"  Vocabulary size: {result['vocab_size']:,}")
        print(f"  Perplexity: {result['summary']['perplexity']:.2f}")
        print(f"  Backend: {result['backend']}")
        if valid_pmis:
            print(f"  Tokens with PMI: {len(valid_pmis)}")
            print(f"  Average PMI: {np.mean(valid_pmis):.3f}")
        
        return result
    
    def analyze_openwebtext_with_unsloth_pmi(self, 
                                           dataset_name: Optional[str] = None,
                                           num_samples: Optional[int] = None,
                                           chunk_size: Optional[int] = None,
                                           return_full_vocab: Optional[bool] = None,
                                           top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Analyze OpenWebText with Unsloth speed + PMI tracking"""
        
        # Use config values as defaults
        dataset_name = dataset_name or self.config.get('dataset_name', 'Bingsu/openwebtext_20p')
        num_samples = num_samples or self.config.get('num_samples', 100)
        chunk_size = chunk_size or self.config.get('chunk_size', 512)
        return_full_vocab = return_full_vocab if return_full_vocab is not None else self.config.get('return_full_vocab', False)
        top_k = top_k or self.config.get('top_k', 1000)
        
        print(f"Starting Unsloth + PMI OpenWebText analysis...")
        print(f"  Dataset: {dataset_name}")
        print(f"  Samples: {num_samples}")
        print(f"  🚀 Unsloth optimizations: {UNSLOTH_AVAILABLE}")
        print(f"  📊 PMI tracking: {self.calculate_pmi}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create run metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short_name = self.model_name.split('/')[-1].replace('-bnb-4bit', '')
        run_id = f"{model_short_name}_unsloth_pmi_{timestamp}"
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
            start_time = datetime.now()
            
            for i, example in enumerate(tqdm(dataset.take(num_samples), desc="Processing samples", total=num_samples)):
                text = example.get('text', '').strip()
                if not text or len(text) < 10:
                    continue
                    
                # Truncate text to reasonable length
                if len(text) > chunk_size * 4:
                    text = text[:chunk_size * 4]
                
                # Show progress with token stats
                if i % 10 == 0 and i > 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    samples_per_sec = i / elapsed if elapsed > 0 else 0
                    token_stats = self.token_counter.get_statistics()
                    print(f"\nSample {i+1}/{num_samples} - {samples_per_sec:.2f} samples/sec")
                    print(f"Token stats: {token_stats['total_tokens']} tokens, {token_stats['unique_tokens']} unique")
                
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
                    
                    current_batch = []
                    file_counter += 1
                    
                    # Save token counts periodically
                    if self.save_token_counts and i % 50 == 0:
                        token_counts_file = run_dir / f"token_counts_checkpoint_{i+1}.json"
                        self.token_counter.save(str(token_counts_file))
                    
                    # Memory cleanup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Save final token counts
            if self.save_token_counts:
                final_token_counts = run_dir / "final_token_counts.json"
                self.token_counter.save(str(final_token_counts))
            
            # Create summary
            total_time = (datetime.now() - start_time).total_seconds()
            summary = {
                "run_id": run_id,
                "model_name": self.model_name,
                "backend": "unsloth" if UNSLOTH_AVAILABLE else "transformers",
                "dataset_name": dataset_name,
                "total_samples": len(all_results),
                "total_files": file_counter,
                "processing_time_seconds": total_time,
                "samples_per_second": len(all_results) / total_time if total_time > 0 else 0,
                "token_frequency_stats": self.token_counter.get_statistics(),
                "config": self.config,
                "timestamp": timestamp,
                "optimizations": {
                    "unsloth": UNSLOTH_AVAILABLE,
                    "4bit_quantization": self.load_in_4bit,
                    "mixed_precision": True,
                    "pmi_tracking": self.calculate_pmi
                }
            }
            
            summary_file = run_dir / "run_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n🎉 Unsloth + PMI analysis complete!")
            print(f"  Total samples: {len(all_results)}")
            print(f"  Total time: {total_time:.1f} seconds")
            print(f"  Speed: {summary['samples_per_second']:.2f} samples/sec")
            print(f"  Token stats: {self.token_counter.get_statistics()}")
            print(f"  Files created: {file_counter}")
            print(f"  Output directory: {run_dir}")
            
            return all_results
            
        except Exception as e:
            print(f"❌ Error in Unsloth PMI analysis: {e}")
            logging.error(f"Unsloth PMI analysis error: {e}", exc_info=True)
            return []
    
    def _calculate_perplexity(self, tokens: List[Dict]) -> float:
        """Calculate perplexity from token probabilities"""
        if not tokens:
            return float('inf')
        log_probs = [t["logprob"] for t in tokens]
        avg_log_prob = np.mean(log_probs)
        return math.exp(-avg_log_prob)
    
    def _estimate_file_size_mb(self, results: List[Dict]) -> float:
        """Estimate file size in MB"""
        if not results:
            return 0.0
        sample_json = json.dumps(results[0])
        avg_size_per_sample = len(sample_json.encode('utf-8'))
        total_size_bytes = avg_size_per_sample * len(results)
        return total_size_bytes / (1024 * 1024)
    
    def _save_results(self, results: List[Dict], output_path: str):
        """Save results to JSON file"""
        try:
            serializable_results = []
            for result in results:
                serializable_result = result.copy()
                for token in serializable_result.get("tokens", []):
                    if "full_logprobs" in token and token["full_logprobs"] is not None:
                        token["full_logprobs"] = token["full_logprobs"].tolist()
                serializable_results.append(serializable_result)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"❌ Error saving results: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Unsloth PMI-enhanced logits analyzer")
    parser.add_argument("--config", "-c", help="Path to config YAML file")
    parser.add_argument("--model", help="Unsloth model name (overrides config)")
    parser.add_argument("--text", help="Text to analyze")
    parser.add_argument("--openwebtext", action="store_true", help="Analyze OpenWebText dataset")
    parser.add_argument("--num-samples", type=int, help="Number of samples (overrides config)")
    parser.add_argument("--top-k", type=int, help="Top-k alternatives per position (overrides config)")
    parser.add_argument("--full-vocab", action="store_true", help="Save full vocabulary logits (overrides config)")
    parser.add_argument("--max-seq-length", type=int, help="Maximum sequence length")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--calculate-pmi", action="store_true", help="Enable PMI calculation")
    parser.add_argument("--min-token-count", type=int, help="Minimum token count for PMI")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    if args.config:
        # Load from config file with overrides
        overrides = {
            'model_name': args.model,
            'num_samples': args.num_samples,
            'top_k': args.top_k,
            'return_full_vocab': args.full_vocab,
            'max_sequence_length': args.max_seq_length,
            'load_in_4bit': not args.no_4bit if args.no_4bit else None,
            'calculate_pmi': args.calculate_pmi,
            'min_token_count': args.min_token_count
        }
        analyzer = UnslothPMIAnalyzer.from_config(args.config, **overrides)
    else:
        # Use command line arguments
        config = {
            'calculate_pmi': args.calculate_pmi if args.calculate_pmi is not None else True,
            'min_token_count': args.min_token_count or 5,
            'save_token_counts': True
        }
        analyzer = UnslothPMIAnalyzer(
            model_name=args.model or "unsloth/llama-2-7b-bnb-4bit",
            max_seq_length=args.max_seq_length or 2048,
            load_in_4bit=not args.no_4bit,
            config=config
        )
    
    if args.openwebtext:
        # Analyze OpenWebText with Unsloth + PMI
        results = analyzer.analyze_openwebtext_with_unsloth_pmi(
            num_samples=args.num_samples,
            return_full_vocab=args.full_vocab,
            top_k=args.top_k
        )
        print(f"\n🚀 Completed Unsloth + PMI analysis of {len(results)} samples")
        
    elif args.text:
        # Analyze single text
        result = analyzer.get_full_logits_with_pmi(
            text=args.text,
            return_full_vocab=args.full_vocab or False,
            top_k=args.top_k or 1000
        )
        
        print(f"\nUnsloth + PMI Results for: '{args.text}'")
        print(f"Tokens analyzed: {len(result['tokens'])}")
        print(f"Backend: {result['backend']}")
        print(f"Token frequency stats: {result['token_frequency_stats']}")
        
        # Show first few tokens with PMI
        for token in result['tokens'][:5]:
            pmi = token.get('pmi', 'N/A')
            marginal = token.get('marginal_probability', 0)
            print(f"  '{token['token']}': prob={token['probability']:.4f}, marginal={marginal:.6f}, PMI={pmi}")
    
    else:
        print("Please specify either --text or --openwebtext")
        print("Example: python unsloth_pmi_analyzer.py --openwebtext --num-samples 100")


if __name__ == "__main__":
    main()