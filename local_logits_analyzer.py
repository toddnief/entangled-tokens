#!/usr/bin/env python3
"""
Local Model Logits Analyzer for A100
Provides full vocabulary logprobs using Hugging Face Transformers or vLLM
"""

import os
import json
import math
import yaml
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from datasets import load_dataset
import logging
from datetime import datetime
from tqdm import tqdm

# Optional imports - will check availability
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: Transformers not available. Install with: pip install transformers")

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not available. Install with: pip install vllm")


@dataclass
class FullTokenAnalysis:
    """Structure to hold complete token analysis with full vocabulary logits"""
    token: str
    token_id: int
    logprob: float
    probability: float
    position: int
    rank: int  # Rank in vocabulary (0 = most likely)
    full_logprobs: Optional[np.ndarray] = None  # Full vocabulary logprobs
    top_k_alternatives: Optional[List[Dict]] = None


class LocalLogitsAnalyzer:
    """Analyzer for examining full vocabulary logprobs using local models"""
    
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-2-7b-hf",
                 device: str = "auto",
                 backend: str = "transformers",
                 max_memory_gb: Optional[float] = None,
                 dtype: str = "float16",
                 config: Optional[Dict] = None):
        """
        Initialize local model analyzer
        
        Args:
            model_name: HuggingFace model name/path
            device: Device to use ("auto", "cuda", "cpu")
            backend: "transformers" or "vllm" 
            max_memory_gb: Maximum GPU memory to use
            dtype: Model dtype (float16, bfloat16, float32)
            config: Optional config dictionary
        """
        self.model_name = model_name
        self.backend = backend
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, dtype)
        self.config = config or {}
        
        # File management settings
        self.samples_per_file = self.config.get('samples_per_file', 50)  # Split files every 50 samples
        self.max_file_size_mb = self.config.get('max_file_size_mb', 500)  # Split if file > 500MB
        self.output_dir = Path(self.config.get('output_dir', 'full_logits_outputs'))
        self.compress = self.config.get('compress', True)
        
        print(f"Initializing LocalLogitsAnalyzer:")
        print(f"  Model: {model_name}")
        print(f"  Backend: {backend}")
        print(f"  Device: {self.device}")
        print(f"  Dtype: {dtype}")
        print(f"  Output dir: {self.output_dir}")
        print(f"  Samples per file: {self.samples_per_file}")
        
        if max_memory_gb:
            print(f"  Max memory: {max_memory_gb}GB")
            
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.llm = None  # For vLLM
        
        self._load_model(max_memory_gb)
        
    @classmethod
    def from_config(cls, config_path: str, **overrides):
        """Create analyzer from config file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Apply any command-line overrides
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
        
    def _load_model(self, max_memory_gb: Optional[float] = None):
        """Load model based on selected backend"""
        
        if self.backend == "vllm" and VLLM_AVAILABLE:
            self._load_vllm_model(max_memory_gb)
        elif self.backend == "transformers" and HF_AVAILABLE:
            self._load_hf_model(max_memory_gb)
        else:
            raise ValueError(f"Backend {self.backend} not available. Install required packages.")
    
    def _load_vllm_model(self, max_memory_gb: Optional[float] = None):
        """Load model using vLLM for faster inference"""
        print("Loading model with vLLM...")
        
        gpu_memory_utilization = 0.9  # Default
        if max_memory_gb:
            # Estimate based on A100 80GB
            gpu_memory_utilization = min(0.95, max_memory_gb / 80.0)
            
        self.llm = LLM(
            model=self.model_name,
            dtype=self.dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=4096,  # Adjust based on your needs
        )
        
        # Get tokenizer from vLLM
        self.tokenizer = self.llm.get_tokenizer()
        print(f"✓ vLLM model loaded successfully")
        print(f"  Vocabulary size: {len(self.tokenizer)}")
        
    def _load_hf_model(self, max_memory_gb: Optional[float] = None):
        """Load model using Hugging Face Transformers"""
        print("Loading model with Hugging Face Transformers...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Configure device map and memory
        device_map = "auto" if self.device == "cuda" else None
        max_memory = None
        
        if max_memory_gb and self.device == "cuda":
            max_memory = {0: f"{max_memory_gb}GiB"}
            
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=device_map,
            max_memory=max_memory,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        self.model.eval()
        print(f"✓ Transformers model loaded successfully")
        print(f"  Device: {next(self.model.parameters()).device}")
        print(f"  Vocabulary size: {len(self.tokenizer)}")
        
    def get_full_logits(self, text: str, return_full_vocab: bool = False, top_k: int = 1000) -> Dict[str, Any]:
        """
        Get complete logits for every token position
        
        Args:
            text: Input text to analyze
            return_full_vocab: If True, returns full vocab logits (memory intensive)
            top_k: Number of top alternatives to return per position
            
        Returns:
            Dictionary with complete logits analysis
        """
        print(f"Analyzing text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        
        if self.backend == "vllm":
            return self._get_logits_vllm(text, return_full_vocab, top_k)
        else:
            return self._get_logits_transformers(text, return_full_vocab, top_k)
    
    def _get_logits_transformers(self, text: str, return_full_vocab: bool, top_k: int) -> Dict[str, Any]:
        """Get logits using Hugging Face Transformers"""
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        input_ids = inputs["input_ids"].to(self.model.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=False)
            logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
        
        # Remove batch dimension
        logits = logits[0]  # Shape: [seq_len, vocab_size]
        input_ids = input_ids[0]  # Shape: [seq_len]
        
        # Convert to log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)
        
        result = {
            "model": self.model_name,
            "backend": "transformers", 
            "text": text,
            "tokens": [],
            "vocab_size": logits.shape[-1],
            "sequence_length": len(input_ids),
            "device": str(self.model.device)
        }
        
        print(f"Processing {len(input_ids)} tokens with vocab size {logits.shape[-1]}...")
        
        # Process each token position
        for pos in tqdm(range(len(input_ids)), desc="Extracting logits"):
            token_id = input_ids[pos].item()
            token = self.tokenizer.decode([token_id])
            
            # Get logprobs for this position
            position_logprobs = log_probs[pos]  # Shape: [vocab_size]
            token_logprob = position_logprobs[token_id].item()
            
            # Get top-k alternatives
            top_k_logprobs, top_k_indices = torch.topk(position_logprobs, min(top_k, len(position_logprobs)))
            
            # Find rank of actual token
            sorted_indices = torch.argsort(position_logprobs, descending=True)
            rank = (sorted_indices == token_id).nonzero(as_tuple=True)[0].item()
            
            # Create alternatives list
            alternatives = []
            for i, (alt_logprob, alt_token_id) in enumerate(zip(top_k_logprobs, top_k_indices)):
                alt_token = self.tokenizer.decode([alt_token_id.item()])
                alternatives.append({
                    "rank": i,
                    "token": alt_token,
                    "token_id": alt_token_id.item(),
                    "logprob": alt_logprob.item(),
                    "probability": math.exp(alt_logprob.item())
                })
            
            token_analysis = {
                "position": pos,
                "token": token,
                "token_id": token_id,
                "logprob": token_logprob,
                "probability": math.exp(token_logprob),
                "rank": rank,
                "top_k_alternatives": alternatives
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
        
        print(f"✓ Analysis complete!")
        print(f"  Total tokens: {result['summary']['total_tokens']}")
        print(f"  Vocabulary size: {result['vocab_size']:,}")
        print(f"  Perplexity: {result['summary']['perplexity']:.2f}")
        
        return result
    
    def _get_logits_vllm(self, text: str, return_full_vocab: bool, top_k: int) -> Dict[str, Any]:
        """Get logits using vLLM (implementation needed)"""
        # Note: vLLM doesn't directly expose logits like Transformers
        # This would require modifying vLLM or using a different approach
        raise NotImplementedError("vLLM full logits extraction not yet implemented. Use 'transformers' backend.")
    
    def _calculate_perplexity(self, tokens: List[Dict]) -> float:
        """Calculate perplexity from token probabilities"""
        if not tokens:
            return float('inf')
        
        log_probs = [t["logprob"] for t in tokens]
        avg_log_prob = np.mean(log_probs)
        return math.exp(-avg_log_prob)
    
    def analyze_openwebtext(self, 
                          dataset_name: Optional[str] = None,
                          num_samples: Optional[int] = None,
                          output_path: Optional[str] = None,
                          chunk_size: Optional[int] = None,
                          return_full_vocab: Optional[bool] = None,
                          top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Analyze OpenWebText dataset with full vocabulary logits
        Uses config file settings as defaults, can be overridden by parameters
        
        Args:
            dataset_name: HuggingFace dataset name
            num_samples: Number of samples to process
            output_path: Path to save results (deprecated - uses output_dir from config)
            chunk_size: Maximum tokens per chunk
            return_full_vocab: Whether to save full vocab logits
            top_k: Top-k alternatives to save
            
        Returns:
            List of analysis results
        """
        # Use config values as defaults
        dataset_name = dataset_name or self.config.get('dataset_name', 'Bingsu/openwebtext_20p')
        num_samples = num_samples or self.config.get('num_samples', 100)
        chunk_size = chunk_size or self.config.get('chunk_size', 512)
        return_full_vocab = return_full_vocab if return_full_vocab is not None else self.config.get('return_full_vocab', False)
        top_k = top_k or self.config.get('top_k', 1000)
        
        print(f"Starting OpenWebText analysis...")
        print(f"  Dataset: {dataset_name}")
        print(f"  Samples: {num_samples}")
        print(f"  Chunk size: {chunk_size}")
        print(f"  Top-k alternatives: {top_k}")
        print(f"  Full vocab: {return_full_vocab}")
        print(f"  Output dir: {self.output_dir}")
        print(f"  Samples per file: {self.samples_per_file}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create run metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{self.model_name.split('/')[-1]}_{timestamp}"
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
                if len(text) > chunk_size * 4:  # Rough character estimate
                    text = text[:chunk_size * 4]
                
                if i % 10 == 0:
                    print(f"\nProcessing sample {i+1}/{num_samples}")
                
                analysis = self.get_full_logits(
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
                    len(current_batch) >= self.samples_per_file or  # Samples limit
                    i == num_samples - 1 or  # Last sample
                    self._estimate_file_size_mb(current_batch) >= self.max_file_size_mb  # Size limit
                )
                
                if should_save:
                    output_file = run_dir / f"batch_{file_counter:04d}_{len(current_batch):03d}_samples.json"
                    self._save_results(current_batch, str(output_file))
                    
                    print(f"  Saved batch {file_counter} with {len(current_batch)} samples to {output_file.name}")
                    
                    current_batch = []
                    file_counter += 1
                    
                    # Clean up memory periodically
                    if i % self.config.get('memory_cleanup_interval', 50) == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            
            # Create summary file
            summary = {
                "run_id": run_id,
                "model_name": self.model_name,
                "dataset_name": dataset_name,
                "total_samples": len(all_results),
                "total_files": file_counter,
                "samples_per_file": self.samples_per_file,
                "config": self.config,
                "timestamp": timestamp,
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
            
            print(f"\n✅ Analysis complete!")
            print(f"  Total samples processed: {len(all_results)}")
            print(f"  Files created: {file_counter}")
            print(f"  Output directory: {run_dir}")
            print(f"  Summary file: {summary_file}")
            
            return all_results
            
        except Exception as e:
            print(f"Error in OpenWebText analysis: {e}")
            logging.error(f"OpenWebText analysis error: {e}", exc_info=True)
            return []
    
    def _estimate_file_size_mb(self, results: List[Dict]) -> float:
        """Estimate file size in MB for a list of results"""
        if not results:
            return 0.0
        
        # Sample first result to estimate average size
        sample_json = json.dumps(results[0])
        avg_size_per_sample = len(sample_json.encode('utf-8'))
        
        # Estimate total size
        total_size_bytes = avg_size_per_sample * len(results)
        return total_size_bytes / (1024 * 1024)
    
    def _save_results(self, results: List[Dict], output_path: str):
        """Save results to JSON file"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = []
            for result in results:
                serializable_result = result.copy()
                for token in serializable_result.get("tokens", []):
                    if "full_logprobs" in token and token["full_logprobs"] is not None:
                        token["full_logprobs"] = token["full_logprobs"].tolist()
                serializable_results.append(serializable_result)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {output_path}")
            
        except Exception as e:
            print(f"Error saving results: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Local model logits analyzer for A100")
    parser.add_argument("--config", "-c", help="Path to config YAML file")
    parser.add_argument("--model", help="Model name/path (overrides config)")
    parser.add_argument("--backend", choices=["transformers", "vllm"], help="Backend to use (overrides config)")
    parser.add_argument("--text", help="Text to analyze")
    parser.add_argument("--openwebtext", action="store_true", help="Analyze OpenWebText dataset")
    parser.add_argument("--num-samples", type=int, help="Number of samples (overrides config)")
    parser.add_argument("--output", help="Output file path (deprecated - use config output_dir)")
    parser.add_argument("--top-k", type=int, help="Top-k alternatives per position (overrides config)")
    parser.add_argument("--full-vocab", action="store_true", help="Save full vocabulary logits (overrides config)")
    parser.add_argument("--max-memory", type=float, help="Max GPU memory in GB (overrides config)")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    if args.config:
        # Load from config file with overrides
        overrides = {
            'model_name': args.model,
            'backend': args.backend,
            'num_samples': args.num_samples,
            'top_k': args.top_k,
            'return_full_vocab': args.full_vocab,
            'max_memory_gb': args.max_memory
        }
        analyzer = LocalLogitsAnalyzer.from_config(args.config, **overrides)
    else:
        # Use command line arguments
        analyzer = LocalLogitsAnalyzer(
            model_name=args.model or "meta-llama/Llama-2-7b-hf",
            backend=args.backend or "transformers",
            max_memory_gb=args.max_memory
        )
    
    if args.openwebtext:
        # Analyze OpenWebText
        results = analyzer.analyze_openwebtext(
            num_samples=args.num_samples,
            return_full_vocab=args.full_vocab,
            top_k=args.top_k
        )
        print(f"\nCompleted analysis of {len(results)} samples")
        
    elif args.text:
        # Analyze single text
        result = analyzer.get_full_logits(
            text=args.text,
            return_full_vocab=args.full_vocab or False,
            top_k=args.top_k or 1000
        )
        
        print(f"\nResults for: '{args.text}'")
        print(f"Tokens analyzed: {len(result['tokens'])}")
        print(f"Vocabulary size: {result['vocab_size']:,}")
        print(f"Perplexity: {result['summary']['perplexity']:.2f}")
        
        # Show first few tokens
        for token in result['tokens'][:5]:
            print(f"  '{token['token']}': {token['probability']:.4f} (rank {token['rank']:,})")
            
        if args.output:
            analyzer._save_results([result], args.output)
    
    else:
        print("Please specify either --text or --openwebtext")
        print("Example with config: python local_logits_analyzer.py --config configs/a100_full_logits_config.yaml --openwebtext")


if __name__ == "__main__":
    main()