#!/usr/bin/env python3
"""
Config-based runner for OpenWebText logits extraction
Runs multiple models with configurable parameters
"""

import os
import yaml
import json
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from together_logits_analyzer import TogetherLogitsAnalyzer
from dotenv import load_dotenv


class LogitsExtractionRunner:
    """Runner for extracting logits using configuration files"""
    
    def __init__(self, config_path: str):
        """Initialize runner with configuration file"""
        self.config_path = config_path
        self.config = self.load_config()
        self.api_key = self.get_api_key()
        
        # Set random seed for reproducibility
        if 'random_seed' in self.config:
            random.seed(self.config['random_seed'])
        
        # Create output directory
        self.output_dir = Path(self.config.get('output_dir', 'outputs'))
        self.output_dir.mkdir(exist_ok=True)
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Loaded configuration from: {self.config_path}")
            return config
        except Exception as e:
            raise Exception(f"Failed to load config file {self.config_path}: {e}")
    
    def get_api_key(self) -> str:
        """Get Together.ai API key from environment"""
        load_dotenv()
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise Exception("TOGETHER_API_KEY not found in environment variables")
        return api_key
    
    def create_run_metadata(self) -> Dict[str, Any]:
        """Create metadata for this run"""
        return {
            "timestamp": datetime.now().isoformat(),
            "config_file": str(self.config_path),
            "config": self.config,
            "models_count": len(self.config.get('models', [])),
            "total_samples": self.config.get('num_samples', 0)
        }
    
    def run_model_extraction(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run logits extraction for a single model"""
        model_name = model_config['name']
        model_id = model_config['model_id']
        
        print(f"\n=== Starting extraction for model: {model_name} ({model_id}) ===")
        
        # Initialize analyzer with model-specific settings
        analyzer = TogetherLogitsAnalyzer(
            api_key=self.api_key,
            model=model_id
        )
        
        # Run extraction with top_logprobs from model config
        top_logprobs = model_config.get('top_logprobs', 5)
        results = analyzer.analyze_openwebtext(
            dataset_name=self.config.get('dataset_name', 'Bingsu/openwebtext_20p'),
            num_samples=self.config.get('num_samples', 100),
            chunk_size=self.config.get('chunk_size', 500),
            split=self.config.get('split', 'train'),
            top_logprobs=top_logprobs
        )
        
        # Add model-specific metadata to each result
        for result in results:
            result['model_name'] = model_name
            result['model_config'] = model_config
            result['extraction_timestamp'] = datetime.now().isoformat()
        
        # Save individual model results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"{model_name}_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(results)} results to: {output_file}")
        
        # Print summary
        analyzer.print_analysis_summary(results)
        
        return {
            'model_name': model_name,
            'model_id': model_id,
            'results_count': len(results),
            'output_file': str(output_file),
            'results': results
        }
    
    def run_all_models(self) -> Dict[str, Any]:
        """Run logits extraction for all configured models"""
        print(f"Starting logits extraction run with {len(self.config.get('models', []))} models")
        print(f"Samples per model: {self.config.get('num_samples', 'N/A')}")
        print(f"Dataset: {self.config.get('dataset_name', 'N/A')}")
        
        run_metadata = self.create_run_metadata()
        all_results = []
        model_summaries = []
        
        # Process each model
        for model_config in self.config.get('models', []):
            try:
                model_result = self.run_model_extraction(model_config)
                all_results.extend(model_result['results'])
                model_summaries.append({
                    'model_name': model_result['model_name'],
                    'model_id': model_result['model_id'],
                    'results_count': model_result['results_count'],
                    'output_file': model_result['output_file']
                })
                
            except Exception as e:
                print(f"ERROR: Failed to process model {model_config.get('name', 'unknown')}: {e}")
                model_summaries.append({
                    'model_name': model_config.get('name', 'unknown'),
                    'model_id': model_config.get('model_id', 'unknown'),
                    'error': str(e),
                    'results_count': 0
                })
        
        # Save combined results and metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save all results combined
        if self.config.get('save_intermediate', True):
            combined_file = self.output_dir / f"combined_results_{timestamp}.json"
            with open(combined_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"\nSaved combined results ({len(all_results)} total) to: {combined_file}")
        
        # Save run summary
        summary = {
            'run_metadata': run_metadata,
            'model_summaries': model_summaries,
            'total_results': len(all_results),
            'successful_models': len([s for s in model_summaries if 'error' not in s]),
            'failed_models': len([s for s in model_summaries if 'error' in s])
        }
        
        summary_file = self.output_dir / f"run_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n=== RUN COMPLETE ===")
        print(f"Total results: {len(all_results)}")
        print(f"Successful models: {summary['successful_models']}")
        print(f"Failed models: {summary['failed_models']}")
        print(f"Run summary saved to: {summary_file}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Run logits extraction with configuration file")
    parser.add_argument("--config", "-c", required=True,
                       help="Path to configuration YAML file")
    parser.add_argument("--num-samples", "-n", type=int,
                       help="Override number of samples from config")
    
    args = parser.parse_args()
    
    try:
        # Initialize runner
        runner = LogitsExtractionRunner(args.config)
        
        # Override num_samples if provided
        if args.num_samples:
            runner.config['num_samples'] = args.num_samples
            print(f"Overriding num_samples to: {args.num_samples}")
        
        # Run extraction
        summary = runner.run_all_models()
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    exit(main())