#!/usr/bin/env python3
"""
Together.ai Logits Analyzer
Analyzes token probabilities over a corpus of text using Together.ai API
"""

import os
import json
import math
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path
import together
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class TokenAnalysis:
    """Structure to hold token analysis results"""
    token: str
    logprob: float
    probability: float
    token_id: int
    position: int


class TogetherLogitsAnalyzer:
    """Analyzer for examining token probabilities using Together.ai"""
    
    def __init__(self, api_key: str, model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"):
        """
        Initialize the analyzer
        
        Args:
            api_key: Together.ai API key
            model: Model to use for analysis
        """
        self.client = together.Together(api_key=api_key)
        self.model = model
        
    def get_token_probabilities(self, text: str, max_tokens: int = 50) -> Dict[str, Any]:
        """
        Get token probabilities for given text
        
        Args:
            text: Input text to analyze
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary containing token analysis results
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": text}
                ],
                max_tokens=max_tokens,
                logprobs=1,
                temperature=0.7
            )
            
            return self._parse_logprobs_response(response)
            
        except Exception as e:
            print(f"Error getting token probabilities: {e}")
            return {}
    
    def _parse_logprobs_response(self, response) -> Dict[str, Any]:
        """Parse the logprobs response from Together.ai"""
        result = {
            "model": self.model,
            "tokens": [],
            "full_text": "",
            "token_count": 0
        }
        
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            
            if hasattr(choice, 'logprobs') and choice.logprobs:
                logprobs_data = choice.logprobs
                
                # Extract token information
                if hasattr(logprobs_data, 'content') and logprobs_data.content:
                    for i, token_data in enumerate(logprobs_data.content):
                        token_analysis = TokenAnalysis(
                            token=token_data.token,
                            logprob=token_data.logprob,
                            probability=math.exp(token_data.logprob),
                            token_id=getattr(token_data, 'token_id', None),
                            position=i
                        )
                        result["tokens"].append({
                            "token": token_analysis.token,
                            "logprob": token_analysis.logprob,
                            "probability": token_analysis.probability,
                            "probability_percent": token_analysis.probability * 100,
                            "token_id": token_analysis.token_id,
                            "position": token_analysis.position
                        })
            
            # Get the full generated text
            if hasattr(choice, 'message') and choice.message:
                result["full_text"] = choice.message.content
            
            result["token_count"] = len(result["tokens"])
        
        return result
    
    def analyze_corpus(self, corpus_path: str, output_path: str = None, chunk_size: int = 500) -> List[Dict[str, Any]]:
        """
        Analyze a corpus of text files
        
        Args:
            corpus_path: Path to corpus file or directory
            output_path: Optional path to save results
            chunk_size: Size of text chunks to analyze
            
        Returns:
            List of analysis results
        """
        corpus_path = Path(corpus_path)
        results = []
        
        if corpus_path.is_file():
            files = [corpus_path]
        elif corpus_path.is_dir():
            files = list(corpus_path.glob("*.txt")) + list(corpus_path.glob("*.md"))
        else:
            print(f"Error: {corpus_path} is not a valid file or directory")
            return results
        
        for file_path in files:
            print(f"Processing: {file_path}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split content into chunks
                chunks = self._split_text(content, chunk_size)
                
                for i, chunk in enumerate(chunks):
                    print(f"  Analyzing chunk {i+1}/{len(chunks)}")
                    
                    analysis = self.get_token_probabilities(chunk)
                    analysis["source_file"] = str(file_path)
                    analysis["chunk_index"] = i
                    analysis["chunk_text"] = chunk
                    
                    results.append(analysis)
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Save results if output path provided
        if output_path:
            self._save_results(results, output_path)
        
        return results
    
    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks of approximately chunk_size characters"""
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _save_results(self, results: List[Dict[str, Any]], output_path: str):
        """Save analysis results to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {output_path}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def print_analysis_summary(self, results: List[Dict[str, Any]]):
        """Print a summary of the analysis results"""
        if not results:
            print("No results to summarize")
            return
        
        total_tokens = sum(result["token_count"] for result in results)
        total_chunks = len(results)
        
        print(f"\n=== Analysis Summary ===")
        print(f"Total chunks analyzed: {total_chunks}")
        print(f"Total tokens generated: {total_tokens}")
        print(f"Average tokens per chunk: {total_tokens/total_chunks:.2f}")
        
        # Find tokens with highest and lowest confidence
        all_tokens = []
        for result in results:
            all_tokens.extend(result["tokens"])
        
        if all_tokens:
            highest_conf = max(all_tokens, key=lambda x: x["probability"])
            lowest_conf = min(all_tokens, key=lambda x: x["probability"])
            
            print(f"\nHighest confidence token: '{highest_conf['token']}' ({highest_conf['probability_percent']:.2f}%)")
            print(f"Lowest confidence token: '{lowest_conf['token']}' ({lowest_conf['probability_percent']:.2f}%)")


def main():
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Analyze token probabilities using Together.ai")
    parser.add_argument("--api-key", help="Together.ai API key (or set TOGETHER_API_KEY env var)")
    parser.add_argument("--input", required=True, help="Input text file or directory")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--model", help="Model to use for analysis")
    parser.add_argument("--chunk-size", type=int, default=500, 
                       help="Text chunk size for processing")
    parser.add_argument("--max-tokens", type=int, default=50,
                       help="Maximum tokens to generate per chunk")
    
    args = parser.parse_args()
    
    # Get API key from args or environment
    api_key = args.api_key or os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print("Error: API key required. Set TOGETHER_API_KEY environment variable or use --api-key")
        return
    
    # Get model from args or environment
    model = args.model or os.getenv("TOGETHER_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
    
    # Initialize analyzer
    analyzer = TogetherLogitsAnalyzer(api_key, model)
    
    # Check if input is a single text string or file path
    input_path = Path(args.input)
    if input_path.exists():
        # Analyze corpus
        print(f"Analyzing corpus from: {input_path}")
        results = analyzer.analyze_corpus(str(input_path), args.output, args.chunk_size)
        analyzer.print_analysis_summary(results)
    else:
        # Treat as direct text input
        print(f"Analyzing direct text input")
        result = analyzer.get_token_probabilities(args.input, args.max_tokens)
        
        if result:
            print(f"\n=== Token Analysis ===")
            print(f"Generated text: {result['full_text']}")
            print(f"Token count: {result['token_count']}")
            print(f"\nToken probabilities:")
            
            for token_data in result["tokens"]:
                print(f"  '{token_data['token']}': {token_data['probability_percent']:.2f}% "
                      f"(logprob: {token_data['logprob']:.4f})")
            
            if args.output:
                analyzer._save_results([result], args.output)


if __name__ == "__main__":
    main()