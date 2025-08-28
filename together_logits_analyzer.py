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
from datasets import load_dataset
import logging


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
        
    def get_token_probabilities_full_text(self, text: str, max_tokens: int = 50, top_logprobs: int = 5) -> Dict[str, Any]:
        """
        Get token probabilities for every position in text by treating it as completion
        This simulates getting logprobs for input tokens by making the text a completion
        
        Args:
            text: Input text to analyze  
            max_tokens: Maximum additional tokens to generate
            top_logprobs: Number of top alternatives to return per position (max 5 for Together.ai)
            
        Returns:
            Dictionary containing token analysis results for every token position
        """
        try:
            # Strategy: Use empty prompt and treat the text as the completion
            # This way we get logprobs for every token in the original text
            response = self.client.completions.create(
                model=self.model,
                prompt="",  # Empty prompt
                max_tokens=len(text.split()) + max_tokens,  # Estimate tokens needed
                logprobs=top_logprobs,
                temperature=0.7,
                # Force the model to generate our text by using it as suffix
                suffix=text
            )
            
            return self._parse_completions_logprobs_response(response)
            
        except Exception as e:
            # If suffix isn't supported, fall back to standard approach
            print(f"Suffix approach failed, trying standard completions: {e}")
            return self.get_token_probabilities_standard(text, max_tokens, top_logprobs)
    
    def get_token_probabilities_standard(self, text: str, max_tokens: int = 50, top_logprobs: int = 5) -> Dict[str, Any]:
        """
        Standard approach: Get token probabilities for generated completions only
        
        Args:
            text: Input text to analyze
            max_tokens: Maximum tokens to generate
            top_logprobs: Number of top alternatives to return per position (max 5 for Together.ai)
            
        Returns:
            Dictionary containing token analysis results
        """
        try:
            response = self.client.completions.create(
                model=self.model,
                prompt=text,
                max_tokens=max_tokens,
                logprobs=top_logprobs,
                temperature=0.7
            )
            
            return self._parse_completions_logprobs_response(response)
            
        except Exception as e:
            print(f"Error getting token probabilities: {e}")
            return {}
    
    def get_token_probabilities_progressive(self, text: str, top_logprobs: int = 5) -> Dict[str, Any]:
        """
        Get token probabilities for every position by progressively analyzing each token.
        This works by:
        1. Tokenizing the input text
        2. For each position, use the prefix as prompt and analyze next token logprobs
        3. Combine all position results
        
        Args:
            text: Input text to analyze
            top_logprobs: Number of top alternatives to return per position (max 5 for Together.ai)
            
        Returns:
            Dictionary containing token analysis results for every token position
        """
        print(f"Starting progressive token analysis for: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # First, get a rough tokenization by using the model to complete with max_tokens=0
        try:
            # Get the full completion to see how the text gets tokenized
            initial_response = self.client.completions.create(
                model=self.model,
                prompt=text,
                max_tokens=1,
                logprobs=top_logprobs,
                temperature=0.0
            )
        except Exception as e:
            print(f"Failed to get initial tokenization: {e}")
            return {}
        
        # Split text into rough tokens by words and characters
        # This is approximate - real tokenization would be more complex
        import re
        
        # Simple tokenization approach: split on whitespace and punctuation
        tokens = re.findall(r'\S+|\s+', text)
        
        result = {
            "model": self.model,
            "tokens": [],
            "full_text": text,
            "token_count": 0,
            "position_alternatives": [],
            "method": "progressive"
        }
        
        print(f"Analyzing {len(tokens)} approximate tokens...")
        
        # Analyze each token position
        for i, token in enumerate(tokens):
            if i == 0:
                prefix = ""
            else:
                prefix = "".join(tokens[:i])
            
            print(f"Position {i:2d}: Analyzing token '{token}' with prefix '{prefix[-20:]}{'...' if len(prefix) > 20 else ''}'")
            
            try:
                # Use the prefix as prompt and analyze what the model predicts for the next token
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prefix,
                    max_tokens=5,  # Generate a few tokens to capture alternatives
                    logprobs=top_logprobs,
                    temperature=0.0
                )
                
                if hasattr(response, 'choices') and response.choices:
                    choice = response.choices[0]
                    
                    if hasattr(choice, 'logprobs') and choice.logprobs:
                        logprobs_data = choice.logprobs
                        
                        # Extract the first token's alternatives (what model would generate next)
                        if (hasattr(logprobs_data, 'tokens') and logprobs_data.tokens and
                            hasattr(logprobs_data, 'token_logprobs') and logprobs_data.token_logprobs and
                            hasattr(logprobs_data, 'top_logprobs') and logprobs_data.top_logprobs):
                            
                            first_token = logprobs_data.tokens[0] if logprobs_data.tokens else None
                            first_logprob = logprobs_data.token_logprobs[0] if logprobs_data.token_logprobs else None
                            first_alternatives = logprobs_data.top_logprobs[0] if logprobs_data.top_logprobs else {}
                            
                            # Check if the predicted token matches our actual token
                            actual_match = False
                            actual_logprob = None
                            
                            if first_alternatives:
                                for alt_token, alt_logprob in first_alternatives.items():
                                    # Check various token matching strategies
                                    if (alt_token == token or 
                                        alt_token.strip() == token.strip() or
                                        token.startswith(alt_token) or
                                        alt_token.startswith(token)):
                                        actual_match = True
                                        actual_logprob = alt_logprob
                                        break
                            
                            # If not found in alternatives, check the main prediction
                            if not actual_match and first_token:
                                if (first_token == token or 
                                    first_token.strip() == token.strip() or
                                    token.startswith(first_token) or
                                    first_token.startswith(token)):
                                    actual_match = True
                                    actual_logprob = first_logprob
                            
                            # Add token data
                            if actual_logprob is not None:
                                result["tokens"].append({
                                    "token": token,
                                    "logprob": actual_logprob,
                                    "probability": math.exp(actual_logprob),
                                    "probability_percent": math.exp(actual_logprob) * 100,
                                    "position": i,
                                    "is_input": True,
                                    "matched": actual_match
                                })
                            else:
                                # Estimate a low probability if token not found
                                estimated_logprob = -10.0  # Very low probability
                                result["tokens"].append({
                                    "token": token,
                                    "logprob": estimated_logprob,
                                    "probability": math.exp(estimated_logprob),
                                    "probability_percent": math.exp(estimated_logprob) * 100,
                                    "position": i,
                                    "is_input": True,
                                    "matched": False,
                                    "estimated": True
                                })
                            
                            # Add alternatives for this position
                            if first_alternatives:
                                alternatives = []
                                for alt_token, alt_logprob in first_alternatives.items():
                                    alternatives.append({
                                        "token": alt_token,
                                        "logprob": alt_logprob,
                                        "probability": math.exp(alt_logprob),
                                        "probability_percent": math.exp(alt_logprob) * 100
                                    })
                                
                                alternatives.sort(key=lambda x: x["probability"], reverse=True)
                                result["position_alternatives"].append({
                                    "position": i,
                                    "alternatives": alternatives
                                })
                
                # Small delay to avoid rate limiting
                import time
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error analyzing position {i}: {e}")
                continue
        
        result["token_count"] = len(result["tokens"])
        print(f"Progressive analysis complete. Analyzed {result['token_count']} tokens.")
        
        return result
    
    def get_token_probabilities(self, text: str, max_tokens: int = 50, top_logprobs: int = 5) -> Dict[str, Any]:
        """
        Get token probabilities - uses progressive analysis for full coverage
        """
        return self.get_token_probabilities_progressive(text, top_logprobs)
    
    def _parse_completions_logprobs_response(self, response) -> Dict[str, Any]:
        """Parse the logprobs response from Together.ai completions API"""
        result = {
            "model": self.model,
            "tokens": [],
            "full_text": "",
            "token_count": 0,
            "position_alternatives": []
        }
        
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            
            # Get the full text (includes both prompt and completion)
            if hasattr(choice, 'text'):
                result["full_text"] = choice.text
            
            # Parse logprobs data
            if hasattr(choice, 'logprobs') and choice.logprobs:
                logprobs_data = choice.logprobs
                
                # Extract tokens and their logprobs
                if hasattr(logprobs_data, 'tokens') and logprobs_data.tokens:
                    tokens = logprobs_data.tokens
                    token_logprobs = getattr(logprobs_data, 'token_logprobs', [])
                    top_logprobs = getattr(logprobs_data, 'top_logprobs', [])
                    
                    # Process each token position
                    for i, token in enumerate(tokens):
                        # Main token data
                        logprob = token_logprobs[i] if i < len(token_logprobs) and token_logprobs[i] is not None else None
                        
                        if logprob is not None:
                            result["tokens"].append({
                                "token": token,
                                "logprob": logprob,
                                "probability": math.exp(logprob),
                                "probability_percent": math.exp(logprob) * 100,
                                "token_id": None,
                                "position": i,
                                "is_input": True  # Will determine later based on text offset
                            })
                        
                        # Top alternatives for this position
                        if i < len(top_logprobs) and top_logprobs[i]:
                            alternatives = []
                            for alt_token, alt_logprob in top_logprobs[i].items():
                                alternatives.append({
                                    "token": alt_token,
                                    "logprob": alt_logprob,
                                    "probability": math.exp(alt_logprob),
                                    "probability_percent": math.exp(alt_logprob) * 100
                                })
                            
                            # Sort alternatives by probability (descending)
                            alternatives.sort(key=lambda x: x["probability"], reverse=True)
                            
                            result["position_alternatives"].append({
                                "position": i,
                                "alternatives": alternatives
                            })
                
            result["token_count"] = len(result["tokens"])
        
        return result
    
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
                
                # Extract token information - Together.ai format
                if hasattr(logprobs_data, 'tokens') and logprobs_data.tokens:
                    # Handle generated tokens
                    for i, token in enumerate(logprobs_data.tokens):
                        logprob = logprobs_data.token_logprobs[i] if hasattr(logprobs_data, 'token_logprobs') and i < len(logprobs_data.token_logprobs) else None
                        if logprob is not None:
                            result["tokens"].append({
                                "token": token,
                                "logprob": logprob,
                                "probability": math.exp(logprob),
                                "probability_percent": math.exp(logprob) * 100,
                                "token_id": None,
                                "position": i,
                                "is_generated": True
                            })
                
                # Extract top logprobs alternatives at each position
                if hasattr(logprobs_data, 'top_logprobs') and logprobs_data.top_logprobs:
                    result["position_alternatives"] = []
                    for pos, alternatives in enumerate(logprobs_data.top_logprobs):
                        if alternatives:
                            pos_data = {
                                "position": pos,
                                "alternatives": []
                            }
                            for alt_token, alt_logprob in alternatives.items():
                                pos_data["alternatives"].append({
                                    "token": alt_token,
                                    "logprob": alt_logprob,
                                    "probability": math.exp(alt_logprob),
                                    "probability_percent": math.exp(alt_logprob) * 100
                                })
                            # Sort by probability (descending)
                            pos_data["alternatives"].sort(key=lambda x: x["probability"], reverse=True)
                            result["position_alternatives"].append(pos_data)
                else:
                    # Try alternate format for content-based logprobs
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
    
    def analyze_openwebtext(self, dataset_name: str = "Bingsu/openwebtext_20p", 
                           num_samples: int = 100, output_path: str = None, 
                           chunk_size: int = 500, split: str = "train", 
                           top_logprobs: int = 20) -> List[Dict[str, Any]]:
        """
        Analyze OpenWebText dataset using Together.ai
        
        Args:
            dataset_name: HuggingFace dataset name for OpenWebText
            num_samples: Number of samples to analyze from the dataset
            output_path: Optional path to save results
            chunk_size: Size of text chunks to analyze
            split: Dataset split to use ('train', 'test', etc.)
            
        Returns:
            List of analysis results
        """
        print(f"Loading OpenWebText dataset: {dataset_name}")
        print(f"Will analyze {num_samples} samples...")
        
        try:
            # Load the dataset
            dataset = load_dataset(dataset_name, split=split, streaming=True)
            results = []
            
            # Take the specified number of samples
            for i, example in enumerate(dataset.take(num_samples)):
                print(f"Processing sample {i+1}/{num_samples}")
                
                text = example.get('text', '')
                if not text:
                    print(f"  Warning: Empty text in sample {i+1}")
                    continue
                
                # Split text into chunks if it's too long
                chunks = self._split_text(text, chunk_size)
                
                for j, chunk in enumerate(chunks):
                    if len(chunk.strip()) < 10:  # Skip very short chunks
                        continue
                        
                    print(f"  Analyzing chunk {j+1}/{len(chunks)}")
                    
                    analysis = self.get_token_probabilities(chunk, top_logprobs=top_logprobs)
                    analysis["dataset_name"] = dataset_name
                    analysis["sample_index"] = i
                    analysis["chunk_index"] = j
                    analysis["chunk_text"] = chunk
                    analysis["original_text_length"] = len(text)
                    
                    results.append(analysis)
                    
                    # Optional: Add a small delay to avoid rate limiting
                    # time.sleep(0.1)
            
            # Save results if output path provided
            if output_path:
                self._save_results(results, output_path)
            
            return results
            
        except Exception as e:
            print(f"Error analyzing OpenWebText dataset: {e}")
            logging.error(f"OpenWebText analysis error: {e}", exc_info=True)
            return []
    
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
        
        total_tokens = sum(result.get("token_count", 0) for result in results)
        total_chunks = len(results)
        
        print(f"\n=== Analysis Summary ===")
        print(f"Total chunks analyzed: {total_chunks}")
        print(f"Total tokens generated: {total_tokens}")
        print(f"Average tokens per chunk: {total_tokens/total_chunks:.2f}")
        
        # Find tokens with highest and lowest confidence
        all_tokens = []
        for result in results:
            all_tokens.extend(result.get("tokens", []))
        
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
    parser.add_argument("--input", help="Input text file or directory")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--model", help="Model to use for analysis")
    parser.add_argument("--chunk-size", type=int, default=500, 
                       help="Text chunk size for processing")
    parser.add_argument("--max-tokens", type=int, default=50,
                       help="Maximum tokens to generate per chunk")
    parser.add_argument("--openwebtext", action="store_true",
                       help="Analyze OpenWebText dataset instead of local files")
    parser.add_argument("--dataset", default="Bingsu/openwebtext_20p",
                       help="OpenWebText dataset name (default: Bingsu/openwebtext_20p)")
    parser.add_argument("--num-samples", type=int, default=100,
                       help="Number of samples to analyze from OpenWebText")
    
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
    
    if args.openwebtext:
        # Analyze OpenWebText dataset
        print(f"Analyzing OpenWebText dataset: {args.dataset}")
        results = analyzer.analyze_openwebtext(
            dataset_name=args.dataset,
            num_samples=args.num_samples,
            output_path=args.output,
            chunk_size=args.chunk_size
        )
        analyzer.print_analysis_summary(results)
    elif args.input:
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
    else:
        print("Error: Either --input or --openwebtext flag is required")
        parser.print_help()


if __name__ == "__main__":
    main()