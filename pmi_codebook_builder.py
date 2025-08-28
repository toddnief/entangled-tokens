#!/usr/bin/env python3
"""
PMI Codebook Builder
Creates a mapping from each token to its highest PMI alternative token
"""

import os
import json
import math
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass
import argparse


@dataclass
class PMIMapping:
    """Structure for token-to-highest-PMI-token mapping"""
    original_token: str
    original_token_id: int
    best_alternative_token: str
    best_alternative_token_id: int
    max_pmi: float
    contexts_count: int
    avg_pmi: float
    marginal_prob_original: float
    marginal_prob_alternative: float


class PMICodebookBuilder:
    """Builds codebook mapping each token to its highest PMI alternative"""
    
    def __init__(self, min_occurrences: int = 10, min_contexts: int = 3):
        """
        Initialize codebook builder
        
        Args:
            min_occurrences: Minimum times a token must appear to be included
            min_contexts: Minimum contexts where alternative must appear for reliable PMI
        """
        self.min_occurrences = min_occurrences
        self.min_contexts = min_contexts
        
        # Data structures for building codebook
        self.token_alternatives = defaultdict(list)  # token_id -> List[(alt_token_id, pmi, context_info)]
        self.token_marginals = {}  # token_id -> marginal_probability
        self.token_strings = {}  # token_id -> string
        self.token_counts = Counter()  # token_id -> total_count
        
        print(f"Initializing PMI Codebook Builder:")
        print(f"  Min occurrences: {min_occurrences}")
        print(f"  Min contexts: {min_contexts}")
    
    def load_results_directory(self, results_dir: str):
        """Load all results from a directory containing batch files"""
        results_dir = Path(results_dir)
        
        print(f"Loading PMI results from: {results_dir}")
        
        # Load token counts first
        token_counts_file = results_dir / "final_token_counts.json"
        if token_counts_file.exists():
            with open(token_counts_file, 'r') as f:
                counts_data = json.load(f)
                self.token_counts = Counter({int(k): v for k, v in counts_data["token_counts"].items()})
                self.token_strings = {int(k): v for k, v in counts_data["token_strings"].items()}
                print(f"✅ Loaded token counts: {len(self.token_counts)} unique tokens")
        else:
            print("⚠️  No token counts file found")
        
        # Load all batch files
        batch_files = list(results_dir.glob("batch_*.json"))
        print(f"Found {len(batch_files)} batch files")
        
        for batch_file in batch_files:
            print(f"Loading {batch_file.name}...")
            with open(batch_file, 'r') as f:
                batch_data = json.load(f)
            
            self._process_batch_data(batch_data)
        
        print(f"✅ Loaded data for {len(self.token_alternatives)} tokens with alternatives")
    
    def _process_batch_data(self, batch_data: List[Dict]):
        """Process a batch of results to extract PMI alternatives"""
        
        for sample in batch_data:
            # Get marginal probabilities from token frequency stats
            if "token_frequency_stats" in sample:
                # This gives us the marginal probabilities
                pass
            
            # Process each token in the sample
            for token_data in sample.get("tokens", []):
                token_id = token_data["token_id"]
                token_str = token_data["token"]
                marginal_prob = token_data.get("marginal_probability", 0)
                
                # Store token info
                self.token_strings[token_id] = token_str
                if marginal_prob > 0:
                    self.token_marginals[token_id] = marginal_prob
                
                # Process alternatives for this token position
                for alt in token_data.get("top_k_alternatives", []):
                    alt_token_id = alt["token_id"]
                    alt_token_str = alt["token"]
                    alt_conditional_prob = alt["probability"]
                    
                    # Store alternative token info
                    self.token_strings[alt_token_id] = alt_token_str
                    
                    # Calculate PMI for this alternative in this context
                    # We need the marginal probability of the alternative token
                    # For now, we'll calculate PMI when we have enough data
                    
                    # Store this alternative occurrence
                    context_info = {
                        "conditional_prob": alt_conditional_prob,
                        "position": token_data["position"],
                        "sample_info": {
                            "sample_index": sample.get("sample_index"),
                            "original_token": token_str,
                            "original_token_id": token_id
                        }
                    }
                    
                    self.token_alternatives[token_id].append((alt_token_id, alt_conditional_prob, context_info))
    
    def build_codebook(self) -> Dict[int, PMIMapping]:
        """Build the final codebook mapping each token to its highest PMI alternative"""
        
        print(f"Building PMI codebook...")
        print(f"Processing {len(self.token_alternatives)} tokens...")
        
        codebook = {}
        
        for original_token_id, alternatives_list in self.token_alternatives.items():
            # Skip tokens that don't appear frequently enough
            if self.token_counts[original_token_id] < self.min_occurrences:
                continue
                
            original_token_str = self.token_strings.get(original_token_id, f"<{original_token_id}>")
            original_marginal = self.token_marginals.get(original_token_id, 0)
            
            if original_marginal == 0:
                continue  # Can't calculate PMI without marginal probability
            
            # Group alternatives by token ID and calculate PMI for each
            alt_pmis = defaultdict(list)  # alt_token_id -> list of PMI values
            
            for alt_token_id, conditional_prob, context_info in alternatives_list:
                alt_marginal = self.token_marginals.get(alt_token_id, 0)
                
                if alt_marginal > 0 and conditional_prob > 0:
                    # PMI = log(P(alt|context) / P(alt))
                    pmi = math.log(conditional_prob / alt_marginal)
                    alt_pmis[alt_token_id].append(pmi)
            
            # Find the alternative with the highest average PMI
            best_alt_token_id = None
            max_avg_pmi = float('-inf')
            best_alt_info = None
            
            for alt_token_id, pmi_values in alt_pmis.items():
                if len(pmi_values) < self.min_contexts:
                    continue  # Not enough contexts for reliable estimate
                
                avg_pmi = np.mean(pmi_values)
                max_pmi = np.max(pmi_values)
                
                if avg_pmi > max_avg_pmi:
                    max_avg_pmi = avg_pmi
                    best_alt_token_id = alt_token_id
                    best_alt_info = {
                        "avg_pmi": avg_pmi,
                        "max_pmi": max_pmi,
                        "contexts_count": len(pmi_values),
                        "marginal_prob": self.token_marginals.get(alt_token_id, 0)
                    }
            
            # Create mapping if we found a good alternative
            if best_alt_token_id is not None and best_alt_info is not None:
                best_alt_token_str = self.token_strings.get(best_alt_token_id, f"<{best_alt_token_id}>")
                
                mapping = PMIMapping(
                    original_token=original_token_str,
                    original_token_id=original_token_id,
                    best_alternative_token=best_alt_token_str,
                    best_alternative_token_id=best_alt_token_id,
                    max_pmi=best_alt_info["max_pmi"],
                    contexts_count=best_alt_info["contexts_count"],
                    avg_pmi=best_alt_info["avg_pmi"],
                    marginal_prob_original=original_marginal,
                    marginal_prob_alternative=best_alt_info["marginal_prob"]
                )
                
                codebook[original_token_id] = mapping
        
        print(f"✅ Built codebook with {len(codebook)} token mappings")
        return codebook
    
    def save_codebook(self, codebook: Dict[int, PMIMapping], output_path: str):
        """Save codebook to JSON file"""
        
        # Convert to JSON-serializable format
        json_codebook = {}
        
        for token_id, mapping in codebook.items():
            json_codebook[str(token_id)] = {
                "original_token": mapping.original_token,
                "original_token_id": mapping.original_token_id,
                "best_alternative_token": mapping.best_alternative_token,
                "best_alternative_token_id": mapping.best_alternative_token_id,
                "max_pmi": mapping.max_pmi,
                "avg_pmi": mapping.avg_pmi,
                "contexts_count": mapping.contexts_count,
                "marginal_prob_original": mapping.marginal_prob_original,
                "marginal_prob_alternative": mapping.marginal_prob_alternative
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_codebook, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Codebook saved to: {output_path}")
    
    def save_simple_mapping(self, codebook: Dict[int, PMIMapping], output_path: str):
        """Save a simple token -> token mapping file"""
        
        simple_mapping = {}
        
        for token_id, mapping in codebook.items():
            simple_mapping[mapping.original_token] = mapping.best_alternative_token
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(simple_mapping, f, indent=2, ensure_ascii=False, sort_keys=True)
        
        print(f"📋 Simple mapping saved to: {output_path}")
    
    def print_codebook_stats(self, codebook: Dict[int, PMIMapping]):
        """Print statistics about the codebook"""
        
        print(f"\n=== CODEBOOK STATISTICS ===")
        print(f"Total mappings: {len(codebook)}")
        
        pmis = [mapping.max_pmi for mapping in codebook.values()]
        contexts = [mapping.contexts_count for mapping in codebook.values()]
        
        print(f"PMI statistics:")
        print(f"  Mean: {np.mean(pmis):.3f}")
        print(f"  Std:  {np.std(pmis):.3f}")
        print(f"  Min:  {np.min(pmis):.3f}")
        print(f"  Max:  {np.max(pmis):.3f}")
        
        print(f"Contexts per mapping:")
        print(f"  Mean: {np.mean(contexts):.1f}")
        print(f"  Min:  {np.min(contexts)}")
        print(f"  Max:  {np.max(contexts)}")
        
        # Show top 20 highest PMI mappings
        sorted_mappings = sorted(codebook.values(), key=lambda x: x.max_pmi, reverse=True)
        
        print(f"\n=== TOP 20 HIGHEST PMI MAPPINGS ===")
        for i, mapping in enumerate(sorted_mappings[:20]):
            print(f"{i+1:2d}. '{mapping.original_token}' -> '{mapping.best_alternative_token}' "
                  f"(PMI: {mapping.max_pmi:.3f}, contexts: {mapping.contexts_count})")
    
    def analyze_results_directory(self, results_dir: str, output_dir: str):
        """Complete pipeline: load results and build codebook"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.load_results_directory(results_dir)
        
        # Build codebook
        codebook = self.build_codebook()
        
        if not codebook:
            print("❌ No codebook mappings created. Check your data and parameters.")
            return
        
        # Save results
        codebook_file = output_dir / "pmi_codebook.json"
        simple_mapping_file = output_dir / "token_to_highest_pmi_token.json"
        
        self.save_codebook(codebook, str(codebook_file))
        self.save_simple_mapping(codebook, str(simple_mapping_file))
        
        # Print statistics
        self.print_codebook_stats(codebook)
        
        print(f"\n🎉 PMI Codebook complete!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📊 Full codebook: {codebook_file}")
        print(f"📋 Simple mapping: {simple_mapping_file}")
        
        return codebook


def main():
    parser = argparse.ArgumentParser(description="Build PMI codebook from logits analysis results")
    parser.add_argument("--results-dir", "-r", required=True, 
                       help="Directory containing PMI analysis results")
    parser.add_argument("--output-dir", "-o", default="pmi_codebook_output",
                       help="Output directory for codebook files")
    parser.add_argument("--min-occurrences", type=int, default=10,
                       help="Minimum token occurrences for inclusion")
    parser.add_argument("--min-contexts", type=int, default=3,
                       help="Minimum contexts for reliable PMI estimate")
    
    args = parser.parse_args()
    
    # Build codebook
    builder = PMICodebookBuilder(
        min_occurrences=args.min_occurrences,
        min_contexts=args.min_contexts
    )
    
    builder.analyze_results_directory(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()