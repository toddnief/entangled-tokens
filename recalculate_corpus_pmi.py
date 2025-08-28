#!/usr/bin/env python3
"""
Recalculate PMI with Corpus-Wide Statistics
Post-processes existing results to calculate proper PMI using true corpus marginal probabilities
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
from tqdm import tqdm


@dataclass
class CorpusPMIMapping:
    """Structure for token-to-highest-PMI-token mapping with corpus statistics"""
    original_token: str
    original_token_id: int
    best_alternative_token: str
    best_alternative_token_id: int
    max_pmi_corpus: float  # PMI calculated with true corpus marginals
    avg_pmi_corpus: float
    contexts_count: int
    corpus_marginal_original: float
    corpus_marginal_alternative: float


class CorpusPMIRecalculator:
    """Recalculates PMI using true corpus-wide marginal probabilities"""
    
    def __init__(self, min_occurrences: int = 15, min_contexts: int = 5):
        """
        Initialize PMI recalculator
        
        Args:
            min_occurrences: Minimum times a token must appear in corpus
            min_contexts: Minimum contexts where alternative must appear
        """
        self.min_occurrences = min_occurrences
        self.min_contexts = min_contexts
        
        # Corpus-wide statistics
        self.corpus_token_counts = Counter()  # token_id -> total count across corpus
        self.corpus_total_tokens = 0
        self.token_strings = {}  # token_id -> string representation
        
        # PMI calculation data
        self.token_context_data = defaultdict(list)  # token_id -> [(context_info, alternatives)]
        
        print(f"Initializing Corpus PMI Recalculator:")
        print(f"  Min occurrences: {min_occurrences}")
        print(f"  Min contexts: {min_contexts}")
    
    def load_and_process_results(self, results_dir: str):
        """Load results and build corpus-wide statistics"""
        results_dir = Path(results_dir)
        
        print(f"Loading results from: {results_dir}")
        
        # Step 1: Load final token counts to get TRUE corpus marginals
        self._load_corpus_token_counts(results_dir)
        
        # Step 2: Load all batch files to get context data
        self._load_context_data(results_dir)
        
        print(f"✅ Corpus statistics loaded:")
        print(f"   Total tokens in corpus: {self.corpus_total_tokens:,}")
        print(f"   Unique tokens: {len(self.corpus_token_counts):,}")
        print(f"   Tokens with context data: {len(self.token_context_data):,}")
    
    def _load_corpus_token_counts(self, results_dir: Path):
        """Load the final token counts - these are our TRUE corpus marginals"""
        token_counts_file = results_dir / "final_token_counts.json"
        
        if not token_counts_file.exists():
            raise FileNotFoundError(f"Token counts file not found: {token_counts_file}")
        
        print("Loading corpus-wide token counts...")
        with open(token_counts_file, 'r') as f:
            counts_data = json.load(f)
        
        self.corpus_token_counts = Counter({int(k): v for k, v in counts_data["token_counts"].items()})
        self.token_strings = {int(k): v for k, v in counts_data["token_strings"].items()}
        self.corpus_total_tokens = counts_data["total_tokens"]
        
        print(f"✅ Loaded {len(self.corpus_token_counts)} unique tokens from corpus")
        print(f"   Total tokens: {self.corpus_total_tokens:,}")
    
    def _load_context_data(self, results_dir: Path):
        """Load context data from all batch files"""
        batch_files = list(results_dir.glob("batch_*.json"))
        print(f"Loading context data from {len(batch_files)} batch files...")
        
        for batch_file in tqdm(batch_files, desc="Processing batch files"):
            with open(batch_file, 'r') as f:
                batch_data = json.load(f)
            
            for sample in batch_data:
                for token_data in sample.get("tokens", []):
                    token_id = token_data["token_id"]
                    token_position = token_data["position"]
                    
                    # Store context information and alternatives for this token occurrence
                    context_info = {
                        "sample_index": sample.get("sample_index", -1),
                        "position": token_position,
                        "conditional_prob": token_data["probability"]
                    }
                    
                    # Store all alternatives that appeared in this context
                    alternatives = []
                    for alt in token_data.get("top_k_alternatives", []):
                        alternatives.append({
                            "token_id": alt["token_id"],
                            "token": alt["token"],
                            "conditional_prob": alt["probability"],
                            "rank": alt["rank"]
                        })
                    
                    self.token_context_data[token_id].append({
                        "context_info": context_info,
                        "alternatives": alternatives
                    })
    
    def calculate_corpus_pmi_codebook(self) -> Dict[int, CorpusPMIMapping]:
        """Calculate PMI codebook using true corpus marginal probabilities"""
        print(f"Calculating corpus PMI for {len(self.token_context_data)} tokens...")
        
        codebook = {}
        
        for original_token_id, context_list in tqdm(self.token_context_data.items(), desc="Building PMI codebook"):
            # Skip tokens that don't appear frequently enough
            if self.corpus_token_counts[original_token_id] < self.min_occurrences:
                continue
            
            original_token_str = self.token_strings.get(original_token_id, f"<{original_token_id}>")
            
            # Calculate TRUE corpus marginal for original token
            original_corpus_marginal = self.corpus_token_counts[original_token_id] / self.corpus_total_tokens
            
            # Collect all alternative tokens and their conditional probabilities
            alternative_data = defaultdict(list)  # alt_token_id -> list of conditional_probs
            
            for context_data in context_list:
                for alt in context_data["alternatives"]:
                    alt_token_id = alt["token_id"]
                    conditional_prob = alt["conditional_prob"]
                    
                    if conditional_prob > 0:  # Valid probability
                        alternative_data[alt_token_id].append(conditional_prob)
            
            # Find the alternative with highest PMI using corpus marginals
            best_alt_token_id = None
            max_avg_pmi = float('-inf')
            best_alt_info = None
            
            for alt_token_id, conditional_probs in alternative_data.items():
                if len(conditional_probs) < self.min_contexts:
                    continue
                
                # Calculate TRUE corpus marginal for alternative token
                alt_corpus_marginal = self.corpus_token_counts[alt_token_id] / self.corpus_total_tokens
                
                if alt_corpus_marginal == 0:
                    continue
                
                # Calculate PMI for each occurrence using corpus marginal
                pmis = []
                for cond_prob in conditional_probs:
                    pmi = math.log(cond_prob / alt_corpus_marginal)
                    pmis.append(pmi)
                
                avg_pmi = np.mean(pmis)
                max_pmi = np.max(pmis)
                
                if avg_pmi > max_avg_pmi:
                    max_avg_pmi = avg_pmi
                    best_alt_token_id = alt_token_id
                    best_alt_info = {
                        "avg_pmi": avg_pmi,
                        "max_pmi": max_pmi,
                        "contexts_count": len(pmis),
                        "corpus_marginal": alt_corpus_marginal
                    }
            
            # Create mapping if we found a good alternative
            if best_alt_token_id is not None and best_alt_info is not None:
                best_alt_token_str = self.token_strings.get(best_alt_token_id, f"<{best_alt_token_id}>")
                
                mapping = CorpusPMIMapping(
                    original_token=original_token_str,
                    original_token_id=original_token_id,
                    best_alternative_token=best_alt_token_str,
                    best_alternative_token_id=best_alt_token_id,
                    max_pmi_corpus=best_alt_info["max_pmi"],
                    avg_pmi_corpus=best_alt_info["avg_pmi"],
                    contexts_count=best_alt_info["contexts_count"],
                    corpus_marginal_original=original_corpus_marginal,
                    corpus_marginal_alternative=best_alt_info["corpus_marginal"]
                )
                
                codebook[original_token_id] = mapping
        
        print(f"✅ Built corpus PMI codebook with {len(codebook)} mappings")
        return codebook
    
    def save_codebook(self, codebook: Dict[int, CorpusPMIMapping], output_path: str):
        """Save corpus PMI codebook to JSON"""
        json_codebook = {}
        
        for token_id, mapping in codebook.items():
            json_codebook[str(token_id)] = {
                "original_token": mapping.original_token,
                "original_token_id": mapping.original_token_id,
                "best_alternative_token": mapping.best_alternative_token,
                "best_alternative_token_id": mapping.best_alternative_token_id,
                "max_pmi_corpus": mapping.max_pmi_corpus,
                "avg_pmi_corpus": mapping.avg_pmi_corpus,
                "contexts_count": mapping.contexts_count,
                "corpus_marginal_original": mapping.corpus_marginal_original,
                "corpus_marginal_alternative": mapping.corpus_marginal_alternative
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_codebook, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Corpus PMI codebook saved to: {output_path}")
    
    def save_simple_mapping(self, codebook: Dict[int, CorpusPMIMapping], output_path: str):
        """Save simple token -> highest PMI token mapping"""
        simple_mapping = {}
        
        for mapping in codebook.values():
            simple_mapping[mapping.original_token] = mapping.best_alternative_token
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(simple_mapping, f, indent=2, ensure_ascii=False, sort_keys=True)
        
        print(f"📋 Simple token mapping saved to: {output_path}")
    
    def save_detailed_analysis(self, codebook: Dict[int, CorpusPMIMapping], output_path: str):
        """Save detailed analysis with corpus statistics"""
        
        # Calculate statistics
        pmis = [m.max_pmi_corpus for m in codebook.values()]
        contexts = [m.contexts_count for m in codebook.values()]
        marginals_orig = [m.corpus_marginal_original for m in codebook.values()]
        marginals_alt = [m.corpus_marginal_alternative for m in codebook.values()]
        
        analysis = {
            "corpus_statistics": {
                "total_tokens_in_corpus": self.corpus_total_tokens,
                "unique_tokens": len(self.corpus_token_counts),
                "tokens_with_mappings": len(codebook)
            },
            "pmi_statistics": {
                "mean_pmi": float(np.mean(pmis)),
                "std_pmi": float(np.std(pmis)),
                "min_pmi": float(np.min(pmis)),
                "max_pmi": float(np.max(pmis)),
                "median_pmi": float(np.median(pmis))
            },
            "context_statistics": {
                "mean_contexts": float(np.mean(contexts)),
                "min_contexts": int(np.min(contexts)),
                "max_contexts": int(np.max(contexts))
            },
            "marginal_probability_statistics": {
                "original_tokens": {
                    "mean": float(np.mean(marginals_orig)),
                    "min": float(np.min(marginals_orig)),
                    "max": float(np.max(marginals_orig))
                },
                "alternative_tokens": {
                    "mean": float(np.mean(marginals_alt)),
                    "min": float(np.min(marginals_alt)),
                    "max": float(np.max(marginals_alt))
                }
            },
            "top_20_highest_pmi": []
        }
        
        # Add top 20 mappings
        sorted_mappings = sorted(codebook.values(), key=lambda x: x.max_pmi_corpus, reverse=True)
        for i, mapping in enumerate(sorted_mappings[:20]):
            analysis["top_20_highest_pmi"].append({
                "rank": i + 1,
                "original_token": mapping.original_token,
                "best_alternative": mapping.best_alternative_token,
                "pmi": mapping.max_pmi_corpus,
                "contexts": mapping.contexts_count,
                "original_freq": mapping.corpus_marginal_original,
                "alternative_freq": mapping.corpus_marginal_alternative
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Detailed analysis saved to: {output_path}")
    
    def print_results_summary(self, codebook: Dict[int, CorpusPMIMapping]):
        """Print summary of results"""
        pmis = [m.max_pmi_corpus for m in codebook.values()]
        
        print(f"\n=== CORPUS PMI CODEBOOK RESULTS ===")
        print(f"Corpus size: {self.corpus_total_tokens:,} tokens")
        print(f"Unique tokens: {len(self.corpus_token_counts):,}")
        print(f"Token mappings created: {len(codebook):,}")
        print(f"Coverage: {len(codebook)/len(self.corpus_token_counts)*100:.1f}% of unique tokens")
        
        print(f"\nPMI Statistics:")
        print(f"  Mean: {np.mean(pmis):.3f}")
        print(f"  Std:  {np.std(pmis):.3f}")
        print(f"  Range: [{np.min(pmis):.3f}, {np.max(pmis):.3f}]")
        
        # Show top 10 mappings
        sorted_mappings = sorted(codebook.values(), key=lambda x: x.max_pmi_corpus, reverse=True)
        print(f"\nTop 10 Highest PMI Mappings:")
        for i, mapping in enumerate(sorted_mappings[:10]):
            print(f"{i+1:2d}. '{mapping.original_token}' → '{mapping.best_alternative_token}' "
                  f"(PMI: {mapping.max_pmi_corpus:.3f})")
    
    def process_results_directory(self, results_dir: str, output_dir: str):
        """Complete pipeline: load results and calculate corpus PMI"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and process
        self.load_and_process_results(results_dir)
        
        # Calculate PMI with corpus statistics
        codebook = self.calculate_corpus_pmi_codebook()
        
        if not codebook:
            print("❌ No PMI mappings created. Check your parameters.")
            return
        
        # Save results
        codebook_file = output_dir / "corpus_pmi_codebook.json"
        simple_mapping_file = output_dir / "token_to_highest_pmi_token.json"
        analysis_file = output_dir / "corpus_pmi_analysis.json"
        
        self.save_codebook(codebook, str(codebook_file))
        self.save_simple_mapping(codebook, str(simple_mapping_file))
        self.save_detailed_analysis(codebook, str(analysis_file))
        
        # Print results
        self.print_results_summary(codebook)
        
        print(f"\n🎉 Corpus PMI Analysis Complete!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📊 Full codebook: {codebook_file}")
        print(f"📋 Simple mapping: {simple_mapping_file}")
        print(f"📈 Detailed analysis: {analysis_file}")
        
        return codebook


def main():
    parser = argparse.ArgumentParser(description="Recalculate PMI with corpus-wide marginal probabilities")
    parser.add_argument("--results-dir", "-r", required=True,
                       help="Directory containing original PMI analysis results")
    parser.add_argument("--output-dir", "-o", default="corpus_pmi_codebook",
                       help="Output directory for corpus PMI results")
    parser.add_argument("--min-occurrences", type=int, default=15,
                       help="Minimum token occurrences in corpus")
    parser.add_argument("--min-contexts", type=int, default=5,
                       help="Minimum contexts for reliable PMI")
    
    args = parser.parse_args()
    
    recalculator = CorpusPMIRecalculator(
        min_occurrences=args.min_occurrences,
        min_contexts=args.min_contexts
    )
    
    recalculator.process_results_directory(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()