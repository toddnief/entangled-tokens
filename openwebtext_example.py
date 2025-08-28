#!/usr/bin/env python3
"""
Example script for analyzing OpenWebText dataset with Together.ai
"""

from together_logits_analyzer import TogetherLogitsAnalyzer
import os
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print("Error: Please set TOGETHER_API_KEY in your .env file")
        return
    
    # Initialize analyzer
    analyzer = TogetherLogitsAnalyzer(api_key)
    
    # Analyze a small sample of OpenWebText
    print("Starting OpenWebText analysis...")
    results = analyzer.analyze_openwebtext(
        dataset_name="Bingsu/openwebtext_20p",  # Parquet-based dataset for testing
        num_samples=5,  # Just 5 samples for quick testing
        output_path="openwebtext_logits_sample.json",
        chunk_size=300
    )
    
    # Print summary
    analyzer.print_analysis_summary(results)
    
    print(f"\nAnalyzed {len(results)} chunks from OpenWebText")
    print("Results saved to openwebtext_logits_sample.json")

if __name__ == "__main__":
    main()