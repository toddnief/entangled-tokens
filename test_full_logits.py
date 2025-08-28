#!/usr/bin/env python3
"""
Test script to verify we can get logits for every token position
"""

import os
from together_logits_analyzer import TogetherLogitsAnalyzer
from dotenv import load_dotenv

def test_full_token_logits():
    load_dotenv()
    api_key = os.getenv("TOGETHER_API_KEY")
    
    if not api_key:
        print("Error: TOGETHER_API_KEY not found")
        return
    
    analyzer = TogetherLogitsAnalyzer(api_key)
    test_text = "The quick brown fox"
    
    print(f"Testing logits extraction for: '{test_text}'")
    print("Using completions API with echo=True to get ALL token logits...")
    
    result = analyzer.get_token_probabilities(
        text=test_text, 
        max_tokens=10, 
        top_logprobs=5
    )
    
    if result:
        print(f"\n✅ Success!")
        print(f"Full text returned: '{result['full_text']}'")
        print(f"Total tokens with logits: {len(result.get('tokens', []))}")
        print(f"Position alternatives: {len(result.get('position_alternatives', []))}")
        
        print(f"\nAll token logits:")
        for i, token_data in enumerate(result.get('tokens', [])):
            is_input = token_data.get('is_input', False)
            token_type = "INPUT" if is_input else "GENERATED"
            print(f"  {i:2d}: '{token_data['token']}' -> {token_data['probability_percent']:.2f}% ({token_type})")
        
        print(f"\nTop alternatives for each position:")
        for pos_data in result.get('position_alternatives', []):
            pos = pos_data['position']
            alts = pos_data['alternatives'][:3]  # Show top 3
            print(f"  Position {pos:2d}:")
            for alt in alts:
                print(f"    '{alt['token']}': {alt['probability_percent']:.2f}%")
    else:
        print("❌ No result returned")

if __name__ == "__main__":
    test_full_token_logits()