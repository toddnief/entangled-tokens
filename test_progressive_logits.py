#!/usr/bin/env python3
"""
Test the progressive token analysis approach to get logits for every position
"""

import os
from together_logits_analyzer import TogetherLogitsAnalyzer
from dotenv import load_dotenv

def test_progressive_logits():
    load_dotenv()
    api_key = os.getenv("TOGETHER_API_KEY")
    
    if not api_key:
        print("Error: TOGETHER_API_KEY not found")
        return
    
    analyzer = TogetherLogitsAnalyzer(api_key)
    test_text = "The quick brown fox"
    
    print(f"Testing progressive logits extraction for: '{test_text}'")
    print("This will make multiple API calls to analyze each token position...")
    
    result = analyzer.get_token_probabilities_progressive(
        text=test_text, 
        top_logprobs=5
    )
    
    if result:
        print(f"\n✅ Success!")
        print(f"Method used: {result.get('method', 'unknown')}")
        print(f"Full text: '{result['full_text']}'")
        print(f"Total tokens analyzed: {len(result.get('tokens', []))}")
        print(f"Position alternatives: {len(result.get('position_alternatives', []))}")
        
        print(f"\nToken-by-token analysis:")
        for i, token_data in enumerate(result.get('tokens', [])):
            matched = token_data.get('matched', False)
            estimated = token_data.get('estimated', False)
            status = "ESTIMATED" if estimated else ("MATCHED" if matched else "NOT_FOUND")
            print(f"  {i:2d}: '{token_data['token']}' -> {token_data['probability_percent']:.2f}% [{status}]")
        
        print(f"\nTop alternatives for first few positions:")
        for pos_data in result.get('position_alternatives', [])[:3]:
            pos = pos_data['position']
            alts = pos_data['alternatives'][:3]  # Show top 3
            print(f"  Position {pos:2d}:")
            for alt in alts:
                print(f"    '{alt['token']}': {alt['probability_percent']:.2f}%")
    else:
        print("❌ No result returned")

if __name__ == "__main__":
    test_progressive_logits()