#!/usr/bin/env python3
"""
Test script to get logits primarily for input tokens with minimal generation
"""

import os
from together_logits_analyzer import TogetherLogitsAnalyzer
from dotenv import load_dotenv

def test_input_focused_logits():
    load_dotenv()
    api_key = os.getenv("TOGETHER_API_KEY")
    
    if not api_key:
        print("Error: TOGETHER_API_KEY not found")
        return
    
    analyzer = TogetherLogitsAnalyzer(api_key)
    test_text = "The quick brown fox jumps over the lazy dog"
    
    print(f"Testing input-focused logits extraction for: '{test_text}'")
    print("Using completions API with echo=True and max_tokens=1 to focus on input...")
    
    result = analyzer.get_token_probabilities(
        text=test_text, 
        max_tokens=1,  # Generate minimal tokens to focus on input logits
        top_logprobs=5
    )
    
    if result:
        print(f"\n✅ Success!")
        print(f"Full text returned: '{result['full_text']}'")
        print(f"Total tokens with logits: {len(result.get('tokens', []))}")
        print(f"Position alternatives: {len(result.get('position_alternatives', []))}")
        
        print(f"\nAll token logits:")
        for i, token_data in enumerate(result.get('tokens', [])):
            print(f"  {i:2d}: '{token_data['token']}' -> {token_data['probability_percent']:.2f}%")
        
        print(f"\nFirst 5 position alternatives:")
        for pos_data in result.get('position_alternatives', [])[:5]:
            pos = pos_data['position']
            alts = pos_data['alternatives'][:3]  # Show top 3
            print(f"  Position {pos:2d}:")
            for alt in alts:
                print(f"    '{alt['token']}': {alt['probability_percent']:.2f}%")
                
        # Analyze which tokens are likely input vs generated
        original_words = test_text.lower().split()
        print(f"\nAnalyzing input vs generated tokens:")
        print(f"Original text words: {original_words}")
        
    else:
        print("❌ No result returned")

if __name__ == "__main__":
    test_input_focused_logits()