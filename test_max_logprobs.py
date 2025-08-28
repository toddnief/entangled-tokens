#!/usr/bin/env python3
"""
Test script to find Together.ai's maximum logprobs limit
and verify top-k logprobs extraction
"""

import os
from together_logits_analyzer import TogetherLogitsAnalyzer
from dotenv import load_dotenv

def test_max_logprobs():
    load_dotenv()
    api_key = os.getenv("TOGETHER_API_KEY")
    
    if not api_key:
        print("Error: TOGETHER_API_KEY not found")
        return
    
    analyzer = TogetherLogitsAnalyzer(api_key)
    test_text = "The quick brown fox jumps over"
    
    # Test different logprobs values to find the maximum
    test_values = [5, 10, 20, 50, 100, 200]
    
    for logprobs_val in test_values:
        print(f"\n=== Testing logprobs={logprobs_val} ===")
        try:
            result = analyzer.get_token_probabilities(
                text=test_text, 
                max_tokens=10, 
                top_logprobs=logprobs_val
            )
            
            if result:
                print(f"✅ Success with logprobs={logprobs_val}")
                print(f"Generated tokens: {len(result.get('tokens', []))}")
                print(f"Position alternatives: {len(result.get('position_alternatives', []))}")
                
                # Show first position alternatives if available
                if result.get('position_alternatives'):
                    first_pos = result['position_alternatives'][0]
                    print(f"Position 0 has {len(first_pos['alternatives'])} alternatives")
                    print("Top 3 alternatives:")
                    for alt in first_pos['alternatives'][:3]:
                        print(f"  '{alt['token']}': {alt['probability_percent']:.2f}%")
            else:
                print(f"❌ No result returned for logprobs={logprobs_val}")
                
        except Exception as e:
            print(f"❌ Failed with logprobs={logprobs_val}: {e}")
            if "maximum" in str(e).lower():
                print("Found maximum limit!")
                break

if __name__ == "__main__":
    test_max_logprobs()