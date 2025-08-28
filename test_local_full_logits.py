#!/usr/bin/env python3
"""
Test script for local full vocabulary logits extraction
"""

import os
import torch
from local_logits_analyzer import LocalLogitsAnalyzer

def test_local_full_logits():
    """Test the local analyzer with small model first"""
    
    print("=== Testing Local Full Logits Analyzer ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Start with a smaller model for testing
    model_name = "gpt2"  # Small model for testing
    # For A100, use: "meta-llama/Llama-2-7b-hf" or "meta-llama/Llama-2-13b-hf"
    
    print(f"\nInitializing analyzer with model: {model_name}")
    
    try:
        analyzer = LocalLogitsAnalyzer(
            model_name=model_name,
            backend="transformers",
            device="auto",
            dtype="float16"  # Use float16 for efficiency on A100
        )
        
        # Test with simple text
        test_text = "The quick brown fox jumps"
        print(f"\nTesting with: '{test_text}'")
        
        result = analyzer.get_full_logits(
            text=test_text,
            return_full_vocab=False,  # Set to True for full vocab (memory intensive)
            top_k=50  # Get top 50 alternatives per position
        )
        
        print(f"\n✅ SUCCESS!")
        print(f"Model: {result['model']}")
        print(f"Vocabulary size: {result['vocab_size']:,} tokens")
        print(f"Sequence length: {result['sequence_length']} tokens")
        print(f"Perplexity: {result['summary']['perplexity']:.2f}")
        
        print(f"\nFirst 5 tokens with full logits:")
        for i, token_data in enumerate(result['tokens'][:5]):
            print(f"  Position {i}: '{token_data['token']}'")
            print(f"    Probability: {token_data['probability']:.6f}")
            print(f"    Rank in vocab: {token_data['rank']:,}")
            print(f"    Top 3 alternatives:")
            for j, alt in enumerate(token_data['top_k_alternatives'][:3]):
                print(f"      {j+1}. '{alt['token']}': {alt['probability']:.6f}")
            print()
        
        print("🎉 Full vocabulary logits extraction working!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_a100_setup():
    """Test with A100-optimized settings"""
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - skipping A100 test")
        return
        
    gpu_name = torch.cuda.get_device_name()
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"\n=== A100 Setup Test ===")
    print(f"GPU: {gpu_name}")
    print(f"Memory: {gpu_memory_gb:.1f}GB")
    
    if "A100" not in gpu_name:
        print(f"⚠️  Not running on A100, but testing with available GPU")
    
    # Use larger model for A100
    model_name = "meta-llama/Llama-2-7b-hf"
    max_memory = min(70, gpu_memory_gb - 5)  # Leave 5GB free
    
    print(f"Testing with: {model_name}")
    print(f"Max memory: {max_memory}GB")
    
    try:
        analyzer = LocalLogitsAnalyzer(
            model_name=model_name,
            backend="transformers",
            device="cuda",
            max_memory_gb=max_memory,
            dtype="float16"
        )
        
        # Test with longer text
        test_text = "Artificial intelligence and machine learning have revolutionized the way we process information."
        
        result = analyzer.get_full_logits(
            text=test_text,
            return_full_vocab=False,
            top_k=1000  # Get top 1000 alternatives
        )
        
        print(f"\n✅ A100 test successful!")
        print(f"Processed {result['sequence_length']} tokens")
        print(f"Vocabulary size: {result['vocab_size']:,}")
        print(f"Got top-{len(result['tokens'][0]['top_k_alternatives'])} alternatives per position")
        
        return True
        
    except Exception as e:
        print(f"❌ A100 test failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting local logits analyzer tests...\n")
    
    # Test 1: Basic functionality
    basic_success = test_local_full_logits()
    
    # Test 2: A100 optimized (if available)
    if basic_success:
        test_a100_setup()
    else:
        print("Skipping A100 test due to basic test failure")
    
    print("\n" + "="*50)
    print("Test complete!")
    if basic_success:
        print("✅ Ready for full vocabulary logits extraction!")
        print("\nNext steps:")
        print("1. Install requirements: pip install -r requirements.txt")
        print("2. Run on A100: python local_logits_analyzer.py --openwebtext --num-samples 1000")
        print("3. For full vocab: add --full-vocab flag (warning: very large files)")
    else:
        print("❌ Setup needs attention - check error messages above")