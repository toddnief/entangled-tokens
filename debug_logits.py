#!/usr/bin/env python3
"""
Debug script to understand logits extraction and perplexity issues
"""

import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM

def debug_logits_extraction():
    """Debug the logits extraction process step by step"""
    
    # Use a small model for debugging
    model_name = "gpt2"  # Small and fast
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    
    # Test with simple text
    text = "The quick brown fox"
    print(f"\nAnalyzing text: '{text}'")
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"][0]
    
    print(f"Tokens: {[tokenizer.decode([t]) for t in input_ids]}")
    print(f"Token IDs: {input_ids.tolist()}")
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(input_ids.unsqueeze(0))
        logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]
    
    print(f"Logits shape: {logits.shape}")
    
    # The key insight: logits[i] predicts token at position i+1
    # So to get probability of token at position i, we need logits[i-1]
    
    print(f"\n=== DEBUGGING LOGIT POSITIONS ===")
    
    for pos in range(len(input_ids)):
        token_id = input_ids[pos].item()
        token = tokenizer.decode([token_id])
        
        print(f"\nPosition {pos}: Token '{token}' (ID: {token_id})")
        
        if pos == 0:
            print("  First token - no previous context to predict from")
            print("  Skipping probability calculation for first token")
            continue
        
        # For token at position i, we use logits from position i-1
        # because logits[i-1] contains predictions for position i
        prev_logits = logits[pos - 1]  # Logits that predict current token
        
        # Convert to probabilities
        probs = torch.softmax(prev_logits, dim=-1)
        
        # Get probability of current token
        token_prob = probs[token_id].item()
        token_logprob = math.log(token_prob)
        
        print(f"  Probability: {token_prob:.6f}")
        print(f"  Log probability: {token_logprob:.4f}")
        
        # Find rank of this token
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        rank = (sorted_indices == token_id).nonzero(as_tuple=True)[0].item()
        
        print(f"  Rank in vocabulary: {rank + 1}")
        
        # Show top 5 alternatives that model would have predicted
        print(f"  Top 5 alternatives model would predict:")
        for i in range(5):
            alt_token_id = sorted_indices[i].item()
            alt_token = tokenizer.decode([alt_token_id])
            alt_prob = sorted_probs[i].item()
            print(f"    {i+1}. '{alt_token}': {alt_prob:.6f}")
    
    # Calculate corrected perplexity (skip first token)
    valid_tokens = []
    valid_logprobs = []
    
    for pos in range(1, len(input_ids)):  # Skip first token
        token_id = input_ids[pos].item()
        prev_logits = logits[pos - 1]
        log_probs = torch.log_softmax(prev_logits, dim=-1)
        token_logprob = log_probs[token_id].item()
        
        valid_tokens.append(tokenizer.decode([token_id]))
        valid_logprobs.append(token_logprob)
    
    if valid_logprobs:
        avg_logprob = sum(valid_logprobs) / len(valid_logprobs)
        perplexity = math.exp(-avg_logprob)
        
        print(f"\n=== CORRECTED CALCULATION ===")
        print(f"Valid tokens (excluding first): {valid_tokens}")
        print(f"Average log probability: {avg_logprob:.4f}")
        print(f"Perplexity: {perplexity:.2f}")
        
        print(f"\nToken-by-token log probabilities:")
        for token, logprob in zip(valid_tokens, valid_logprobs):
            print(f"  '{token}': {logprob:.4f}")
    
    print(f"\n=== COMPARISON WITH COMMON ISSUE ===")
    # Show what happens if we incorrectly use same-position logits
    wrong_logprobs = []
    for pos in range(len(input_ids)):
        token_id = input_ids[pos].item()
        same_pos_logits = logits[pos]  # WRONG: this predicts NEXT token
        log_probs = torch.log_softmax(same_pos_logits, dim=-1)
        wrong_logprob = log_probs[token_id].item()
        wrong_logprobs.append(wrong_logprob)
    
    wrong_avg = sum(wrong_logprobs) / len(wrong_logprobs)
    wrong_perplexity = math.exp(-wrong_avg)
    
    print(f"Wrong method (same position) average log prob: {wrong_avg:.4f}")
    print(f"Wrong method perplexity: {wrong_perplexity:.2f}")
    print("^ This is likely what's causing your high perplexity!")

if __name__ == "__main__":
    debug_logits_extraction()