# Together.ai Logits Analyzer

A Python script to analyze token probabilities over a corpus of text using Together.ai's API with logprobs support.

## Features

- Get full token probabilities (logits) from Together.ai models
- Process single text inputs or entire corpus directories
- Export results to JSON format
- Detailed probability analysis and summaries
- Support for various text file formats (.txt, .md)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Get your Together.ai API key from [together.ai](https://api.together.ai)

## Usage

### Analyze Direct Text Input
```bash
python together_logits_analyzer.py --api-key YOUR_API_KEY --input "What is the meaning of life?"
```

### Analyze Text Files
```bash
python together_logits_analyzer.py --api-key YOUR_API_KEY --input sample_text.txt --output results.json
```

### Analyze Corpus Directory
```bash
python together_logits_analyzer.py --api-key YOUR_API_KEY --input /path/to/corpus/ --output corpus_analysis.json
```

### Advanced Options
```bash
python together_logits_analyzer.py \
  --api-key YOUR_API_KEY \
  --input corpus/ \
  --output results.json \
  --model "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo" \
  --chunk-size 1000 \
  --max-tokens 100
```

## Command Line Arguments

- `--api-key`: Your Together.ai API key (required)
- `--input`: Input text, file path, or directory (required)
- `--output`: Output JSON file path (optional)
- `--model`: Model to use (default: meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo)
- `--chunk-size`: Text chunk size for processing (default: 500)
- `--max-tokens`: Maximum tokens to generate per chunk (default: 50)

## Output Format

The script outputs detailed token analysis including:
- Token text
- Log probability
- Probability percentage
- Token ID
- Position in sequence

Example output:
```json
{
  "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
  "tokens": [
    {
      "token": "The",
      "logprob": -0.1234,
      "probability": 0.8841,
      "probability_percent": 88.41,
      "token_id": 791,
      "position": 0
    }
  ],
  "full_text": "Generated response text",
  "token_count": 10
}
```

## Example Test

Test with the provided sample:
```bash
python together_logits_analyzer.py --api-key YOUR_API_KEY --input sample_text.txt
```