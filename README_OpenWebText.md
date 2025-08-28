# OpenWebText Logits Extraction with Together.ai

This project provides tools to extract token logits (probabilities) from the OpenWebText dataset using Together.ai's API.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your Together.ai API key in `.env`:
   ```
   TOGETHER_API_KEY=your_api_key_here
   ```

## Usage

### Command Line Interface

To analyze OpenWebText samples:

```bash
# Analyze 10 samples from OpenWebText
python together_logits_analyzer.py --openwebtext --num-samples 10 --output openwebtext_results.json

# Use a different dataset (20% of full OpenWebText)
python together_logits_analyzer.py --openwebtext --dataset "Bingsu/openwebtext_20p" --num-samples 50

# Customize chunk size and model
python together_logits_analyzer.py --openwebtext --num-samples 5 --chunk-size 300 --model "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
```

### Python API

```python
from together_logits_analyzer import TogetherLogitsAnalyzer
import os

# Initialize analyzer
analyzer = TogetherLogitsAnalyzer(os.getenv("TOGETHER_API_KEY"))

# Analyze OpenWebText samples
results = analyzer.analyze_openwebtext(
    dataset_name="Bingsu/openwebtext_20p",
    num_samples=100,
    output_path="results.json",
    chunk_size=500
)

# Print summary
analyzer.print_analysis_summary(results)
```

### Quick Example

```bash
python openwebtext_example.py
```

## Available OpenWebText Datasets

Due to compatibility issues with the newer datasets library, we use parquet-formatted datasets:

- **Bingsu/openwebtext_20p** (default): First 20% of OpenWebText with 33.2M rows (4.94 GB)
- **stas/openwebtext-synthetic-testing**: Smaller testing dataset

## Output Format

The tool generates JSON files with the following structure:

```json
[
  {
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "tokens": [
      {
        "token": "When",
        "logprob": -1.0859375,
        "probability": 0.3375851514160396,
        "probability_percent": 33.758515141603965,
        "token_id": null,
        "position": 0
      }
    ],
    "full_text": "Generated continuation...",
    "token_count": 50,
    "dataset_name": "Bingsu/openwebtext_20p",
    "sample_index": 0,
    "chunk_index": 0,
    "chunk_text": "Original text chunk...",
    "original_text_length": 228
  }
]
```

## Options

- `--openwebtext`: Enable OpenWebText analysis mode
- `--dataset`: Specify the HuggingFace dataset name (default: Bingsu/openwebtext_20p)
- `--num-samples`: Number of samples to analyze (default: 100)
- `--chunk-size`: Size of text chunks in characters (default: 500)
- `--output`: Output JSON file path
- `--model`: Together.ai model to use

## Notes

- The tool uses streaming to handle large datasets efficiently
- Logprobs are limited to 5 by Together.ai API
- Text is automatically chunked for processing
- Progress is displayed during processing