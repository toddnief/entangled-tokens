# Configuration-Based Logits Extraction Runner

This system allows you to run logits extraction on multiple models with configurable parameters using YAML configuration files.

## Quick Start

### 1. Run with Default Config (1,000 samples across multiple models)
```bash
python run_logits_extraction.py -c configs/default_config.yaml
```

### 2. Run with Small Test Config (10 samples)
```bash
python run_logits_extraction.py -c configs/small_test_config.yaml
```

### 3. Override Number of Samples
```bash
# Run 1,000 samples instead of config default
python run_logits_extraction.py -c configs/small_test_config.yaml -n 1000
```

## Configuration Files

### Creating Custom Configs

Copy and modify `configs/default_config.yaml`:

```yaml
# General settings
dataset_name: "Bingsu/openwebtext_20p"
num_samples: 1000  # Number of samples to analyze
chunk_size: 500    # Text chunk size
max_tokens: 50     # Max tokens to generate
split: "train"     # Dataset split
output_dir: "outputs"

# Model configurations
models:
  - name: "my-model-1"
    model_id: "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    temperature: 0.7
    logprobs: 5
  
  - name: "my-model-2" 
    model_id: "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    temperature: 0.7
    logprobs: 5

# Advanced settings
random_seed: 42
save_intermediate: true
verbose: true
```

### Available Models

Common Together.ai models you can use:

- `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`
- `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo`
- `meta-llama/Llama-3.2-3B-Instruct-Turbo`
- `mistralai/Mixtral-8x7B-Instruct-v0.1`
- `mistralai/Mixtral-8x22B-Instruct-v0.1`

## Output Structure

The runner creates timestamped files in the `outputs/` folder:

```
outputs/
├── llama-3.1-8b_20250828_135607.json      # Individual model results
├── llama-3.1-70b_20250828_135607.json     # Individual model results  
├── combined_results_20250828_135607.json   # All results combined
└── run_summary_20250828_135607.json       # Run metadata and summary
```

### Output File Contents

**Individual Model Results** (`{model_name}_{timestamp}.json`):
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
    "original_text_length": 228,
    "model_name": "llama-3.1-8b",
    "model_config": {...},
    "extraction_timestamp": "2025-08-28T13:56:07.123456"
  }
]
```

**Run Summary** (`run_summary_{timestamp}.json`):
```json
{
  "run_metadata": {
    "timestamp": "2025-08-28T13:55:56.591663",
    "config_file": "configs/default_config.yaml",
    "config": {...},
    "models_count": 3,
    "total_samples": 1000
  },
  "model_summaries": [
    {
      "model_name": "llama-3.1-8b",
      "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", 
      "results_count": 1000,
      "output_file": "outputs/llama-3.1-8b_20250828_135607.json"
    }
  ],
  "total_results": 3000,
  "successful_models": 3,
  "failed_models": 0
}
```

## Usage Examples

### Large Scale Run (1,000+ samples)
```bash
# Create custom config for large run
cp configs/default_config.yaml configs/large_run.yaml
# Edit configs/large_run.yaml to set num_samples: 5000

# Run extraction
python run_logits_extraction.py -c configs/large_run.yaml
```

### Single Model Test
```yaml
# configs/single_model.yaml
dataset_name: "Bingsu/openwebtext_20p"
num_samples: 100
models:
  - name: "test-model"
    model_id: "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    temperature: 0.7
    logprobs: 5
```

```bash
python run_logits_extraction.py -c configs/single_model.yaml
```

### Error Recovery
If a model fails, the runner continues with other models and logs the error in the summary file.

## File Organization

- **`configs/`** - Configuration files (version controlled)
- **`outputs/`** - Generated results (ignored by git)
- **`run_logits_extraction.py`** - Main runner script
- **`together_logits_analyzer.py`** - Core analysis logic

## API Rate Limiting

The system processes samples sequentially to avoid rate limits. For large runs:
- Consider adding delays between API calls
- Monitor your Together.ai API usage
- Split very large runs across multiple sessions