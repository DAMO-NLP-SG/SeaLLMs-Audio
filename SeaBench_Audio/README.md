# SeaBench Audio

SeaBench Audio is a benchmarking suite for evaluating multilingual audio language models. It provides scripts and data for generating model responses, automated and human judgements, and summary reports across various tasks and languages.

## Project Structure

- `run_pipeline.sh` — Example pipeline for generating responses, judgements, and summaries.
- `prepare_data.py` — Downloads and prepares the evaluation dataset from HuggingFace.
- `gen_responses.py` — Generates model responses to audio/text queries using specified models.
- `gen_judgements.py` — Uses an LLM (e.g., Gemini) to judge model responses against references.
- `gen_summary.py` — Aggregates judgement results into summary CSVs.
- `utils.py` — Utility functions for Gemini API interaction and model inference.
- `data/` — Contains evaluation datasets:
  - `seabench_audio.json` — Main evaluation set with audio/text queries and references.
  - `rules.json` — Evaluation scoring rules for different tasks.
  - `audio_data/` — Audio files for evaluation, organized by language (e.g., `en/`, `th/`, `id/`, `vi/`).
- `responses/` — Stores model outputs (JSON files per model).
- `judgements/` — Stores judgement results (JSON files per model-judge pair).
- `summary/` — Contains aggregated summary CSVs (by language, task, and details).
- `scripts/` — Additional scripts for specific models:
  - `gen_responses_MERaLion.py` — Response generation for MERaLiON models.
  - `gen_responses_MERaLion2.py` — Response generation for MERaLiON-2 models.
  - `gen_responses_qwen_omni.py` — Response generation for Qwen2.5-Omni models.

## Requirements

- Python 3.10+
- See `environment.yml` for the complete conda environment specification

**Setup:**
```bash
conda env create -f environment.yml
conda activate SeaBench-Audio
```

**Environment Variables:**
- `GEMINI_API_KEY`: Google API key for Gemini models (used in judgements)

## Data Preparation

First, prepare the evaluation dataset:

```bash
python prepare_data.py
```

This downloads the dataset from HuggingFace, extracts audio files to `data/audio_data/`, and creates `data/seabench_audio.json`.

## Usage

### 1. Generate Model Responses

```bash
python gen_responses.py --model_path <path_to_model> [--data_path <data/seabench_audio.json>] [--output_dir <responses>]
```

**Parameters:**
- `--model_path` (required): Path to the model (e.g., `SeaLLMs/SeaLLMs-Audio-7B`, `Qwen/Qwen2-Audio-7B-Instruct`)
- `--data_path` (optional): Path to the evaluation data JSON file (default: `data/seabench_audio.json`)
- `--output_dir` (optional): Output directory for model responses (default: `responses`)

**Output:** A JSON file in `responses/` named `<model_name>.json` with model predictions.

**Alternative scripts for specific models:**
```bash
python scripts/gen_responses_MERaLion.py
python scripts/gen_responses_MERaLion2.py
python scripts/gen_responses_qwen_omni.py
```

### 2. Generate Judgements

```bash
python gen_judgements.py --model <model_name> --judgement_model <llm_name> --output_folder <judgements> --max_workers <num> [--api_key <key>] [--languages <lang>] [--tasks <task>] [--indexs <ids>] [--post]
```

**Parameters:**
- `--model` (required): Name of the model to evaluate (e.g., `SeaLLMs-Audio-7B`)
- `--judgement_model` (required): Name of the judgement model (e.g., `gemini-2.5-flash`)
- `--output_folder` (optional): Output folder for judgements (default: `judgements`)
- `--input_folder` (optional): Input folder containing model outputs (default: `responses`)
- `--max_workers` (optional): Maximum number of parallel workers (default: 5)
- `--api_key` (optional): API key for the judgement model (default: uses `GEMINI_API_KEY` environment variable)
- `--languages` (optional): Comma-separated list of languages to evaluate (e.g., `en,th,id`)
- `--tasks` (optional): Comma-separated list of tasks to evaluate
- `--indexs` (optional): Comma-separated list of indices to evaluate
- `--post` (flag): Post-process judgements to retry failed evaluations

**Output:** A JSON file in `judgements/` named `<judgement_model>_eval_<model>.json` with evaluation results.

### 3. Generate Summary

```bash
python gen_summary.py --judgement_folder <judgements> --summary_folder <summary>
```

**Parameters:**
- `--judgement_folder` (optional): Folder containing judgement JSON files (default: `judgements`)
- `--summary_folder` (optional): Folder to save summary CSVs (default: `summary`)

**Output:** Three CSV files in `summary/`:
- `summary.csv` — Average ratings by language
- `task.csv` — Average ratings by task
- `details.csv` — Detailed ratings for each question

## Example Pipeline

See `run_pipeline.sh` for a full example workflow:

```bash
# Prepare data
python prepare_data.py

# Generate model responses
python gen_responses.py --model_path SeaLLMs/SeaLLMs-Audio-7B

# Generate judgements
export GEMINI_API_KEY=<your_api_key>
python gen_judgements.py --model SeaLLMs-Audio-7B --judgement_model gemini-2.5-flash --output_folder judgements --max_workers 2 --api_key $GEMINI_API_KEY

# Generate summary
python gen_summary.py --judgement_folder judgements --summary_folder summary
```


