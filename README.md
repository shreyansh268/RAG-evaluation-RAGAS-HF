# RAG Evaluation with Ragas + Hugging Face

Evaluate a RAG (Retrieval-Augmented Generation) system using [Ragas 0.3](https://docs.ragas.io) and Hugging Face Inference API as the LLM backend.

Results File with DicreteMetric Scores (Correctness & Tone)

<img width="327" height="83" alt="image" src="https://github.com/user-attachments/assets/9245459b-bcb2-48f4-af21-111648d19ac8" />

Sample RAG_run_log_file

<img width="397" height="416" alt="image" src="https://github.com/user-attachments/assets/974a3888-60f8-4171-945d-224e79eba5cc" />


## How it works

```
Dataset (questions + grading notes)
        │
        ▼
  ExampleRAG.query()          ← rag.py
  SimpleKeywordRetriever      ← keyword match scoring
  LLM via HF Inference API    ← generates answer
        │
        ▼
  DiscreteMetric.score()      ← evals.py
  LLM judge → pass/fail       ← correctness vs grading notes
  LLM judge → formal/…        ← tone classification
        │
        ▼
  evals/experiments/<run>.csv  ← final scores
  evals/logs/<run_id>.json     ← per-query trace
```

## Project structure

```
rag_eval/
├── rag.py              # RAG system: SimpleKeywordRetriever + ExampleRAG + tracing
├── evals.py            # Evaluation workflow: dataset, metrics, experiment runner
├── pyproject.toml      # Dependencies (ragas>=0.3, openai SDK for HF routing)
├── evals/
│   ├── datasets/       # test_dataset.csv — questions & grading notes
│   ├── experiments/    # Output CSVs, one per experiment run (git-ignored)
│   └── logs/           # Per-query JSON trace files (git-ignored)
└── README.md
```

## Quick start

### 1. Set your Hugging Face token

```bash
# Windows
set HF_TOKEN=hf_...

# macOS / Linux
export HF_TOKEN=hf_...
```

Never commit your token. It is read at runtime via `os.environ["HF_TOKEN"]`.

### 2. Install dependencies

```bash
# recommended
uv sync

# or pip
pip install -e .
```

### 3. Run the evaluation

```bash
# uv
uv run python evals.py

# plain Python
python evals.py
```

Results are written to `evals/experiments/<run-name>.csv`.

## Scores explained

### Evaluation scores (in the CSV)

| Column | Values | What it measures |
|---|---|---|
| `score` | `pass` / `fail` | **Correctness** — LLM judge checks if the answer covers the key points in `grading_notes` |
| `tone_score` | `formal` / `informal` / `neutral` | **Tone** — LLM judge classifies the writing style of the answer |

Both use `DiscreteMetric` from Ragas: the judge LLM is given a prompt and constrained to return one of the `allowed_values`.

### Retrieval scores (in the JSON trace logs)

The `scores` array in each `retrieve_complete` trace event (e.g. `[3, 3, 1]`) is a **keyword match count** — how many query words appear in a document. Documents scoring `0` are excluded from the context. These are diagnostic only, not evaluation quality metrics.

## Customization

### Change the model

In `evals.py`, update `HF_MODEL`:

```python
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3:featherless-ai"
```

Any model available via the [Hugging Face Inference Router](https://huggingface.co/docs/inference-providers) works here.

### Add test cases

Edit `load_dataset()` in `evals.py`:

```python
{
    "question": "Your question here",
    "grading_notes": "- key point 1\n- key point 2",
}
```

### Add a metric

```python
from ragas.metrics import DiscreteMetric

my_metric = DiscreteMetric(
    name="my_metric",
    prompt="Your judge prompt. Response: {response}",
    allowed_values=["value_a", "value_b"],
)
```

## Documentation

- Ragas: https://docs.ragas.io
- HF Inference Providers: https://huggingface.co/docs/inference-providers
