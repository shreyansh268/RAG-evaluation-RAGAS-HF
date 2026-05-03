import csv
import json
import os
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from ragas import Dataset, experiment
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric

sys.path.insert(0, str(Path(__file__).parent))
from rag import default_rag_client

RAG_MODEL = "HuggingFaceH4/zephyr-7b-alpha:featherless-ai"
HF_BASE_URL = "https://router.huggingface.co/v1"
MODELS_CONFIG_PATH = Path(__file__).parent / "models_config.json"

CORRECTNESS_NUMERIC = {"pass": 1, "fail": 0}

# Fixed RAG client — model and client are not swapped between runs
hf_client_rag = OpenAI(base_url=HF_BASE_URL, api_key=os.environ["HF_TOKEN"])
rag_client = default_rag_client(llm_client=hf_client_rag, logdir="evals/logs", model=RAG_MODEL)

my_metric = DiscreteMetric(
    name="correctness",
    prompt="Check if the response contains points mentioned from the grading notes and return 'pass' or 'fail'.\nResponse: {response} Grading Notes: {grading_notes}",
    allowed_values=["pass", "fail"],
)

my_tone_metric = DiscreteMetric(
    name="tone",
    prompt="Evaluate the tone of the response and return 'formal', 'informal', or 'neutral'.\nResponse: {response}",
    allowed_values=["formal", "informal", "neutral"],
)


def load_judge_models() -> list[str]:
    with open(MODELS_CONFIG_PATH) as f:
        config = json.load(f)
    return config["judge_models"]


def load_dataset():
    dataset = Dataset(
        name="test_dataset",
        backend="local/csv",
        root_dir="evals",
    )

    data_samples = [
        {
            "question": "What is ragas 0.3",
            "grading_notes": "- experimentation as the central pillar - provides abstraction for datasets, experiments and metrics - supports evals for RAG, LLM workflows and Agents",
        },
        {
            "question": "What is Ragas?",
            "grading_notes": "Ragas is an evaluation framework for LLM applications",
        },
        {
            "question": "how are experiment results stored in ragas 0.3?",
            "grading_notes": "- configured using different backends like local, gdrive, etc - stored under experiments/ folder in the backend storage",
        },
        {
            "question": "What metrics are supported in ragas 0.3?",
            "grading_notes": "- provides abstraction for discrete, numerical and ranking metrics",
        },
    ]

    for sample in data_samples:
        dataset.append({"question": sample["question"], "grading_notes": sample["grading_notes"]})

    dataset.save()
    return dataset


def make_experiment(llm, row_results: list):
    @experiment()
    async def run_experiment(row):
        response = rag_client.query(row["question"])
        answer = response.get("answer", "")

        score = my_metric.score(llm=llm, response=answer, grading_notes=row["grading_notes"])
        tone_score = my_tone_metric.score(llm=llm, response=answer)

        result = {
            **row,
            "response": answer,
            "score": score.value,
            "tone_score": tone_score.value,
            "log_file": response.get("logs", " "),
        }
        row_results.append(result)
        return result

    return run_experiment


def compute_cv(values: list[float]) -> float:
    """Coefficient of variation = std / mean. Returns 0 for single-value lists."""
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    if mean == 0:
        return float("inf")
    return statistics.stdev(values) / mean


def generate_score_report(model_results: dict, report_dir: str = "evals/reports"):
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    per_model_rows = []
    model_pass_rates = {}

    for judge_model, results in model_results.items():
        correctness_nums = [CORRECTNESS_NUMERIC.get(r["score"], 0) for r in results]
        tone_counts = Counter(r["tone_score"] for r in results)
        pass_rate = statistics.mean(correctness_nums) if correctness_nums else 0.0
        model_pass_rates[judge_model] = pass_rate

        per_model_rows.append({
            "judge_model": judge_model,
            "n_samples": len(results),
            "pass_count": sum(correctness_nums),
            "fail_count": len(correctness_nums) - sum(correctness_nums),
            "pass_rate": round(pass_rate, 4),
            "tone_formal": tone_counts.get("formal", 0),
            "tone_informal": tone_counts.get("informal", 0),
            "tone_neutral": tone_counts.get("neutral", 0),
        })

    pass_rates = list(model_pass_rates.values())
    mean_pr = statistics.mean(pass_rates) if pass_rates else 0.0
    std_pr = statistics.stdev(pass_rates) if len(pass_rates) > 1 else 0.0
    cv = compute_cv(pass_rates)

    csv_path = Path(report_dir) / f"score_report_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_model_rows[0].keys())
        writer.writeheader()
        writer.writerows(per_model_rows)
        writer.writerow({
            "judge_model": "SUMMARY",
            "n_samples": "",
            "pass_count": "",
            "fail_count": "",
            "pass_rate": f"mean={mean_pr:.4f} std={std_pr:.4f} CV={cv:.4f}",
            "tone_formal": "",
            "tone_informal": "",
            "tone_neutral": "",
        })

    print("\n" + "=" * 60)
    print("SCORE DISTRIBUTION REPORT")
    print("=" * 60)
    for row in per_model_rows:
        print(f"\nJudge : {row['judge_model']}")
        print(f"  Correctness : {row['pass_count']}/{row['n_samples']} pass  ({row['pass_rate'] * 100:.1f}%)")
        print(f"  Tone        : formal={row['tone_formal']}  informal={row['tone_informal']}  neutral={row['tone_neutral']}")

    print(f"\n--- Cross-Judge Correctness Statistics ---")
    print(f"  Mean pass rate : {mean_pr:.4f}")
    print(f"  Std deviation  : {std_pr:.4f}")
    print(f"  CV  (σ/μ)      : {cv:.4f}")
    print(f"\nReport saved to : {csv_path.resolve()}")
    print("=" * 60)

    return str(csv_path)


async def main():
    dataset = load_dataset()
    print("Dataset loaded successfully:", dataset)

    judge_models = load_judge_models()
    print(f"Judge models loaded: {judge_models}")

    model_results = {}

    for judge_model in judge_models:
        print(f"\n--- Running evaluation with judge: {judge_model} ---")

        hf_client_judge = OpenAI(base_url=HF_BASE_URL, api_key=os.environ["HF_TOKEN"])
        llm = llm_factory(judge_model, client=hf_client_judge)

        row_results = []
        run_experiment = make_experiment(llm, row_results)
        experiment_results = await run_experiment.arun(dataset)
        experiment_results.save()

        model_results[judge_model] = row_results
        print(f"Completed {len(row_results)} samples for judge: {judge_model}")

    generate_score_report(model_results)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
