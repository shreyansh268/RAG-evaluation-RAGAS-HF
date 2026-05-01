import os
import sys
from pathlib import Path

from openai import OpenAI

from ragas import Dataset, experiment
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric

# Add the current directory to the path so we can import rag module when run as a script
sys.path.insert(0, str(Path(__file__).parent))
from rag import default_rag_client

HF_MODEL = "HuggingFaceH4/zephyr-7b-alpha:featherless-ai"

hf_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)
rag_client = default_rag_client(llm_client=hf_client, logdir="evals/logs", model=HF_MODEL)
llm = llm_factory(HF_MODEL, client=hf_client)


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
        row = {"question": sample["question"], "grading_notes": sample["grading_notes"]}
        dataset.append(row)

    # make sure to save it
    dataset.save()
    return dataset


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

@experiment()
async def run_experiment(row):
    response = rag_client.query(row["question"])
    answer = response.get("answer", "")

    score = my_metric.score(
        llm=llm,
        response=answer,
        grading_notes=row["grading_notes"],
    )

    tone_score = my_tone_metric.score(
        llm=llm,
        response=answer,
    )

    experiment_view = {
        **row,
        "response": answer,
        "score": score.value,
        "tone_score": tone_score.value,
        "log_file": response.get("logs", " "),
    }
    return experiment_view


async def main():
    dataset = load_dataset()
    print("dataset loaded successfully", dataset)
    experiment_results = await run_experiment.arun(dataset)
    print("Experiment completed successfully!")
    print("Experiment results:", experiment_results)

    # Save experiment results to CSV
    experiment_results.save()
    csv_path = Path(".") / "experiments" / f"{experiment_results.name}.csv"
    print(f"\nExperiment results saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
