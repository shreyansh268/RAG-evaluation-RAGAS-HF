import json
import os
import sys
from datetime import datetime
from pathlib import Path

from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings
from openai import OpenAI
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.metrics import AnswerRelevancy, Faithfulness

sys.path.insert(0, str(Path(__file__).parent))
from rag import default_rag_client

RAG_MODEL = "HuggingFaceH4/zephyr-7b-alpha:featherless-ai"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2:hf-inference"
HF_BASE_URL = "https://router.huggingface.co/v1"
MODELS_CONFIG_PATH = Path(__file__).parent / "models_config.json"

hf_client_rag = OpenAI(base_url=HF_BASE_URL, api_key=os.environ["HF_TOKEN"])
rag_client = default_rag_client(llm_client=hf_client_rag, logdir="evals/logs", model=RAG_MODEL)

DATA_SAMPLES = [
    {
        "question": "What is ragas 0.3",
        "reference": "Ragas 0.3 places experimentation as the central pillar, providing abstraction for datasets, experiments and metrics. It supports evals for RAG, LLM workflows and Agents.",
    },
    {
        "question": "What is Ragas?",
        "reference": "Ragas is an evaluation framework for LLM applications.",
    },
    {
        "question": "how are experiment results stored in ragas 0.3?",
        "reference": "Experiment results are configured using different backends like local or gdrive, and stored under the experiments/ folder in the backend storage.",
    },
    {
        "question": "What metrics are supported in ragas 0.3?",
        "reference": "Ragas 0.3 provides abstraction for discrete, numerical and ranking metrics.",
    },
]


def load_judge_models() -> list[str]:
    with open(MODELS_CONFIG_PATH) as f:
        return json.load(f)["judge_models"]


def build_eval_dataset() -> EvaluationDataset:
    """Run RAG for each question and return an EvaluationDataset of SingleTurnSamples."""
    print("Running RAG pipeline...")
    samples = []
    for item in DATA_SAMPLES:
        response = rag_client.query(item["question"])
        samples.append(
            SingleTurnSample(
                user_input=item["question"],
                response=response.get("answer", ""),
                retrieved_contexts=[response.get("context", "")],
                reference=item["reference"],
            )
        )
    return EvaluationDataset(samples=samples)


def main():
    judge_models = load_judge_models()
    eval_dataset = build_eval_dataset()

    embeddings = LangchainEmbeddingsWrapper(
        LangchainOpenAIEmbeddings(
            model=EMBED_MODEL,
            openai_api_base=HF_BASE_URL,
            openai_api_key=os.environ["HF_TOKEN"],
        )
    )

    metrics = [Faithfulness(), AnswerRelevancy()]

    results_dir = Path("evals/experiments")
    results_dir.mkdir(parents=True, exist_ok=True)

    for judge_model in judge_models:
        print(f"\n--- Evaluating with judge: {judge_model} ---")
        hf_client_judge = OpenAI(base_url=HF_BASE_URL, api_key=os.environ["HF_TOKEN"])
        llm = llm_factory(judge_model, client=hf_client_judge)

        result = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
        )

        print(result)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        judge_slug = judge_model.split("/")[-1].replace(":", "_")
        csv_path = results_dir / f"eval_{judge_slug}_{timestamp}.csv"
        result.to_pandas().to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
