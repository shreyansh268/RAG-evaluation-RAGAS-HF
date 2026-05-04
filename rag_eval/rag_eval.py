import asyncio
import os
import numpy as np
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings

from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

HF_TOKEN = os.environ["HF_TOKEN"]

# Generator: Qwen3-8B via nscale
_generator_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-8B",
    provider="nscale",
    huggingfacehub_api_token=HF_TOKEN,
)
llm = ChatHuggingFace(llm=_generator_endpoint)

# Embeddings: BAAI/bge-small-en-v1.5 via HF Inference (no OpenAI key needed)
embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-small-en-v1.5",
        huggingfacehub_api_token=HF_TOKEN,
    )
)


class RAG:
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
        self.docs: list[str] = []
        self.doc_embeddings: list[list[float]] = []

    def load_documents(self, docs: list[str]) -> None:
        self.docs = docs
        self.doc_embeddings = asyncio.run(self.embeddings.embed_texts(docs))

    def get_most_relevant_docs(self, query: str) -> list[str]:
        q_vec = np.array(asyncio.run(self.embeddings.embed_text(query)))
        scores = [
            np.dot(q_vec, np.array(d)) / (np.linalg.norm(q_vec) * np.linalg.norm(d))
            for d in self.doc_embeddings
        ]
        return [self.docs[int(np.argmax(scores))]]

    def generate_answer(self, query: str, relevant_docs: list[str]) -> str:
        messages = [
            SystemMessage(content="Answer in one concise sentence using only facts from the documents. Be precise."),
            HumanMessage(
                content=f"Question: {query}\n\nDocuments:\n" + "\n".join(relevant_docs)
            ),
        ]
        return self.llm.invoke(messages).content


# ---------------------------------------------------------------------------
# Sample corpus
# ---------------------------------------------------------------------------
sample_docs = [
    "Albert Einstein developed the theory of relativity, one of the two pillars of modern physics.",
    "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity.",
    "Isaac Newton formulated the laws of motion and universal gravitation.",
    "Charles Darwin proposed the theory of evolution by natural selection.",
    "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's Analytical Engine.",
]

rag = RAG(llm=llm, embeddings=embeddings)
rag.load_documents(sample_docs)

# Smoke test
_test_query = "Who introduced the theory of relativity?"
_retrieved = rag.get_most_relevant_docs(_test_query)
_answer = rag.generate_answer(_test_query, _retrieved)
print(f"Query   : {_test_query}")
print(f"Retrieved: {_retrieved}")
print(f"Answer  : {_answer}\n")

# ---------------------------------------------------------------------------
# Build evaluation dataset
# ---------------------------------------------------------------------------
sample_queries = [
    "Who introduced the theory of relativity?",
    "Who was the first to research radioactivity?",
    "Who formulated the laws of motion?",
    "Who proposed the theory of evolution by natural selection?",
    "Who is regarded as the first computer programmer?",
]

expected_responses = [
    "Albert Einstein developed the theory of relativity.",
    "Marie Curie conducted pioneering research on radioactivity.",
    "Isaac Newton formulated the laws of motion and universal gravitation.",
    "Charles Darwin proposed the theory of evolution by natural selection.",
    "Ada Lovelace is regarded as the first computer programmer.",
]

dataset = []
for query, reference in zip(sample_queries, expected_responses):
    relevant_docs = rag.get_most_relevant_docs(query)
    response = rag.generate_answer(query, relevant_docs)
    dataset.append(
        {
            "user_input": query,
            "retrieved_contexts": relevant_docs,
            "response": response,
            "reference": reference,
        }
    )

evaluation_dataset = EvaluationDataset.from_list(dataset)

# ---------------------------------------------------------------------------
# Judge LLM: Llama-3.1-8B-Instruct via novita
# ---------------------------------------------------------------------------
_judge_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    provider="novita",
    huggingfacehub_api_token=HF_TOKEN,
)
evaluator_llm = LangchainLLMWrapper(ChatHuggingFace(llm=_judge_endpoint))

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
result = evaluate(
    dataset=evaluation_dataset,
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
    llm=evaluator_llm,
)

print(result)
