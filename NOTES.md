# Notes

Improvement ideas and observations captured during sessions. Add entries with `## YYYY-MM-DD` headings.

---

## 2026-05-04

### RAG retrieval improvements (`rag_eval.py`)

Current retrieval returns only top-1 document (`np.argmax`). Two changes not yet applied:

**1. Top-k retrieval (top-2)**

Return the 2 most similar documents instead of just 1. More context reduces hallucination pressure on the generator.

```python
def get_most_relevant_docs(self, query: str, k: int = 2) -> list[str]:
    q_vec = np.array(asyncio.run(self.embeddings.embed_text(query)))
    scores = [
        np.dot(q_vec, np.array(d)) / (np.linalg.norm(q_vec) * np.linalg.norm(d))
        for d in self.doc_embeddings
    ]
    top_indices = np.argsort(scores)[-k:][::-1]
    return [self.docs[i] for i in top_indices]
```

Targets: `faithfulness`, `factual_correctness`

**2. Low-similarity fallback**

If the best match scores below a threshold (e.g. 0.5), the corpus has no relevant document. Return empty rather than feeding irrelevant context to the LLM.

```python
top_idx = int(np.argmax(scores))
if scores[top_idx] < 0.5:
    return []
return [self.docs[top_idx]]
```

Targets: `factual_correctness` (avoids confident wrong answers from irrelevant context)
