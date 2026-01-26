# evaluate.py 
import csv
import json
import time
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_FILE = "alice_index.json"
QUESTIONS_FILE = "eval_questions.txt"
OUTPUT_CSV = "eval_results.csv"

TOP_K = 6
MIN_SCORE = 0.30
ANSWER_CARDS = 3

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:mini"

REFUSAL_TEXT = "I can’t answer that from the current knowledge cards."

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def load_index(path: str):
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    for it in items:
        it["embedding"] = np.array(it["embedding"], dtype=np.float32)
    return items

def embed_text(model: SentenceTransformer, text: str) -> np.ndarray:
    vec = model.encode(text)
    return normalize(np.array(vec, dtype=np.float32))

def retrieve(index, qvec: np.ndarray, k: int = TOP_K, min_score: float = MIN_SCORE):
    scored = []
    for it in index:
        score = cosine(qvec, it["embedding"])
        scored.append((score, it))
    scored.sort(key=lambda x: x[0], reverse=True)

    filtered = [(s, it) for (s, it) in scored[:k] if s >= min_score]
    if not filtered:
        filtered = scored[:2]
    return filtered

def ollama_generate(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0.1,
            "num_predict": 90,
        },
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    return (r.json().get("response") or "").strip()

def build_prompt(question: str, cards) -> str:
    primary = cards[:ANSWER_CARDS]
    context = "\n---\n".join(
        [
            f"[{c['entry_id']}] {c['title']} (Chapter {c['chapter']})\n"
            f"Summary: {c['summary']}\n"
            f"Ambiguity: {c['ambiguity']}\n"
            f"Guardrail: {c['guardrail_note']}\n"
            for c in primary
        ]
    )

    return f"""You are a strictly grounded assistant for Alice’s Adventures in Wonderland.

Rules:
- Use ONLY the context cards for factual claims.
- If the answer is not explicitly stated in the cards, say exactly:
  "{REFUSAL_TEXT}"
- Do NOT invent explanations or interpretations.
- Every factual sentence MUST end with citations like [A-03].
- Keep answers short and literal.

Context cards:
{context}

Question:
{question}

Answer:
"""

def load_questions(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [q.strip() for q in f if q.strip()]

def main():
    model = SentenceTransformer(MODEL_NAME)
    index = load_index(INDEX_FILE)
    questions = load_questions(QUESTIONS_FILE)

    # warm-up
    warm_q = "How does Alice enter Wonderland?"
    warm_vec = embed_text(model, warm_q)
    warm_cards = [it for _, it in retrieve(index, warm_vec)]
    _ = ollama_generate(build_prompt(warm_q, warm_cards))

    rows = []
    for i, q in enumerate(questions, start=1):
        t0 = time.time()

        qvec = embed_text(model, q)
        scored_cards = retrieve(index, qvec)
        cards = [it for _, it in scored_cards]

        top_ids = [it["entry_id"] for _, it in scored_cards]
        top_scores = [round(float(s), 3) for s, _ in scored_cards]

        answer = ""
        error = ""
        try:
            answer = ollama_generate(build_prompt(q, cards))
        except Exception as e:
            error = str(e)

        latency_ms = int((time.time() - t0) * 1000)
        refused = REFUSAL_TEXT in answer

        rows.append({
            "n": i,
            "question": q,
            "latency_ms": latency_ms,
            "top_ids": "|".join(top_ids),
            "top_scores": "|".join(map(str, top_scores)),
            "refused": refused,
            "answer": answer,
            "error": error,
        })

        print(f"{i}/{len(questions)} done, {latency_ms} ms, refused={refused}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["n", "question", "latency_ms", "top_ids", "top_scores", "refused", "answer", "error"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote results to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
