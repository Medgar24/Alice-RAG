import json
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_FILE = "alice_index.json"

TOP_K = 6
MIN_SCORE = 0.30
DISPLAY_CARDS = 3
ANSWER_CARDS = 3

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:mini"

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

def format_cards(cards):
    lines = []
    for c in cards:
        lines.append(
            f"[{c['entry_id']}] {c['title']} (Chapter {c['chapter']})\n"
            f"Summary: {c['summary']}\n"
            f"Ambiguity: {c['ambiguity']}\n"
            f"Guardrail: {c['guardrail_note']}\n"
        )
    return "\n---\n".join(lines)

def ollama_generate(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 220
        },
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    return (r.json().get("response") or "").strip()

def answer_with_ollama(question: str, cards) -> str:
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

    prompt = f"""You are a strictly grounded assistant for Alice’s Adventures in Wonderland.

Hard rules (must follow exactly):
- Use ONLY the context cards for factual claims.
- If the answer is not explicitly stated in the cards, say exactly:
  "I can’t answer that from the current knowledge cards."
- Do NOT use words like symbol, symbolic, metaphor, metaphorical, theme, or meaning unless the cards explicitly use those words.
- Do NOT interpret or explain motives, symbolism, or abstract ideas.
- Do NOT invent mechanisms, timelines, or causes.
- Every factual sentence MUST end with citations in brackets using card IDs, e.g. [A-05].
- Keep answers short, factual, and literal (max 5 sentences).

Context cards:
{context}

Question:
{question}

Answer:
"""
    
    return ollama_generate(prompt)

def main():
    model = SentenceTransformer(MODEL_NAME)
    index = load_index(INDEX_FILE)

    print("Alice RAG (local + Ollama) ready. Type a question, or 'quit'.")

    while True:
        q = input("\nQuestion> ").strip()
        if q.lower() in {"quit", "exit"}:
            break
        if not q:
            continue

        qvec = embed_text(model, q)
        scored_cards = retrieve(index, qvec, k=TOP_K, min_score=MIN_SCORE)
        cards = [it for _, it in scored_cards]

        print("\nTop matches:")
        for score, it in scored_cards:
            print(f"  {score:.3f} -> [{it['entry_id']}] {it['title']} (Chapter {it['chapter']})")

        print("\nTop context cards:\n")
        print(format_cards(cards[:DISPLAY_CARDS]))

        print("\nAnswer:\n")
        try:
            print(answer_with_ollama(q, cards))
        except requests.exceptions.RequestException as e:
            print("Ollama request failed. Is Ollama running, and did you pull the model?")
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
