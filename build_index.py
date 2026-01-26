import json
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
INPUT_CSV = "alice_kb.csv"
OUTPUT_INDEX = "alice_index.json"

REQUIRED_COLUMNS = [
    "entry_id",
    "title",
    "category",
    "chapter",
    "summary",
    "tags",
    "ambiguity",
    "guardrail_note",
]

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def build_embed_text(row: pd.Series) -> str:
    return (
        f"Title: {row['title']}\n"
        f"Category: {row['category']}\n"
        f"Chapter: {row['chapter']}\n"
        f"Summary: {row['summary']}\n"
        f"Tags: {row['tags']}\n"
        f"Ambiguity: {row['ambiguity']}\n"
        f"Guardrail: {row['guardrail_note']}\n"
    )

def main():
    if not os.path.exists(INPUT_CSV):
        raise SystemExit(f"Missing {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV).fillna("")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing columns: {missing}")

    model = SentenceTransformer(MODEL_NAME)

    records = []
    total = len(df)

    for i, row in df.iterrows():
        text = build_embed_text(row)
        emb = model.encode(text)
        vec = normalize(np.array(emb, dtype=np.float32))

        records.append({
            "entry_id": str(row["entry_id"]),
            "title": str(row["title"]),
            "category": str(row["category"]),
            "chapter": str(row["chapter"]),
            "tags": str(row["tags"]),
            "summary": str(row["summary"]),
            "ambiguity": str(row["ambiguity"]),
            "guardrail_note": str(row["guardrail_note"]),
            "embedding": vec.tolist(),
        })

        if (i + 1) % 5 == 0 or (i + 1) == total:
            print(f"Embedded {i + 1}/{total}")

    with open(OUTPUT_INDEX, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Done. Wrote {len(records)} records to {OUTPUT_INDEX}")

if __name__ == "__main__":
    main()
