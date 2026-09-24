"""Small synthetic smoke evaluation. Not a general QA quality benchmark."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from rag.core import Engine, Ollama, load_chunks

ROOT = Path(__file__).resolve().parent


def evaluate(engine, questions, mode, k):
    rows = []
    for q in questions:
        result = engine.ask(q["question"], mode, k)
        relevant = set(q["relevant"])
        ids = [c["id"] for c in result["evidence"]]
        rows.append({"id": q["id"], "question": q["question"], "relevant": q["relevant"],
            "retrieved": ids, "status": result["status"], "overlap": result["overlap"],
            "recall_at_k": len(relevant & set(ids)) / len(relevant) if relevant else None,
            "reciprocal_rank": next((1/(i+1) for i, cid in enumerate(ids) if cid in relevant), 0) if relevant else None,
            "total_ms": result["total_ms"]})
    positive = [r for r in rows if r["relevant"]]
    negative = [r for r in rows if not r["relevant"]]
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    return {"mode": mode, "k": k, "summary": {
        "recall_at_k": mean([r["recall_at_k"] for r in positive]),
        "mrr_at_k": mean([r["reciprocal_rank"] for r in positive]),
        "unanswerable_abstention_rate": mean([r["status"] == "abstained" for r in negative]),
        "answerable_acceptance_rate": mean([r["status"] == "evidence" for r in positive]),
        "mean_ms": mean([r["total_ms"] for r in rows]),
        "p95_ms": sorted(r["total_ms"] for r in rows)[max(0, __import__('math').ceil(.95 * len(rows))-1)],
        "questions": len(rows)}, "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/corpus.json")
    parser.add_argument("--questions", type=Path, default=ROOT / "data/questions.json")
    parser.add_argument("--modes", nargs="+", choices=["bm25", "dense", "hybrid"], default=["bm25"])
    parser.add_argument("--embedding-model")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/smoke.json")
    args = parser.parse_args()
    chunks = load_chunks(args.corpus)
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    if not questions or any(set(q["relevant"]) - {c.id for c in chunks} for q in questions):
        parser.error("Empty questions or unknown relevant chunk IDs")
    provider = Ollama(embedding_model=args.embedding_model) if args.embedding_model else None
    results = [evaluate(Engine(chunks, provider), questions, mode, args.k) for mode in args.modes]
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "dataset": "Original synthetic Atlas smoke set; NOT a held-out benchmark",
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
        "embedding_model": args.embedding_model, "generation": False,
        "latency_note": "Sequential end-to-end retrieval and extractive response; first dense query includes indexing.",
        "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([r["summary"] for r in results], indent=2))
