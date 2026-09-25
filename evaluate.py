"""Offline retrieval/abstention evaluation; never a generated-answer quality score."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
from rag.core import Engine, Ollama, load_chunks

ROOT = Path(__file__).resolve().parent


def validate_questions(questions, chunk_ids):
    if not isinstance(questions, list) or not questions:
        raise ValueError("Questions must be a nonempty array")
    seen = set()
    topics, evidence_splits = {}, {}
    for q in questions:
        if not isinstance(q, dict) or any(not isinstance(q.get(f), str) or not q[f].strip()
                                         for f in ("id", "question")):
            raise ValueError("Each question requires id and question")
        refs = q.get("relevant")
        if not isinstance(refs, list) or any(not isinstance(r, str) for r in refs):
            raise ValueError("Relevant IDs must be an array of strings")
        if q["id"] in seen or len(set(refs)) != len(refs) or set(refs) - set(chunk_ids):
            raise ValueError("Duplicate question/relevance IDs or unknown evidence")
        seen.add(q["id"])
        split = q.get("split")
        if split is not None:
            if split not in ("dev", "test") or not isinstance(q.get("topic"), str) or not q["topic"].strip():
                raise ValueError("Split must be dev/test with a topic")
            for key, mapping in [(q["topic"], topics), *((r, evidence_splits) for r in refs)]:
                if key in mapping and mapping[key] != split:
                    raise ValueError("Topic or relevant evidence leaks across splits")
                mapping[key] = split


def summarize(rows):
    positive = [r for r in rows if r["relevant"]]
    negative = [r for r in rows if not r["relevant"]]
    rejected = [r for r in rows if r["status"] == "abstained"]
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    return {
        "questions": len(rows), "answerable": len(positive), "unanswerable": len(negative),
        "recall_at_k": mean([r["recall_at_k"] for r in positive]),
        "mrr_at_k": mean([r["reciprocal_rank"] for r in positive]),
        "all_evidence_recall_at_k": mean([set(r["relevant"]) <= set(r["retrieved"]) for r in positive]),
        "unanswerable_abstention_rate": mean([r["status"] == "abstained" for r in negative]),
        "unanswerable_false_acceptance_rate": mean([r["status"] == "evidence" for r in negative]),
        "answerable_acceptance_rate": mean([r["status"] == "evidence" for r in positive]),
        "abstention_precision": mean([not r["relevant"] for r in rejected]),
        "answer_coverage": mean([r["status"] == "evidence" for r in rows]),
        "mean_ms": mean([r["total_ms"] for r in rows]),
        "p95_ms": sorted(r["total_ms"] for r in rows)[math.ceil(.95 * len(rows))-1] if rows else None,
    }


def evaluate(engine, questions, mode, k, min_overlap=.15):
    if not questions:
        raise ValueError("No questions selected")
    rows = []
    for q in questions:
        result = engine.ask(q["question"], mode, k, min_overlap=min_overlap)
        relevant = set(q["relevant"])
        ids = [c["id"] for c in result["evidence"]]
        rows.append({"id": q["id"], "question": q["question"], "relevant": q["relevant"],
            "retrieved": ids, "status": result["status"], "overlap": result["overlap"],
            "recall_at_k": len(relevant & set(ids)) / len(relevant) if relevant else None,
            "reciprocal_rank": next((1/(i+1) for i, cid in enumerate(ids) if cid in relevant), 0) if relevant else None,
            "category": q.get("category", "unspecified"), "split": q.get("split", "unsplit"),
            "reference_answer": q.get("reference_answer"), "answer": result["answer"],
            "total_ms": result["total_ms"]})
    return {"mode": mode, "k": k, "min_overlap": min_overlap, "summary": summarize(rows),
            "by_category": {c: summarize([r for r in rows if r["category"] == c])
                            for c in sorted({r["category"] for r in rows})}, "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/corpus.json")
    parser.add_argument("--questions", type=Path, default=ROOT / "data/questions.json")
    parser.add_argument("--modes", nargs="+", choices=["bm25", "dense", "hybrid"], default=["bm25"])
    parser.add_argument("--embedding-model")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--split", choices=["all", "dev", "test"], default="all")
    parser.add_argument("--min-overlap", type=float, default=.15)
    parser.add_argument("--dataset-name", help="Human-readable dataset version label")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/smoke.json")
    args = parser.parse_args()
    chunks = load_chunks(args.corpus)
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    try:
        validate_questions(questions, {c.id for c in chunks})
    except ValueError as exc:
        parser.error(str(exc))
    if args.split != "all":
        questions = [q for q in questions if q.get("split") == args.split]
    if not questions:
        parser.error("No questions in requested split")
    if not 1 <= args.k <= 10 or not 0 <= args.min_overlap <= 1:
        parser.error("k must be 1..10 and min-overlap must be 0..1")
    provider = Ollama(embedding_model=args.embedding_model) if args.embedding_model else None
    results = [evaluate(Engine(chunks, provider), questions, mode, args.k, args.min_overlap) for mode in args.modes]
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "dataset": args.dataset_name or ("Original synthetic Atlas smoke set; NOT a held-out benchmark"
                    if args.questions.resolve() == (ROOT / "data/questions.json").resolve() else args.questions.name),
        "split": args.split,
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
        "embedding_model": args.embedding_model, "generation": False,
        "latency_note": "Sequential end-to-end retrieval and extractive response; first dense query includes indexing.",
        "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([r["summary"] for r in results], indent=2))
