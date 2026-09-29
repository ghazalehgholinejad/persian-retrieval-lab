import hashlib
import math
import platform
import sqlite3
from pathlib import Path

from . import __version__
from .engine import Retriever, validate_corpus


def ranking_metrics(retrieved, relevant, k):
    if k < 1 or not relevant:
        raise ValueError("Ranking metrics require positive k and at least one relevant passage")
    if len(set(retrieved)) != len(retrieved):
        raise ValueError("Retrieved IDs must be unique")
    relevant = set(relevant)
    ranks = [i for i, pid in enumerate(retrieved[:k], 1) if pid in relevant]
    ideal = sum(1/math.log2(i+1) for i in range(1, min(k, len(relevant))+1))
    return {"recall": len(ranks)/len(relevant),
            "reciprocal_rank": 1/ranks[0] if ranks else 0.0,
            "ndcg": sum(1/math.log2(i+1) for i in ranks)/ideal}


def evaluate(passages, queries, k=3):
    validate_corpus(passages)
    if k < 1 or not queries:
        raise ValueError("Evaluation requires queries and positive k")
    ids = {p["id"] for p in passages}
    seen = set()
    for q in queries:
        if not isinstance(q.get("id"), str) or not q["id"] or q["id"] in seen:
            raise ValueError("Query IDs must be nonempty and unique")
        seen.add(q["id"])
        if not isinstance(q.get("query"), str) or not q["query"].strip():
            raise ValueError("Queries must be nonempty strings")
        rel = q.get("relevant_ids")
        if not isinstance(rel, list) or any(not isinstance(x, str) for x in rel):
            raise ValueError("relevant_ids must be a list of strings")
        if len(set(rel)) != len(rel) or not set(rel) <= ids:
            raise ValueError("Relevance labels must be unique and reference existing passages")
    results = {}
    for mode in ("raw", "normalized"):
        rows = []
        with Retriever(passages, mode) as engine:
            for q in queries:
                retrieved = [h["id"] for h in engine.search(q["query"], k)]
                metrics = ranking_metrics(retrieved, q["relevant_ids"], k) if q["relevant_ids"] else None
                rows.append({**q, "retrieved_ids": retrieved, "metrics": metrics})
        answerable = [r for r in rows if r["metrics"] is not None]
        unanswerable = [r for r in rows if r["metrics"] is None]
        results[mode] = {
            "answerable_queries": len(answerable), "unanswerable_queries": len(unanswerable),
            "mean_recall": sum(r["metrics"]["recall"] for r in answerable)/len(answerable) if answerable else None,
            "mrr": sum(r["metrics"]["reciprocal_rank"] for r in answerable)/len(answerable) if answerable else None,
            "mean_ndcg": sum(r["metrics"]["ndcg"] for r in answerable)/len(answerable) if answerable else None,
            "empty_result_rate_unanswerable": sum(not r["retrieved_ids"] for r in unanswerable)/len(unanswerable) if unanswerable else None,
            "queries": rows,
        }
    return {"tool_version": __version__, "python": platform.python_version(),
            "sqlite": sqlite3.sqlite_version, "k": k, "passages": len(passages), "results": results}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def markdown_report(report):
    k = report["k"]
    lines = ["# Retrieval evaluation", "", report["dataset_note"], "",
             f"Passages: {report['passages']} · k: {k}", "",
             f"| Method | Recall@{k} | MRR@{k} | nDCG@{k} | Empty results on unanswerable queries |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for mode, result in report["results"].items():
        values = [result[key] for key in ("mean_recall", "mrr", "mean_ndcg", "empty_result_rate_unanswerable")]
        lines.append("| "+mode+" | "+" | ".join(f"{v:.4f}" if v is not None else "N/A" for v in values)+" |")
    lines += ["", "Metrics are macro-averaged over answerable queries. Unanswerable queries are reported separately.",
              "Empty-result rate is not calibrated abstention or answer correctness. No LLM is executed.", "",
              f"Environment: Python {report['python']}, SQLite {report['sqlite']}, package {report['tool_version']}.", "",
              "See report.json for per-query rankings, labels, and input hashes."]
    return "\n".join(lines)+"\n"
