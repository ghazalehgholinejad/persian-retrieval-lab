import argparse
import json
from pathlib import Path
import sqlite3

from .engine import Retriever, read_jsonl, context_bundle
from .evaluation import evaluate, markdown_report, sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare Persian BM25 retrieval and export cited RAG context")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("search", "context", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--corpus", required=True, type=Path)
        command.add_argument("-k", type=int, default=3)
        if name == "evaluate":
            command.add_argument("--queries", required=True, type=Path)
            command.add_argument("--output", required=True, type=Path)
            command.add_argument("--dataset-note", default="User-provided evaluation set; provenance and representativeness must be assessed separately.")
        else:
            command.add_argument("query")
            command.add_argument("--mode", choices=("raw", "normalized"), default="normalized")
            if name == "context":
                command.add_argument("--max-chars", type=int, default=4000)
    args = parser.parse_args(argv)
    try:
        passages = read_jsonl(args.corpus)
        if args.command == "evaluate":
            if args.output.exists():
                raise ValueError("Output already exists; choose a fresh directory")
            report = evaluate(passages, read_jsonl(args.queries), args.k)
            report.update(dataset_note=args.dataset_note, corpus_sha256=sha256(args.corpus), queries_sha256=sha256(args.queries))
            args.output.mkdir(parents=True, exist_ok=False)
            (args.output/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
            (args.output/"report.md").write_text(markdown_report(report), encoding="utf-8")
            print(f"Saved {args.output/'report.md'}")
        else:
            with Retriever(passages, args.mode) as engine:
                hits = engine.search(args.query, args.k)
            result = context_bundle(args.query, hits, args.max_chars) if args.command == "context" else hits
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
