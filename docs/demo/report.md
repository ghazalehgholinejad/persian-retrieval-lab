# Retrieval evaluation

Author-created synthetic Persian passages and queries. Orthographic stress test, not a held-out or real-world benchmark.

Passages: 16 · k: 3

| Method | Recall@3 | MRR@3 | nDCG@3 | Empty results on unanswerable queries |
| --- | ---: | ---: | ---: | ---: |
| raw | 0.5476 | 0.5714 | 0.5530 | 0.3333 |
| normalized | 1.0000 | 1.0000 | 1.0000 | 0.3333 |

Metrics are macro-averaged over answerable queries. Unanswerable queries are reported separately.
Empty-result rate is not calibrated abstention or answer correctness. No LLM is executed.

Environment: Python 3.12.14, SQLite 3.53.1, package 0.1.0.

See report.json for per-query rankings, labels, and input hashes.
