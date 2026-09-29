# Persian Retrieval Lab

[![tests](https://github.com/ghazalehgholinejad/persian-retrieval-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/ghazalehgholinejad/persian-retrieval-lab/actions/workflows/tests.yml)

**A small experiment in the retrieval layer of Persian RAG.** Compare SQLite BM25 before and after Persian text normalization, inspect per-query results, and export source-linked context for a downstream language model.

Runs locally on CPU with Python's standard library. No API key, model download, or paid service is required. This project implements retrieval and evaluation; it does not generate answers or claim semantic understanding.

## The problem

The same Persian word can be written with Persian or Arabic letters: `کلید` versus `كليد`. Different digit forms, diacritics, and half-spaces can also affect matching. Before adding an LLM, it is useful to measure whether the retriever can find the intended passage at all.

```text
Corpus JSONL → normalization (optional) → SQLite FTS5 / BM25
                                                ↑
Query → same preprocessing → quoted token query ─┘
                                                ↓
                              ranked passages → cited context JSON
                                      ↓
                      Recall@k / MRR@k / nDCG@k evaluation
```

## Quick start

Requires **Python 3.10+ with SQLite FTS5 enabled**. Standard Python distributions commonly include FTS5; the tests exercise it directly.

```bash
git clone https://github.com/ghazalehgholinejad/persian-retrieval-lab.git
cd persian-retrieval-lab
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install .

persian-search search --corpus examples/corpus.jsonl 'كليد'
persian-search search --corpus examples/corpus.jsonl --mode raw 'كليد'
```

The normalized query retrieves passage `api`, whose original text contains `کلید`. The raw query returns no results for this example. Search results preserve the original passage text and include its ID, source, rank, and SQLite BM25 score. **Lower scores rank first**; scores are not probabilities and should not be compared as confidence across queries.

## Export context for RAG

```bash
persian-search context --corpus examples/corpus.jsonl \
  --max-chars 1200 -k 3 'كليد' > runs-context.json
```

The output includes a context string with labels such as `[S1]`, a separate citation map, and IDs omitted because of the character budget. Complete passages are retained; partial passages are never silently cut. The budget includes titles and labels but excludes the JSON wrapper, query, and citation map. It is a **character budget**, not a model-token budget.

Retrieved text is untrusted source material. A downstream application must keep it separate from system instructions, check answer support, and handle unsupported questions. Exporting citation IDs alone does not guarantee an LLM will cite or answer correctly.

## Run the comparison

```bash
persian-search evaluate \
  --corpus examples/corpus.jsonl \
  --queries examples/queries.jsonl \
  --output runs/evaluation \
  --dataset-note 'Synthetic orthographic stress test; not a held-out benchmark.'
```

The command writes `report.md` and `report.json`, including per-query rankings, relevance labels, input SHA-256 hashes, and Python/SQLite versions. Existing output directories are never overwritten.

### Measured demo results

The included **author-created synthetic set** has 16 short passages, 21 answerable queries, and 3 unanswerable queries. The fictional handbook is not an actual organization's policy. Queries intentionally exercise spelling variants that normalization addresses; the set was authored with the implementation and is **not held out**.

| Method | Recall@3 | MRR@3 | nDCG@3 |
| --- | ---: | ---: | ---: |
| Raw BM25 | 0.5476 | 0.5714 | 0.5530 |
| Normalized BM25 | 1.0000 | 1.0000 | 1.0000 |

These are actual local-run results, not estimates. They illustrate an orthographic effect on a tiny controlled corpus, not production or general Persian RAG performance. See the [saved report](docs/demo/report.md) and [per-query results](docs/demo/report.json).

Both methods return no results for only **1 of the 3 unanswerable queries**. Common overlapping words still retrieve irrelevant passages. This is a deliberate example of why retrieval scores and nonempty results are insufficient for answerability detection.

## Use your own passages

Corpus JSONL contains one complete passage per line:

```json
{"id":"doc-1-p1","title":"راهنمای اتصال","text":"متن یک بخش از سند...","source":"handbook/network.md#connection"}
```

Query JSONL contains binary relevance labels:

```json
{"id":"q-1","query":"پرسش نمونه","relevant_ids":["doc-1-p1"]}
{"id":"q-2","query":"پرسشی بدون پاسخ در این مجموعه","relevant_ids":[]}
```

IDs must be unique; labels must reference existing passage IDs. Sources are identifiers only: the tool never fetches their URLs. Split long documents into passages before indexing and use stable IDs. **Metrics are passage-level**, not document-level.

For a meaningful evaluation, create a separate query set from real usage, review the labels, keep it out of normalization/retrieval tuning, and report performance across query types and corpus sizes.

## Methods and metric definitions

- **Raw:** case-fold and tokenize title + text; index the resulting tokens with FTS5 `unicode61 remove_diacritics 0`.
- **Normalized:** additionally apply Unicode NFKC, map Arabic `ي/ك` to Persian `ی/ک`, map Persian/Arabic digits to ASCII, remove combining marks/tatweel, and replace ZWNJ/ZWJ with spaces. Apply the same preprocessing to documents and queries.
- Both methods use OR over distinct quoted query tokens and the same SQLite BM25 implementation. Ties are ordered by passage ID.
- **Recall@k:** relevant passages retrieved / all relevant passages.
- **MRR@k:** reciprocal rank of the first relevant passage, or zero, averaged across answerable queries.
- **nDCG@k:** binary relevance discounted by `log2(rank+1)`, divided by the ideal ranking, averaged across answerable queries.
- Unanswerable queries are excluded from those ranking metrics and reported separately using empty-result rate. This is not a calibrated abstention metric.

## Tests

```bash
python -m unittest discover -s tests -v
```

16 tests cover normalization, spelling variants, safe query construction, deterministic ranking, invalid labels, citation mapping, context budgets, hand-calculated metrics, and CLI output protection. GitHub Actions runs the tests and evaluation on Python 3.10 and 3.12.

## Scope and limitations

- Lexical retrieval only: no embeddings, reranker, stemming, synonym expansion, stop-word removal, or automatic answer generation.
- Half-spaces become boundaries; joined and separated compound words are not always equivalent. Normalization can also collapse distinctions that matter in some domains.
- The index is rebuilt in memory for each command. This favors simple reproducible experiments over production throughput; large corpora need persistent indexing and performance measurement.
- Python's Unicode handling and SQLite versions can affect tokenization and ranking. The report records both versions for reproducibility.
- Input documents are assumed to be UTF-8 text. PDF extraction and document chunking are outside this version's scope.

## References

- [SQLite FTS5 and its BM25 implementation](https://www.sqlite.org/fts5.html#the_bm25_function)
- [FTS5 query syntax and tokenizers](https://www.sqlite.org/fts5.html)

Original code and the synthetic example corpus/query set are MIT licensed.
