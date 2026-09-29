import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

from persian_retrieval.text import normalize
from persian_retrieval.engine import Retriever, context_bundle, read_jsonl
from persian_retrieval.evaluation import evaluate, ranking_metrics
from persian_retrieval.cli import main


def passage(pid, text):
    return {"id": pid, "title": pid, "text": text, "source": "test/"+pid}


class TextTests(unittest.TestCase):
    def test_letters_digits_diacritics_and_halfspace(self):
        self.assertEqual(normalize(' كِتاب\u200cهاي ۱۲٣  '), 'کتاب های 123')

    def test_normalization_is_idempotent(self):
        s = 'پشتيباني\u200cها و ٣٠ API'
        self.assertEqual(normalize(normalize(s)), normalize(s))

    def test_normalization_preserves_negation(self):
        self.assertNotEqual(normalize('فعال است'), normalize('فعال نیست'))


class SearchTests(unittest.TestCase):
    def test_orthographic_ablation(self):
        corpus = [passage('a', 'کلید'), passage('b', 'جلسه')]
        with Retriever(corpus, 'raw') as engine:
            self.assertEqual(engine.search('كليد'), [])
        with Retriever(corpus, 'normalized') as engine:
            hits = engine.search('كليد')
            self.assertEqual(hits[0]['id'], 'a')
            self.assertEqual(hits[0]['text'], 'کلید')

    def test_query_syntax_cannot_become_sql(self):
        with Retriever([passage('a', 'کلید')]) as engine:
            engine.search('" OR body:*; DROP TABLE passages; --')
            self.assertEqual(engine.search('کلید')[0]['id'], 'a')

    def test_empty_and_unmatched_queries(self):
        with Retriever([passage('a', 'کلید')]) as engine:
            self.assertEqual(engine.search('!!!'), [])
            self.assertEqual(engine.search('فوتبال'), [])

    def test_top_k_and_deterministic_ties(self):
        with Retriever([passage('b', 'کلید'), passage('a', 'کلید')]) as engine:
            self.assertEqual([h['id'] for h in engine.search('کلید', 1)], ['a'])
            self.assertEqual(engine.search('کلید'), engine.search('کلید'))

    def test_bad_corpus_and_parameters(self):
        for corpus in ([], [passage('a','x'),passage('a','y')], [{'id':'a'}]):
            with self.assertRaises(ValueError):
                Retriever(corpus)
        with self.assertRaises(ValueError):
            Retriever([passage('a','x')], 'unknown')
        with Retriever([passage('a','x')]) as engine:
            with self.assertRaises(ValueError):
                engine.search('x', 0)

    def test_context_budget_and_source_mapping(self):
        hits = [passage('a','x'*500), passage('b','یک متن کوتاه')]
        bundle = context_bundle('پرسش', hits, 40)
        self.assertLessEqual(len(bundle['context']), 40)
        self.assertEqual(bundle['context_chars'], len(bundle['context']))
        self.assertEqual(bundle['omitted_passage_ids'], ['a'])
        self.assertEqual(bundle['citations'], [{'label':'S1','passage_id':'b','source':'test/b'}])
        self.assertIn('یک متن کوتاه', bundle['context'])

    def test_context_never_truncates_passages(self):
        bundle = context_bundle('q', [passage('a','x'*40)], 10)
        self.assertEqual(bundle['context'], '')
        self.assertEqual(bundle['citations'], [])


class EvaluationTests(unittest.TestCase):
    def test_hand_calculated_metrics(self):
        m = ranking_metrics(['x','a','b'], ['a','b'], 2)
        self.assertEqual(m['recall'], 0.5)
        self.assertEqual(m['reciprocal_rank'], 0.5)
        self.assertAlmostEqual(m['ndcg'], (1/math.log2(3))/(1+1/math.log2(3)))

    def test_unanswerable_separate_and_null_when_absent(self):
        corpus = [passage('a','کلید')]
        q = [{'id':'1','query':'کلید','relevant_ids':['a']},
             {'id':'2','query':'فوتبال','relevant_ids':[]}]
        result = evaluate(corpus,q)['results']['normalized']
        self.assertEqual(result['mrr'], 1)
        self.assertEqual(result['empty_result_rate_unanswerable'], 1)
        result = evaluate(corpus,q[:1])['results']['normalized']
        self.assertIsNone(result['empty_result_rate_unanswerable'])

    def test_invalid_labels_fail_loudly(self):
        corpus = [passage('a','کلید')]
        for relevant in (['missing'], ['a','a'], 'a', [1]):
            with self.assertRaises(ValueError):
                evaluate(corpus,[{'id':'q','query':'کلید','relevant_ids':relevant}])

    def test_missing_corpus_fields_are_validated(self):
        with self.assertRaises(ValueError):
            evaluate([{'text':'x'}],[{'id':'q','query':'x','relevant_ids':[]}])

    def test_cli_writes_metrics_and_prevents_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            corpus, queries = root/'c.jsonl', root/'q.jsonl'
            corpus.write_text(json.dumps(passage('a','کلید')),encoding='utf-8')
            queries.write_text(json.dumps({'id':'q','query':'كليد','relevant_ids':['a']}),encoding='utf-8')
            args = ['evaluate','--corpus',str(corpus),'--queries',str(queries),'--output',str(root/'out')]
            with contextlib.redirect_stdout(io.StringIO()):
                main(args)
            report = json.loads((root/'out/report.json').read_text())
            self.assertEqual(report['results']['normalized']['mrr'],1)
            self.assertEqual(report['results']['raw']['mrr'],0)
            self.assertEqual(len(report['corpus_sha256']),64)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                main(args)
            self.assertEqual(raised.exception.code,2)

    def test_jsonl_reports_line_numbers(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'bad.jsonl'
            path.write_text('{}\nnot-json\n')
            with self.assertRaisesRegex(ValueError,'line 2'):
                read_jsonl(path)


if __name__ == '__main__':
    unittest.main()
