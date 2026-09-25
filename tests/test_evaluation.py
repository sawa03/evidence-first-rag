import copy
import json
from pathlib import Path
import unittest

from evaluate import summarize, validate_questions
from rag.core import load_chunks

ROOT = Path(__file__).resolve().parents[1]


class EvaluationTests(unittest.TestCase):
    def test_public_dataset_integrity(self):
        chunks = load_chunks(ROOT / 'data/python_docs/corpus.json')
        questions = json.loads((ROOT / 'data/python_docs/questions.json').read_text(encoding='utf-8'))
        validate_questions(questions, {c.id for c in chunks})
        self.assertEqual(len(questions), 100)
        self.assertEqual(len({q['question'] for q in questions}), 100)
        for split in ('dev', 'test'):
            subset = [q for q in questions if q['split'] == split]
            self.assertEqual(len(subset), 50)
            self.assertEqual(sum(not q['relevant'] for q in subset), 16)
            self.assertTrue(all(q['reference_answer'] for q in subset))

    def test_split_leakage_and_bad_labels_rejected(self):
        base = [{'id': 'a', 'question': 'one', 'relevant': ['x'], 'split': 'dev', 'topic': 'one'},
                {'id': 'b', 'question': 'two', 'relevant': ['y'], 'split': 'test', 'topic': 'two'}]
        for field, value in [('topic', 'one'), ('relevant', ['x']), ('relevant', ['missing']),
                             ('id', 'a'), ('relevant', ['y', 'y']), ('split', 'train')]:
            data = copy.deepcopy(base)
            data[1][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_questions(data, {'x', 'y'})

    def test_missing_denominators_are_null(self):
        row = dict(relevant=['a', 'b'], retrieved=['a'], recall_at_k=.5,
                   reciprocal_rank=1, status='evidence', total_ms=2)
        summary = summarize([row])
        self.assertEqual(summary['all_evidence_recall_at_k'], 0)
        self.assertIsNone(summary['unanswerable_abstention_rate'])
        self.assertIsNone(summary['abstention_precision'])
        self.assertIsNone(summarize([])['p95_ms'])

    def test_abstention_precision_uses_rejections(self):
        rows = [dict(relevant=r, retrieved=[], recall_at_k=0 if r else None,
                     reciprocal_rank=0 if r else None, status=s, total_ms=1)
                for r, s in [(['a'], 'abstained'), ([], 'abstained'), ([], 'evidence')]]
        result = summarize(rows)
        self.assertEqual(result['abstention_precision'], .5)
        self.assertEqual(result['unanswerable_false_acceptance_rate'], .5)
        self.assertEqual(result['answer_coverage'], 1/3)


if __name__ == '__main__':
    unittest.main()
