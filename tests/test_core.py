import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer
from app import make_handler
from evaluate import evaluate
from rag.core import Chunk, Engine, Ollama, cosine, load_chunks, rrf


class FakeModel:
    embedding_model = "test-only"
    generation_model = "test-only"
    def __init__(self, answer="网络错误最多重试三次 [retry]"):
        self.answer = answer
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[1., 0.] if any(t in s for t in ("重试", "retry", "重新尝试")) else [0., 1.] for s in texts]

    def generate(self, question, evidence):
        return self.answer, {"eval_count": 10}


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [Chunk("retry", "失败重试", "test", "网络错误最多重试三次。"),
                       Chunk("export", "结果导出", "test", "实验结果可以导出 CSV 文件。")]
        self.engine = Engine(self.chunks)

    def test_keyword_and_citation(self):
        result = self.engine.ask("网络错误会重试几次？", k=1)
        self.assertEqual(result["evidence"][0]["id"], "retry")
        self.assertIn("[retry]", result["answer"])
        self.assertEqual(result["answer_mode"], "extractive")

    def test_unrelated_query_abstains(self):
        self.assertEqual(self.engine.ask("火星直径")['status'], "abstained")

    def test_dense_ranks_semantic_match_and_indexes_once(self):
        model = FakeModel()
        engine = Engine(self.chunks, model)
        self.assertEqual(engine.retrieve("重新尝试", "dense", 1)[0]["id"], "retry")
        engine.retrieve("重新尝试", "dense", 1)
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(len(model.calls[0]), 2)

    def test_hybrid_fusion(self):
        engine = Engine(self.chunks, FakeModel())
        self.assertEqual(engine.retrieve("重试", "hybrid", 1)[0]["id"], "retry")
        ranks = rrf([[('a', 99), ('b', 1)], [('b', .9)]])
        self.assertEqual(ranks[0][0], 'b')

    def test_generation_rejects_invented_or_missing_citations(self):
        for answer in ("三次 [invented]", "三次", "三次 [retry] [invented]"):
            result = Engine(self.chunks, FakeModel(answer)).ask("网络错误重试", generate=True)
            self.assertEqual(result['status'], 'citation_rejected')

    def test_valid_citation_does_not_claim_semantic_verification(self):
        result = Engine(self.chunks, FakeModel()).ask("网络错误重试", generate=True)
        self.assertEqual(result['status'], 'generated_unverified')
        self.assertEqual(result['usage']['eval_count'], 10)

    def test_model_can_abstain(self):
        result = Engine(self.chunks, FakeModel("证据不足，无法回答。")).ask("重试", generate=True)
        self.assertEqual(result['status'], 'abstained')

    def test_validation(self):
        for query in (None, '', '  ', 'a'*2001):
            with self.assertRaises(ValueError):
                self.engine.ask(query)
        for k in (0, 11, True, "3"):
            with self.assertRaises(ValueError):
                self.engine.ask("重试", k=k)
        with self.assertRaises(ValueError):
            self.engine.retrieve("重试", mode="unknown")
        with self.assertRaises(ValueError):
            self.engine.retrieve("重试", mode="dense")

    def test_invalid_vectors_fail_explicitly(self):
        for vector in ([], [0., 0.], [float('nan'), 1], [1., 2., 3.]):
            with self.assertRaises(ValueError):
                cosine([1., 0.], vector)

    def test_duplicate_corpus_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'corpus.json'
            row = {"id": "same", "title": "x", "source": "test", "text": "example"}
            path.write_text(json.dumps([row, row]), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_chunks(path)

    def test_metrics_multi_relevance_and_negatives(self):
        questions = [{"id": "one", "question": "重试导出", "relevant": ["retry", "export"]},
                     {"id": "two", "question": "火星", "relevant": []}]
        result = evaluate(self.engine, questions, 'bm25', 1)
        self.assertEqual(result['summary']['recall_at_k'], .5)
        self.assertEqual(result['summary']['unanswerable_abstention_rate'], 1)

    def test_ollama_payload_contract_without_network(self):
        model = Ollama("embedding-test", "generation-test")
        calls = []
        def post(endpoint, payload):
            calls.append((endpoint, payload))
            return {"embeddings": [[1, 0]], "response": "结果 [retry]", "eval_count": 5}
        model._post = post
        self.assertEqual(model.embed(["test"]), [[1, 0]])
        answer, usage = model.generate("test", [{"id": "retry", "text": "data"}])
        self.assertFalse(calls[0][1]['truncate'])
        self.assertFalse(calls[1][1]['stream'])
        self.assertEqual(usage['eval_count'], 5)


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(
            Engine([Chunk('a', '重试', 'test', '重试三次')]) ))
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join()

    def test_page_and_query(self):
        with urlopen(self.url) as response:
            self.assertIn(b'EVIDENCE / FIRST', response.read())
        request = Request(self.url+'/api/ask', data=json.dumps({'question':'重试'}).encode(),
                          headers={'Content-Type':'application/json'})
        with urlopen(request) as response:
            self.assertEqual(json.load(response)['status'], 'evidence')

    def test_bad_requests_and_cross_origin(self):
        for body, headers, code in [
            (b'{}', {'Content-Type':'text/plain'}, 415),
            (b'{}', {'Content-Type':'application/json','Origin':'https://example.com'}, 403),
            (b'[]', {'Content-Type':'application/json'}, 400),
            (b'{', {'Content-Type':'application/json'}, 400),
            (b'{}', {'Content-Type':'application/json'}, 400)]:
            with self.assertRaises(HTTPError) as caught:
                urlopen(Request(self.url+'/api/ask', data=body, headers=headers))
            self.assertEqual(caught.exception.code, code)
            caught.exception.close()


if __name__ == '__main__':
    unittest.main()
