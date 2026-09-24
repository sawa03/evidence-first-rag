import io
import json
import os
import unittest
import threading
from http.server import ThreadingHTTPServer
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app import make_handler

from rag.core import Chunk, Engine
from rag.openai_provider import OpenAIProvider, ModelServiceError


def response(text="最多重试三次 [retry]", **changes):
    data = {"status": "completed", "output": [
        {"type": "reasoning", "summary": []},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}],
        "usage": {"input_tokens": 30, "output_tokens": 10, "total_tokens": 40}}
    data.update(changes)
    return io.BytesIO(json.dumps(data).encode())


class OpenAITests(unittest.TestCase):
    def setUp(self):
        self.provider = OpenAIProvider(api_key="test-placeholder-not-a-real-key")
        self.evidence = [{"id": "retry", "title": "重试", "text": "最多重试三次", "source": "private-path", "score": 1}]

    def test_request_contract_and_usage(self):
        with patch("rag.openai_provider.urlopen", return_value=response()) as send:
            answer, usage = self.provider.generate("重试几次", self.evidence)
        request = send.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "https://api.openai.com/v1/responses")
        self.assertEqual(payload["model"], "gpt-4.1-mini")
        self.assertFalse(payload["store"])
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["max_output_tokens"], 1024)
        self.assertNotIn("private-path", payload["input"])
        self.assertNotIn("score", payload["input"])
        self.assertNotIn("test-placeholder", payload["input"])
        self.assertEqual(usage, {"input_tokens": 30, "output_tokens": 10, "total_tokens": 40})
        self.assertIn("[retry]", answer)
        self.assertEqual(send.call_count, 1)

    def test_key_and_model_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                OpenAIProvider()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "OPENAI_MODEL": "configured-model"}):
            self.assertEqual(OpenAIProvider().generation_model, "configured-model")
            self.assertEqual(OpenAIProvider("explicit-model").generation_model, "explicit-model")
        for limit in (0, 4097, True, "100"):
            with self.assertRaises(ValueError):
                OpenAIProvider(api_key="test", max_output_tokens=limit)

    def test_unconfigured_startup_and_key_validation(self):
        provider = OpenAIProvider(api_key="", allow_unconfigured=True)
        self.assertFalse(provider.ready)
        with self.assertRaises(ModelServiceError):
            provider.generate("test", [])
        for key in (None, "", "  ", "key\nvalue", "a" * 513):
            with self.assertRaises(ValueError):
                provider.configure_key(key)
        provider.configure_key("test-placeholder")
        self.assertTrue(provider.ready)

    def test_http_errors_are_sanitized_and_not_retried(self):
        for code in (400, 401, 403, 404, 429, 500):
            error = HTTPError("https://api.openai.com/v1/responses", code, "secret-in-upstream-error", {}, io.BytesIO(b'secret'))
            with patch("rag.openai_provider.urlopen", side_effect=error) as send:
                with self.assertRaises(ModelServiceError) as caught:
                    self.provider.generate("重试", self.evidence)
                self.assertNotIn("secret", str(caught.exception))
                self.assertEqual(send.call_count, 1)

    def test_network_errors_are_sanitized(self):
        with patch("rag.openai_provider.urlopen", side_effect=URLError("secret")):
            with self.assertRaises(ModelServiceError) as caught:
                self.provider.generate("重试", self.evidence)
            self.assertNotIn("secret", str(caught.exception))

    def test_incomplete_empty_refusal_and_invalid_responses(self):
        variants = [response(status="incomplete"), response(output=[]), io.BytesIO(b'invalid'),
                    response(output=[{"type": "message", "role": "assistant", "content": [{"type": "refusal"}]}])]
        for value in variants:
            with patch("rag.openai_provider.urlopen", return_value=value):
                with self.assertRaises(ModelServiceError):
                    self.provider.generate("重试", self.evidence)

    def test_input_limit_prevents_request(self):
        with patch("rag.openai_provider.urlopen") as send:
            with self.assertRaises(ValueError):
                self.provider.generate("a" * 24001, self.evidence)
            send.assert_not_called()

    def test_citation_validation_still_applies(self):
        engine = Engine([Chunk("retry", "重试", "test", "重试三次")], self.provider)
        for text, expected in [("三次 [retry]", "generated_unverified"),
                               ("三次 [invented]", "citation_rejected"),
                               ("证据不足，无法回答。", "abstained")]:
            with patch("rag.openai_provider.urlopen", return_value=response(text)):
                self.assertEqual(engine.ask("重试", generate=True)["status"], expected)

    def test_offline_and_abstained_queries_never_call_api(self):
        engine = Engine([Chunk("retry", "重试", "test", "重试三次")], self.provider)
        with patch("rag.openai_provider.urlopen") as send:
            engine.ask("重试", generate=False)
            engine.ask("火星直径", generate=True)
            send.assert_not_called()

    def test_http_status_and_generation_keep_key_server_side(self):
        provider = OpenAIProvider(api_key="", allow_unconfigured=True)
        engine = Engine([Chunk("retry", "重试", "test", "重试三次")], provider)
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(engine))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(base + '/api/status') as res:
                status = json.load(res)
            self.assertEqual(status['provider'], 'openai')
            self.assertFalse(status['generation'])
            configure = Request(base + '/api/configure', data=json.dumps({'api_key': 'test-placeholder'}).encode(),
                                headers={'Content-Type': 'application/json'})
            with patch('rag.openai_provider.urlopen') as send:
                with urlopen(configure) as res:
                    self.assertEqual(json.load(res), {'configured': True})
                send.assert_not_called()
            with urlopen(base + '/api/status') as res:
                status = json.load(res)
            self.assertTrue(status['generation'])
            self.assertTrue(status['key_configured'])
            configure.add_header('Origin', 'https://example.com')
            with self.assertRaises(HTTPError) as blocked:
                urlopen(configure)
            self.assertEqual(blocked.exception.code, 403)
            blocked.exception.close()
            self.assertFalse(status['embedding'])
            self.assertNotIn('test-placeholder', json.dumps(status))
            request = Request(base + '/api/ask', data=json.dumps({'question': '重试', 'generate': True}).encode(),
                              headers={'Content-Type': 'application/json'})
            with patch('rag.openai_provider.urlopen', return_value=response()):
                with urlopen(request) as res:
                    result = json.load(res)
            self.assertEqual(result['status'], 'generated_unverified')
            self.assertEqual(result['usage']['total_tokens'], 40)
            with patch('rag.openai_provider.urlopen', side_effect=URLError('private-detail')):
                with self.assertRaises(HTTPError) as caught:
                    urlopen(request)
            self.assertEqual(caught.exception.code, 502)
            self.assertNotIn('private-detail', caught.exception.read().decode())
            caught.exception.close()
        finally:
            server.shutdown()
            server.server_close()
            worker.join()


if __name__ == "__main__":
    unittest.main()
