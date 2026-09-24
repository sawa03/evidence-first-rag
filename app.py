"""A localhost-only demonstration server, not a production deployment."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from rag.core import Engine, Ollama, load_chunks

ROOT = Path(__file__).resolve().parent


def make_handler(engine):
    class Handler(BaseHTTPRequestHandler):
        def send(self, code, payload, content_type="application/json; charset=utf-8"):
            body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.send(200, (ROOT / "web/index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/status":
                self.send(200, {"chunks": len(engine.chunks),
                    "embedding": bool(engine.provider and engine.provider.embedding_model),
                    "generation": bool(engine.provider and engine.provider.generation_model)})
            else:
                self.send(404, {"error": "Not found"})

        def do_POST(self):
            if self.path != "/api/ask":
                return self.send(404, {"error": "Not found"})
            # JSON-only endpoint; avoid cross-origin form submissions to local models.
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.send(415, {"error": "Use application/json"})
            origin = self.headers.get("Origin")
            expected = f"http://127.0.0.1:{self.server.server_port}"
            if origin and origin != expected:
                return self.send(403, {"error": "Cross-origin requests are disabled"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16000:
                    raise ValueError("Invalid request length")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict) or type(data.get("generate", False)) is not bool:
                    raise ValueError("Expected an object and a boolean generate field")
                result = engine.ask(data.get("question"), data.get("mode", "bm25"),
                                    data.get("k", 3), data.get("generate", False))
                self.send(200, result)
            except (ValueError, TypeError) as exc:
                self.send(400, {"error": str(exc)})
            except Exception:
                self.send(503, {"error": "Local model unavailable or invalid response; check Ollama and model names."})

        def log_message(self, fmt, *args):
            pass  # Do not log user questions or model responses.
    return Handler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/corpus.json")
    parser.add_argument("--embedding-model")
    parser.add_argument("--generation-model")
    args = parser.parse_args()
    engine = Engine(load_chunks(args.corpus), Ollama(args.embedding_model, args.generation_model))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(engine))
    print(f"Evidence First RAG: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
