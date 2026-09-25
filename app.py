"""A localhost-only demonstration server, not a production deployment."""
import argparse
import getpass
import json
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from rag.core import Engine, Ollama, load_chunks
from rag.openai_provider import OpenAIProvider, ModelServiceError

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
                self.send(200, {"service": "evidence-first-rag", "chunks": len(engine.chunks),
                    "provider": getattr(engine.provider, "name", "offline"),
                    "model": getattr(engine.provider, "generation_model", None),
                    "key_configured": getattr(engine.provider, "ready", False),
                    "embedding": bool(engine.provider and engine.provider.embedding_model),
                    "generation": bool(engine.provider and engine.provider.generation_model and getattr(engine.provider, "ready", True))})
            else:
                self.send(404, {"error": "Not found"})

        def do_POST(self):
            if self.path not in ("/api/ask", "/api/configure"):
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
                if self.path == "/api/configure":
                    if not isinstance(engine.provider, OpenAIProvider):
                        return self.send(400, {"error": "此服务不是 OpenAI 模式。"})
                    if not isinstance(data, dict):
                        raise ValueError("Expected a JSON object")
                    engine.provider.configure_key(data.get("api_key"))
                    return self.send(200, {"configured": True})
                if not isinstance(data, dict) or type(data.get("generate", False)) is not bool:
                    raise ValueError("Expected an object and a boolean generate field")
                result = engine.ask(data.get("question"), data.get("mode", "bm25"),
                                    data.get("k", 3), data.get("generate", False))
                self.send(200, result)
            except ModelServiceError as exc:
                self.send(502, {"error": str(exc)})
            except (ValueError, TypeError) as exc:
                self.send(400, {"error": str(exc)})
            except Exception:
                self.send(503, {"error": "模型服务不可用或响应异常，请检查服务配置。"})

        def log_message(self, fmt, *args):
            pass  # Do not log user questions or model responses.
    return Handler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/corpus.json")
    parser.add_argument("--embedding-model")
    parser.add_argument("--generation-model")
    parser.add_argument("--provider", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--prompt-api-key", action="store_true", help="Hidden local key input for OpenAI")
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    args = parser.parse_args()
    if args.prompt_api_key and args.provider != "openai":
        parser.error("--prompt-api-key requires --provider openai")
    if args.provider == "openai" and args.embedding_model:
        parser.error("OpenAI 模式当前只支持本地 BM25 检索，请去掉 --embedding-model")
    try:
        if args.provider == "openai":
            api_key = None
            if args.prompt_api_key:
                # Fail rather than falling back to echoed input in a noninteractive terminal.
                with warnings.catch_warnings():
                    warnings.simplefilter("error", getpass.GetPassWarning)
                    api_key = getpass.getpass("OpenAI API key (hidden): ")
            provider = OpenAIProvider(args.generation_model, api_key, args.max_output_tokens, allow_unconfigured=not args.prompt_api_key)
            del api_key
            print("OpenAI mode: generation sends your question and retrieved text to OpenAI and uses API quota.", flush=True)
        else:
            provider = Ollama(args.embedding_model, args.generation_model)
    except (ValueError, getpass.GetPassWarning, EOFError) as exc:
        parser.error(str(exc) or "请在本机交互式终端中输入密钥。")
    engine = Engine(load_chunks(args.corpus), provider)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(engine))
    print(f"Evidence First RAG: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
