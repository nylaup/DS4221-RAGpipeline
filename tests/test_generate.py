"""Generation against a mock OpenAI-compatible server (no real LLM needed)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from recipe_rag import generate as g

CHUNKS = [
    {"chunk_id": "Corny_Soufflé:1:section:2", "title": "Corny Soufflé",
     "text": "Recipe: Corny Soufflé\nSection: Procedure\n1. Preheat oven to 325°F."},
    {"chunk_id": "Cajun_Burger:2:section:1", "title": "Cajun Burger", "text": "cook to 165°F"},
]


class _Handler(BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.seen.append((self.path, self.headers.get("Authorization"), body))
        reply = {"choices": [{"message": {"content":
                 "Preheat to 325°F [Corny_Soufflé:1:section:2]. Also [made-up:id]."}}]}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_generate_and_citations(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("LLM_BASE_URL", f"http://127.0.0.1:{srv.server_port}/v1")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "gemma-test")
    try:
        ans = g.generate("What oven temperature does the corn soufflé use?", CHUNKS)
    finally:
        srv.shutdown()
    path, auth, body = _Handler.seen[-1]
    assert path == "/v1/chat/completions" and auth == "Bearer k"
    assert body["model"] == "gemma-test" and body["messages"][0]["role"] == "system"
    assert "[Corny_Soufflé:1:section:2]" in body["messages"][1]["content"]
    # only ids that were in the prompt count as citations
    assert g.parse_citations(ans, CHUNKS) == ["Corny_Soufflé:1:section:2"]


def test_echo_mode(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    out = g.generate("q?", CHUNKS)
    assert out.startswith("[echo mode") and "Question: q?" in out
