"""Exercise relay retention through real local HTTP, without external model calls."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import threading

import httpx

from clawbench.alignment.service import ServiceState, make_server


def test_concurrent_responses_get_distinct_evidence_files(tmp_path, monkeypatch):
    barrier = threading.Barrier(2)

    @contextmanager
    def upstream(*args, **kwargs):
        probe = json.loads(kwargs["content"])["probe"]

        class Response:
            status_code = 200
            headers = {"content-type": "text/event-stream"}

            def iter_bytes(self):
                barrier.wait(timeout=5)
                yield f"data: {probe}\n\n".encode()
                barrier.wait(timeout=5)

        yield Response()

    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    monkeypatch.setattr("clawbench.alignment.service.httpx.stream", upstream)
    state = ServiceState("assets", "preserve", tmp_path, "test-model")
    server = make_server(state, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:

        def request(probe):
            with httpx.Client(trust_env=False, timeout=10) as client:
                return client.post(
                    f"http://127.0.0.1:{server.server_port}/v1/responses",
                    json={"model": "test-model", "probe": probe},
                ).text

        with ThreadPoolExecutor(max_workers=2) as pool:
            replies = list(pool.map(request, ["one", "two"]))
        captured = [p.read_text() for p in tmp_path.glob("model-response-*.txt")]
        assert sorted(captured) == sorted(replies)
        assert len(captured) == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_stream_failure_keeps_the_received_prefix(tmp_path, monkeypatch):
    @contextmanager
    def upstream(*args, **kwargs):
        class Response:
            status_code = 200
            headers = {"content-type": "text/event-stream"}

            def iter_bytes(self):
                yield b"data: partial evidence\n\n"
                raise httpx.ReadError("upstream stream interrupted")

        yield Response()

    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    monkeypatch.setattr("clawbench.alignment.service.httpx.stream", upstream)
    state = ServiceState("assets", "preserve", tmp_path, "test-model")
    server = make_server(state, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with httpx.Client(trust_env=False, timeout=10) as client:
            result = client.post(
                f"http://127.0.0.1:{server.server_port}/v1/responses", json={"model": "test-model"}
            )
        assert result.content == b"data: partial evidence\n\n"
        captured = list(tmp_path.glob("model-response-*.txt"))
        assert len(captured) == 1
        assert captured[0].read_bytes() == result.content
        events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
        assert events[-1]["type"] == "model_error"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
