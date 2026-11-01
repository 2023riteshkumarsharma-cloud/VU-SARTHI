from fastapi.testclient import TestClient

from advisor import graph, server


def test_semester_one_without_batch_asks_then_uses_reply(monkeypatch):
    server.PENDING_BATCH_QUESTIONS.clear()
    monkeypatch.setattr(server.config, "chat_ready", lambda: True)
    calls = []

    def fake_ask(message, thread_id, profile):
        calls.append((message, thread_id, profile))
        return {"answer": "Semester 1 course list for the selected batch.", "sources": [],
                "latency_ms": 1, "flags": {"tools": [], "cited": []}}

    monkeypatch.setattr(graph, "ask", fake_ask)
    client = TestClient(server.app)
    first = client.post("/api/chat", json={"message": "What courses are in semster 1?", "thread_id": "sem1-followup"})
    assert first.status_code == 200
    assert first.json()["flags"]["needs_batch"] is True
    assert "2022, 2023, 2024, 2025, 2026" in first.json()["answer"]
    assert calls == []

    second = client.post("/api/chat", json={"message": "2024", "thread_id": "sem1-followup"})
    assert second.status_code == 200
    assert len(calls) == 1
    assert "What courses are in semster 1?" in calls[0][0]
    assert "batch 2024" in calls[0][0]
    assert calls[0][1] == "sem1-followup:none"
    server.PENDING_BATCH_QUESTIONS.clear()


def test_semester_one_with_batch_does_not_ask(monkeypatch):
    server.PENDING_BATCH_QUESTIONS.clear()
    monkeypatch.setattr(server.config, "chat_ready", lambda: True)
    calls = []
    monkeypatch.setattr(graph, "ask", lambda message, thread_id, profile: calls.append(message) or
                        {"answer": "For batch 2025.", "sources": [], "latency_ms": 1, "flags": {"tools": [], "cited": []}})
    response = TestClient(server.app).post("/api/chat", json={"message": "What courses are in semester 1 for 2025?"})
    assert response.status_code == 200
    assert "Which batch" not in response.json()["answer"]
    assert calls == ["What courses are in semester 1 for 2025?"]
