import json

from fastapi.testclient import TestClient

from fakes import FailingInvoker, make_draft, make_hit, make_triage

MESSAGE = "京东店铺授权失效了，订单下不来"


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        name = None
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        if name:
            events.append((name, json.loads("\n".join(data_lines))))
    return events


def test_create_get_review_flow(make_app):
    app = make_app(make_triage(feature_name="店铺授权"), make_draft(), hits=[make_hit()])
    with TestClient(app) as client:
        created = client.post("/tickets", json={"message": MESSAGE})
        assert created.status_code == 200
        body = created.json()
        assert body["status"] == "awaiting_review"
        assert body["pending_review"]["resolution"] == "auto_reply"
        thread_id = body["thread_id"]

        fetched = client.get(f"/tickets/{thread_id}")
        assert fetched.status_code == 200
        assert fetched.json() == body

        reviewed = client.post(f"/tickets/{thread_id}/review", json={"action": "approve"})
        assert reviewed.status_code == 200
        assert reviewed.json()["status"] == "resolved"
        assert reviewed.json()["actions"][0]["tool"] == "send_reply"
        assert client.get(f"/tickets/{thread_id}").json()["status"] == "resolved"

        assert client.get("/health").json() == {"ok": True}


def test_unknown_thread_returns_404(make_app):
    app = make_app(make_triage(), make_draft(), hits=[make_hit()])
    with TestClient(app) as client:
        assert client.get("/tickets/nope").status_code == 404
        assert client.post("/tickets/nope/review", json={"action": "approve"}).status_code == 404


def test_review_non_awaiting_returns_409(make_app):
    app = make_app(make_triage(feature_name="店铺授权"), make_draft(), hits=[make_hit()])
    with TestClient(app) as client:
        thread_id = client.post("/tickets", json={"message": MESSAGE}).json()["thread_id"]
        assert client.post(f"/tickets/{thread_id}/review", json={"action": "approve"}).status_code == 200
        conflict = client.post(f"/tickets/{thread_id}/review", json={"action": "approve"})
        assert conflict.status_code == 409


def test_invalid_action_returns_422(make_app):
    app = make_app(make_triage(feature_name="店铺授权"), make_draft(), hits=[make_hit()])
    with TestClient(app) as client:
        thread_id = client.post("/tickets", json={"message": MESSAGE}).json()["thread_id"]
        resp = client.post(f"/tickets/{thread_id}/review", json={"action": "banana"})
        assert resp.status_code == 422


def test_sse_create_and_review(make_app):
    app = make_app(make_triage(feature_name="店铺授权"), make_draft(), hits=[make_hit()])
    with TestClient(app) as client:
        resp = client.post("/tickets?stream=true", json={"message": MESSAGE})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(resp.text)

        names = [name for name, _ in events]
        assert names == ["node", "node", "node", "result"]
        nodes = [data["node"] for name, data in events if name == "node"]
        assert nodes == ["classify", "retrieve", "draft_reply"]
        summaries = {data["node"]: data["summary"] for name, data in events if name == "node"}
        assert summaries["classify"] == {"category": "feature_howto", "severity": "medium"}
        assert summaries["retrieve"] == {"hits": 1}
        assert summaries["draft_reply"] == {"resolution": "auto_reply"}

        result = events[-1][1]
        assert result["status"] == "awaiting_review"
        # result 与同 thread 的非流式响应完全一致
        assert result == client.get(f"/tickets/{result['thread_id']}").json()

        reviewed = client.post(f"/tickets/{result['thread_id']}/review?stream=true", json={"action": "approve"})
        assert reviewed.status_code == 200
        review_events = parse_sse(reviewed.text)
        assert [data["node"] for name, data in review_events if name == "node"] == ["human_review", "execute_actions"]
        assert [name for name, _ in review_events] == ["node", "node", "result"]
        assert review_events[-1][1]["status"] == "resolved"
        assert review_events[-1][1] == client.get(f"/tickets/{result['thread_id']}").json()


def test_sse_error_event_when_node_fails(make_app):
    app = make_app(
        make_triage(feature_name="店铺授权"),
        FailingInvoker(RuntimeError("draft boom")),
        hits=[make_hit()],
    )
    with TestClient(app) as client:
        resp = client.post("/tickets?stream=true", json={"message": MESSAGE})
        assert resp.status_code == 200
        events = parse_sse(resp.text)
        names = [name for name, _ in events]
        assert names == ["node", "node", "error"]
        assert events[-1][1]["message"] == "draft boom"
        assert "result" not in names
