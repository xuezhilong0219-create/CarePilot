"""工作台页面：自包含、无服务端插值、身份来自用户表。"""

from pathlib import Path

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from carepilot.api.app import create_app
from carepilot.api.ui import DEMO_USERS
from carepilot.config import Settings
from carepilot.graph.nodes import Models
from carepilot.graph.state import ToolPlan
from carepilot.retrieval.base import Chunk, Hit
from carepilot.sandbox import tools as sbx
from carepilot.sandbox.data import build_world
from carepilot.sandbox.logistics import LogisticsClient
from carepilot.store import TicketStore, seed_demo_users
from carepilot.toolbox import Toolbox

from fakes import FakeRetriever, ScriptedInvoker, make_draft, make_triage

H = lambda uid: {"X-User-Id": uid}  # noqa: E731
CHUNK = Chunk(id="企业版/库存同步#0", title="库存同步", text="库存同步 > 常见问题 > 失败\n重新授权后再同步。",
              doc_id="企业版/库存同步", meta={"top_dir": "企业版", "title_key": "库存同步"})
UI_SOURCE = Path(__file__).resolve().parents[1] / "src" / "carepilot" / "api" / "ui.py"


def build(tmp_path):
    store = TicketStore(tmp_path / "t.sqlite")
    seed_demo_users(store)
    world = build_world(7)
    logistics = LogisticsClient(lambda oid: sbx.query_shipment(world, oid), timeout_s=0.3, retries=1, sleep=lambda s: None)
    tb = Toolbox(world=world, store=store, logistics=logistics, db_path=tmp_path / "g.sqlite")
    app = create_app(
        settings=Settings(llm_api_key="t", llm_base_url="http://localhost:0/v1", trace_dir=tmp_path / "traces"),
        models=Models(triage=ScriptedInvoker(make_triage()), draft=ScriptedInvoker(make_draft()),
                      planner=ScriptedInvoker(ToolPlan(tool="none"))),
        retriever=FakeRetriever([Hit(CHUNK, 1.0)]),
        checkpointer=InMemorySaver(), governed=True, store=store, toolbox=tb, chunks=[CHUNK],
    )
    return app


def test_index_serves_a_self_contained_page(tmp_path):
    app = build(tmp_path)
    with TestClient(app) as c:
        res = c.get("/")
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/html")
        body = res.text
        # 「运行轨迹」是页面上真有的 tab，不是只有接口
        for marker in ('id="tickets"', 'id="messages"', 'id="side"', '"tab"', "loadTrace", "/traces", "CarePilot",
                    "等待人工确认", "正在等坐席确认后回复你"):
            assert marker in body
        # 页面上没有任何外部资源：CSP 里也把 default-src 收成 none
        assert "http://" not in body and "https://" not in body
        assert "<script src" not in body and "<link " not in body
        csp = res.headers["content-security-policy"]
        assert "default-src 'none'" in csp and "connect-src 'self'" in csp
        assert "script-src 'unsafe-inline'" in csp


def test_page_never_embeds_ticket_text(tmp_path):
    """页面是静态的：客户消息只经 JSON 接口回来，由前端 textContent 渲染。"""
    app = build(tmp_path)
    payload = '<img src=x onerror="window.__xss=1">'
    with TestClient(app) as c:
        created = c.post("/tickets", json={"message": payload}, headers=H("c1"))
        assert created.status_code == 200, created.text
        ticket_id = created.json()["ticket"]["ticket_id"]
        view = c.get(f"/tickets/{ticket_id}", headers=H("c1")).json()
        assert any(m["body"] == payload for m in view["messages"])  # 原样存下来、原样给出
        page = c.get("/").text
        assert payload not in page and "onerror" not in page


def test_page_builds_dom_without_html_injection_helpers():
    """结构性保证：源码里不出现把字符串当 HTML 塞进 DOM 的 API。"""
    source = UI_SOURCE.read_text(encoding="utf-8")
    for banned in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert banned not in source, banned
    assert "textContent" in source


def test_whoami_reports_the_role_from_the_user_table(tmp_path):
    app = build(tmp_path)
    with TestClient(app) as c:
        assert c.get("/whoami").status_code == 401
        assert c.get("/whoami", headers=H("nobody")).status_code == 401
        customer = c.get("/whoami", headers=H("c1")).json()
        assert customer == {"user_id": "c1", "role": "customer", "shops": ["S1", "S2"]}
        assert c.get("/whoami", headers=H("a1")).json()["role"] == "agent"
        assert c.get("/whoami", headers=H("admin1")).json()["role"] == "admin"
        # 自报角色无效：请求头里写 role 也不影响结果
        spoofed = c.get("/whoami", headers={"X-User-Id": "c1", "X-Role": "admin"}).json()
        assert spoofed["role"] == "customer"


def test_demo_users_in_the_page_match_the_seeded_accounts(tmp_path):
    app = build(tmp_path)
    with TestClient(app) as c:
        for user in DEMO_USERS:
            assert c.get("/whoami", headers=H(user)).status_code == 200, user
        assert '"' + ",".join(DEMO_USERS) + '"' in c.get("/").text
