"""端到端冒烟：真实调用模型，走一遍 FastAPI + LangGraph（含 interrupt / resume）。"""
import json
import time

from fastapi.testclient import TestClient

from carepilot.api.app import app


def show(title, resp, keys=("status", "followup_question", "triage", "pending_review", "actions")):
    d = resp.json()
    print(f"\n### {title}  [{resp.status_code}]")
    brief = {k: d.get(k) for k in keys}
    if brief.get("triage"):
        brief["triage"] = {k: v for k, v in brief["triage"].items() if v}
    print(json.dumps(brief, ensure_ascii=False, indent=1)[:1500])
    return d


def timed(fn):
    t = time.time()
    r = fn()
    print(f"({time.time() - t:.1f}s)")
    return r


with TestClient(app) as c:
    d = timed(lambda: show("1 信息不全 -> 追问", c.post("/tickets", json={"message": "我的店铺有问题，帮我看看"})))

    d = timed(lambda: show("2 可自动回复 -> 等待审核", c.post("/tickets", json={"message": "京东店铺授权已失效，订单下载不下来，怎么重新授权？店铺名叫小熊数码"})))
    if d["status"] == "awaiting_review":
        show("2b 审核通过", c.post(f"/tickets/{d['thread_id']}/review", json={"action": "approve"}))

    d = timed(lambda: show("3 需要查客户数据 -> 升级", c.post("/tickets", json={"message": "我昨天有3笔淘宝订单在系统里不见了，帮我查一下是什么原因"})))
    if d["status"] == "awaiting_review":
        show("3b 审核确认升级", c.post(f"/tickets/{d['thread_id']}/review", json={"action": "approve", "note": "需要查订单日志"}))

    d = timed(lambda: c.post("/tickets", json={"message": "库存同步到京东一直失败，日志里提示请求过于频繁"}).json())
    if d["status"] == "awaiting_review":
        show("4 审核驳回自动回复 -> 升级", c.post(f"/tickets/{d['thread_id']}/review", json={"action": "reject", "note": "客户情绪激动，转人工"}))

    print("\nsent replies:", len(app.state.system.replies), "escalations:", len(app.state.system.escalations))
