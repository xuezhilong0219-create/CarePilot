"""端到端冒烟：真实调用模型，走一遍 FastAPI + LangGraph（含 interrupt / resume）。"""
import json
import time

from fastapi.testclient import TestClient

from carepilot.api.app import app


def show(title, resp):
    d = resp.json() if hasattr(resp, "json") else resp
    print(f"\n### {title}  [{resp.status_code}]")
    print(json.dumps(d, ensure_ascii=False, indent=2)[:1400])
    return d


with TestClient(app) as c:
    t = time.time()
    d = show("1 缺信息 -> 追问", c.post("/tickets", json={"message": "我的设备坏了，帮我看看"}))
    print(f"({time.time()-t:.1f}s)")

    t = time.time()
    d = show("2 可自动回复 -> 等待审核", c.post("/tickets", json={"message": "我的 X200 路由器开机完全没反应，指示灯也不亮，怎么办？"}))
    print(f"({time.time()-t:.1f}s)")
    if d["status"] == "awaiting_review":
        show("2b 审核通过", c.post(f"/tickets/{d['thread_id']}/review", json={"action": "approve"}))

    t = time.time()
    d = show("3 知识库没有 -> 升级路由", c.post("/tickets", json={"message": "X200 路由器连接 5G 频段时频繁掉线，已经三天了，影响了公司的视频会议，非常紧急"}))
    print(f"({time.time()-t:.1f}s)")
    if d["status"] == "awaiting_review":
        show("3b 审核确认升级", c.post(f"/tickets/{d['thread_id']}/review", json={"action": "approve", "note": "客户为企业用户"}))

    d = c.post("/tickets", json={"message": "X200 路由器怎么恢复出厂设置？"}).json()
    if d["status"] == "awaiting_review":
        show("4 审核驳回自动回复 -> 升级", c.post(f"/tickets/{d['thread_id']}/review", json={"action": "reject", "note": "客户情绪激动，转人工"}))

    print("\nsent replies:", len(app.state.system.replies), "escalations:", len(app.state.system.escalations))
