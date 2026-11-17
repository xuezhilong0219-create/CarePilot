"""MCP 通道的契约：schema 从 ToolSpec 生成，执行一律落在网关。

这里不测 MCP 框架本身，测的是「把工具接到 MCP 上之后，网关的保证还在不在」：
- 暴露的工具必须和注册表一一对应（不能多暴露一个没经过闸的工具）
- inputSchema 必须带着 pydantic 的约束（pattern/maxLength），否则模型不知道边界
- 只读调用照常执行并写审计
- 写操作在这条通道上只能拿到 needs_approval，不能真的执行
- 权限闸对 MCP 调用同样生效
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import anyio
import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from carepilot.config import ROOT
from carepilot.mcpserver import build_server, build_toolbox, main


@pytest.fixture
def env():
    # 与 conftest 一致：沙箱里只有仓库根目录可写
    tmp = Path(tempfile.mkdtemp(dir=ROOT, prefix=".tmp-mcp-"))
    toolbox, store = build_toolbox(data_dir=tmp)
    yield toolbox, store, tmp
    shutil.rmtree(tmp, ignore_errors=True)


def call(server, name: str, arguments: dict) -> dict:
    """走 MCP 的 call_tool（不是直接调 gateway），这样连 schema 校验一起测到。"""
    result = anyio.run(lambda: server.call_tool(name, arguments))
    return json.loads(result.content[0].text)


def schemas(server) -> dict[str, dict]:
    return {tool.name: tool.input_schema for tool in anyio.run(server.list_tools)}


def test_exposed_tools_are_exactly_the_registry(env):
    toolbox, store, _ = env
    server = build_server(toolbox, store.get_principal("a1"), run_id="run-tools")

    listed = [tool.name for tool in anyio.run(server.list_tools)]

    assert listed == list(toolbox.registry)
    assert len(listed) == 9


def test_every_tool_has_a_description(env):
    toolbox, store, _ = env
    server = build_server(toolbox, store.get_principal("a1"), run_id="run-desc")

    missing = [tool.name for tool in anyio.run(server.list_tools) if not tool.description]

    assert missing == []


def test_input_schema_keeps_the_pydantic_constraints(env):
    """schema 是给模型看的契约。丢了 pattern，模型就会开始编单号。"""
    toolbox, store, _ = env
    schema = schemas(build_server(toolbox, store.get_principal("a1"), run_id="run-schema"))

    send = schema["send_reply"]
    assert send["properties"]["ticket_id"]["pattern"] == r"^T-\d{6}$"
    assert send["properties"]["text"]["maxLength"] == 4000
    assert send["required"] == ["ticket_id", "text"]

    escalate = schema["escalate_ticket"]["properties"]
    assert escalate["team"]["pattern"] == r"^[a-z][a-z0-9_]{2,40}$"


def test_read_tool_executes_and_lands_in_the_audit_log(env):
    toolbox, store, _ = env
    agent = store.get_principal("a1")
    order_id = next(iter(toolbox.world.orders))
    server = build_server(toolbox, agent, run_id="run-read")

    out = call(server, "query_order", {"order_id": order_id})

    assert out["outcome"] == "executed"
    assert out["result"]["found"] is True
    # 一次执行留两条审计：先写意图（execution_intent）再写结果，中间隔着副作用，进程崩了才查得清
    entries = toolbox.gateway.audit(run_id="run-read")
    assert [e["decision"] for e in entries] == ["approved", "executed"]
    assert entries[0]["code"] == "execution_intent"
    assert {e["actor_id"] for e in entries} == {"a1"}


def test_same_call_twice_replays_instead_of_running_again(env):
    """一次服务器进程 = 一个 run，所以同一个调用重复两次应当命中断言账本而不是再查一遍。"""
    toolbox, store, _ = env
    order_id = next(iter(toolbox.world.orders))
    server = build_server(toolbox, store.get_principal("a1"), run_id="run-replay")

    first = call(server, "query_order", {"order_id": order_id})
    second = call(server, "query_order", {"order_id": order_id})

    assert first["replayed"] is False
    assert second["replayed"] is True
    # 第二次没有再走一遍工具：审计里只有一条 replayed，没有第二条 execution_intent
    assert [e["decision"] for e in toolbox.gateway.audit(run_id="run-replay")] == ["approved", "executed", "replayed"]


def test_state_change_returns_needs_approval_and_does_not_change_anything(env):
    toolbox, store, _ = env
    customer, agent = store.get_principal("c1"), store.get_principal("a1")
    ticket = store.create_ticket(customer, subject="导出失败", shop_id="S1", thread_id="t-mcp-1")
    store.claim(agent, ticket.ticket_id)
    server = build_server(toolbox, agent, run_id="run-approve")

    out = call(server, "set_ticket_status", {"ticket_id": ticket.ticket_id, "to_status": "open"})

    assert out["outcome"] == "needs_approval"
    assert out["stage"] == "approve"
    assert out["approval_id"]
    assert store.get_ticket(agent, ticket.ticket_id).status == "new"  # 没被悄悄改掉
    entries = toolbox.gateway.audit(run_id="run-approve")
    assert [e["decision"] for e in entries] == ["needs_approval"]
    assert entries[0]["code"] == "approval_required"


def test_send_reply_cannot_be_sent_over_mcp(env):
    """对外发送要人工批准 + 二次确认，而 MCP 通道上没有第二个人。"""
    toolbox, store, _ = env
    customer, agent = store.get_principal("c1"), store.get_principal("a1")
    ticket = store.create_ticket(customer, subject="授权过期", shop_id="S1", thread_id="t-mcp-2")
    store.claim(agent, ticket.ticket_id)
    server = build_server(toolbox, agent, run_id="run-send")

    text = "已为您重新授权。"
    out = call(server, "send_reply", {"ticket_id": ticket.ticket_id, "text": text})

    assert out["outcome"] == "needs_approval"
    # 预览里带正文摘要（长度 + 8 位 sha1），不带正文本身——审批人看到的是「发了什么」而不是全文回显
    assert re.fullmatch(
        rf"ticket={re.escape(ticket.ticket_id)}, len={len(text)}, sha1=[0-9a-f]{{8}}", out["preview"]["args_summary"]
    )


def test_post_followup_requires_approval_over_mcp(env):
    toolbox, store, _ = env
    customer, admin = store.get_principal("c1"), store.get_principal("admin1")
    ticket = store.create_ticket(customer, subject="缺少订单号", shop_id="S1", thread_id="t-mcp-followup")
    server = build_server(toolbox, admin, run_id="run-followup")

    out = call(server, "post_followup", {
        "ticket_id": ticket.ticket_id,
        "question": "请提供您的账号密码和短信验证码以便我们核实。",
    })

    assert out["outcome"] == "needs_approval"
    assert out["stage"] == "approve"
    assert store.get_ticket(admin, ticket.ticket_id).status == "new"
    assert store.list_messages(customer, ticket.ticket_id) == []


def test_permission_gate_applies_to_mcp_callers(env):
    """客户身份连写工具的边都摸不到，返回的是拒绝而不是异常。"""
    toolbox, store, _ = env
    customer = store.get_principal("c1")
    ticket = store.create_ticket(customer, subject="导出失败", shop_id="S1", thread_id="t-mcp-3")
    server = build_server(toolbox, customer, run_id="run-customer")

    out = call(server, "send_reply", {"ticket_id": ticket.ticket_id, "text": "我自己回的"})

    assert out["outcome"] == "rejected"
    assert out["gate"] == "permission"
    assert out["code"] == "role_denied"


def test_scope_filter_applies_to_mcp_callers(env):
    """别人的店铺在 MCP 上也查不到，而且和「不存在」返回一样的东西。"""
    toolbox, store, _ = env
    customer = store.get_principal("c2")  # 只有 S3
    order_id = next(oid for oid, o in toolbox.world.orders.items() if o.shop_id == "S1")
    server = build_server(toolbox, customer, run_id="run-scope")

    out = call(server, "query_order", {"order_id": order_id})

    assert out["outcome"] == "executed"
    assert out["result"] == {"found": False}


def test_unknown_user_refuses_to_start(env):
    _, _, tmp = env
    assert main(["--user", "nobody", "--data-dir", str(tmp)]) == 2


def test_unwritable_data_dir_is_one_line_not_a_traceback(tmp_path, capsys):
    """目录建不出来时给一句话就退出。MCP 客户端只看到进程没了，栈对它没意义。"""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    rc = main(["--user", "a1", "--data-dir", str(blocker / "sub")])
    assert rc == 2
    err = capsys.readouterr().err
    assert err.startswith("cannot prepare data dir:")
    assert "Traceback" not in err


def test_stdio_transport_actually_speaks_mcp(env):
    """上面那些都直接调 server.call_tool，绕过了传输层。这一条起一个真子进程走 stdio，
    确认 `--user` / `--run-id` 能一路走到网关，而不是只有函数调得通。"""
    toolbox, _, tmp = env
    order_id = next(iter(toolbox.world.orders))
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "carepilot.mcpserver", "--user", "a1", "--data-dir", str(tmp), "--run-id", "run-stdio"],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )

    async def handshake():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                result = await session.call_tool("query_order", {"order_id": order_id})
                return listed, result

    listed, result = anyio.run(handshake)

    assert {tool.name for tool in listed.tools} == set(toolbox.registry)
    payload = json.loads(result.content[0].text)
    assert payload["outcome"] == "executed"
    assert payload["replayed"] is False


def test_mcp_channel_does_not_require_llm_settings(monkeypatch, tmp_path):
    """这条通道从不调用模型，就不该被 LLM 密钥挡住。

    `Settings` 把 llm_api_key / llm_base_url 设成必填，一旦有人把 `or Settings()`
    加回来，在没有 .env 的环境里（比如容器）启动就会直接崩。这里把 `__init__`
    换成炸弹：只要 `build_toolbox` 真的去实例化它，就会在这里炸出来。
    """
    from carepilot import mcpserver

    def boom(self):  # pragma: no cover - 只是为了在出错时给出清楚的信息
        raise AssertionError("MCP 通道不该实例化 Settings（它要求 LLM 密钥）")

    monkeypatch.setattr(mcpserver.Settings, "__init__", boom)

    toolbox, store = mcpserver.build_toolbox(data_dir=tmp_path)

    assert len(toolbox.registry) == 9
    assert store.get_principal("a1") is not None
