"""治理模式的工作台页面：一个自包含的 HTML，服务端不往里拼任何业务数据。

为什么不做成前端工程：这个项目刻意保持「无构建链」——装好 Python 起一个进程就能用，
不引入 node/pnpm/webpack。页面全部是原生 DOM，数据一律通过 JSON 接口取。

安全约束（与 viewer/render.py 同一套）：
- HTML 里**没有任何服务端插值**：工单正文、客户消息、知识库片段都是 JS 取回来之后
  用 textContent 塞进 DOM 的，所以「客户消息里写 <img onerror>」不会变成 DOM 节点；
- CSP 只放行同源 fetch（`connect-src 'self'`），没有外部脚本/样式/图片；
- 服务端只注入一个常量：演示用户列表（`__USERS__`），不来自用户输入。
"""

from __future__ import annotations

# 演示账号：与 store/service.py 的 seed_demo_users 一致
DEMO_USERS = ("c1", "c2", "c3", "a1", "a2", "admin1")

CSP = (
    "default-src 'none'; connect-src 'self'; style-src 'unsafe-inline'; "
    "script-src 'unsafe-inline'; img-src 'none'; base-uri 'none'; "
    "form-action 'none'; frame-ancestors 'none'"
)

_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="__CSP__">
<title>CarePilot · 售后工单台</title>
<style>
:root {
  --paper: #f2efe9;
  --panel: #fffdf9;
  --panel-2: #f8f5ef;
  --ink: #1c1a16;
  --muted: #6d675c;
  --line: #e0dbd0;
  --line-strong: #cfc8ba;
  --accent: #2f6b52;
  --accent-soft: #e7f0ea;
  --warn: #9a5b0b;
  --warn-soft: #fbf0dc;
  --danger: #a13a2b;
  --danger-soft: #f9e9e6;
  --mono: ui-monospace, SFMono-Regular, "Cascadia Mono", Consolas, monospace;
  --radius: 12px;
  --shadow: 0 1px 2px rgba(28, 26, 22, .06), 0 8px 24px -18px rgba(28, 26, 22, .5);
}
@media (prefers-color-scheme: dark) {
  :root {
    --paper: #16171a;
    --panel: #1d1f23;
    --panel-2: #212429;
    --ink: #e8e4dc;
    --muted: #9a958a;
    --line: #303338;
    --line-strong: #3d4147;
    --accent: #7fbf9c;
    --accent-soft: #22302a;
    --warn: #d9a45a;
    --warn-soft: #2f2718;
    --danger: #d98a7c;
    --danger-soft: #33211e;
    --shadow: 0 1px 2px rgba(0, 0, 0, .4), 0 8px 24px -18px rgba(0, 0, 0, .9);
  }
}
* { box-sizing: border-box; }
[hidden] { display: none !important; }  /* 作者样式里的 display 会盖掉 UA 的 [hidden] */
html, body { height: 100%; overflow: hidden; }
body {
  margin: 0; background: var(--paper); color: var(--ink);
  font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
  -webkit-font-smoothing: antialiased;
  display: grid; grid-template-rows: auto minmax(0, 1fr); height: 100vh;
}
button, input, select, textarea { font: inherit; color: inherit; }
button { cursor: pointer; }
button:disabled { cursor: not-allowed; opacity: .5; }

/* ---------------------------------------------------------------- 顶栏 */
.top {
  display: flex; align-items: center; justify-content: space-between; gap: 16px;
  padding: 14px 22px; background: var(--panel); border-bottom: 1px solid var(--line);
  position: sticky; top: 0; z-index: 20;
}
.brand { display: flex; align-items: center; gap: 12px; }
.mark {
  width: 34px; height: 34px; border-radius: 9px; display: grid; place-items: center;
  background: var(--accent); color: #fff; font: 600 13px/1 var(--mono); letter-spacing: .5px;
}
.brand h1 { margin: 0; font-size: 16px; font-weight: 650; letter-spacing: .2px; }
.brand p { margin: 1px 0 0; font-size: 12px; color: var(--muted); }
.who { display: flex; align-items: center; gap: 10px; font-size: 12px; color: var(--muted); }
.who select {
  appearance: none; background: var(--panel-2); border: 1px solid var(--line);
  border-radius: 8px; padding: 6px 28px 6px 10px; font-family: var(--mono); font-size: 12.5px;
  background-image: linear-gradient(45deg, transparent 50%, var(--muted) 50%), linear-gradient(135deg, var(--muted) 50%, transparent 50%);
  background-position: calc(100% - 14px) 12px, calc(100% - 9px) 12px;
  background-size: 5px 5px, 5px 5px; background-repeat: no-repeat;
}
.pill {
  padding: 3px 9px; border-radius: 999px; font-size: 11.5px; font-weight: 600;
  background: var(--accent-soft); color: var(--accent); border: 1px solid transparent;
}
.pill.customer { background: var(--panel-2); color: var(--muted); border-color: var(--line); }

/* ---------------------------------------------------------------- 三栏 */
main {
  display: grid; grid-template-columns: 286px minmax(380px, 1fr) 372px;
  gap: 0; min-height: 0; align-items: stretch; overflow: hidden;
}
.col { min-width: 0; display: flex; flex-direction: column; }
.list { border-right: 1px solid var(--line); background: var(--panel-2); }
.thread { background: var(--paper); }
.side { border-left: 1px solid var(--line); background: var(--panel-2); overflow-y: auto; }
.colhead {
  display: flex; align-items: center; justify-content: space-between; gap: 10px;
  padding: 14px 16px 10px; border-bottom: 1px solid var(--line);
}
.colhead h2 { margin: 0; font-size: 12px; font-weight: 700; letter-spacing: .9px; text-transform: uppercase; color: var(--muted); }
.ghost {
  background: transparent; border: 1px solid var(--line-strong); border-radius: 8px;
  padding: 5px 10px; font-size: 12px; color: var(--ink);
}
.ghost:hover { border-color: var(--accent); color: var(--accent); }
.ghost.on { border-color: var(--accent); color: var(--accent); background: var(--accent-soft); }
.primary {
  background: var(--accent); color: #fff; border: 0; border-radius: 9px;
  padding: 9px 16px; font-weight: 600; font-size: 13px;
}
.primary:hover:not(:disabled) { filter: brightness(1.06); }
.primary.warn { background: var(--warn); }
.danger { background: transparent; color: var(--danger); border: 1px solid var(--danger); border-radius: 9px; padding: 9px 14px; }

/* ---------------------------------------------------------------- 工单列表 */
.items { overflow-y: auto; padding: 8px; display: flex; flex-direction: column; gap: 4px; }
.item {
  text-align: left; width: 100%; background: transparent; border: 1px solid transparent;
  border-radius: 10px; padding: 10px 11px; display: grid; gap: 5px;
}
.item:hover { background: var(--panel); border-color: var(--line); }
.item.on { background: var(--panel); border-color: var(--line-strong); box-shadow: var(--shadow); }
.item .l1 { display: flex; align-items: center; gap: 8px; }
.item .id { font: 600 12px/1 var(--mono); color: var(--muted); }
.item .subj {
  font-size: 13px; color: var(--ink); overflow: hidden; text-overflow: ellipsis;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
}
.item .l2 { display: flex; align-items: center; gap: 6px; font-size: 11.5px; color: var(--muted); }
.empty { padding: 26px 14px; color: var(--muted); font-size: 12.5px; text-align: center; line-height: 1.8; }
.hero { flex: 1; display: grid; place-items: center; padding: 30px; }
.hero .card { max-width: 380px; text-align: center; }
.hero h3 { margin: 0 0 8px; font-size: 17px; font-weight: 650; }
.hero p { margin: 0 0 14px; color: var(--muted); font-size: 13px; line-height: 1.7; }
.hero code { font: 12px var(--mono); background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 1px 6px; }

.badge {
  display: inline-flex; align-items: center; gap: 4px; padding: 2px 7px; border-radius: 6px;
  font-size: 11px; font-weight: 650; letter-spacing: .2px; white-space: nowrap;
  background: var(--panel-2); color: var(--muted); border: 1px solid var(--line);
}
.badge[data-tone="run"] { background: var(--accent-soft); color: var(--accent); border-color: transparent; }
.badge[data-tone="wait"] { background: var(--warn-soft); color: var(--warn); border-color: transparent; }
.badge[data-tone="bad"] { background: var(--danger-soft); color: var(--danger); border-color: transparent; }

/* ---------------------------------------------------------------- 会话 */
.t-head { padding: 14px 20px; border-bottom: 1px solid var(--line); background: var(--panel); }
.t-head .row1 { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.t-head h3 { margin: 0; font-size: 15px; font-weight: 650; }
.t-head .meta { margin-top: 6px; display: flex; gap: 14px; flex-wrap: wrap; font-size: 12px; color: var(--muted); }
.t-head .meta b { font-weight: 600; color: var(--ink); }

.progress { display: flex; gap: 0; padding: 10px 20px; background: var(--panel); border-bottom: 1px solid var(--line); overflow-x: auto; }
.step { display: flex; align-items: center; gap: 7px; font-size: 11.5px; color: var(--muted); padding-right: 12px; }
.step + .step::before { content: ""; width: 14px; height: 1px; background: var(--line-strong); margin-right: 4px; }
.step .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--line-strong); flex: none; }
.step.done { color: var(--ink); }
.step.done .dot { background: var(--accent); }
.step.now { color: var(--accent); font-weight: 600; }
.step.now .dot { background: var(--accent); box-shadow: 0 0 0 0 var(--accent); animation: pulse 1.4s infinite; }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(47,107,82,.5); } 70% { box-shadow: 0 0 0 7px rgba(47,107,82,0); } 100% { box-shadow: 0 0 0 0 rgba(47,107,82,0); } }

.messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 14px; }
.msg { max-width: 74%; display: flex; flex-direction: column; gap: 5px; }
.msg .who { font-size: 11px; color: var(--muted); font-family: var(--mono); letter-spacing: .3px; }
.msg .body {
  padding: 10px 13px; border-radius: 12px; background: var(--panel);
  border: 1px solid var(--line); white-space: pre-wrap; word-break: break-word; font-size: 13.5px;
}
.msg.customer { align-self: flex-end; align-items: flex-end; }
.msg.customer .body { background: var(--accent-soft); border-color: transparent; }
.msg.internal .body { background: repeating-linear-gradient(135deg, var(--panel-2), var(--panel-2) 6px, transparent 6px, transparent 12px); border-style: dashed; color: var(--muted); }
.msg.system .body { background: transparent; border-style: dashed; color: var(--muted); font-size: 12.5px; }

.composer { border-top: 1px solid var(--line); background: var(--panel); padding: 12px 16px 14px; display: grid; gap: 8px; }
.composer textarea {
  width: 100%; resize: vertical; min-height: 46px; padding: 10px 12px; border-radius: 10px;
  border: 1px solid var(--line-strong); background: var(--panel-2); outline: none;
}
.composer textarea:focus { border-color: var(--accent); }
.composer .row { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
.hint { font-size: 11.5px; color: var(--muted); }

/* ---------------------------------------------------------------- 处理台 */
.side section { padding: 14px 16px; border-bottom: 1px solid var(--line); }
.side h4 {
  margin: 0 0 10px; font-size: 11px; font-weight: 700; letter-spacing: .9px;
  text-transform: uppercase; color: var(--muted);
}
.kv { display: grid; grid-template-columns: 68px 1fr; gap: 6px 10px; font-size: 12.5px; }
.kv dt { color: var(--muted); }
.kv dd { margin: 0; word-break: break-word; }
.mono { font-family: var(--mono); font-size: 12px; }
.draft {
  white-space: pre-wrap; word-break: break-word; font-size: 13px; line-height: 1.65;
  background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 11px 12px;
}
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 9px; }
.chip {
  border: 1px solid var(--line-strong); background: var(--panel); border-radius: 999px;
  padding: 3px 9px; font-size: 11.5px; color: var(--accent);
}
.chip:hover { border-color: var(--accent); }
.actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.actions button { flex: 1 1 auto; }
.note { width: 100%; padding: 8px 10px; border-radius: 9px; border: 1px solid var(--line-strong); background: var(--panel); margin-top: 10px; }
.callout {
  border-radius: 10px; padding: 11px 12px; font-size: 12.5px; line-height: 1.6;
  background: var(--warn-soft); color: var(--warn); border: 1px solid transparent;
}
.callout.plain { background: var(--panel); color: var(--muted); border-color: var(--line); }
.callout b { display: block; margin-bottom: 4px; }
table.audit { width: 100%; border-collapse: collapse; font-size: 11.5px; }
table.audit td { padding: 5px 4px; border-top: 1px solid var(--line); vertical-align: top; word-break: break-word; }
table.audit tr:first-child td { border-top: 0; }
table.audit td.k { color: var(--muted); white-space: nowrap; font-family: var(--mono); }

/* ---------------------------------------------------------------- 弹层 */
.modal { position: fixed; inset: 0; background: rgba(20, 18, 15, .42); display: grid; place-items: center; z-index: 40; padding: 24px; }
.modal[hidden] { display: none; }
.sheet {
  width: min(720px, 100%); max-height: 82vh; overflow: auto; background: var(--panel);
  border-radius: 14px; border: 1px solid var(--line-strong); box-shadow: 0 24px 60px -30px rgba(0, 0, 0, .6);
}
.sheet header { display: flex; align-items: start; justify-content: space-between; gap: 12px; padding: 16px 18px 10px; }
.sheet h3 { margin: 0; font-size: 14.5px; }
.sheet .sec { padding: 0 18px 4px; color: var(--muted); font-size: 12px; }
.sheet pre {
  margin: 12px 18px 18px; white-space: pre-wrap; word-break: break-word; font: 12.5px/1.7 var(--mono);
  background: var(--panel-2); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px;
}
.toast {
  position: fixed; left: 50%; bottom: 26px; transform: translateX(-50%);
  background: var(--ink); color: var(--panel); padding: 9px 16px; border-radius: 999px;
  font-size: 12.5px; opacity: 0; pointer-events: none; transition: opacity .18s; z-index: 60;
}
.toast.on { opacity: 1; }
.toast.bad { background: var(--danger); color: #fff; }
.spin { display: inline-block; width: 12px; height: 12px; border: 2px solid currentColor; border-right-color: transparent; border-radius: 50%; animation: spin .7s linear infinite; vertical-align: -1px; }
@keyframes spin { to { transform: rotate(360deg); } }

@media (max-width: 1180px) {
  main { grid-template-columns: 250px minmax(320px, 1fr); }
  .side { display: none; }
  .side.open { display: flex; position: fixed; inset: 63px 0 0 auto; width: min(380px, 92vw); z-index: 30; box-shadow: -18px 0 40px -30px rgba(0,0,0,.6); }
}
</style>
</head>
<body>
<header class="top">
  <div class="brand">
    <span class="mark">CP</span>
    <div><h1>CarePilot</h1><p>售后工单台 · 治理模式</p></div>
  </div>
  <div class="who">
    <label for="user">身份</label>
    <select id="user"></select>
    <span id="role" class="pill customer">—</span>
  </div>
</header>

<main>
  <section class="col list">
    <div class="colhead">
      <h2>工单</h2>
      <button id="new" class="ghost" hidden>＋ 新建</button>
    </div>
    <div id="tickets" class="items"></div>
  </section>

  <section class="col thread">
    <div id="t-head" class="t-head"><div class="meta">选一个工单开始。</div></div>
    <div id="progress" class="progress" hidden></div>
    <div id="messages" class="messages"></div>
    <form id="composer" class="composer" hidden>
      <textarea id="message" rows="2" maxlength="2000" placeholder="描述你遇到的问题…（Ctrl/⌘ + Enter 发送）"></textarea>
      <div class="row"><span id="hint" class="hint"></span><button id="send" class="primary" type="submit">发送</button></div>
    </form>
  </section>

  <aside id="side" class="col side"></aside>
</main>

<div id="modal" class="modal" hidden>
  <div class="sheet" role="dialog" aria-modal="true">
    <header><h3 id="m-title"></h3><button id="m-close" class="ghost">关闭</button></header>
    <div id="m-sec" class="sec"></div>
    <pre id="m-body"></pre>
  </div>
</div>
<div id="toast" class="toast" role="status" aria-live="polite"></div>

<script>
"use strict";
const USERS = "__USERS__".split(",");
const STEPS = ["classify", "ask_for_info", "plan_tools", "retrieve", "draft_reply", "route_escalation", "save_draft", "human_review", "execute_actions"];
const STEP_LABEL = {
  classify: "分诊", ask_for_info: "追问槽位", plan_tools: "查数据", retrieve: "检索",
  draft_reply: "起草", route_escalation: "升级", save_draft: "存草稿",
  human_review: "人工审核", execute_actions: "执行动作"
};
const STATUS_LABEL = {
  new: "新工单", open: "处理中", need_info: "等客户补充", awaiting_review: "待人工审核",
  awaiting_confirmation: "待二次确认", resolved: "已解决", escalated: "已升级",
  needs_human: "需人工接手", closed: "已关闭", running: "处理中"
};
const STATUS_TONE = {
  awaiting_review: "wait", awaiting_confirmation: "wait", needs_human: "bad", escalated: "bad",
  resolved: "run", open: "run", running: "run", closed: "run"
};

const state = { user: null, role: null, shops: [], ticketId: null, view: null, lastNodes: [] };
const $ = (id) => document.getElementById(id);

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function toast(msg, bad) {
  const t = $("toast");
  t.textContent = msg;
  t.className = "toast on" + (bad ? " bad" : "");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { t.className = "toast"; }, 3200);
}

async function api(path, opts) {
  const options = Object.assign({ headers: { "X-User-Id": state.user } }, opts || {});
  options.headers = Object.assign({ "X-User-Id": state.user }, (opts && opts.headers) || {});
  if (options.body) options.headers["content-type"] = "application/json";
  const res = await fetch(path, options);
  const ctype = res.headers.get("content-type") || "";
  const payload = ctype.includes("json") ? await res.json().catch(() => null) : await res.text();
  if (!res.ok) {
    const detail = payload && payload.detail ? payload.detail : "请求失败（" + res.status + "）";
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return payload;
}

async function stream(path, body, onNode) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "X-User-Id": state.user, "content-type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!res.ok) {
    const payload = await res.json().catch(() => null);
    throw new Error((payload && payload.detail) || "请求失败（" + res.status + "）");
  }
  if (!(res.headers.get("content-type") || "").includes("text/event-stream")) return res.json();
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const chunk = await reader.read();
    if (chunk.done) break;
    buffer += decoder.decode(chunk.value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const raw = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      let event = "message";
      let data = "";
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (!data) continue;
      let payload;
      try { payload = JSON.parse(data); } catch (err) { continue; }
      if (event === "result") return payload;
      if (event === "error") throw new Error((payload && payload.message) || "流内错误");
      onNode(event, payload);
    }
  }
  return null;
}

/* ------------------------------------------------------------- 渲染：列表 */
function statusBadge(status) {
  const badge = el("span", "badge", STATUS_LABEL[status] || status || "—");
  if (STATUS_TONE[status]) badge.dataset.tone = STATUS_TONE[status];
  return badge;
}

async function loadTickets(keep) {
  let rows;
  try { rows = await api("/tickets"); } catch (err) { toast(err.message, true); return; }
  const box = $("tickets");
  box.replaceChildren();
  if (!rows.length) {
    const hint = state.role === "customer" ? "还没有工单。点「新建」描述一个问题。" : "当前没有工单。换成 c1/c2/c3 建一个。";
    box.append(el("div", "empty", hint));
  }
  const wanted = keep === undefined ? state.ticketId : keep;
  for (const row of rows) {
    const b = el("button", "item" + (row.ticket_id === wanted ? " on" : ""));
    const l1 = el("div", "l1");
    l1.append(el("span", "id", row.ticket_id), statusBadge(row.status));
    if (row.priority === "high" || row.priority === "critical") {
      const p = el("span", "badge", row.priority === "critical" ? "紧急" : "高");
      p.dataset.tone = "bad";
      l1.append(p);
    }
    const subj = el("div", "subj", row.subject || "（无标题）");
    const l2 = el("div", "l2", (row.category || "未分类") + (row.owner_id ? " · " + row.owner_id : ""));
    b.append(l1, subj, l2);
    b.addEventListener("click", () => openTicket(row.ticket_id));
    box.append(b);
  }
  if (wanted && rows.some((r) => r.ticket_id === wanted)) await openTicket(wanted);
  else if (rows.length && !state.ticketId) await openTicket(rows[0].ticket_id);
}

/* ------------------------------------------------------------- 渲染：会话 */
function renderEmptyState() {
  const box = $("messages");
  box.replaceChildren();
  const hero = el("div", "hero");
  const card = el("div", "card");
  const isCustomer = state.role === "customer";
  card.append(el("h3", null, isCustomer ? "描述一个问题，剩下的交给它" : "从左边选一个工单"));
  card.append(el("p", null, isCustomer
    ? "系统先判断问题类型：笼统的会追问，需要查订单的会调用只读工具，其余去知识库找依据。对外发送一律要人工审核。"
    : "坐席视角能看到分诊结果、草稿与引用、审计流水；对外发送要人工批准，再点一次二次确认才真的发出去。"));
  if (isCustomer) {
    const tip = el("p");
    tip.append(document.createTextNode("点左上角 "));
    tip.append(el("code", null, "＋ 新建"));
    tip.append(document.createTextNode(" 开始。"));
    card.append(tip);
  }
  hero.append(card);
  box.append(hero);
  const side = $("side");
  side.replaceChildren();
  const sec = el("section");
  sec.append(el("h4", null, "处理台"));
  sec.append(el("div", "callout plain", isCustomer
    ? "你的工单会自动分类并回复；需要人工时会转给坐席。"
    : "选中工单后，这里会出现待审草稿、二次确认按钮、交接摘要和审计流水。"));
  side.append(sec);
}

function renderHead(view) {
  const head = $("t-head");
  head.replaceChildren();
  const ticket = view.ticket;
  const row = el("div", "row1");
  row.append(el("h3", null, ticket.ticket_id + " · " + (ticket.subject || "（无标题）")), statusBadge(view.status));
  head.append(row);
  const meta = el("div", "meta");
  const add = (k, v) => { const s = el("span", null, k + " "); s.append(el("b", null, v || "—")); meta.append(s); };
  add("分类", ticket.category);
  add("优先级", ticket.priority);
  if (state.role !== "customer") {
    add("负责人", ticket.owner_id || "未认领");
    if (ticket.sla_remaining_minutes !== undefined && ticket.sla_remaining_minutes !== null) add("SLA", remain(ticket.sla_remaining_minutes));
  }
  head.append(meta);
  if (view.followup_question) {
    const q = el("div", "callout", "");
    q.append(el("b", null, "还需要客户补充"), document.createTextNode(view.followup_question));
    q.style.marginTop = "10px";
    head.append(q);
  }
}

const AUTHOR = { customer: "客户", agent: "坐席", admin: "系统", system: "系统" };
const KIND = { draft: "草稿", note: "内部备注" };

function authorLabel(m) {
  const base = AUTHOR[m.author_role] || m.author_role || "系统";
  return m.kind && KIND[m.kind] ? base + " · " + KIND[m.kind] : base;
}

function remain(minutes) {
  const total = Math.max(0, Math.round(minutes));
  const h = Math.floor(total / 60), m = total % 60;
  if (h >= 24) return Math.floor(h / 24) + " 天 " + (h % 24) + " 小时";
  return h ? h + " 小时 " + m + " 分" : m + " 分";
}

function clock(ts) {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  return String(d.getHours()).padStart(2, "0") + ":" + String(d.getMinutes()).padStart(2, "0");
}

function renderMessages(view) {
  const box = $("messages");
  box.replaceChildren();
  if (!view.messages.length) { box.append(el("div", "empty", "还没有消息。")); return; }
  for (const m of view.messages) {
    const role = m.author_role === "customer" ? "customer" : (m.kind === "note" || m.kind === "draft" ? "internal" : (m.author_role === "system" ? "system" : "agent"));
    const wrap = el("div", "msg " + role);
    const who = el("div", "who", authorLabel(m) + " · " + clock(m.created_at));
    const body = el("div", "body", m.body);
    wrap.append(who, body);
    box.append(wrap);
  }
  box.scrollTop = box.scrollHeight;
}

function renderProgress(nodes) {
  state.lastNodes = nodes;
  const bar = $("progress");
  if (!nodes.length) { bar.hidden = true; bar.replaceChildren(); return; }
  bar.hidden = false;
  bar.replaceChildren();
  const seen = nodes.map((n) => n.node);
  const order = STEPS.filter((s) => seen.includes(s));
  order.forEach((name, i) => {
    const item = nodes.filter((n) => n.node === name).pop();
    const step = el("div", "step " + (i === order.length - 1 ? "now" : "done"));
    step.append(el("span", "dot"));
    const label = STEP_LABEL[name] || name;
    const detail = stepDetail(name, item && item.summary);
    step.append(el("span", null, detail ? label + " · " + detail : label));
    bar.append(step);
  });
}

function stepDetail(node, summary) {
  if (!summary) return "";
  if (node === "classify") return [summary.category, summary.severity].filter(Boolean).join("/");
  if (node === "retrieve") return (summary.hits || 0) + " 命中";
  if (node === "draft_reply") return summary.resolution || "";
  if (node === "route_escalation") return (summary.route_to || []).join(",");
  if (node === "ask_for_info") return ((summary.missing_fields || []).join(",")) || "";
  if (node === "execute_actions") return summary.status || "";
  return "";
}

async function openTicket(id) {
  state.ticketId = id;
  let view;
  try { view = await api("/tickets/" + encodeURIComponent(id)); } catch (err) { toast(err.message, true); return; }
  state.view = view;
  renderHead(view);
  renderMessages(view);
  renderProgress([]);
  renderSide(view);
  renderComposer(view);
  document.querySelectorAll(".item").forEach((b) => b.classList.remove("on"));
  const list = $("tickets").children;
  for (const node of list) if (node.textContent && node.textContent.includes(id)) node.classList.add("on");
}

function renderComposer(view) {
  const form = $("composer");
  const isCustomer = state.role === "customer";
  const closed = !!(view && view.ticket && view.ticket.status === "closed");
  form.hidden = !view || !isCustomer || closed;
  $("hint").textContent = closed ? "工单已关闭" : "消息会立刻进入自动处理流程";
}

/* ------------------------------------------------------------- 渲染：处理台 */
function renderSide(view) {
  const side = $("side");
  side.replaceChildren();
  const isStaff = state.role !== "customer";

  const sec0 = el("section");
  sec0.append(el("h4", null, "工单状态"));
  const kv = el("dl", "kv");
  const put = (k, v, mono) => { kv.append(el("dt", null, k), el("dd", mono ? "mono" : null, v)); };
  put("编号", view.ticket.ticket_id, true);
  put("流程", STATUS_LABEL[view.status] || view.status || "—");
  put("优先级", view.ticket.priority || "—");
  if (isStaff) {
    put("负责人", view.ticket.owner_id || "未认领");
    put("SLA", view.ticket.sla_remaining_minutes === null || view.ticket.sla_remaining_minutes === undefined ? "不限" : remain(view.ticket.sla_remaining_minutes));
  }
  sec0.append(kv);
  side.append(sec0);

  if (!isStaff) {
    const sec = el("section");
    sec.append(el("h4", null, "你能做什么"));
    const c = el("div", "callout plain", "");
    c.append(el("b", null, "自动处理"), document.createTextNode(
      view.status === "resolved" ? "这单已经解决了。" :
      view.status === "escalated" ? "已转给人工，稍后会有同事跟进。" :
      view.status === "need_info" ? "系统需要你补充信息，看上方提示。" :
      "系统正在处理，结果会以消息形式出现在会话里。"
    ));
    sec.append(c);
    side.append(sec);
    return;
  }

  const sec1 = el("section");
  sec1.append(el("h4", null, "分诊"));
  if (view.triage) {
    const mk = el("dl", "kv");
    const kv2 = (k, v) => { mk.append(el("dt", null, k), el("dd", null, v || "—")); };
    kv2("类别", view.triage.category);
    kv2("严重度", view.triage.severity);
    kv2("产品线", view.triage.product_edition);
    kv2("笼统", view.triage.is_vague ? "是" : "否");
    const missing = view.triage.missing_fields || [];
    kv2("缺槽位", missing.length ? missing.join(", ") : "无");
    sec1.append(mk);
  } else {
    sec1.append(el("div", "callout plain", "本轮还没有分诊结果。"));
  }
  side.append(sec1);

  if (view.abort_reason) {
    const sec = el("section");
    const c = el("div", "callout", "");
    c.append(el("b", null, "自动处理已中止"), document.createTextNode(view.abort_reason));
    sec.append(c);
    side.append(sec);
  }

  if (view.pending_review) {
    const card = view.pending_review;
    const sec = el("section");
    sec.append(el("h4", null, "待审核"));
    const res = card.resolution === "auto_reply" ? "自动回复草稿" : "建议升级";
    sec.append(el("div", "callout plain", res));
    if (card.draft) sec.append(el("div", "draft", card.draft));
    if (card.escalate_reason) sec.append(el("div", "callout plain", "升级原因：" + card.escalate_reason));
    const details = card.citation_details || [];
    if (details.length) {
      const chips = el("div", "chips");
      details.forEach((c, i) => {
        const chip = el("button", "chip", "[" + (i + 1) + "] " + (c.doc_title || c.id));
        chip.title = (c.section || "") + (c.edition ? " · " + c.edition : "");
        chip.addEventListener("click", () => showCitation(c.link || ("/citations/" + c.id)));
        chips.append(chip);
      });
      sec.append(chips);
    }
    const note = el("input", "note");
    note.placeholder = "审核备注（可选，会写进审计）";
    sec.append(note);
    const acts = el("div", "actions");
    const auto = card.resolution === "auto_reply";
    const approve = el("button", "primary", auto ? "通过并发送" : "确认升级");
    const edit = auto ? el("button", "ghost", "改后发送") : null;
    const reject = el("button", "danger", auto ? "驳回并升级" : "驳回");
    acts.append(approve);
    if (edit) acts.append(edit);
    acts.append(reject);
    sec.append(acts);
    approve.addEventListener("click", () => doReview("approve", null, note.value, approve));
    reject.addEventListener("click", () => doReview("reject", null, note.value, reject));
    if (edit) edit.addEventListener("click", () => {
      const box = el("div", "draft");
      const ta = el("textarea");
      ta.className = "draft";
      ta.style.width = "100%";
      ta.style.minHeight = "120px";
      ta.value = card.draft || "";
      const send = el("button", "primary", "用这段文字发送");
      send.style.marginTop = "8px";
      send.addEventListener("click", () => doReview("edit", ta.value, note.value, send));
      box.replaceChildren(ta, send);
      sec.insertBefore(box, acts);
      ta.focus();
    });
    side.append(sec);
  }

  if (view.pending_confirmation) {
    const pc = view.pending_confirmation;
    const sec = el("section");
    sec.append(el("h4", null, "二次确认"));
    const c = el("div", "callout", "");
    c.append(el("b", null, "这一步会真的发出去"),
      document.createTextNode("已批准的动作还在等第二个人点确认。批准与确认分开，是为了让「手滑」不至于变成「已发送」。"));
    sec.append(c);
    const kv3 = el("dl", "kv");
    kv3.append(el("dt", null, "动作"), el("dd", "mono", pc.tool || "send_reply"));
    kv3.append(el("dt", null, "审批"), el("dd", "mono", pc.approval_id || "—"));
    sec.append(kv3);
    const go = el("button", "primary warn", "确认发送（真实动作）");
    go.style.marginTop = "10px";
    go.style.width = "100%";
    go.addEventListener("click", () => doConfirm(go));
    sec.append(go);
    side.append(sec);
  }

  const acted = (view.actions || []).filter((a) => a && (a.ok === true || !(a.stage === "confirm" && !view.pending_confirmation)));
  if (acted.length) {
    const sec = el("section");
    sec.append(el("h4", null, "动作记录"));
    sec.append(el("div", "draft mono", JSON.stringify(acted, null, 2)));
    side.append(sec);
  }

  const sec3 = el("section");
  sec3.append(el("h4", null, "交接与审计"));
  const bar = el("div", "actions");
  const hb = el("button", "ghost", "交接摘要");
  const ab = el("button", "ghost", "审计流水");
  bar.append(hb, ab);
  sec3.append(bar);
  const out = el("div");
  out.style.marginTop = "10px";
  sec3.append(out);
  hb.addEventListener("click", async () => {
    out.replaceChildren(el("div", "empty", "加载中…"));
    try {
      const h = await api("/tickets/" + encodeURIComponent(view.ticket.ticket_id) + "/handoff");
      out.replaceChildren();
      out.append(el("div", "callout plain", "升级原因：" + (h.escalation_reason || "—")));
      const done = el("div", "draft", (h.actions_done || []).join("\n") || "（无已执行动作）");
      done.style.marginTop = "8px";
      out.append(done);
      if ((h.open_questions || []).length) {
        const q = el("div", "draft", h.open_questions.join("\n"));
        q.style.marginTop = "8px";
        out.append(q);
      }
    } catch (err) { out.replaceChildren(el("div", "empty", err.message)); }
  });
  ab.addEventListener("click", async () => {
    out.replaceChildren(el("div", "empty", "加载中…"));
    try {
      const rows = (await api("/tickets/" + encodeURIComponent(view.ticket.ticket_id) + "/audit")).slice(-40);
      const table = el("table", "audit");
      for (const row of rows) {
        const tr = el("tr");
        const when = typeof row.ts === "number" ? new Date(row.ts * 1000).toLocaleTimeString() : "—";
        tr.append(el("td", "k", when));
        const td = el("td");
        td.append(el("b", null, (row.decision || "") + " " + (row.tool || "")));
        td.append(document.createTextNode(" " + (row.gate && row.gate !== "none" ? "[" + row.gate + "] " : "") + (row.args_summary || row.code || "")));
        tr.append(td);
        table.append(tr);
      }
      out.replaceChildren(table.children.length ? table : el("div", "empty", "还没有审计记录。"));
    } catch (err) { out.replaceChildren(el("div", "empty", err.message)); }
  });
  side.append(sec3);
}

/* ------------------------------------------------------------- 动作 */
async function doReview(action, reply, note, button) {
  const view = state.view;
  if (!view) return;
  const body = { action: action };
  if (reply !== null && reply !== undefined) body.reply = reply;
  if (note) body.note = note;
  await runAction(button, async () => {
    const result = await stream("/tickets/" + encodeURIComponent(view.ticket.ticket_id) + "/review?stream=true", body,
      (ev, payload) => { if (payload.node) state.lastNodes = state.lastNodes.concat(payload); renderProgress(state.lastNodes); });
    if (result) applyView(result);
    toast(action === "reject" ? "已驳回，转人工升级" : "审核已提交");
    loadTickets(view.ticket.ticket_id);
  });
}

async function doConfirm(button) {
  const view = state.view;
  if (!view) return;
  await runAction(button, async () => {
    const result = await api("/tickets/" + encodeURIComponent(view.ticket.ticket_id) + "/confirm", { method: "POST" });
    applyView(result);
    toast("已确认并发送");
    loadTickets(view.ticket.ticket_id);
  });
}

async function runAction(button, work) {
  const label = button ? button.textContent : "";
  if (button) { button.disabled = true; button.replaceChildren(el("span", "spin"), document.createTextNode(" 处理中")); }
  try { await work(); }
  catch (err) { toast(err.message, true); }
  finally { if (button) { button.disabled = false; button.textContent = label; } }
}

function applyView(view) {
  state.view = view;
  renderHead(view);
  renderMessages(view);
  renderSide(view);
  renderComposer(view);
}

async function showCitation(path) {
  $("m-title").textContent = "引用原文";
  $("m-sec").textContent = "加载中…";
  $("m-body").textContent = "";
  $("modal").hidden = false;
  try {
    const c = await api(path);
    $("m-title").textContent = c.doc_title + (c.section ? " › " + c.section : "");
    $("m-sec").textContent = [c.edition, c.doc_id].filter(Boolean).join(" · ");
    $("m-body").textContent = c.text;
  } catch (err) {
    $("m-sec").textContent = err.message;
  }
}

/* ------------------------------------------------------------- 新建 / 切换身份 */
function openNew() {
  $("m-title").textContent = "新建工单";
  $("m-sec").textContent = "描述你遇到的问题。系统会先分诊，再决定是追问、查数据还是检索资料。";
  $("m-body").replaceChildren();
  const ta = el("textarea");
  ta.rows = 5;
  ta.maxLength = 2000;
  ta.style.cssText = "width:100%;padding:10px 12px;border-radius:10px;border:1px solid var(--line-strong);background:var(--panel);";
  ta.placeholder = "例如：企业中台导订单报错，提示库存同步失败…";
  const go = el("button", "primary", "提交");
  go.style.marginTop = "10px";
  go.addEventListener("click", () => {
    const text = ta.value.trim();
    if (!text) { toast("先写点内容", true); return; }
    $("modal").hidden = true;
    runAction(go, async () => {
      const result = await stream("/tickets?stream=true", { message: text }, (ev, payload) => {
        if (payload.node) { state.lastNodes = state.lastNodes.concat(payload); renderProgress(state.lastNodes); }
      });
      if (result) applyView(result);
      toast("工单已创建");
      await loadTickets(result ? result.ticket.ticket_id : undefined);
    });
  });
  const box = $("m-body");
  box.appendChild(ta);
  box.appendChild(go);
  $("modal").hidden = false;
  ta.focus();
}

async function switchUser(user) {
  state.user = user;
  try { localStorage.setItem("carepilot.user", user); } catch (err) {}
  let who;
  try { who = await api("/whoami"); } catch (err) { toast(err.message, true); return; }
  state.role = who.role;
  state.shops = who.shops || [];
  const pill = $("role");
  pill.textContent = who.role === "customer" ? "客户 · " + state.shops.join("/")
    : who.role === "agent" ? "坐席" : "管理员";
  pill.className = "pill" + (who.role === "customer" ? " customer" : "");
  $("new").hidden = who.role !== "customer";
  state.ticketId = null;
  state.view = null;
  renderProgress([]);
  $("t-head").replaceChildren(el("div", "meta", "选一个工单开始。"));
  renderEmptyState();
  renderComposer(null);
  await loadTickets(undefined);
}

function boot() {
  const select = $("user");
  for (const user of USERS) select.append(el("option", null, user));
  select.addEventListener("change", () => switchUser(select.value));
  $("new").addEventListener("click", openNew);
  $("m-close").addEventListener("click", () => { $("modal").hidden = true; });
  $("modal").addEventListener("click", (ev) => { if (ev.target === $("modal")) $("modal").hidden = true; });
  document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") $("modal").hidden = true; });
  $("composer").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const text = $("message").value.trim();
    if (!text || !state.ticketId) return;
    const button = $("send");
    await runAction(button, async () => {
      $("message").value = "";
      const result = await stream("/tickets/" + encodeURIComponent(state.ticketId) + "/messages?stream=true", { message: text },
        (ev, payload) => { if (payload.node) { state.lastNodes = state.lastNodes.concat(payload); renderProgress(state.lastNodes); } });
      if (result) applyView(result);
      loadTickets(state.ticketId);
    });
  });
  $("message").addEventListener("keydown", (ev) => {
    if ((ev.metaKey || ev.ctrlKey) && ev.key === "Enter") $("composer").requestSubmit();
  });
  let saved = "c1";
  try { saved = localStorage.getItem("carepilot.user") || "c1"; } catch (err) {}
  select.value = USERS.includes(saved) ? saved : "c1";
  switchUser(select.value);
}

document.addEventListener("DOMContentLoaded", boot);
</script>
</body>
</html>
"""


def render_ui() -> str:
    """返回工作台页面。只注入两个常量（演示用户列表、CSP），不注入任何业务数据。"""
    return _HTML.replace("__CSP__", CSP).replace("__USERS__", ",".join(DEMO_USERS))
