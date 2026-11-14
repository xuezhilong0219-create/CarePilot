"""把 RunTrace 列表渲染成一个自包含的 HTML 页面。

安全约束：摘要来自客户消息 / 知识库 / 订单留言，都是不可信文本。
- 数据只进 `application/json` 脚本块，并且 `< > & U+2028 U+2029` 都转成 \\uXXXX，
  这样即使文本里出现 `</script>` 也不会提前闭合标签；
- 前端只用 createElement / createTextNode / setAttribute / textContent 渲染，
  源码里不出现任何 HTML 注入 API。
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from ..observability.spec import RunTrace


def _json_for_script(data: Any) -> str:
    # ensure_ascii=False 保留中文；随后转义会破坏 HTML/JS 语法的字符。
    raw = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return (
        raw.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'">
<title>CarePilot Trace Viewer</title>
<style>
:root {
  --bg: #f5f6f8;
  --panel: #ffffff;
  --border: #d9dee5;
  --text: #1f2933;
  --muted: #66727f;
  --accent: #2b6cb0;
  --error: #b42318;
  --error-bg: #fef3f2;
  --warn: #9a6700;
  --warn-bg: #fff8e5;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
}
header { padding: 18px 24px 8px; }
h1 { margin: 0 0 12px; font-size: 20px; }
#summary { display: flex; flex-wrap: wrap; gap: 10px; }
.summary-card {
  min-width: 120px; padding: 10px 14px; border: 1px solid var(--border);
  border-radius: 8px; background: var(--panel);
}
.summary-label { color: var(--muted); font-size: 12px; }
.summary-value { font-size: 20px; font-weight: 600; }
main { display: grid; grid-template-columns: 360px minmax(0, 1fr); gap: 16px; padding: 8px 24px 24px; }
.panel { border: 1px solid var(--border); border-radius: 10px; background: var(--panel); overflow: hidden; }
.toolbar { display: flex; gap: 8px; padding: 12px; border-bottom: 1px solid var(--border); }
.toolbar select, .toolbar input {
  min-width: 0; padding: 6px 8px; border: 1px solid var(--border); border-radius: 6px; background: #fff;
}
.toolbar input { flex: 1; }
#list-note { padding: 8px 12px; color: var(--muted); font-size: 12px; border-bottom: 1px solid var(--border); }
#run-list { max-height: 70vh; overflow-y: auto; }
.run-item {
  display: block; width: 100%; padding: 10px 12px; border: 0; border-bottom: 1px solid var(--border);
  background: #fff; text-align: left; cursor: pointer; font: inherit; color: inherit;
}
.run-item:hover { background: #f0f5fb; }
.run-item[aria-current="true"] { background: #e8f1fb; }
.run-title { display: flex; justify-content: space-between; gap: 8px; font-weight: 600; }
.run-sub { color: var(--muted); font-size: 12px; }
.badge { display: inline-block; padding: 1px 7px; border-radius: 999px; color: #fff; font-size: 11px; }
.status-running { background: #2b6cb0; }
.status-resolved { background: #1f7a46; }
.status-escalated { background: #b45309; }
.status-need_info { background: #6b46c1; }
.status-error { background: #b42318; }
.status-budget_exceeded { background: #9a3412; }
.status-unknown { background: #4a5568; }
#detail { padding: 16px; }
.placeholder { color: var(--muted); }
.run-head { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px; margin-bottom: 14px; }
.run-head .run-id { font-size: 18px; font-weight: 600; }
.meta { color: var(--muted); font-size: 12px; }
.span-card { border: 1px solid var(--border); border-radius: 8px; margin-bottom: 10px; overflow: hidden; }
.span-card.has-error { border-color: var(--error); background: var(--error-bg); }
.span-head {
  display: flex; align-items: center; gap: 10px; width: 100%; padding: 9px 12px;
  border: 0; background: transparent; text-align: left; cursor: pointer; font: inherit; color: inherit;
}
.span-name { font-weight: 600; min-width: 110px; }
.span-duration { color: var(--muted); font-size: 12px; white-space: nowrap; }
.bar { flex: 1; height: 8px; min-width: 60px; border-radius: 999px; background: #e6eaf0; overflow: hidden; }
.bar-fill { height: 100%; background: var(--accent); }
.span-body { padding: 0 12px 12px; border-top: 1px solid var(--border); }
.summary-block { margin-top: 10px; }
.summary-label { margin-bottom: 3px; font-size: 12px; color: var(--muted); }
.summary-text {
  margin: 0; padding: 8px 10px; border: 1px solid var(--border); border-radius: 6px;
  background: #fbfcfd; white-space: pre-wrap; overflow-wrap: anywhere; font: 13px/1.5 inherit;
}
.error-text {
  margin-top: 10px; padding: 8px 10px; border-radius: 6px;
  background: var(--error-bg); color: var(--error); white-space: pre-wrap; overflow-wrap: anywhere;
}
table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 12px; }
th, td { padding: 5px 7px; border-bottom: 1px solid var(--border); text-align: left; vertical-align: top; }
th { color: var(--muted); font-weight: 600; background: #f8fafc; }
td.warn { color: var(--warn); background: var(--warn-bg); font-weight: 600; }
@media (max-width: 820px) {
  main { grid-template-columns: 1fr; }
  #run-list { max-height: 40vh; }
}
</style>
</head>
<body>
<header>
  <h1>CarePilot Trace Viewer</h1>
  <div id="summary"></div>
</header>
<main>
  <section class="panel" aria-label="run 列表">
    <div class="toolbar">
      <select id="status-filter" aria-label="按状态过滤"></select>
      <input id="search" type="search" placeholder="搜索 run_id / ticket_id" aria-label="搜索">
    </div>
    <div id="list-note"></div>
    <div id="run-list"></div>
  </section>
  <section class="panel" id="detail" aria-live="polite"></section>
</main>
<script type="application/json" id="trace-data">__TRACE_DATA__</script>
<script>
(function () {
  "use strict";

  var RUNS = [];
  var dataNode = document.getElementById("trace-data");
  try {
    var parsed = dataNode ? JSON.parse(dataNode.textContent || "[]") : [];
    RUNS = Array.isArray(parsed) ? parsed : [];
  } catch (err) {
    RUNS = [];
  }

  var state = { status: "all", query: "", selected: null };

  function num(value) {
    var n = Number(value);
    return isFinite(n) ? n : 0;
  }

  function str(value) {
    return value === null || value === undefined ? "" : String(value);
  }

  function make(tag, className, value) {
    var node = document.createElement(tag);
    if (className) {
      node.setAttribute("class", className);
    }
    if (value !== undefined && value !== null) {
      node.appendChild(document.createTextNode(str(value)));
    }
    return node;
  }

  function clear(node) {
    while (node.firstChild) {
      node.removeChild(node.firstChild);
    }
  }

  function setText(node, value) {
    clear(node);
    node.appendChild(document.createTextNode(str(value)));
  }

  function formatMs(ms) {
    var n = num(ms);
    return n >= 1000 ? (n / 1000).toFixed(2) + " s" : Math.round(n) + " ms";
  }

  function spansOf(run) {
    return Array.isArray(run.spans) ? run.spans : [];
  }

  function callsOf(run) {
    var result = [];
    spansOf(run).forEach(function (span) {
      var calls = Array.isArray(span.llm_calls) ? span.llm_calls : [];
      calls.forEach(function (call) { result.push(call); });
    });
    return result;
  }

  function totals(run) {
    var prompt = 0;
    var completion = 0;
    callsOf(run).forEach(function (call) {
      prompt += num(call.prompt_tokens);
      completion += num(call.completion_tokens);
    });
    return { prompt: prompt, completion: completion };
  }

  function runDuration(run) {
    var start = num(run.started_at);
    var end = num(run.ended_at);
    if (end > start) {
      return (end - start) * 1000;
    }
    return spansOf(run).reduce(function (total, span) {
      return total + num(span.duration_ms);
    }, 0);
  }

  function statusClass(status) {
    var known = {
      running: "status-running",
      resolved: "status-resolved",
      escalated: "status-escalated",
      need_info: "status-need_info",
      error: "status-error",
      budget_exceeded: "status-budget_exceeded"
    };
    return known[str(status)] || "status-unknown";
  }

  function statusBadge(status) {
    return make("span", "badge " + statusClass(status), str(status) || "unknown");
  }

  function card(label, value) {
    var node = make("div", "summary-card");
    node.appendChild(make("div", "summary-label", label));
    node.appendChild(make("div", "summary-value", value));
    return node;
  }

  function renderSummary() {
    var box = document.getElementById("summary");
    if (!box) { return; }
    clear(box);
    var prompt = 0;
    var completion = 0;
    var counts = {};
    RUNS.forEach(function (run) {
      var token = totals(run);
      prompt += token.prompt;
      completion += token.completion;
      var status = str(run.status) || "unknown";
      counts[status] = (counts[status] || 0) + 1;
    });
    box.appendChild(card("run 数", RUNS.length));
    Object.keys(counts).sort().forEach(function (status) {
      box.appendChild(card("状态 " + status, counts[status]));
    });
    box.appendChild(card("prompt tokens", prompt));
    box.appendChild(card("completion tokens", completion));
  }

  function buildFilters() {
    var select = document.getElementById("status-filter");
    if (select) {
      clear(select);
      var all = document.createElement("option");
      all.setAttribute("value", "all");
      all.appendChild(document.createTextNode("全部状态"));
      select.appendChild(all);
      var seen = {};
      RUNS.forEach(function (run) {
        var status = str(run.status) || "unknown";
        if (!seen[status]) {
          seen[status] = true;
          var option = document.createElement("option");
          option.setAttribute("value", status);
          option.appendChild(document.createTextNode(status));
          select.appendChild(option);
        }
      });
      select.addEventListener("change", function (event) {
        state.status = str(event.target.value);
        renderRunList();
      });
    }
    var search = document.getElementById("search");
    if (search) {
      search.addEventListener("input", function (event) {
        state.query = str(event.target.value).toLowerCase();
        renderRunList();
      });
    }
  }

  function matches(run) {
    if (state.status !== "all" && (str(run.status) || "unknown") !== state.status) {
      return false;
    }
    if (!state.query) {
      return true;
    }
    var haystack = (str(run.run_id) + " " + str(run.ticket_id)).toLowerCase();
    return haystack.indexOf(state.query) !== -1;
  }

  function renderRunList() {
    var list = document.getElementById("run-list");
    var note = document.getElementById("list-note");
    if (!list) { return; }
    clear(list);
    var visible = RUNS.filter(matches);
    if (note) {
      setText(note, visible.length + " / " + RUNS.length + " 个 run");
    }
    if (!RUNS.length) {
      list.appendChild(make("p", "placeholder", "没有 trace"));
      return;
    }
    if (!visible.length) {
      list.appendChild(make("p", "placeholder", "没有匹配的 run"));
      return;
    }
    visible.forEach(function (run) {
      var item = make("button", "run-item");
      item.setAttribute("type", "button");
      item.setAttribute("aria-current", state.selected === run ? "true" : "false");
      var title = make("div", "run-title");
      title.appendChild(make("span", null, str(run.run_id) || "(无 run_id)"));
      title.appendChild(statusBadge(run.status));
      item.appendChild(title);
      var token = totals(run);
      item.appendChild(make(
        "div",
        "run-sub",
        "ticket " + (str(run.ticket_id) || "-") +
          " · " + (str(run.category) || "-") +
          " · " + formatMs(runDuration(run)) +
          " · " + (token.prompt + token.completion) + " tokens"
      ));
      item.addEventListener("click", function () {
        state.selected = run;
        renderRunList();
        renderDetail();
      });
      list.appendChild(item);
    });
  }

  function summaryBlock(label, value) {
    var block = make("div", "summary-block");
    block.appendChild(make("div", "summary-label", label));
    block.appendChild(make("pre", "summary-text", str(value) || "（空）"));
    return block;
  }

  function renderCalls(calls) {
    if (!calls.length) {
      return null;
    }
    var table = document.createElement("table");
    var thead = document.createElement("thead");
    var headRow = document.createElement("tr");
    ["purpose", "model", "prompt", "completion", "耗时", "重试", "usage"].forEach(function (label) {
      headRow.appendChild(make("th", null, label));
    });
    thead.appendChild(headRow);
    table.appendChild(thead);
    var tbody = document.createElement("tbody");
    calls.forEach(function (call) {
      var row = document.createElement("tr");
      row.appendChild(make("td", null, str(call.purpose) || "-"));
      row.appendChild(make("td", null, str(call.model) || "-"));
      row.appendChild(make("td", null, num(call.prompt_tokens)));
      row.appendChild(make("td", null, num(call.completion_tokens)));
      row.appendChild(make("td", null, formatMs(call.latency_ms)));
      row.appendChild(make("td", null, num(call.retries)));
      var usage = make("td", call.usage_missing ? "warn" : null, call.usage_missing ? "usage 缺失" : "ok");
      row.appendChild(usage);
      tbody.appendChild(row);
    });
    table.appendChild(tbody);
    return table;
  }

  function renderSpan(span, totalDuration) {
    var cardNode = make("section", "span-card" + (span.error ? " has-error" : ""));
    var head = make("button", "span-head");
    head.setAttribute("type", "button");
    head.setAttribute("aria-expanded", "true");
    head.appendChild(make("span", "span-name", str(span.name) || "(未命名节点)"));
    var duration = num(span.duration_ms);
    var pct = totalDuration > 0 ? (duration / totalDuration) * 100 : 0;
    pct = Math.max(0, Math.min(100, pct));
    var bar = make("span", "bar");
    var fill = make("span", "bar-fill");
    fill.setAttribute("style", "width: " + pct.toFixed(2) + "%");
    bar.appendChild(fill);
    head.appendChild(bar);
    head.appendChild(make("span", "span-duration", formatMs(duration)));
    head.appendChild(make("span", "span-duration", "重试 " + num(span.retries)));
    cardNode.appendChild(head);

    var body = make("div", "span-body");
    body.appendChild(summaryBlock("输入摘要", span.input_summary));
    body.appendChild(summaryBlock("输出摘要", span.output_summary));
    if (span.error) {
      body.appendChild(make("div", "error-text", "错误：" + str(span.error)));
    }
    var table = renderCalls(Array.isArray(span.llm_calls) ? span.llm_calls : []);
    if (table) {
      body.appendChild(table);
    }
    cardNode.appendChild(body);

    head.addEventListener("click", function () {
      var expanded = head.getAttribute("aria-expanded") === "true";
      head.setAttribute("aria-expanded", expanded ? "false" : "true");
      if (expanded) {
        body.setAttribute("style", "display: none");
      } else {
        body.removeAttribute("style");
      }
    });
    return cardNode;
  }

  function renderDetail() {
    var detail = document.getElementById("detail");
    if (!detail) { return; }
    clear(detail);
    if (!state.selected) {
      detail.appendChild(make("p", "placeholder", RUNS.length ? "选择左侧的 run 查看节点时间线" : "没有 trace"));
      return;
    }
    var run = state.selected;
    var head = make("div", "run-head");
    head.appendChild(make("div", "run-id", str(run.run_id) || "(无 run_id)"));
    head.appendChild(statusBadge(run.status));
    head.appendChild(make("div", "meta", "ticket " + (str(run.ticket_id) || "-")));
    head.appendChild(make("div", "meta", "category " + (str(run.category) || "-")));
    head.appendChild(make("div", "meta", "总耗时 " + formatMs(runDuration(run))));
    var token = totals(run);
    head.appendChild(make("div", "meta", "tokens " + (token.prompt + token.completion)));
    detail.appendChild(head);

    var spans = spansOf(run);
    if (!spans.length) {
      detail.appendChild(make("p", "placeholder", "这个 run 没有节点"));
      return;
    }
    var total = runDuration(run);
    spans.forEach(function (span) {
      detail.appendChild(renderSpan(span, total));
    });
  }

  renderSummary();
  buildFilters();
  renderRunList();
  renderDetail();
})();
</script>
</body>
</html>
"""


def render_trace_viewer(traces: Sequence[RunTrace]) -> str:
    data = [trace.to_dict() for trace in traces]
    return _HTML.replace("__TRACE_DATA__", _json_for_script(data))
