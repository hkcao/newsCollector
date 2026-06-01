"""调试追踪器 —— --debug 时记录流水线每一步筛选前/后的内容，落盘供分析定位。

全局单例 TRACE，仿 core.ranker.USAGE 模式。未 enable 时所有方法都是 no-op，
对正常运行零开销、零侵入。enable 后在各筛选/改写边界记录：
  - snapshot：某一步的全量候选快照（筛选前/后内容）
  - drop：被丢弃的条目 + 原因
  - change：被改写的字段（改写前/后）
  - note：自由文本 / 结构化数据（如 LLM 原始响应）
run 结束时 dump 成 debug-<时间>.json（完整）+ debug-<时间>.md（可读）。
"""
from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

# HTML 里每个步骤最多渲染多少条 item（避免上万条快照把页面拖卡）；完整数据看同名 JSON
_HTML_ROW_CAP = 300


def _snap(item: dict) -> dict:
    """把一条 item 投影成调试用紧凑字段（摘要截断，避免快照过大）。"""
    sig = item.get("signals", {}) or {}
    return {
        "id": item.get("id"),
        "title": item.get("title"),
        "display_title": item.get("display_title"),
        "source": item.get("source"),
        "category": item.get("category"),
        "llm_category": item.get("llm_category"),
        "url": item.get("url"),
        "published": item.get("published"),
        "score": round(float(item.get("score") or 0), 3),
        "matched_keywords": item.get("matched_keywords"),
        "signals": {k: v for k, v in sig.items() if v},
        "summary": (item.get("summary") or "")[:300],
        "watchlisted": item.get("watchlisted", False),
        "theme_gated": item.get("theme_gated", False),
    }


def _esc(x) -> str:
    return html.escape("" if x is None else str(x))


def _item_row(it: dict) -> str:
    """渲染一条 item 为 HTML 行：分数 / 来源 / 命中词 / 标题 / 摘要 / 丢弃原因。"""
    flags = ("⭐" if it.get("watchlisted") else "") + ("🔎" if it.get("theme_gated") else "")
    mk = it.get("matched_keywords") or []
    meta = f'<span class="score">{_esc(it.get("score"))}</span>'
    meta += f' <span class="src">{_esc(it.get("source"))}</span>'
    if mk:
        meta += f' <span class="kw">{_esc("／".join(mk))}</span>'
    if flags:
        meta += f' <span class="flag">{flags}</span>'
    title = _esc(it.get("display_title") or it.get("title"))
    url = it.get("url")
    if url:
        title = f'<a href="{_esc(url)}" target="_blank">{title}</a>'
    reason = (f'<span class="reason">← {_esc(it["reason"])}</span>'
              if it.get("reason") else "")
    summary = _esc(it.get("summary"))
    sm_html = f'<div class="summary">{summary}</div>' if summary else ""
    return (f'<div class="row"><div class="meta">{meta} {reason}</div>'
            f'<div class="title">{title}</div>{sm_html}</div>')


def _rows_html(items: list[dict]) -> str:
    shown = items[:_HTML_ROW_CAP]
    body = "".join(_item_row(it) for it in shown)
    if len(items) > _HTML_ROW_CAP:
        body += (f'<div class="more">… 还有 {len(items) - _HTML_ROW_CAP} 条未在此展示，'
                 f'完整数据见同名 JSON</div>')
    return body


_HTML_TEMPLATE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body{{font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;
       margin:0;background:#f5f6f8;color:#1d2530}}
  header{{padding:18px 24px;background:#1d2530;color:#fff}}
  header h1{{margin:0;font-size:18px}}
  header .sub{{opacity:.7;font-size:13px;margin-top:4px}}
  .wrap{{max-width:1100px;margin:0 auto;padding:16px}}
  .toc{{background:#fff;border:1px solid #e2e6ec;border-radius:8px;padding:10px 14px;margin-bottom:16px}}
  .toc ul{{margin:0;padding-left:0;list-style:none;columns:2;column-gap:28px}}
  .toc li{{margin:2px 0}}
  .toc a{{text-decoration:none;color:#2b6cb0}}
  .toc .cnt{{color:#94a3b8;font-size:12px}}
  .toc li.drop a{{color:#b04a4a}} .toc li.change a{{color:#7a5ec0}}
  details.step{{background:#fff;border:1px solid #e2e6ec;border-left:4px solid #cbd5e1;
       border-radius:8px;margin:10px 0;padding:4px 14px}}
  details.snapshot{{border-left-color:#3b82f6}}
  details.drop{{border-left-color:#e26d6d}}
  details.change{{border-left-color:#9b7ad6}}
  details.note{{border-left-color:#64748b}}
  summary{{cursor:pointer;font-weight:600;padding:8px 0;list-style:none}}
  summary .cnt{{font-weight:400;color:#64748b;font-size:13px}}
  .note{{color:#475569;font-size:13px;margin:2px 0 8px}}
  .row{{padding:6px 0;border-top:1px dashed #eef1f5}}
  .row .meta{{font-size:12px;color:#64748b}}
  .row .meta .score{{display:inline-block;min-width:46px;font-weight:600;color:#0f766e}}
  .row .meta .src{{color:#475569}}
  .row .meta .kw{{color:#a16207}}
  .row .meta .reason{{color:#b04a4a;font-weight:600}}
  .row .title{{margin:1px 0}} .row .title a{{color:#1d4ed8;text-decoration:none}}
  .row .summary{{font-size:12px;color:#64748b;margin-top:2px}}
  .more{{padding:8px 0;color:#94a3b8;font-style:italic}}
  .chg{{padding:8px 0;border-top:1px dashed #eef1f5}}
  .ba span{{display:inline-block;font-size:12px;color:#64748b;margin:4px 0 2px}}
  .before{{background:#fdf0f0;padding:6px 8px;border-radius:4px;white-space:pre-wrap}}
  .after{{background:#eefaf2;padding:6px 8px;border-radius:4px;white-space:pre-wrap}}
  pre.data{{background:#0f172a;color:#e2e8f0;padding:10px;border-radius:6px;overflow:auto;font-size:12px;white-space:pre-wrap}}
  code{{background:#eef1f5;padding:1px 4px;border-radius:3px}}
</style></head><body>
<header><h1>{title}</h1>
<div class="sub">共 {n} 个记录点，按执行顺序排列。📋 阶段快照　🗑 被丢弃　✏️ 被改写　📝 备注　|　⭐ 重点关注　🔎 主题门召回</div></header>
<div class="wrap">
<div class="toc"><ul>
{toc}
</ul></div>
{body}
</div></body></html>"""


class DebugTrace:
    def __init__(self) -> None:
        self.enabled = False
        self.out_dir: Path | None = None
        self.started_at: datetime | None = None
        self.steps: list[dict] = []

    def enable(self, out_dir) -> None:
        self.enabled = True
        self.out_dir = Path(out_dir)
        self.started_at = datetime.now()
        self.steps = []

    # ---------- 记录原语（未 enable 时一律 no-op）----------

    def snapshot(self, stage: str, items: list[dict], note: str = "") -> None:
        if not self.enabled:
            return
        self.steps.append({
            "type": "snapshot", "stage": stage, "note": note,
            "count": len(items), "items": [_snap(it) for it in items],
        })

    def drops(self, stage: str, dropped, note: str = "") -> None:
        """dropped: list[(item, reason)]。"""
        if not self.enabled:
            return
        recs = []
        for it, reason in dropped:
            rec = _snap(it)
            rec["reason"] = reason
            recs.append(rec)
        self.steps.append({
            "type": "drop", "stage": stage, "note": note,
            "count": len(recs), "items": recs,
        })

    def changes(self, stage: str, changes: list[dict], note: str = "") -> None:
        """changes: list[{title, field, reason, before, after}]。"""
        if not self.enabled or not changes:
            return
        self.steps.append({
            "type": "change", "stage": stage, "note": note,
            "count": len(changes), "items": changes,
        })

    def note(self, stage: str, text: str, data=None) -> None:
        if not self.enabled:
            return
        self.steps.append({"type": "note", "stage": stage, "note": text, "data": data})

    # ---------- 落盘 ----------

    def dump(self) -> Path | None:
        if not self.enabled:
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        ts = self.started_at.strftime("%Y-%m-%d-%H%M%S")
        json_path = self.out_dir / f"debug-{ts}.json"
        html_path = self.out_dir / f"debug-{ts}.html"

        json_path.write_text(
            json.dumps({"started_at": self.started_at.isoformat(), "steps": self.steps},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        html_path.write_text(self._render_html(), encoding="utf-8")
        return html_path

    # ---------- HTML 渲染 ----------

    _ICON = {"snapshot": "📋", "drop": "🗑", "change": "✏️", "note": "📝"}

    def _render_html(self) -> str:
        toc, body = [], []
        for i, s in enumerate(self.steps, 1):
            t, stage, note = s["type"], s["stage"], s.get("note", "")
            icon = self._ICON.get(t, "•")
            count = s.get("count")
            cnt_txt = (f"{count} 条" if t == "snapshot"
                       else f"丢弃 {count} 条" if t == "drop"
                       else f"改写 {count} 条" if t == "change" else "")
            head = f'{i}. {icon} {_esc(stage)}'
            head_full = head + (f'　<span class="cnt">{cnt_txt}</span>' if cnt_txt else "")
            toc.append(f'<li class="{t}"><a href="#s{i}">{head}</a> '
                       f'<span class="cnt">{cnt_txt}</span></li>')

            inner = self._step_inner(s)
            # 大快照默认折叠；小步骤直接展开
            open_attr = "" if (count or 0) > 20 else " open"
            note_html = f'<div class="note">{_esc(note)}</div>' if note else ""
            body.append(
                f'<details id="s{i}" class="step {t}"{open_attr}>'
                f'<summary>{head_full}</summary>{note_html}{inner}</details>'
            )

        return _HTML_TEMPLATE.format(
            title=f"newsCollector 调试追踪 {self.started_at.strftime('%Y-%m-%d %H:%M:%S')}",
            n=len(self.steps),
            toc="\n".join(toc),
            body="\n".join(body),
        )

    def _step_inner(self, s: dict) -> str:
        t = s["type"]
        if t in ("snapshot", "drop"):
            return _rows_html(s["items"])
        if t == "change":
            out = []
            for c in s["items"]:
                head = (f'<b>{_esc(c.get("title"))}</b> · 字段 <code>{_esc(c.get("field"))}</code>'
                        + (f'（原因：{_esc(c["reason"])}）' if c.get("reason") else ""))
                out.append(
                    f'<div class="chg"><div class="chg-head">{head}</div>'
                    f'<div class="ba"><span>改写前</span><div class="before">{_esc(c.get("before"))}</div></div>'
                    f'<div class="ba"><span>改写后</span><div class="after">{_esc(c.get("after"))}</div></div>'
                    f'</div>'
                )
            return "".join(out)
        if t == "note":
            data = s.get("data")
            if data is None:
                return ""
            txt = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, indent=2)
            return f'<pre class="data">{_esc(txt)}</pre>'
        return ""


# 全局单例 —— 每次 run_once 开始时由 main.py 在 --debug 下 enable()（同时清零）
TRACE = DebugTrace()
