"""HTML 报告渲染 —— jinja2 模板 → 按日期落盘到 reports/。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from core.timeutil import fmt_local

# 与 core.ranker.CATEGORIES 保持一致的展示顺序
_CATEGORY_ORDER = [
    "存储/基建产品",
    "模型/框架/算法",
    "基准评测",
    "学术论文",
    "政策导向",
    "github趋势",
]


def _group_by_category(
    grouped: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    """把按关键词分组的结果重排成按 llm_category 分组，并给每条带上 matched_kw。"""
    bucket: dict[str, list[dict]] = {c: [] for c in _CATEGORY_ORDER}
    for kw, picks in grouped.items():
        for p in picks:
            item = dict(p)
            item["matched_kw"] = kw
            cat = p.get("llm_category") or "存储/基建产品"
            bucket.setdefault(cat, []).append(item)
    # 丢掉空类别，未知类别附在最后
    out: dict[str, list[dict]] = {}
    for c in _CATEGORY_ORDER:
        if bucket.get(c):
            out[c] = bucket[c]
    for c, items in bucket.items():
        if c not in _CATEGORY_ORDER and items:
            out[c] = items
    return out


def render_html(
    grouped: dict[str, list[dict]],
    window_from: str,
    window_to: str,
) -> str:
    """返回渲染后的 HTML 字符串。

    grouped: 按关键词分组的 dict；模板内按 llm_category 重新聚合呈现。
    每条 item 需要含: title, url, source, published, summary, llm_category
    """
    tpl_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(tpl_dir)),
        autoescape=select_autoescape(["html"]),
    )
    env.filters["localtime"] = fmt_local
    tpl = env.get_template("daily.html")
    total = sum(len(v) for v in grouped.values())
    by_category = _group_by_category(grouped)
    now = datetime.now()
    return tpl.render(
        grouped=grouped,
        by_category=by_category,
        total=total,
        window_from=window_from,
        window_to=window_to,
        date=now.strftime("%Y-%m-%d"),
        generated_at=now.strftime("%Y-%m-%d %H:%M:%S"),
    )


def write_report(html: str, reports_dir: Path) -> Path:
    """落盘到 reports/YYYY-MM-DD-HHMM.html，返回写入路径。"""
    reports_dir.mkdir(parents=True, exist_ok=True)
    fname = datetime.now().strftime("%Y-%m-%d-%H%M") + ".html"
    out = reports_dir / fname
    out.write_text(html, encoding="utf-8")
    return out
