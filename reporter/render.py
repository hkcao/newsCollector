"""HTML 报告渲染 —— jinja2 模板 → 按日期落盘到 reports/。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape


def render_html(
    grouped: dict[str, list[dict]],
    window_from: str,
    window_to: str,
) -> str:
    """返回渲染后的 HTML 字符串。

    grouped 中每条 item 需要含: title, url, source, published, summary
    """
    tpl_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(tpl_dir)),
        autoescape=select_autoescape(["html"]),
    )
    tpl = env.get_template("daily.html")
    total = sum(len(v) for v in grouped.values())
    now = datetime.now()
    return tpl.render(
        grouped=grouped,
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
