"""端到端抓取-排序-自检 pipeline。供 CLI (main.py) 与 GUI (app.py) 共享调用。

返回结构化结果而不是打印，便于 UI 渲染。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from collectors.rss import fetch_all
from collectors.url_decoder import decode_items_inplace
from core.db import DB
from core.normalizer import normalize
from core.ranker import rank_all, translate_all, verify_all
from core.summary import clean_rss_summary


LAST_RUN_KEY = "last_run_at"


def _noop(*_, **__):
    pass


def compute_since_from_db(
    db: DB,
    window_hours: float | None,
    use_all: bool,
) -> datetime | None:
    """与 main.py:compute_since 同一套规则，避免行为漂移。"""
    if use_all:
        return None
    now = datetime.now(timezone.utc)
    if window_hours is not None:
        return now - timedelta(hours=window_hours)
    last = db.get_meta(LAST_RUN_KEY)
    if last:
        try:
            return datetime.fromisoformat(last)
        except ValueError:
            pass
    return now - timedelta(hours=24)


def collect_news(
    keywords: list[dict],
    sources: list[dict],
    llm_cfg: dict | None,
    db_path: Path,
    window_hours: float | None = None,
    use_all: bool = False,
    use_llm: bool = True,
    update_last_run: bool = True,
    log: Callable[[str], None] = _noop,
) -> dict:
    """执行完整流程，返回:

      {
        "items": [...],              # 命中且时窗内的全部条目
        "grouped": {kw: [picks...]}, # LLM 选中的 top-K（每条含 summary/url/...）
        "stats": {
          "raw": int,                # 原始抓取条数
          "matched": int,            # 命中关键词且在时窗内
          "rss_used": int,
          "llm_used": int,
          "verify_fixed": int,
        },
        "window_from": str | None,
        "window_to": str,
      }
    """
    db = DB(db_path)
    since = compute_since_from_db(db, window_hours, use_all)
    now = datetime.now(timezone.utc)

    if since is None:
        log("时间窗: 不限 (--all)")
    else:
        hours = (now - since).total_seconds() / 3600
        log(f"时间窗: {since.isoformat()[:16]} ~ {now.isoformat()[:16]}  ({hours:.1f}h)")

    log(f"[1/5] 抓取 {len(sources)} 个源…")
    raw = list(fetch_all(sources, keywords))
    log(f"  抓取完毕：{len(raw)} 条原始数据")

    log("[2/5] 去重 + 关键词匹配 + 客观打分（SQLite delta）…")
    items = normalize(raw, keywords, db=db, since=since)
    log(f"  命中且在窗口内：{len(items)} 条")

    stats = {"raw": len(raw), "matched": len(items),
             "rss_used": 0, "llm_used": 0, "verify_fixed": 0, "translated": 0}
    grouped: dict[str, list[dict]] = {}

    if use_llm and items and llm_cfg is not None:
        log("[3/5] LLM 按重要性筛选 (top_k 每关键词)…")
        log(f"  模型: {llm_cfg['model']}  base_url: {llm_cfg.get('base_url') or 'OpenAI default'}")
        grouped_raw = rank_all(items, keywords, llm_cfg)

        log("[4/5] 摘要选择 (RSS 优先) + URL 解码…")
        for picks in grouped_raw.values():
            for p in picks:
                p["raw_rss_summary"] = p.get("summary", "")
                rss = clean_rss_summary(p["raw_rss_summary"])
                if rss:
                    p["summary"] = rss
                    p["summary_source"] = "rss"
                    stats["rss_used"] += 1
                else:
                    p["summary"] = p.get("llm_summary", "")
                    p["summary_source"] = "llm"
                    stats["llm_used"] += 1
            decode_items_inplace(picks)

        if stats["llm_used"]:
            log("[5/5] 二次自检 (仅 LLM 生成的摘要)…")
            verify_all(grouped_raw, llm_cfg)
            stats["verify_fixed"] = sum(
                1 for picks in grouped_raw.values() for p in picks if p.get("verified")
            )

        # RSS 原文摘要翻译（仅非中文）
        if stats["rss_used"]:
            log("[5/5] 翻译非中文的 RSS 摘要…")
            translate_all(grouped_raw, llm_cfg)
            stats["translated"] = sum(
                1 for picks in grouped_raw.values() for p in picks if p.get("translated")
            )
        else:
            stats["translated"] = 0
        grouped = grouped_raw

    if update_last_run and not use_all and window_hours is None:
        db.set_meta(LAST_RUN_KEY, now.isoformat())
    db.close()

    return {
        "items": items,
        "grouped": grouped,
        "stats": stats,
        "window_from": since.isoformat() if since else None,
        "window_to": now.isoformat(),
    }
