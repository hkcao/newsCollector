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
from core.digest import build_digest
from core.normalizer import normalize
from core.ranker import rank_all, translate_all, verify_all, verify_titles
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
    # 默认固定 24h 窗口：不再以 last_run_at 收窄，避免高频运行时各分类凑不齐数据
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
    user_preference: str | None = None,
    themes: list[dict] | None = None,
    watchlist: list[str] | None = None,
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

    log("[2/5] 去重 + 关键词匹配 + 语义主题门 + 客观打分（SQLite delta）…")
    # themes + llm_cfg 都在、且本次用 LLM 时，启用语义主题门按内容召回未命中关键词的媒体条目
    items = normalize(raw, keywords, db=db, since=since,
                      themes=themes, llm_cfg=(llm_cfg if use_llm else None),
                      watchlist=watchlist)
    log(f"  命中且在窗口内：{len(items)} 条")

    stats = {"raw": len(raw), "matched": len(items),
             "rss_used": 0, "llm_used": 0, "verify_fixed": 0,
             "title_fixed": 0, "translated": 0}
    grouped: dict[str, list[dict]] = {}
    digest: dict = {"overview": "", "trends": [], "advice": []}

    if use_llm and items and llm_cfg is not None:
        log("[3/5] LLM 按重要性筛选 (每个分类独立排序)…")
        log(f"  模型: {llm_cfg['model']}  base_url: {llm_cfg.get('base_url') or 'OpenAI default'}")
        grouped_raw = rank_all(items, keywords, llm_cfg, user_preference=user_preference)

        log("[4/5] 摘要选择 (RSS 优先) + URL 解码…")
        for picks in grouped_raw.values():
            for p in picks:
                p["raw_rss_summary"] = p.get("summary", "")
                rss = clean_rss_summary(p["raw_rss_summary"])
                # 意见领袖发言用 LLM 中文转述（含"为何值得知道"），不回退原始英文推文
                if rss and p.get("llm_category") != "AI意见领袖":
                    p["summary"] = rss
                    p["summary_source"] = "rss"
                    stats["rss_used"] += 1
                else:
                    p["summary"] = p.get("llm_summary", "")
                    p["summary_source"] = "llm"
                    stats["llm_used"] += 1
            decode_items_inplace(picks)
        # 解码后二次过滤黑名单域名（Google News 此时才暴露真实域名）
        from core.normalizer import is_blacklisted_url
        dropped_bl = 0
        for cat, picks in list(grouped_raw.items()):
            kept = [p for p in picks if not is_blacklisted_url(p.get("url", ""))]
            dropped_bl += len(picks) - len(kept)
            grouped_raw[cat] = kept
        if dropped_bl:
            log(f"  [二次过滤] 解码后丢弃 {dropped_bl} 条命中黑名单域名的 picks")

        if stats["llm_used"]:
            log("[5/5] 二次自检 (仅 LLM 生成的摘要)…")
            verify_all(grouped_raw, llm_cfg)
            stats["verify_fixed"] = sum(
                1 for picks in grouped_raw.values() for p in picks if p.get("verified")
            )

        # 标题幻觉核查（独立 pass，针对所有 picks 不只是 LLM 摘要）
        log("[5/5] 标题反幻觉核查…")
        verify_titles(grouped_raw, llm_cfg, keywords=keywords)
        stats["title_fixed"] = sum(
            1 for picks in grouped_raw.values() for p in picks if p.get("title_fixed")
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

        log("[5/5] 综合层：今日概览 / 趋势 / 建议…")
        digest = build_digest(grouped_raw, llm_cfg)

    if update_last_run and not use_all and window_hours is None:
        db.set_meta(LAST_RUN_KEY, now.isoformat())
    db.close()

    return {
        "items": items,
        "grouped": grouped,
        "digest": digest,
        "stats": stats,
        "window_from": since.isoformat() if since else None,
        "window_to": now.isoformat(),
    }
