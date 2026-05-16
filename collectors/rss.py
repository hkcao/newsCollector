"""RSS collector —— 覆盖 HN / Reddit / arXiv / Google News 等所有 RSS 源。"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Iterable
from urllib.parse import quote_plus

import feedparser


HN_POINTS_RE = re.compile(r"Points:\s*(\d+)")
HN_COMMENTS_RE = re.compile(r"Comments:\s*(\d+)")


def _parse_published(entry) -> str:
    for key in ("published_parsed", "updated_parsed"):
        t = entry.get(key)
        if t:
            return datetime(*t[:6], tzinfo=timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()


def _extract_signals(entry, source_name: str) -> dict:
    """从 RSS 条目里抽取客观信号 —— 不同源字段不同，做最小努力。"""
    signals: dict = {}
    summary = entry.get("summary", "") or ""

    if "hnrss" in source_name.lower() or "HN" in source_name:
        m = HN_POINTS_RE.search(summary)
        if m:
            signals["points"] = int(m.group(1))
        m = HN_COMMENTS_RE.search(summary)
        if m:
            signals["comments"] = int(m.group(1))
    return signals


def _make_id(url: str, title: str) -> str:
    return hashlib.sha1(f"{url}|{title}".encode()).hexdigest()[:16]


def fetch_rss(url: str, source_name: str, category: str) -> list[dict]:
    """同步抓单个 RSS URL，返回统一 schema 的 item 列表。"""
    feed = feedparser.parse(url)
    items = []
    for entry in feed.entries:
        title = entry.get("title", "").strip()
        link = entry.get("link", "").strip()
        if not title or not link:
            continue
        items.append({
            "id": _make_id(link, title),
            "source": source_name,
            "category": category,
            "title": title,
            "url": link,
            "summary": entry.get("summary", "").strip(),
            "published": _parse_published(entry),
            "signals": _extract_signals(entry, source_name),
            "matched_keywords": [],  # 由 normalizer 填
        })
    return items


def fetch_all(sources: list[dict], keywords: list[dict]) -> Iterable[dict]:
    """按 sources.yaml 配置遍历所有源；keyword_search / tavily 会对每个关键词逐个查询。"""
    for src in sources:
        if src.get("enabled", True) is False:
            print(f"  [SKIP] {src.get('name')} (enabled=false)")
            continue
        stype = src.get("type", "rss")
        if stype == "rss":
            print(f"  [RSS] {src['name']}")
            try:
                yield from fetch_rss(src["url"], src["name"], src.get("category", ""))
            except Exception as e:
                print(f"    ! 失败: {e}")
        elif stype == "keyword_search":
            for kw in keywords:
                q = quote_plus(kw["name"])
                url = src["url"].format(q=q)
                tag = f"{src['name']}({kw['name']})"
                print(f"  [SEARCH] {tag}")
                try:
                    yield from fetch_rss(url, tag, src.get("category", ""))
                except Exception as e:
                    print(f"    ! 失败: {e}")
        elif stype == "tavily":
            from collectors.web_search import fetch_tavily, resolve_api_key
            env_name = src.get("api_key_env", "TAVILY_API_KEY")
            api_key = resolve_api_key(env_name)
            if not api_key:
                print(f"  [TAVILY] 跳过 {src['name']}：环境变量 {env_name} 未设置")
                continue
            days = src.get("days", 2)
            max_results = src.get("max_results", 10)
            topic = src.get("topic", "news")
            for kw in keywords:
                tag = f"{src['name']}({kw['name']})"
                print(f"  [TAVILY] {tag}")
                try:
                    yield from fetch_tavily(
                        kw["name"], tag, src.get("category", ""),
                        api_key=api_key, days=days,
                        max_results=max_results, topic=topic,
                    )
                except Exception as e:
                    print(f"    ! 失败: {e}")
        else:
            print(f"  [WARN] 未知 source type: {stype} ({src.get('name')})")
