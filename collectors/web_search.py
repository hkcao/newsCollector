"""Tavily Search API collector —— 覆盖近似浏览器搜索结果，弥补 Google News RSS 的盲区。

文档: https://docs.tavily.com/docs/rest-api/api-reference
免费档约 1000 次/月，按关键词数 × 每日触发次数计费。
"""
from __future__ import annotations

import hashlib
import os
import random
import threading
import time

import httpx

from core.timeutil import parse_dt


TAVILY_URL = "https://api.tavily.com/search"

# 免费档限流严格：限制并发 + 对 429 退避重试。432（月配额耗尽）无法靠重试解决，直接放弃。
_TAVILY_RETRIES = int(os.getenv("TAVILY_RETRIES", "3"))
_TAVILY_SEM = threading.Semaphore(int(os.getenv("TAVILY_CONCURRENCY", "2")))


def _make_id(url: str, title: str) -> str:
    return hashlib.sha1(f"{url}|{title}".encode()).hexdigest()[:16]


def _normalize_published(raw: str | None) -> str:
    """Tavily 的 published_date 形如 "2024-05-12T08:30:00Z" 或 "2024-05-12"。
    拿不到 / 解析失败返回 ""（不伪造 now，下游按"未知"处理）。"""
    dt = parse_dt(raw)
    return dt.isoformat() if dt else ""


def fetch_tavily(
    query: str,
    source_name: str,
    category: str,
    api_key: str,
    days: int = 2,
    max_results: int = 10,
    topic: str = "news",
) -> list[dict]:
    """对单个关键词查一次 Tavily。返回与 RSS collector 同 schema 的 item 列表。"""
    if not api_key:
        return []
    payload = {
        "query": query,
        "topic": topic,
        "days": days,
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    data = None
    last = ""
    for attempt in range(_TAVILY_RETRIES):
        try:
            with _TAVILY_SEM:
                r = httpx.post(TAVILY_URL, json=payload, headers=headers, timeout=30)
            if r.status_code == 200:
                data = r.json()
                break
            # 429=限流可重试；其他（含 432 配额耗尽）放弃
            if r.status_code != 429:
                print(f"    ! Tavily HTTP {r.status_code}: {r.text[:160]}")
                return []
            last = "HTTP 429"
        except Exception as e:
            last = f"{e}"
        time.sleep((2 ** attempt) * 1.0 + random.random() * 0.5)
    if data is None:
        print(f"    ! Tavily 重试 {_TAVILY_RETRIES} 次仍失败: {last}")
        return []

    items: list[dict] = []
    for res in data.get("results", []):
        url = (res.get("url") or "").strip()
        title = (res.get("title") or "").strip()
        if not url or not title:
            continue
        items.append({
            "id": _make_id(url, title),
            "source": source_name,
            "category": category,
            "title": title,
            "url": url,
            "summary": res.get("content") or "",
            "published": _normalize_published(res.get("published_date")),
            "signals": {"tavily_score": res.get("score")} if res.get("score") is not None else {},
            "matched_keywords": [],
        })
    return items


def resolve_api_key(env_name: str = "TAVILY_API_KEY") -> str:
    return os.environ.get(env_name, "").strip()
