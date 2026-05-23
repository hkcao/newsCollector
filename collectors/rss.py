"""RSS collector —— 覆盖 HN / Reddit / arXiv / Google News 等所有 RSS 源。

并发：所有源（含 keyword_search 的每个关键词查询、Tavily 的每个关键词查询）
作为独立 task 丢进线程池并行拉取。feedparser 是阻塞 IO，线程池足够，无需 asyncio。
"""
from __future__ import annotations

import hashlib
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable
from urllib.parse import quote_plus, urlparse

import feedparser
import httpx

from core.timeutil import parse_dt


HN_POINTS_RE = re.compile(r"Points:\s*(\d+)")
HN_COMMENTS_RE = re.compile(r"Comments:\s*(\d+)")


# ---------- HTTP 抓取层（替代 feedparser 内置的裸 urllib） ----------
# 裸 urllib 没超时/没重试/没 UA/不限每主机并发，对 news.google.com、hnrss.org 这类
# 单 host 并发几十次的源会大面积 SSL EOF / 502。改走 httpx：浏览器 UA + 超时 +
# 跟随跳转（修好一批 301/302 迁移的 feed）+ 对瞬时错误指数退避重试 + 每主机限流。
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
_HEADERS = {
    "User-Agent": _UA,
    "Accept": ("application/rss+xml, application/atom+xml, application/xml, "
               "text/xml, text/html;q=0.9, */*;q=0.8"),
}
_HTTP_TIMEOUT = float(os.getenv("NEWS_HTTP_TIMEOUT", "20"))
_HTTP_RETRIES = int(os.getenv("NEWS_HTTP_RETRIES", "3"))
_PER_HOST_LIMIT = int(os.getenv("NEWS_PER_HOST_LIMIT", "4"))

_HOST_SEMAPHORES: dict[str, threading.Semaphore] = {}
_HOST_LOCK = threading.Lock()


def _host_semaphore(url: str) -> threading.Semaphore:
    """同一 host 最多 _PER_HOST_LIMIT 个并发请求，避免把单站打到限流。"""
    host = urlparse(url).netloc
    with _HOST_LOCK:
        sem = _HOST_SEMAPHORES.get(host)
        if sem is None:
            sem = threading.Semaphore(_PER_HOST_LIMIT)
            _HOST_SEMAPHORES[host] = sem
        return sem


def http_get_bytes(url: str) -> bytes:
    """抓 URL 原始字节供 feedparser 解析。

    - 4xx（除 429）视为永久失败，立即抛出不重试（源已死，归类别 B）。
    - 429 / 5xx / 网络错误 / 超时视为瞬时，指数退避重试。
    退避在释放 host 信号量之后进行，避免占着名额睡觉。
    """
    sem = _host_semaphore(url)
    last = ""
    for attempt in range(_HTTP_RETRIES):
        try:
            with sem:
                r = httpx.get(url, headers=_HEADERS, timeout=_HTTP_TIMEOUT,
                              follow_redirects=True)
            if r.status_code < 400:
                return r.content
            if r.status_code != 429 and r.status_code < 500:
                raise RuntimeError(f"HTTP {r.status_code}")  # 永久失败，不重试
            last = f"HTTP {r.status_code}"  # 429/5xx → 瞬时
        except RuntimeError:
            raise
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep((2 ** attempt) * 0.5 + random.random() * 0.3)
    raise RuntimeError(f"重试 {_HTTP_RETRIES} 次仍失败: {last}")


def _parse_published(entry) -> str:
    """返回 UTC ISO 字符串；拿不到可用日期返回 ""。

    绝不伪造 now() —— 否则无日期的旧闻会被当成"刚发布"，永远通过时间窗。
    优先用 feedparser 解析好的 *_parsed，失败再回退原始字符串（RFC822 等）。
    """
    for key in ("published_parsed", "updated_parsed",
                "published", "updated", "pubDate", "date"):
        dt = parse_dt(entry.get(key))
        if dt:
            return dt.isoformat()
    return ""


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
    feed = feedparser.parse(http_get_bytes(url))
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


def _build_tasks(sources: list[dict], keywords: list[dict]) -> list[dict]:
    """把所有源展开成扁平的 fetch task 列表（每个 task = 一次 HTTP 请求）。"""
    tasks: list[dict] = []
    for src in sources:
        if src.get("enabled", True) is False:
            print(f"  [SKIP] {src.get('name')} (enabled=false)")
            continue
        stype = src.get("type", "rss")
        cat = src.get("category", "")
        if stype == "rss":
            tasks.append({
                "type": "rss", "url": src["url"],
                "tag": src["name"], "category": cat, "label": f"[RSS] {src['name']}",
            })
        elif stype == "keyword_search":
            for kw in keywords:
                q = quote_plus(kw["name"])
                tasks.append({
                    "type": "rss",
                    "url": src["url"].format(q=q),
                    "tag": f"{src['name']}({kw['name']})",
                    "category": cat,
                    "label": f"[SEARCH] {src['name']}({kw['name']})",
                })
        elif stype == "tavily":
            from collectors.web_search import resolve_api_key
            env_name = src.get("api_key_env", "TAVILY_API_KEY")
            api_key = resolve_api_key(env_name)
            if not api_key:
                print(f"  [TAVILY] 跳过 {src['name']}：环境变量 {env_name} 未设置")
                continue
            for kw in keywords:
                tasks.append({
                    "type": "tavily",
                    "keyword": kw["name"],
                    "tag": f"{src['name']}({kw['name']})",
                    "category": cat,
                    "api_key": api_key,
                    "days": src.get("days", 2),
                    "max_results": src.get("max_results", 10),
                    "topic": src.get("topic", "news"),
                    "label": f"[TAVILY] {src['name']}({kw['name']})",
                })
        else:
            print(f"  [WARN] 未知 source type: {stype} ({src.get('name')})")
    return tasks


def _run_task(task: dict) -> tuple[dict, list[dict] | None, str | None]:
    """单个 task 的执行体，返回 (task, items, err)。err 非 None 表示失败。"""
    try:
        if task["type"] == "rss":
            items = fetch_rss(task["url"], task["tag"], task["category"])
            return task, items, None
        if task["type"] == "tavily":
            from collectors.web_search import fetch_tavily
            items = list(fetch_tavily(
                task["keyword"], task["tag"], task["category"],
                api_key=task["api_key"], days=task["days"],
                max_results=task["max_results"], topic=task["topic"],
            ))
            return task, items, None
    except Exception as e:
        return task, None, str(e)
    return task, [], None


def fetch_all(
    sources: list[dict],
    keywords: list[dict],
    max_workers: int | None = None,
) -> Iterable[dict]:
    """并发抓取所有源。线程池大小默认 16，可用环境变量 NEWS_FETCH_WORKERS 覆盖。
    保持 generator 接口不变；上游照旧 list(fetch_all(...))。"""
    if max_workers is None:
        max_workers = int(os.getenv("NEWS_FETCH_WORKERS", "16"))
    tasks = _build_tasks(sources, keywords)
    if not tasks:
        return
    print(f"  并发抓取 {len(tasks)} 个请求 (max_workers={max_workers})…")
    t0 = time.monotonic()
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_run_task, t) for t in tasks]
        for fut in as_completed(futures):
            task, items, err = fut.result()
            if err is not None:
                print(f"    ! {task['label']} 失败: {err}")
                fail += 1
                continue
            ok += 1
            if items:
                yield from items
    print(f"  完成 {ok}/{len(tasks)} 个源（失败 {fail}），耗时 {time.monotonic()-t0:.1f}s")
