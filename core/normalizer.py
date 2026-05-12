"""去重 / 时窗过滤 / 关键词匹配 / 打分。"""
from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone


# ---------- 关键词匹配 ----------

_CJK_RE = re.compile(r"[一-鿿]")
_BOUNDARY = r"[A-Za-z0-9]"


def match_keywords(item: dict, keywords: list[dict]) -> list[str]:
    """在 title+summary 上匹配关键词及 aliases。

    - case_sensitive 默认 False；公司/产品名建议在 yaml 里开 true 避免误伤
    - 英文走单词边界正则；中文直接 substring
    """
    hay = f"{item.get('title','')} {item.get('summary','')}"
    hits: list[str] = []
    for kw in keywords:
        terms = [kw["name"]] + list(kw.get("aliases") or [])
        flags = 0 if kw.get("case_sensitive", False) else re.IGNORECASE
        for t in terms:
            if not t:
                continue
            if _CJK_RE.search(t):
                # 中文：直接子串
                if flags & re.IGNORECASE:
                    if t.lower() in hay.lower():
                        hits.append(kw["name"]); break
                else:
                    if t in hay:
                        hits.append(kw["name"]); break
            else:
                pat = rf"(?<!{_BOUNDARY}){re.escape(t)}(?!{_BOUNDARY})"
                if re.search(pat, hay, flags=flags):
                    hits.append(kw["name"]); break
    return hits


# ---------- 去重 / 时窗 ----------

def dedupe(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for it in items:
        if it["id"] in seen:
            continue
        seen.add(it["id"])
        out.append(it)
    return out


def _parse_pub(item: dict) -> datetime | None:
    try:
        return datetime.fromisoformat(item["published"].replace("Z", "+00:00"))
    except Exception:
        return None


def within_24h(item: dict) -> bool:
    pub = _parse_pub(item)
    if pub is None:
        return True  # 解析失败保留，宁滥勿缺
    return datetime.now(timezone.utc) - pub <= timedelta(hours=24)


def after(item: dict, since: datetime) -> bool:
    """item 是否在 since 时间点之后；published 解析失败时保留。"""
    pub = _parse_pub(item)
    if pub is None:
        return True
    return pub >= since


# ---------- 打分 ----------

def recency_boost(item: dict) -> float:
    """0~1 之间。距今越近越接近 1，>24h 返回 0。"""
    pub = _parse_pub(item)
    if pub is None:
        return 0.0
    hours = (datetime.now(timezone.utc) - pub).total_seconds() / 3600
    if hours < 0 or hours > 24:
        return 0.0
    return 1.0 - (hours / 24)


def compute_score(item: dict) -> float:
    """
    综合打分（值域大致 0~1.5）：

      0.30 * log10(points+1) + 0.10 * log10(comments+1)   # 绝对热度
    + 0.30 * log10(delta_24h+1)                            # 24h 增量
    + 0.15 * log10(cross_source_count)                     # 多源同时报道 (v0.3)
    + 0.15 * recency_boost                                 # 越新越靠前

    取 log 是为了把 HN points 这种数量级（几到几千）压到可比尺度，
    既奖励 100k 级别的大热点，也不让一篇 5000 分把今天 200 分的盖掉太多。
    """
    sig = item.get("signals", {})
    points = sig.get("points", 0)
    comments = sig.get("comments", 0)
    delta = sig.get("delta_24h", 0)
    cross = sig.get("cross_source_count", 1)

    abs_term = 0.30 * math.log10(points + 1) + 0.10 * math.log10(comments + 1)
    delta_term = 0.30 * math.log10(delta + 1)
    cross_term = 0.15 * math.log10(max(cross, 1))
    rec_term = 0.15 * recency_boost(item)
    return abs_term + delta_term + cross_term + rec_term


# ---------- 主流程 ----------

def normalize(
    items: list[dict],
    keywords: list[dict],
    db=None,
    since: datetime | None = None,
) -> list[dict]:
    """去重 → 时窗（since=None 不过滤）→ 关键词匹配 → (可选) 从 db 算 delta → 打分。"""
    items = dedupe(items)
    if since is not None:
        items = [it for it in items if after(it, since)]

    matched: list[dict] = []
    for it in items:
        hits = match_keywords(it, keywords)
        if not hits:
            continue
        it["matched_keywords"] = hits
        matched.append(it)

    if db is not None:
        now_iso = datetime.now(timezone.utc).isoformat()
        for it in matched:
            baseline = db.baseline_signals(it["id"])
            sig = dict(it.get("signals", {}))
            cur_pts = sig.get("points", 0)
            base_pts = (baseline or {}).get("points", 0)
            sig["delta_24h"] = max(0, cur_pts - base_pts) if baseline else 0
            it["signals"] = sig
            db.upsert(it, now_iso)
        db.commit()

    for it in matched:
        it["score"] = compute_score(it)
    return matched
