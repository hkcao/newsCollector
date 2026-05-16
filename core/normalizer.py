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
    综合打分（值域大致 0~2.0），聚合所有源能给出的客观信号：

      0.25 * log10(points+1) + 0.08 * log10(comments+1)    # HN 绝对热度
    + 0.25 * log10(delta_24h+1)                            # HN 24h 增量
    + 0.20 * log2(cross_source_count)                      # 多源同时报道（最强 cross-check）
    + 0.15 * tavily_score                                  # Tavily 自带相关性分（0~1）
    + 0.07 * recency_boost                                 # 越新越靠前

    取 log 是为了把 HN points / cross_source 这类离散数量压到可比尺度。
    cross_source_count 用 log2（增长快）：被两个源同时报道权重明显高于单源，
    被 5 个源覆盖的事件权重接近 HN 1000 分。

    Tavily 没有 HN 那种绝对热度，但 Tavily 自己有一个 0~1 的相关性分（topic=news 的"是否值得当新闻看"
    的判断），作为兜底信号，避免 Tavily-only 条目客观分永远为 0。
    """
    sig = item.get("signals", {})
    points = sig.get("points", 0)
    comments = sig.get("comments", 0)
    delta = sig.get("delta_24h", 0)
    cross = sig.get("cross_source_count", 1)
    tavily = sig.get("tavily_score") or 0.0

    abs_term = 0.25 * math.log10(points + 1) + 0.08 * math.log10(comments + 1)
    delta_term = 0.25 * math.log10(delta + 1)
    cross_term = 0.20 * math.log2(max(cross, 1) + 1)  # +1 防止 1 → log2(1)=0
    tavily_term = 0.15 * float(tavily)
    rec_term = 0.07 * recency_boost(item)
    return abs_term + delta_term + cross_term + tavily_term + rec_term


def _normalize_title(title: str) -> str:
    """跨源同事件检测用：小写 + 去标点 + 折叠空白。粗糙但够用。"""
    t = (title or "").lower()
    t = re.sub(r"[\s\W_]+", " ", t, flags=re.UNICODE)
    return t.strip()


def compute_cross_source_count(items: list[dict]) -> None:
    """原地给每个 item 的 signals 写入 cross_source_count = 同标题在多少个不同源出现。

    用 normalized title 作为聚合 key。同一篇报道在 HN / Reddit / Google News / Tavily
    多重出现时，标题大概率一致或近似，能被聚合到一起。
    """
    title_sources: dict[str, set[str]] = {}
    for it in items:
        key = _normalize_title(it.get("title", ""))
        if not key:
            continue
        src = (it.get("source") or "").split("(")[0].strip()
        title_sources.setdefault(key, set()).add(src)
    for it in items:
        key = _normalize_title(it.get("title", ""))
        sig = dict(it.get("signals", {}))
        sig["cross_source_count"] = len(title_sources.get(key, {"."}))
        it["signals"] = sig


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

    # 跨源聚合：在所有命中条目里数一遍同标题出现在多少个源（包括 HN / Reddit / Google
    # News / Tavily），写进 cross_source_count 后再打分
    compute_cross_source_count(matched)

    for it in matched:
        it["score"] = compute_score(it)
    return matched
