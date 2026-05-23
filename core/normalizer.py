"""去重 / 时窗过滤 / 关键词匹配 / 打分。"""
from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone

from core.timeutil import parse_dt


# ---------- 关键词匹配 ----------

_CJK_RE = re.compile(r"[一-鿿]")
_BOUNDARY = r"[A-Za-z0-9]"


def match_keywords(item: dict, keywords: list[dict]) -> list[str]:
    """在 title+summary 上匹配关键词及 aliases。

    - case_sensitive 默认 False；公司/产品名建议在 yaml 里开 true 避免误伤
    - match_aliases_only=true：只用 aliases 做匹配（避免 "Ray" / "checkpoint" 这类
      通用词裸命中无关内容），keyword name 只作为显示用
    - 英文走单词边界正则；中文直接 substring
    """
    hay = f"{item.get('title','')} {item.get('summary','')}"
    hits: list[str] = []
    for kw in keywords:
        if kw.get("match_aliases_only", False):
            terms = list(kw.get("aliases") or [])
        else:
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


# ---------- 质量过滤（剔除社区讨论 / 求助 / 闲聊，避免污染候选池） ----------

# 站点黑名单：低质量股票分析内容农场 / 聚合新闻列表 / 垃圾页面
# 这些站点对技术规划者没有价值，往往是 SEO 模板文，先在 normalize 阶段彻底拦截
_DOMAIN_BLACKLIST = (
    "simplywall.st",       # 股票估值模板文，每条都是 "A Look at X Valuation"
    "moomoo.com",          # 散户日聚合，多事件混编
    "fool.com",            # The Motley Fool 股评
    "seekingalpha.com",    # 股评，付费墙
    "benzinga.com",        # 股评
    "zacks.com",           # 股评
    "gurufocus.com",
    "investorplace.com",
    "tipranks.com",
    "247wallst.com",
    "wallstreetzen.com",
    # 注意：msn.com / marketwatch.com 不在黑名单 —— 它们偶尔会聚合真实的公司公告
)


def is_blacklisted_url(url: str) -> bool:
    """单独暴露：post-decode 阶段（Google News 解码后才能拿到真实域名）再过一遍。"""
    if not url:
        return False
    url_lc = url.lower()
    return any(bad in url_lc for bad in _DOMAIN_BLACKLIST)

# 标题模板特征：股价分析 / 估值类的纯模板标题（即使域名不在黑名单也拦）
_TEMPLATE_TITLE_PATTERNS = (
    "valuation check", "valuation after", "price target",
    "stock price", "share price", "stock split",
    "should you buy", "should you sell", "is it time to buy",
    "buy or sell", "stock to buy", "stocks to buy",
    "still has room to run", "has more room to run",
    "stock forecast", "stock prediction", "stock soars", "stock surges",
    "stock jumps", "stock plunges", "stock tumbles", "stock rallies",
    "stock could", "stock might", "stock will",
    "wall street analysts", "analyst raises", "analyst lowers",
    "price target raised", "price target lowered",
    "buy the dip", "sell the rip", "trade idea",
    "估值分析", "股价分析", "目标价", "评级",
    "股价飙升", "股价大涨", "股价暴涨", "股价跳水",
)

# 标题以这些前缀开头的，多半是个人闲聊或求助，不是新闻
_LOW_QUALITY_TITLE_PREFIXES = (
    "ask hn", "ask reddit", "ask:", "tell hn", "discussion:",
    "rant:", "meta:", "weekly thread", "weekend thread",
    "what is", "what are", "what's the", "whats the",
    "why is", "why does", "why do", "why are",
    "how do i", "how do you", "how can i", "how to",
    "anyone else", "anyone using", "anyone tried",
    "is there a", "is there any", "is anyone",
    "请问", "求助", "讨论：", "闲聊", "求推荐", "求介绍",
    "有人用过", "有没有人",
)

# 个人 / 营销腔的标题片段
_SPAM_TITLE_HINTS = (
    "i built", "i made", "i created", "look what i", "my first",
    "you won't believe", "you wont believe", "must read", "must-read",
)

# 来源是社区聚合源（这些源的疑问句标题极可能是个人发帖）
_COMMUNITY_SOURCE_PREFIXES = (
    "Reddit", "HN Search", "HackerNews", "Hacker News",
)


def _is_community_source(item: dict) -> bool:
    src = (item.get("source") or "")
    return any(src.startswith(p) for p in _COMMUNITY_SOURCE_PREFIXES)


def _url_host(item: dict) -> str:
    url = (item.get("url") or "").lower()
    m = re.match(r"^https?://([^/]+)/?", url)
    return m.group(1) if m else ""


def is_low_quality(item: dict) -> tuple[bool, str]:
    """判断一条是否应被丢弃在进 LLM 之前。返回 (是否丢弃, 原因)。"""
    title = (item.get("title") or "").strip()
    if not title:
        return True, "empty title"
    title_lc = title.lower()
    host = _url_host(item)
    # 0) 域名黑名单：内容农场 / 股评聚合站
    for bad in _DOMAIN_BLACKLIST:
        if bad in host:
            return True, f"blacklist-domain:{bad}"
    # 1) 标题模板特征：股价/估值模板文
    if any(p in title_lc for p in _TEMPLATE_TITLE_PATTERNS):
        return True, "stock-template"
    # 2) 来自社区源的疑问句标题：私人发帖/求助
    if _is_community_source(item) and title.rstrip().endswith("?"):
        return True, "community-question"
    # 3) 个人闲聊 / 求助前缀
    for p in _LOW_QUALITY_TITLE_PREFIXES:
        if title_lc.startswith(p):
            return True, f"prefix:{p}"
    # 4) 营销腔
    if any(h in title_lc for h in _SPAM_TITLE_HINTS):
        return True, "spammy"
    # 5) 标题过短且无信息（< 12 字符且全英文）—— Reddit 上常见的 1-2 词帖
    if _is_community_source(item) and len(title) < 12 and not _CJK_RE.search(title):
        return True, "too-short"
    return False, ""


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
    return parse_dt(item.get("published"))


def within_24h(item: dict) -> bool:
    pub = _parse_pub(item)
    if pub is None:
        return False  # 无可用日期 → 不算在窗口内
    return datetime.now(timezone.utc) - pub <= timedelta(hours=24)


def after(item: dict, since: datetime) -> bool:
    """item 是否在 since 时间点之后。

    无可用日期 → 丢弃（精确时间窗）：上游已不再伪造时间戳，published 为空即"发布
    时间未知"，限定时间窗的运行里宁缺勿滥，避免无法证明在窗口内的旧闻漏进来。
    """
    pub = _parse_pub(item)
    if pub is None:
        return False
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
    dropped_quality = 0
    for it in items:
        hits = match_keywords(it, keywords)
        if not hits:
            continue
        low, _reason = is_low_quality(it)
        if low:
            dropped_quality += 1
            continue
        it["matched_keywords"] = hits
        matched.append(it)
    if dropped_quality:
        print(f"  [质量过滤] 丢弃 {dropped_quality} 条社区闲聊/疑问/营销条目")

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
