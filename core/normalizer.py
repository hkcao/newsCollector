"""去重 / 时窗过滤 / 关键词匹配 / 打分。"""
from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone

from core.timeutil import parse_dt
from core.debug_trace import TRACE


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

def recency_boost(item: dict, window_hours: float = 24.0) -> float:
    """0~1 之间：在采集周期内距今越近越接近 1，超出周期返回 0。

    window_hours = 本次采集周期（= now - since）。采集窗口本身 >24h 时（如周报、或久未运行
    导致 since 拉得很远），新近度按**同一时间范围**归一化，避免"窗口设 7 天、但凡 >24h 的
    条目 recency 一律为 0"——那会让窗口里靠后的几天完全拿不到新近度信号。
    """
    pub = _parse_pub(item)
    if pub is None:
        return 0.0
    span = max(window_hours, 1.0)
    hours = (datetime.now(timezone.utc) - pub).total_seconds() / 3600
    if hours < 0 or hours > span:
        return 0.0
    return 1.0 - (hours / span)


def source_tier(item: dict) -> float:
    """源质量分级 0~1：精选 RSS（官方/垂直/中文 AI 媒体）> GitHub 趋势 > 搜索兜底。

    现有客观分在 HN 禁用后几乎全平（只剩 recency+cross），导致候选池无有效预排序、
    全靠 LLM 在噪声里硬挑。用源分级把"可信一手/媒体源"和"Google News/Tavily 搜索兜底"
    拉开档次：搜索兜底是内容农场/标题党/近重复的主来源，应整体降权。
    """
    src = (item.get("source") or "").lower()
    if src.startswith("google news") or src.startswith("tavily"):
        return 0.3
    if src.startswith("github trending"):
        return 0.6
    return 1.0  # 具名 RSS：官方博客 / 行业垂直媒体 / 中文 AI 媒体


# Google News 标题形如 "Headline - Outlet"；剥掉尾部 " - 媒体名" 才能让同一事件的
# 多个转载版本归一到一起（否则后缀不同→永远 cross_source_count=1，多源标注也生不出来）
_OUTLET_SUFFIX_RE = re.compile(r"\s+[-–—|]\s+[^-–—|]{1,40}$")


def _strip_outlet(title: str) -> str:
    return _OUTLET_SUFFIX_RE.sub("", title or "").strip()


def outlet_of(item: dict) -> str:
    """真实媒体名：Google News 取标题尾部媒体后缀；具名 RSS 取源名（去掉 '(kw)' 关键词后缀）。"""
    src = (item.get("source") or "").strip()
    if src.lower().startswith("google news"):
        m = _OUTLET_SUFFIX_RE.search(item.get("title", "") or "")
        return m.group(0).lstrip(" -–—|").strip() if m else "Google News"
    return src.split("(")[0].strip()


def _influencer_handle(item: dict) -> str:
    """从 influencer 源名 'X · Karpathy' 提取人名作为 matched_keyword。"""
    src = (item.get("source") or "").strip()
    return src.split("·")[-1].strip() or src or "influencer"


# 已知中文媒体源名片段（源名匹配即判为中文来源；另：标题含 CJK 也算）
_ZH_SOURCE_HINTS = (
    "IT之家", "机器之心", "量子位", "36氪", "InfoQ", "新智元", "钛媒体",
    "虎嗅", "澎湃", "财联社", "财新", "雷锋", "爱范儿", "少数派", "美团", "有赞",
)


def is_chinese_item(item: dict) -> bool:
    """是否为中文来源/中文内容：标题含中日韩表意文字，或源名是已知中文媒体。"""
    if _CJK_RE.search(item.get("title", "") or ""):
        return True
    src = item.get("source") or ""
    return any(z in src for z in _ZH_SOURCE_HINTS)


def compute_score(item: dict, window_hours: float = 24.0) -> float:
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
    rec_term = 0.07 * recency_boost(item, window_hours)  # 新近度按采集周期归一化
    tier_term = 0.30 * source_tier(item)  # 源质量：精选 RSS +0.30，搜索兜底仅 +0.09
    return abs_term + delta_term + cross_term + tavily_term + rec_term + tier_term


def _normalize_title(title: str) -> str:
    """跨源同事件检测用：小写 + 去标点 + 折叠空白。粗糙但够用。"""
    t = (title or "").lower()
    t = re.sub(r"[\s\W_]+", " ", t, flags=re.UNICODE)
    return t.strip()


def _event_key(item: dict) -> str:
    """同事件聚类 key：剥掉媒体名后缀再归一化标题。"""
    return _normalize_title(_strip_outlet(item.get("title", "")))


def compute_cross_source_count(items: list[dict]) -> None:
    """原地给每个 item 的 signals 写入：
      - cross_source_count = 同事件被多少个不同**媒体**报道
      - cross_sources      = 这些媒体名（用于报告里 📎 来源：A / B / C 多源标注）

    用「剥离媒体名后缀的归一化标题」作聚合 key，让 Google News 上同一事件的多个转载
    版本（"… - Bloomberg" / "… - MSN" / "… - Crypto Briefing"）真正归并到一起。
    """
    clusters: dict[str, set[str]] = {}
    for it in items:
        key = _event_key(it)
        if not key:
            continue
        clusters.setdefault(key, set()).add(outlet_of(it))
    for it in items:
        outlets = clusters.get(_event_key(it), set())
        sig = dict(it.get("signals", {}))
        sig["cross_source_count"] = max(len(outlets), 1)
        sig["cross_sources"] = sorted(outlets)
        it["signals"] = sig


def collapse_events(items: list[dict]) -> list[dict]:
    """把同一事件的多个转载折叠成一条代表（优先精选 RSS 源，再按客观分），
    避免同一新闻在候选池里占据多个名额、稀释 LLM 的判断。

    cross_sources 已在折叠前算好并写进每条的 signals，代表条自带完整媒体名单。
    """
    best: dict[str, dict] = {}
    passthrough: list[dict] = []

    def rank(it: dict) -> tuple[int, float, float]:
        # 同一事件多源时：优先中文来源 → 再按源质量 → 再按客观分
        return (int(is_chinese_item(it)), source_tier(it),
                float(it.get("score") or compute_score(it)))

    for it in items:
        key = _event_key(it)
        if not key:
            passthrough.append(it)  # 无可用标题，原样保留
            continue
        cur = best.get(key)
        if cur is None or rank(it) > rank(cur):
            best[key] = it
    return list(best.values()) + passthrough


# ---------- 主流程 ----------

def normalize(
    items: list[dict],
    keywords: list[dict],
    db=None,
    since: datetime | None = None,
    themes: list[dict] | None = None,
    llm_cfg: dict | None = None,
    watchlist: list[str] | None = None,
) -> list[dict]:
    """去重 → 时窗（since=None 不过滤）→ 关键词匹配 → (可选) 语义主题门召回 → 从 db 算 delta → 打分。

    themes + llm_cfg 同时给出时，对"具名 RSS/媒体源里未命中关键词"的条目跑语义主题门
    （core.theme_gate），从内容召回新产品/新名字；任一为空则跳过、退化为纯关键词匹配。
    """
    _before_dedupe = items
    items = dedupe(items)
    if TRACE.enabled:
        TRACE.snapshot("normalize.input", _before_dedupe, note="抓取去重前")
        seen, dups = set(), []
        for it in _before_dedupe:
            if it.get("id") in seen:
                dups.append((it, "重复 id"))
            seen.add(it.get("id"))
        TRACE.drops("normalize.dedupe", dups, note=f"{len(_before_dedupe)}→{len(items)}")
    if since is not None:
        _bw = items
        items = [it for it in items if after(it, since)]
        if TRACE.enabled:
            kept = {id(x) for x in items}
            tw_drops = []
            for it in _bw:
                if id(it) in kept:
                    continue
                pub = parse_dt(it.get("published"))
                reason = "无可解析日期" if pub is None else f"早于时窗起点（{it.get('published')}）"
                tw_drops.append((it, reason))
            TRACE.drops("normalize.time_window", tw_drops, note=f"{len(_bw)}→{len(items)}")

    # 采集周期（小时）：新近度按它归一化。since=None（--all 无窗口）退回日报基准 24h。
    window_hours = (
        max(1.0, (datetime.now(timezone.utc) - since).total_seconds() / 3600)
        if since is not None else 24.0
    )

    matched: list[dict] = []
    dropped_quality = 0
    influencer_kept = 0
    gate_pool: list[dict] = []  # 未命中关键词、但来自具名媒体源、留给语义主题门的候选
    _dbg_drops: list[tuple] = []  # (item, reason) 仅 --debug 用
    for it in items:
        # AI 意见领袖源：绕过关键词匹配与质量过滤（发言本身即精选），但剔除转发/回复
        if (it.get("category") or "") == "influencer":
            t = (it.get("title") or "").lstrip()
            if t.startswith("RT by") or t.startswith("R to ") or t.startswith("R to@"):
                if TRACE.enabled:
                    _dbg_drops.append((it, "意见领袖转发/回复"))
                continue
            it["matched_keywords"] = [_influencer_handle(it)]
            matched.append(it)
            influencer_kept += 1
            continue
        hits = match_keywords(it, keywords)
        if not hits:
            # 仅收集"具名 RSS/媒体源 + 非低质"的未命中条目，交给语义主题门（vocab 粗筛/LLM 在门内做）
            if themes and llm_cfg and source_tier(it) >= 1.0:
                low, reason = is_low_quality(it)
                if not low:
                    gate_pool.append(it)
                elif TRACE.enabled:
                    _dbg_drops.append((it, f"未命中关键词 + 低质（{reason}）"))
            elif TRACE.enabled:
                _dbg_drops.append((it, "未命中关键词（非媒体源，不进主题门）"))
            continue
        low, reason = is_low_quality(it)
        if low:
            dropped_quality += 1
            if TRACE.enabled:
                _dbg_drops.append((it, f"质量过滤：{reason}"))
            continue
        it["matched_keywords"] = hits
        matched.append(it)
    if dropped_quality:
        print(f"  [质量过滤] 丢弃 {dropped_quality} 条社区闲聊/疑问/营销条目")
    if influencer_kept:
        print(f"  [意见领袖] 保留 {influencer_kept} 条大神原创发言（绕过关键词匹配）")
    if TRACE.enabled:
        TRACE.snapshot("normalize.keyword_matched", matched, note="关键词命中（含意见领袖），进主题门前")
        TRACE.snapshot("normalize.gate_pool", gate_pool, note="未命中关键词的媒体源候选，待语义主题门")
        TRACE.drops("normalize.keyword_quality", _dbg_drops)

    # 语义主题门：从未命中关键词的媒体条目里，按内容召回属于关注主题的（如新 KV cache/向量库方案）
    if gate_pool:
        from core.theme_gate import theme_gate  # 延迟导入避免与本模块循环依赖
        matched.extend(theme_gate(gate_pool, themes, llm_cfg,
                                  keywords=keywords, window_hours=window_hours))

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

    # 跨源聚合：数一遍同事件被多少个媒体报道（剥离媒体名后缀归并近重复），
    # 写进 cross_source_count + cross_sources
    compute_cross_source_count(matched)

    # 事件折叠：同一事件的多个转载只留一条代表（优先精选 RSS 源），去除候选池里的近重复
    before = len(matched)
    _bc = list(matched)
    matched = collapse_events(matched)
    if before != len(matched):
        print(f"  [事件折叠] {before} → {len(matched)} 条（合并同事件转载）")
    if TRACE.enabled and before != len(matched):
        kept = {id(x) for x in matched}
        TRACE.drops("normalize.collapse_events",
                    [(it, "同事件转载被折叠") for it in _bc if id(it) not in kept],
                    note=f"{before}→{len(matched)}")

    # 重点关注名单：命中（标题/摘要含 watchlist 词）的条目打标，记下命中的具体词，
    # 供排序阶段保送进 LLM 候选池、并在 LLM 漏选时做保底补回。
    wl_pairs = [(w, w.lower()) for w in (watchlist or []) if w]
    if wl_pairs:
        n_wl = 0
        for it in matched:
            hay = f"{it.get('title','')} {it.get('summary','')}".lower()
            for w_orig, w_low in wl_pairs:
                if w_low in hay:
                    it["watchlisted"] = True
                    it["watchlist_hit"] = w_orig
                    n_wl += 1
                    break
        if n_wl:
            print(f"  [重点关注] 标记 {n_wl} 条命中 watchlist 的条目（保送进 LLM 候选池 + 漏选时保底）")

    for it in matched:
        it["score"] = compute_score(it, window_hours)
    TRACE.snapshot("normalize.output", matched,
                   note="去重 + 时窗 + 关键词 + 主题门 + 折叠 + 打分后的最终候选")
    return matched
