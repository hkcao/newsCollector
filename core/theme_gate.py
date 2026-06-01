"""主题语义门 —— 对「未命中任何关键词」但来自具名 RSS/媒体源的条目，用 LLM 判断它是否
「确实在讲」我们关注的某个主题（KV cache、向量/图数据库、并行文件系统…），属于则以该主题
对应的既有关键词重新纳入候选池。

解决纯字符串精确匹配的召回缺口：从**内容**出发识别，而不是枚举每一个产品名——下一个
Reasonix-类 KV cache 方案、Lakebase-类向量库即使换了名字、标题里不带关键词，也能被捞回。

成本控制（与 normalize 约定）：
  1) 只对**具名 RSS/媒体源**的未命中条目跑（normalize 侧用 source_tier 过滤掉
     Google News / Tavily / GitHub Trending / 意见领袖）；
  2) 先用各主题 vocab 的并集做**本地词表粗筛**，命中粗筛的才送 LLM；
  3) 单次 LLM 候选数设上限（按源质量+新近度排序后截断），超出部分明确告知、不静默丢。

「关键词自动扩展」：每次召回时把让条目相关的核心实体名（如 Reasonix）计数落盘到
reports/theme_finds.json，累计达阈值且尚未进 keywords.yaml 的，打印为"建议固化为关键词"，
让长期反复出现的新名字从语义门（每次花 LLM）沉淀为关键词（精确匹配，零成本）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from openai import OpenAI

from core.normalizer import is_low_quality, recency_boost, source_tier
from core.ranker import _chat
from core.debug_trace import TRACE

# 实体累计出现多少次、且仍不在 keywords.yaml 里，就建议固化为关键词
_PROMOTE_THRESHOLD = 3
_FINDS_PATH = Path(__file__).resolve().parent.parent / "reports" / "theme_finds.json"


def build_vocab(themes: list[dict]) -> set[str]:
    """所有主题 vocab 的并集（小写），用于本地粗筛。"""
    vocab: set[str] = set()
    for t in themes:
        for w in t.get("vocab") or []:
            w = str(w).strip().lower()
            if w:
                vocab.add(w)
    return vocab


def is_media_rss(item: dict) -> bool:
    """是否为具名 RSS/媒体源（排除搜索源 / GitHub 趋势 / 意见领袖）。

    source_tier：Google News / Tavily = 0.3，GitHub Trending = 0.6，具名 RSS = 1.0。
    只有具名 RSS/媒体源（一手博客 / 行业媒体 / 中文 AI 媒体）才值得花 LLM 语义召回。
    """
    return source_tier(item) >= 1.0 and (item.get("category") or "") != "influencer"


def passes_prefilter(item: dict, vocab: set[str]) -> bool:
    hay = f"{item.get('title','')} {item.get('summary','')}".lower()
    return any(w in hay for w in vocab)


def _gist(item: dict, limit: int = 160) -> str:
    s = (item.get("summary") or "").strip().replace("\n", " ")
    return s[:limit]


_PROMPT = """你在为一套「AI 存储技术规划」情报系统做**主题相关性**判定。

下面这些新闻**没有命中任何已知关键词**，请判断每条是否**确实在讲**我们关注的某个主题
（只是顺带提一句、背景里带过，不算）。我们关注的主题及定义：
{themes}

候选新闻（编号 · 标题 · 摘要要点）：
{catalog}

对每条输出一条记录，合成**严格 JSON 数组**（不要 markdown 代码块、不要多余文字），结构：
[{{"i": 编号, "theme": "命中的主题名（必须与上面某个主题名**完全一致**；都不属于填 none）", "entity": "让它相关的最核心产品/系统/公司名，没有具体专有名词就留空字符串"}}]

判定从严：拿不准就填 none，宁缺勿滥。entity 只填**一个**最关键的专有名词。"""


def _parse_json_array(text: str) -> list[dict]:
    cleaned = (text or "").strip()
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def _known_terms(keywords: list[dict]) -> set[str]:
    """已在 keywords.yaml 覆盖的名字（name + aliases，小写），用于过滤"建议固化"。"""
    terms: set[str] = set()
    for kw in keywords:
        terms.add(str(kw.get("name", "")).strip().lower())
        for a in kw.get("aliases") or []:
            terms.add(str(a).strip().lower())
    terms.discard("")
    return terms


def _record_promotions(found: list[tuple[str, str]], keywords: list[dict], log) -> None:
    """累计实体计数落盘；对达阈值且未被关键词覆盖的，打印固化建议。"""
    if not found:
        return
    known = _known_terms(keywords)
    try:
        tally = json.loads(_FINDS_PATH.read_text(encoding="utf-8")) if _FINDS_PATH.exists() else {}
    except (json.JSONDecodeError, OSError):
        tally = {}

    for entity, theme in found:
        key = entity.strip()
        if not key:
            continue
        rec = tally.get(key) or {"count": 0, "theme": theme}
        rec["count"] += 1
        rec["theme"] = theme
        tally[key] = rec

    try:
        _FINDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _FINDS_PATH.write_text(json.dumps(tally, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass

    ripe = [
        (k, v) for k, v in tally.items()
        if v.get("count", 0) >= _PROMOTE_THRESHOLD and k.lower() not in known
    ]
    if ripe:
        log(f"  [主题门] 以下新名字已反复出现，建议固化为 keywords.yaml 关键词（之后零成本精确匹配）：")
        for k, v in sorted(ripe, key=lambda x: -x[1]["count"]):
            log(f"           · {k}（{v['count']} 次，主题：{v['theme']}）")


def theme_gate(
    misses: list[dict],
    themes: list[dict],
    llm_cfg: dict,
    *,
    keywords: list[dict] | None = None,
    max_items: int = 80,
    window_hours: float = 24.0,
    log=print,
) -> list[dict]:
    """对未命中关键词的候选做语义主题判定，返回被召回的条目（已写好 matched_keywords）。

    misses：normalize 已筛过的"具名 RSS/媒体源 + 非低质 + 未命中关键词"条目。
    """
    if not misses or not themes or not llm_cfg:
        return []

    vocab = build_vocab(themes)
    pool = [it for it in misses if passes_prefilter(it, vocab)]
    if not pool:
        return []

    # 按源质量 + 新近度（按采集周期归一化）排序后截断，超上限明确告知
    pool.sort(key=lambda it: (source_tier(it), recency_boost(it, window_hours)), reverse=True)
    if len(pool) > max_items:
        log(f"  [主题门] 粗筛命中 {len(pool)} 条，超上限按源质量+新近度取前 {max_items} 条送判（其余本轮跳过）")
        pool = pool[:max_items]
    else:
        log(f"  [主题门] 粗筛命中 {len(pool)} 条未命中关键词的媒体条目，送 LLM 语义判定…")

    name_set = {t["name"] for t in themes}
    kw_of = {t["name"]: (t.get("keyword") or t["name"]) for t in themes}
    themes_block = "\n".join(f"[{t['name']}] {(t.get('definition') or '').strip()}" for t in themes)
    catalog = "\n".join(f"{i}. {it.get('title','')}｜{_gist(it)}" for i, it in enumerate(pool, 1))
    prompt = _PROMPT.format(themes=themes_block, catalog=catalog)

    client = OpenAI(api_key=llm_cfg["api_key"], base_url=llm_cfg.get("base_url"))
    try:
        raw = _chat(client, llm_cfg["model"], prompt, kind="theme_gate", temperature=0.1)
    except Exception as e:  # 语义门失败不应拖垮主流程
        log(f"  [主题门] 语义判定失败，跳过：{e}")
        return []

    admitted: list[dict] = []
    found: list[tuple[str, str]] = []
    for rec in _parse_json_array(raw):
        if not isinstance(rec, dict):
            continue
        theme = (rec.get("theme") or "").strip()
        if theme not in name_set:
            continue
        try:
            idx = int(rec.get("i")) - 1
        except (TypeError, ValueError):
            continue
        if not (0 <= idx < len(pool)):
            continue
        it = pool[idx]
        it["matched_keywords"] = [kw_of[theme]]
        it["theme_gated"] = True
        entity = (rec.get("entity") or "").strip()
        if entity:
            it["theme_entity"] = entity
            found.append((entity, theme))
        admitted.append(it)

    if TRACE.enabled:
        adm = {id(x) for x in admitted}
        TRACE.snapshot("theme_gate.sent_to_llm", pool, note="vocab 粗筛命中，送 LLM 语义判定")
        TRACE.snapshot("theme_gate.admitted", admitted, note="语义召回（重新纳入候选）")
        TRACE.drops("theme_gate.rejected",
                    [(it, "LLM 判定非关注主题（none）") for it in pool if id(it) not in adm])
    if admitted:
        log(f"  [主题门] 语义召回 {len(admitted)} 条（未命中关键词但确属关注主题）")
    if keywords is not None:
        _record_promotions(found, keywords, log)
    return admitted
