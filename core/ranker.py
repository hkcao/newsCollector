"""LLM 排序器 —— 走 OpenAI 兼容接口 (DeepSeek / Kimi / OpenAI 通用)。

按关键词分组送给 LLM，让它基于"重要性 + 权威性"挑出 top K，可以宁缺毋滥。
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse

import yaml
from openai import OpenAI


# 任何关键词都视为权威首发源
GLOBAL_OFFICIAL_DOMAINS = {"arxiv.org", "github.com"}


IMPORTANCE_RUBRIC = """评判"重要性"的标准（任一即可视为重要）：
1. 产品/技术发布：新产品、新版本、新硬件、官方 release / GA
2. 公司战略动作：收购、合作、战略合作、关键人事、组织变动
3. 技术突破：论文/项目显著超越 SOTA、新架构、突破性结果
4. 重大经营事件：财报超预期/不及预期、监管限制、供应链事件
5. 突发事件：安全事件、官司、出口管制、停服、事故

应降权/丢弃的内容：
- 股价预测、分析师评级、目标价、"will hit $X" 类文章
- 投资组合 listicle（"Top N AI stocks"）
- 二次转载、同质化新闻
- 纯教程、how-to、操作指南（除非是官方首次发布）

权威性偏好（重要！）：
- **优先选择 [官方] 标记的条目**，来自公司官网/官方博客/GitHub/arXiv 等原始发文源
- 同一事件若有多个候选，**优先选官方源版本**，避免选转载或二次报道
  （转载常滞后数小时甚至数天，时效性和准确性都不可控）"""


PROMPT_TEMPLATE = """你在帮用户筛选关于「{keyword}」的最重要资讯。

{rubric}

候选条目共 {n} 条（带 [官方] 标记的来自该关键词的官方源）：
{candidates}

任务：从中选出最重要且最权威的最多 {top_k} 条。**宁缺毋滥** —— 如果都不够重要，可以只返回 1 条甚至空列表。

对每条选中条目，给出 **display_title** 与 **summary** 两个字段，均使用中文：

**display_title**: 中文新闻式短标题 (≤30 字)
- 把事件本质说清楚；论文/技术 paper 要把核心方法 + 主要成果用日常语言写出来，不要保留学术 jargon
- 例: "PARD-2: Target-Aligned Parallel Draft Model for Dual-Mode Speculative Decoding"
  → "PARD-2 提出目标对齐的并行草案模型，加速 LLM 推测解码"
- 不要用"重磅"、"震撼"、"颠覆"、"刷屏"等夸张词
- 如原标题已是简洁的中文新闻式表达，也可直接保留意思翻译过来

**summary**: 中文 3-5 句客观摘要 (150~250 字)，需要**完整覆盖原文已经提及**的以下要素（原文没提的就不写，绝不编造）：
1. 事件本身：发生了什么 / 发布了什么 / 提出了什么方法
2. 涉及主体：公司、产品、技术名、版本号
3. **性能与规模数据（重要！原文有就一定要写进来）**：吞吐、加速比、参数量、容量、价格、对比基线等具体数字或对比
4. 价值与影响：原文中明确说的"解决了什么问题 / 面向什么场景"

严格禁止：
- 扩写原文未提及的信息；加入个人评价；用推测词（"可能"、"或将"、"有望"）替代原文事实
- 把候选信息中的"信号"字段（如 points=410、comments=116）写进摘要
- 夸张修辞、评价性词语（"重大意义"、"格局改变"、"标志着"等）

如原始摘要为空或仅是元数据/占位符（标题信息也很少）：summary 写 "原文仅有标题，未提供详细内容，请点击链接查阅。"，不要编造任何细节。

仅返回 JSON，不要其它任何输出，格式：
{{"selected": [{{"id": <候选编号>, "display_title": "<中文短标题>", "summary": "<3-5句中文摘要>"}}]}}"""


def _domain(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def is_official_for(item: dict, keyword: dict) -> bool:
    """判断该 item 对于此 keyword 是否来自官方源（域名匹配）。"""
    d = _domain(item.get("url", ""))
    if not d:
        return False
    targets = set(GLOBAL_OFFICIAL_DOMAINS) | set(keyword.get("official_domains") or [])
    for t in targets:
        t = t.lower().lstrip(".")
        if d == t or d.endswith("." + t):
            return True
    return False


def _format_candidate(idx: int, item: dict, official: bool) -> str:
    sig = item.get("signals", {})
    sig_str = ", ".join(f"{k}={v}" for k, v in sig.items() if v) or "无客观信号"
    summary = (item.get("summary") or "").strip()
    summary = re.sub(r"<[^>]+>", " ", summary)
    summary = re.sub(r"\s+", " ", summary)[:240]
    tag = "[官方] " if official else ""
    return (
        f"[#{idx}] {tag}源={item['source']} | 域名={_domain(item.get('url', ''))} | {sig_str}\n"
        f"  标题: {item['title']}\n"
        f"  摘要: {summary or '(原 RSS 无摘要)'}"
    )


def _parse_response(text: str) -> list[dict]:
    try:
        return json.loads(text).get("selected", [])
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return []
        try:
            return json.loads(m.group(0)).get("selected", [])
        except json.JSONDecodeError:
            return []


def load_llm_config(path: Path) -> dict:
    cfg = {
        "base_url": None,
        "model": "gpt-4o-mini",
        "api_key": None,
        "top_k_per_keyword": 2,
        "max_candidates_per_keyword": 40,
    }
    if path.exists():
        cfg.update(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    cfg["base_url"] = os.getenv("LLM_BASE_URL", cfg["base_url"])
    cfg["model"] = os.getenv("LLM_MODEL", cfg["model"])
    cfg["api_key"] = os.getenv("LLM_API_KEY", cfg["api_key"])
    if not cfg["api_key"]:
        raise RuntimeError(
            "未配置 LLM API key —— 请设置环境变量 LLM_API_KEY，或用 --no-llm 跳过"
        )
    return cfg


def group_by_keyword(items: list[dict], keywords: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {kw["name"]: [] for kw in keywords}
    for it in items:
        for kw_name in it.get("matched_keywords", []):
            if kw_name in groups:
                groups[kw_name].append(it)
    # 官方源优先，再按 score 倒序 —— 影响进入 LLM 候选池的优先级
    for k, kw_list in [(kw["name"], kw) for kw in keywords]:
        groups[k].sort(
            key=lambda x: (is_official_for(x, kw_list), x.get("score", 0)),
            reverse=True,
        )
    return groups


def rank_per_keyword(
    items: list[dict],
    keyword: dict,
    client: OpenAI,
    model: str,
    top_k: int,
    max_candidates: int,
) -> list[dict]:
    if not items:
        return []
    pool = items[:max_candidates]
    cand_lines = [
        _format_candidate(i, it, is_official_for(it, keyword))
        for i, it in enumerate(pool)
    ]
    prompt = PROMPT_TEMPLATE.format(
        keyword=keyword["name"],
        rubric=IMPORTANCE_RUBRIC,
        n=len(pool),
        candidates="\n\n".join(cand_lines),
        top_k=top_k,
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content or ""
    except Exception as e:
        print(f"    ! LLM 调用失败: {e}")
        return []

    selected = []
    for entry in _parse_response(text):
        idx = entry.get("id")
        if isinstance(idx, int) and 0 <= idx < len(pool):
            it = dict(pool[idx])
            it["llm_summary"] = entry.get("summary") or entry.get("reason") or ""
            it["display_title"] = (entry.get("display_title") or "").strip() or it["title"]
            it["is_official"] = is_official_for(it, keyword)
            selected.append(it)
    return selected[:top_k]


VERIFY_PROMPT = """你是新闻摘要的事实核查员。判断下面这条摘要是否含有"主观推测"或"原文未提及的扩写"。

【原始标题】{title}
【原文摘要 / 可用信息】{original}

【待审核摘要】
{summary}

任务：
1. 检查摘要中是否含有以下问题：
   - 推测词（"可能"、"或将"、"有望"、"预计"、"显示...关注"等），且原文/标题中**没有**出现
   - 评论性词语（"重大意义"、"格局改变"、"标志着"、"凸显"、"反映了"等）
   - 原文/标题里没有的细节、数字、背景或因果解读
2. 如有任一问题：写一个严格基于"原始标题 + 原文摘要"的新版本（中文 3-5 句 150-250 字），**保留原文已经提到的事件、性能数据、影响等具体细节**，只剔除推测/评论/超出原文的内容；
3. 若完全忠实：原样返回原摘要。

仅返回 JSON，不要任何其他内容：
{{"clean": <true|false>, "summary": "<最终摘要>"}}"""


def verify_summary(
    title: str,
    original_text: str,
    generated: str,
    client: OpenAI,
    model: str,
) -> tuple[str, bool]:
    """事实自检 LLM 生成的摘要，返回 (摘要文本, 是否被改写)。失败时静默不改。"""
    if not generated or not generated.strip():
        return generated, False
    prompt = VERIFY_PROMPT.format(
        title=title,
        original=original_text or "(原文 RSS 无可用摘要，仅有标题)",
        summary=generated,
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content or ""
        data = json.loads(text)
    except Exception as e:
        print(f"    [verify] 失败: {e}")
        return generated, False

    new_summary = (data.get("summary") or generated).strip()
    clean = bool(data.get("clean", True))
    return new_summary, (not clean)


TRANSLATE_PROMPT = """请把下面的新闻/论文摘要翻译为中文。

严格规则：
1. **忠实原文**，不增不减事实，不扩写、不评论、不省略
2. 保留所有专有名词（人名、公司、产品、技术名、版本号、模型名）原样不译
3. 数字、单位、对比基线保留原样
4. 仅输出中文翻译，不要任何前缀、解释或 JSON

原文：
{text}"""


def _is_mostly_chinese(text: str) -> bool:
    if not text:
        return True
    cjk = sum(1 for c in text if "一" <= c <= "鿿")
    return cjk / max(len(text), 1) > 0.20


def translate_summary(
    text: str,
    client: OpenAI,
    model: str,
) -> tuple[str, bool]:
    """RSS 原文摘要翻译为中文，返回 (译文, 是否翻译过)。已是中文/空/失败时返回原文。"""
    if not text or not text.strip() or _is_mostly_chinese(text):
        return text, False
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": TRANSLATE_PROMPT.format(text=text)}],
            temperature=0.1,
        )
        out = (resp.choices[0].message.content or "").strip()
        if out:
            return out, True
    except Exception as e:
        print(f"    [translate] 失败: {e}")
    return text, False


def translate_all(
    grouped: dict[str, list[dict]],
    llm_config: dict,
) -> None:
    """对 summary_source=='rss' 且非中文的摘要做翻译，原地标 p['translated']=True。"""
    client = OpenAI(api_key=llm_config["api_key"], base_url=llm_config.get("base_url"))
    model = llm_config["model"]
    n_tried = n_done = 0
    for picks in grouped.values():
        for p in picks:
            if p.get("summary_source") != "rss":
                continue
            n_tried += 1
            translated, ok = translate_summary(p.get("summary", ""), client, model)
            if ok:
                p["summary_original_en"] = p.get("summary", "")
                p["summary"] = translated
                p["translated"] = True
                n_done += 1
    if n_tried:
        print(f"  [翻译] 检查 {n_tried} 条 RSS 原文摘要，翻译 {n_done} 条")


def verify_all(
    grouped: dict[str, list[dict]],
    llm_config: dict,
) -> None:
    """对 LLM 生成的摘要做二次自检，原 RSS 摘要的条目跳过（已是原文）。"""
    from core.summary import clean_rss_summary

    client = OpenAI(api_key=llm_config["api_key"], base_url=llm_config.get("base_url"))
    model = llm_config["model"]
    n_checked = n_fixed = 0
    for picks in grouped.values():
        for p in picks:
            if p.get("summary_source") != "llm":
                continue
            n_checked += 1
            original = clean_rss_summary(p.get("raw_rss_summary", "")) or ""
            new, modified = verify_summary(
                p["title"], original, p.get("summary", ""), client, model
            )
            if modified:
                p["summary"] = new
                p["verified"] = True
                n_fixed += 1
    if n_checked:
        print(f"  [自检] 检查 {n_checked} 条 LLM 生成摘要，修正 {n_fixed} 条")


def rank_all(
    items: list[dict],
    keywords: list[dict],
    llm_config: dict,
) -> dict[str, list[dict]]:
    client = OpenAI(api_key=llm_config["api_key"], base_url=llm_config.get("base_url"))
    model = llm_config["model"]
    top_k = llm_config["top_k_per_keyword"]
    max_cand = llm_config["max_candidates_per_keyword"]

    kw_by_name = {kw["name"]: kw for kw in keywords}
    groups = group_by_keyword(items, keywords)
    result: dict[str, list[dict]] = {}
    for kw_name, cands in groups.items():
        n_official = sum(1 for c in cands if is_official_for(c, kw_by_name[kw_name]))
        print(f"  [LLM] {kw_name}: {len(cands)} 候选 (官方 {n_official}) → ", end="", flush=True)
        sel = rank_per_keyword(
            cands, kw_by_name[kw_name], client, model, top_k, max_cand
        )
        print(f"{len(sel)} 选中")
        result[kw_name] = sel
    return result
