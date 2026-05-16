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

# 用户视角的资讯类型分类（固定四类，便于 UI 聚合与多样性筛选）
CATEGORIES = ["公司新闻", "政策走向", "学术论文", "技术解读"]
CATEGORY_HINTS = """资讯类型分类（必须从下列四个中选一个）：
- 公司新闻：产品/版本发布、官方公告、收购合作、关键人事、财报、出口管制对企业的实际动作
- 政策走向：政府/监管/法律层面的政策、法规、行政命令、行业管制框架（不针对具体一家公司）
- 学术论文：arxiv/会议/期刊论文、研究成果、新基准、新数据集（首发）
- 技术解读：博客文章、技术深度分析、工程实践、issue/PR 讨论、视频解析、上手教程"""


IMPORTANCE_RUBRIC = """用户使用本工具的目的：**追踪技术趋势的最新信号**。
所以你的核心任务是**判断每一条候选的"信息价值程度"**，并按价值高低筛选。

「信息价值」由以下五个维度共同决定（每条都该过一遍这五个维度，再综合打分）：

1. **新增量**（最关键）：对未来趋势判断带来的新信息。
   - 高：新产品/版本/数字/合作/评测/方法。
   - 低：已经众所周知的事实复盘、综述、年终盘点。
2. **首发性**：是该信息的源头，还是 N 手转载？
   - 高：公司官博、官方 GitHub、arXiv 一作。
   - 低：综合新闻媒体对官方稿件的二次复述（往往滞后且可能曲解）。
3. **可验证细节密度**：摘要里能不能数出具体事实？
   - 高：含版本号、benchmark 数字、模型尺寸、推理价格、合同金额、对比基线、日期、人名。
   - 低：通篇形容词、"重要意义"、"格局变化"等空话。
4. **影响范围**：这条信息会撬动多少人/多大场景的决策？
   - 高：开源主流模型、行业标准、监管法规、超大规模硬件发布。
   - 低：小众教程、一次性 demo、限定地区或小公司动作。
5. **可操作性**：用户看完是否能据此做点什么（关注/学习/采购/对标）？
   - 高：能立刻试用/复现/对接的内容。
   - 低：纯舆论、纯八卦、纯股价讨论。

按上述维度综合后，**优先选信息价值最高的几条；价值不够就少选甚至空列表**（宁缺毋滥）。

辅助原则 ——

  **它对未来技术趋势判断是否带来新的信息增量？**

带来增量的内容（视为重要）：
1. 新产品 / 新版本 / 新硬件 / 官方 release / GA（带版本号、能力描述、性能数据）
2. 新方法 / 新架构 / 新论文（显著超越 SOTA、带具体指标）
3. 新公布的合作 / 收购 / 关键人事 / 出口管制等突发事件（带官方确认、具体条款）
4. 新基准 / 新评测结果 / 新数据集发布
5. 新的成本、规模、性能数字（如某模型推理价格降至 X / 训练吞吐达 Y）

**不带增量的内容（即使描述的是真事也要降权或丢弃）**：
- **对既成事实的复盘、综述、行业观察**：把过去几周/几个月已知的事再讲一遍，
  没有新数据、没有新公告。例：
    × "Chinese A.I. Firms Push Beyond Nvidia as DeepSeek Turns to Huawei"
      （此事已被追踪者熟知，文章只是 NYT 的回顾式综述）
    × "How OpenAI Won the AI Race"（盘点/复盘）
    × "X 公司今年的 5 个重要时刻"（年终回顾）
- 第三方解读 / 趋势分析 / 行业评论（"据知情人士 / 业内认为 / 据传"等无官方一手依据）
- 股价预测、分析师评级、目标价、投资 listicle
- how-to、教程、最佳实践（除非是官方首次发布的指引）
- 二次转载、同质化新闻（多家媒体复述同一通稿时只留一份且优先官方源）

判断"是否带新增量"的简单准则：
- 标题/摘要是否含**具体新产品名 / 新版本号 / 新数据 / 新日期 / 新指标**？没有 → 多半是复盘
- 信源域名是否在 [官方] 列表里？是 → 强烈优先；否 → 谨慎，可能是综述
- "DeepSeek 转向华为"这类事实若**两个月前就在传**，今天再出现的报道几乎一定是综述

权威性偏好（重要！）：
- **优先选择 [官方] 标记的条目**，来自公司官网/官方博客/GitHub/arXiv
- 同一事件多个候选时**优先官方源版本**（转载常滞后、且可能曲解）
- **如果某关键词候选里全是复盘/综述/解读，没有带新增量的官方一手内容，宁可返回空列表**"""


PROMPT_TEMPLATE = """你在帮用户筛选关于「{keyword}」的最重要资讯。

{rubric}

{category_hints}
{user_preference_block}

候选条目共 {n} 条（带 [官方] 标记的来自该关键词的官方源）：
{candidates}

任务：从中选出最重要且最权威的最多 {top_k} 条。**宁缺毋滥** —— 如果都不够重要，可以只返回 1 条甚至空列表。

**类别多样性偏好**：如果候选中存在多种类别（如同时有「公司新闻」和「学术论文」），
在不牺牲重要性的前提下，尽量让选中条目覆盖不同类别，避免 top_k 条全是同一类。
但若同类里确有几条都明显更重要（如多个官方发布同时出现），则可全部保留。

对每条选中条目，给出 **display_title** / **summary** / **category** 三个字段：

**display_title**: 中文新闻式短标题 (≤30 字)
- 把事件本质说清楚；论文/技术 paper 要把核心方法 + 主要成果用日常语言写出来，不要保留学术 jargon
- 例: "PARD-2: Target-Aligned Parallel Draft Model for Dual-Mode Speculative Decoding"
  → "PARD-2 提出目标对齐的并行草案模型，加速 LLM 推测解码"
- 不要用"重磅"、"震撼"、"颠覆"、"刷屏"等夸张词
- 如原标题已是简洁的中文新闻式表达，也可直接保留意思翻译过来

**summary**: 中文 8-15 句客观摘要 (**500~1000 字，越接近 1000 字越好，不要为了凑字数编造**)。
摘要的目的是让用户**不点开链接也能完整判断这条信息的价值**，所以信息密度必须高。
要**逐项展开覆盖原文已经提及**的以下要素（原文没提的就不写，绝不编造，绝不重复同一事实）：

1. 事件本身（1-2 句）：发生了什么、发布了什么、提出了什么、宣布了什么。讲清楚动作和触发情境
2. 主体与利益相关方（1 句）：公司、产品、技术名、版本号、合作方、相关人员、所属机构
3. **核心方法/技术细节（重点，3-5 句）**：算法思想、架构特点、训练数据、关键创新点、与已有方案的差异
4. **量化结果（重点，2-4 句）**：吞吐、加速比、accept rate、参数量、上下文长度、容量、价格、对比基线、benchmark 分数、训练 token、数据规模、显存占用等具体数字 —— **凡是原文有的都写出来**
5. 价值与适用场景（1-2 句）：解决了什么实际问题、面向什么场景、什么样的用户/团队会受益
6. 背景与上下文（1 句）：前作、所属系列、是该领域第几次类似工作、和近期其他工作的关系
7. 局限或后续工作（如原文提及，1 句）：作者承认的不足、未来 roadmap

写作要求：
- 不用"重磅"、"震撼"、"颠覆"、"格局"、"凸显"、"标志着"等情绪/评价性词语
- 不写自己的判断或预测，只复述原文
- 避免空泛比喻 ——"性能大幅提升"应改为原文具体的数字
- 不要重复同一信息（避免堆字数）；若原文确实没材料，宁可短也别凑

如果原文长度足够支撑 1000 字，那就尽量写到接近 1000；不够，就写到原文信息量耗尽为止

严格禁止：
- 扩写原文未提及的信息；加入个人评价；用推测词（"可能"、"或将"、"有望"）替代原文事实
- 把候选信息中的"信号"字段（如 points=410、comments=116）写进摘要
- 夸张修辞、评价性词语（"重大意义"、"格局改变"、"标志着"等）

**category**: 从「公司新闻 / 政策走向 / 学术论文 / 技术解读」中选一个，不要新造类别名

【无详情兜底】如原始摘要为空或仅是元数据/占位符（标题信息也很少）：summary 写 "原文仅有标题，未提供详细内容，请点击链接查阅。" 即可，不要编造任何细节、不要凑字数。

仅返回 JSON，不要其它任何输出，格式：
{{"selected": [{{"id": <候选编号>, "display_title": "<中文短标题>", "summary": "<3-5句中文摘要>", "category": "<四选一>"}}]}}"""


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
    # 给 LLM 看的原文摘要：放宽到 1600 字符，覆盖 arxiv abstract / 博客首段 / 详细 RSS
    # 太短会导致 LLM 没素材生成 500-1000 字目标摘要 → 频繁触发"无详情兜底"
    summary = re.sub(r"\s+", " ", summary)[:1600]
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


def _build_user_preference_block(text: str | None) -> str:
    """把用户自填的个性化偏好包成一个独立段落注入 prompt；为空时不输出该段。"""
    if not text or not text.strip():
        return ""
    return (
        "\n用户的个性化偏好（来自界面输入，作为额外的高优先级判断依据）：\n"
        f"  > {text.strip()}\n"
        "把它叠加到上面的「信息价值」五个维度之上 —— "
        "符合用户偏好的条目应被显著加权；与用户偏好明显不符（但仍有价值）的条目可保留 1-2 条不要全部丢。"
    )


def rank_per_keyword(
    items: list[dict],
    keyword: dict,
    client: OpenAI,
    model: str,
    top_k: int,
    max_candidates: int,
    user_preference: str | None = None,
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
        category_hints=CATEGORY_HINTS,
        user_preference_block=_build_user_preference_block(user_preference),
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
            cat = (entry.get("category") or "").strip()
            it["llm_category"] = cat if cat in CATEGORIES else "技术解读"
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
2. 如有任一问题：写一个严格基于"原始标题 + 原文摘要"的新版本（中文 8-15 句 500-1000 字），**保留原文已经提到的事件、性能数据、方法细节、影响等所有具体细节**，只剔除推测/评论/超出原文的内容；如果原文本身就只够写两三句，就老实保留短长度，不要编造内容凑字数
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
    user_preference: str | None = None,
) -> dict[str, list[dict]]:
    client = OpenAI(api_key=llm_config["api_key"], base_url=llm_config.get("base_url"))
    model = llm_config["model"]
    top_k = llm_config["top_k_per_keyword"]
    max_cand = llm_config["max_candidates_per_keyword"]

    kw_by_name = {kw["name"]: kw for kw in keywords}
    groups = group_by_keyword(items, keywords)
    result: dict[str, list[dict]] = {}
    if user_preference:
        print(f"  [LLM] 已注入个性化偏好（{len(user_preference)} 字符）")
    for kw_name, cands in groups.items():
        n_official = sum(1 for c in cands if is_official_for(c, kw_by_name[kw_name]))
        print(f"  [LLM] {kw_name}: {len(cands)} 候选 (官方 {n_official}) → ", end="", flush=True)
        sel = rank_per_keyword(
            cands, kw_by_name[kw_name], client, model, top_k, max_cand,
            user_preference=user_preference,
        )
        print(f"{len(sel)} 选中")
        result[kw_name] = sel
    return result
