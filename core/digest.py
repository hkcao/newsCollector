"""日报综合层 —— 对最终选中的条目做"跨条目综合"，生成：
  - 今日概览（overview）：一段把当天信号串起来的总览
  - 趋势分析（trends）：N 条从多条新闻里归纳出的走向
  - 给从业者的建议（advice）：N 条可落地的行动建议

读者定位与 ranker 一致：**资深 AI 存储技术规划与分析师**。因此趋势/建议都落在
选型、采购、容量/带宽规划、架构演进、TCO 这些决策层面，而不是泛泛而谈。

一次合并 LLM 调用（省往返、对推理模型尤其重要）。失败时优雅降级为空，报告照常出。
"""
from __future__ import annotations

import json
import re

from openai import OpenAI

from core.ranker import _chat


# MiniMax-M2.7 等推理模型会先吐 <think>…</think>，解析前先剥掉
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _parse_json_object(text: str) -> dict:
    """从 LLM 输出里稳健抽取 JSON 对象（容忍 think 段 / 代码块包裹 / 前后噪声）。"""
    cleaned = _THINK_RE.sub("", text or "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _catalog(grouped: dict[str, list[dict]]) -> str:
    """把所有选中条目压成紧凑清单喂给 LLM：分类 · 标题 · 一句话要点 · 来源数。"""
    lines: list[str] = []
    for picks in grouped.values():
        for p in picks:
            title = p.get("display_title") or p.get("title") or ""
            cat = p.get("llm_category") or ""
            summary = (p.get("summary") or p.get("llm_summary") or "").strip()
            # 只取摘要首段/前 160 字，控制 token
            gist = re.split(r"[。\n]", summary, maxsplit=1)[0][:160]
            sig = p.get("signals", {}) or {}
            srcs = sig.get("cross_sources") or ([p.get("source")] if p.get("source") else [])
            src_n = len(srcs)
            lines.append(f"- [{cat}] {title}（{src_n}源）：{gist}")
    return "\n".join(lines)


_PROMPT = """你在为一份面向**资深 AI 存储技术规划与分析师**的每日技术情报日报撰写"综合分析"部分。

读者日常工作：为大规模 AI 训练/推理集群做存储栈选型、容量/带宽规划、架构演进路线图、
TCO 测算、跨厂商对标。读者已经逐条读过下面这些今日入选新闻，现在需要你做**跨条目的提炼**。

今日入选新闻清单（分类 · 标题 · 要点 · 报道源数）：
{catalog}

请输出严格的 JSON（不要 markdown 代码块、不要多余文字），结构如下：
{{
  "overview": "今日概览：一段 3-5 句的总览，把今天多条新闻串成一个判断——今天对存储/AI基建规划最值得注意的是什么、有哪些主线。要具体引用清单里的事件，不要空话套话。"{analysis_keys}
}}

要求：
- overview：基于清单里**真实出现**的事件归纳，禁止编造清单里没有的新闻或数字。
{analysis_rules}- 全部用中文，每条 1-2 句，不加序号前缀（前端会自己编号）。
"""

# 仅当开启「趋势/建议」时拼进 prompt 的 JSON 键与撰写要求
_ANALYSIS_KEYS = """,
  "trends": ["趋势1", "趋势2", "..."],
  "advice": ["建议1", "建议2", "..."]"""

_ANALYSIS_RULES = """- trends：{n_trends} 条以内。每条是从今天多条新闻里**归纳**出的走向（如"分离式推理推动 KV cache 外置存储需求"），点明它对存储/AI 基建规划意味着什么。只有一两条新闻支撑不起来的趋势不要硬凑，宁少勿空泛。
- advice：{n_advice} 条以内。每条是给存储技术规划者的**可落地动作**（纳入选型评估 / 跑 PoC / 调整路线图 / 关注某厂商对标 / 重估某项 TCO 假设），要和今天的新闻直接挂钩，避免"持续关注AI发展"这种废话。
"""


def build_digest(
    grouped: dict[str, list[dict]],
    llm_cfg: dict,
    *,
    n_trends: int = 5,
    n_advice: int = 5,
    analysis: bool | None = None,
) -> dict:
    """返回 {"overview": str, "trends": [str], "advice": [str]}；无内容或失败时各项为空。

    analysis：是否生成「趋势/建议」。None 时读 llm_cfg["digest_analysis"]（默认 False）。
    关闭时只生成「今日概览」，trends/advice 恒为空（也省一截 token）。
    """
    empty = {"overview": "", "trends": [], "advice": []}
    catalog = _catalog(grouped)
    if not catalog.strip():
        return empty
    if analysis is None:
        analysis = bool(llm_cfg.get("digest_analysis", False))

    if analysis:
        prompt = _PROMPT.format(
            catalog=catalog,
            analysis_keys=_ANALYSIS_KEYS,
            analysis_rules=_ANALYSIS_RULES.format(n_trends=n_trends, n_advice=n_advice),
        )
    else:
        prompt = _PROMPT.format(catalog=catalog, analysis_keys="", analysis_rules="")

    client = OpenAI(api_key=llm_cfg["api_key"], base_url=llm_cfg.get("base_url"))
    try:
        raw = _chat(client, llm_cfg["model"], prompt, kind="digest", temperature=0.3)
    except Exception as e:  # 网络/额度/模型异常都不应拖垮整份报告
        print(f"  [digest] 综合层生成失败，跳过：{e}")
        return empty

    data = _parse_json_object(raw)
    overview = (data.get("overview") or "").strip()
    if not analysis:
        return {"overview": overview, "trends": [], "advice": []}
    trends = [str(t).strip() for t in (data.get("trends") or []) if str(t).strip()]
    advice = [str(a).strip() for a in (data.get("advice") or []) if str(a).strip()]
    return {"overview": overview, "trends": trends[:n_trends], "advice": advice[:n_advice]}
