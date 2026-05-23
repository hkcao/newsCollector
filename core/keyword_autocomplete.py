"""通过 LLM 给关键词自动补全同义词和官方域名。

输入：单个关键词名（如 "PureStorage" / "阿里"）
输出：{name, aliases[], official_domains[], case_sensitive}
"""
from __future__ import annotations

import json
import re

from openai import OpenAI


_PROMPT = """你是新闻抓取系统的关键词配置助手。
对下面这个关键词（通常是公司 / 产品 / 技术名），输出它的同义词列表和官方域名，供 RSS 关键词匹配和官方源识别使用。

关键词："{name}"

请给出：
1. **aliases**：所有应当被视为「等价」的写法，便于跨源去重和扩大召回。包含：
   - 中文↔英文别名（NVIDIA ↔ 英伟达）
   - 股票代码 / 缩写（NVDA, BABA, TCEHY）
   - 主要产品 / 旗下大模型 / 关键型号（H100 / H200 / Blackwell 对 NVIDIA；Qwen / 通义千问 对 Alibaba）
   - 子品牌 / 云业务名（阿里云、火山引擎、腾讯云）
2. **official_domains**：公司 / 项目的官方网站域名（裸域名，不含 https:// 和路径）。包括主站、官方博客、开发者站点、官方 hf/github 组织页。
3. **case_sensitive**：当关键词含小写专有名词（vLLM、VAST Data）或非常短易撞日常词时设 true；普通公司名设 false。

仅返回 JSON，无任何其它输出，格式：
{{"aliases": ["..."], "official_domains": ["..."], "case_sensitive": false}}"""


def _clean_domain(d: str) -> str:
    d = d.strip().lower()
    d = re.sub(r"^https?://", "", d)
    d = d.split("/", 1)[0]
    d = d.rstrip(".")
    return d


def autocomplete_keyword(name: str, llm_cfg: dict) -> dict:
    """调用 LLM 补全单个关键词。失败时抛出异常，由调用方处理。"""
    client = OpenAI(api_key=llm_cfg["api_key"], base_url=llm_cfg.get("base_url"))
    resp = client.chat.completions.create(
        model=llm_cfg["model"],
        messages=[{"role": "user", "content": _PROMPT.format(name=name.strip())}],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    text = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}

    aliases = [str(a).strip() for a in (data.get("aliases") or []) if str(a).strip()]
    # 去重保序；剔除与 name 完全相同的 alias
    seen = set()
    aliases_dedup = []
    for a in aliases:
        if a.lower() == name.strip().lower() or a.lower() in seen:
            continue
        seen.add(a.lower())
        aliases_dedup.append(a)

    domains = [_clean_domain(d) for d in (data.get("official_domains") or []) if str(d).strip()]
    domains = list(dict.fromkeys(d for d in domains if d))  # 去重保序

    entry: dict = {"name": name.strip()}
    if bool(data.get("case_sensitive", False)):
        entry["case_sensitive"] = True
    if aliases_dedup:
        entry["aliases"] = aliases_dedup
    if domains:
        entry["official_domains"] = domains
    return entry
