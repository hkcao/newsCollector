"""从原始 RSS 中提取可用的摘要文本 —— 优先用原文，避免 LLM 主观扩写。

可用 = 不是元数据/占位符 + 有实际文本内容（≥ 60 字符）。
- arXiv: summary 是论文 abstract，直接可用，质量最高
- Reddit 文本帖: summary 是帖子正文，多数可用
- HN RSS: summary = "Article URL/Points/Comments URL/# Comments"，全是元数据，不可用
- Google News: summary 是 `<a>源名</a>` 占位符，不可用
- 一般博客 RSS: summary 是首段，可用
"""
from __future__ import annotations

import html
import re


# HN summary 的特征：必带 "Article URL:" 或 "Points:" 或 "Comments URL:"
_HN_METADATA_RE = re.compile(
    r"Article URL:|Comments URL:|# Comments:", re.IGNORECASE
)

# Google News 的 summary 多为单个 <a> 链接 + 来源名，没有正文
_GNEWS_PLACEHOLDER_RE = re.compile(
    r"^\s*<a [^>]*href=[\"'][^\"']+[\"'][^>]*>[^<]+</a>\s*(&nbsp;|\s)*<font", re.I
)

# 网站侧边栏的促销 / 礼物 / "best of" 列表 —— Yahoo Finance / MSN 这类聚合站常把
# sidebar 的内容当成 RSS summary 返回，必须识别为噪声直接丢弃
# 例：- 给嫂子的最佳礼物 / - 给她的最佳生日礼物 / Best gifts for ...
_PROMO_NOISE_PATTERNS = (
    r"给[^。\n]{1,8}的最佳",        # 给嫂子的最佳礼物
    r"最佳礼物",
    r"最佳生日礼物",
    r"结婚纪念日礼物",
    r"best (?:gift|gifts) for",
    r"best (?:birthday|anniversary|holiday) gift",
    r"top \d+ (?:gifts|deals|products)",
    r"sponsored content",
    r"advertisement",
    r"affiliate (?:link|disclosure)",
)
_PROMO_NOISE_RE = re.compile("|".join(_PROMO_NOISE_PATTERNS), re.IGNORECASE)


def _looks_like_promo_noise(text: str) -> bool:
    """如果文本里命中多次"礼物/最佳/sponsored"等模式，视为侧边栏噪声。"""
    if not text:
        return False
    matches = _PROMO_NOISE_RE.findall(text)
    return len(matches) >= 2  # 命中 2 次以上才算 —— 单次提及无害

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_REDDIT_PREFIX_RE = re.compile(r"^submitted by\s+/u/\S+\s*", re.I)
# arXiv RSS 摘要前缀: "arXiv:2605.09370v1 Announce Type: cross Abstract: <正文>"
_ARXIV_PREFIX_RE = re.compile(
    r"^arXiv:\S+\s+Announce Type:\s+\S+\s+Abstract:\s+", re.I
)


MIN_USABLE_CHARS = 60
# 800 字符约等于一篇完整的 arxiv abstract / 一段完整的官方博客引子段；
# 给做技术规划的用户足够多上下文，避免摘要切在 "无需……" 这种半句话上
MAX_SUMMARY_CHARS = 800


def clean_rss_summary(raw: str) -> str | None:
    """提取干净的正文文本，返回 None 表示原文 RSS 没有可用摘要。"""
    if not raw:
        return None
    # 元数据 / 占位符直接判失败
    if _HN_METADATA_RE.search(raw):
        return None
    if _GNEWS_PLACEHOLDER_RE.search(raw):
        return None

    text = _HTML_TAG_RE.sub(" ", raw)
    text = html.unescape(text)
    text = _WS_RE.sub(" ", text).strip()
    text = _REDDIT_PREFIX_RE.sub("", text)
    text = _ARXIV_PREFIX_RE.sub("", text)

    if len(text) < MIN_USABLE_CHARS:
        return None

    # 侧边栏促销噪声：Yahoo Finance / MSN 经常把"最佳礼物"列表当作 summary 推出来
    if _looks_like_promo_noise(text):
        return None

    # 太长时截到 MAX_SUMMARY_CHARS 并尽量收在句号
    if len(text) > MAX_SUMMARY_CHARS:
        cut = text[:MAX_SUMMARY_CHARS]
        # 尝试在最后一个句末符之前切
        for sep in ("。", ". ", "！", "? ", "？"):
            idx = cut.rfind(sep)
            if idx >= MIN_USABLE_CHARS:
                return cut[: idx + len(sep)].strip()
        return cut.rstrip() + "…"
    return text
