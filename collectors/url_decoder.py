"""Google News 跳转 URL 解码 —— 把 base64 加密的中转链接还原为原始文章链接。

只对 LLM 选中的最终 item 调用，避免对 100+ 候选无谓请求。
"""
from __future__ import annotations

from googlenewsdecoder import gnewsdecoder


def _is_gnews_url(url: str) -> bool:
    return url.startswith("https://news.google.com/rss/articles/") or url.startswith(
        "https://news.google.com/articles/"
    )


def decode_gnews_url(url: str) -> str:
    """失败时回退原 URL；保留警告便于调试。"""
    if not _is_gnews_url(url):
        return url
    try:
        res = gnewsdecoder(url)
        if res and res.get("status") and res.get("decoded_url"):
            return res["decoded_url"]
        print(f"    [URL] 解码失败 (无返回): {url[:80]}...")
    except Exception as e:
        print(f"    [URL] 解码异常: {e}")
    return url


def decode_items_inplace(items: list[dict]) -> None:
    """原地把所有 item['url'] 替换成解码后的真实 URL（仅 Google News 域名）。"""
    for it in items:
        u = it.get("url", "")
        if _is_gnews_url(u):
            it["url"] = decode_gnews_url(u)
