"""邮件 (SMTP) + 飞书 webhook + 企业微信群机器人 推送。任一渠道失败不影响其它渠道。"""
from __future__ import annotations

import json
import os
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate
from pathlib import Path

import httpx
import yaml

from core.timeutil import fmt_local


# ---------- 配置 ----------

def load_notify_config(path: Path) -> dict:
    if not path.exists():
        return {"email": {"enabled": False}, "feishu": {"enabled": False}, "wecom": {"enabled": False}}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


# ---------- 邮件 ----------

def send_email(html: str, subject: str, cfg: dict) -> bool:
    """返回是否成功。失败时打印错误不抛异常。"""
    if not cfg.get("enabled"):
        return False
    pwd = os.getenv("EMAIL_PASSWORD")
    if not pwd:
        print("  [email] 跳过：未设置 EMAIL_PASSWORD 环境变量")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = cfg["from_addr"]
    msg["To"] = ", ".join(cfg["to_addrs"])
    msg["Date"] = formatdate(localtime=True)
    msg.attach(MIMEText(html, "html", "utf-8"))

    try:
        if cfg.get("use_ssl", True):
            with smtplib.SMTP_SSL(cfg["smtp_host"], cfg["smtp_port"],
                                  context=ssl.create_default_context()) as s:
                s.login(cfg["username"], pwd)
                s.sendmail(cfg["from_addr"], cfg["to_addrs"], msg.as_string())
        else:
            with smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"]) as s:
                s.starttls(context=ssl.create_default_context())
                s.login(cfg["username"], pwd)
                s.sendmail(cfg["from_addr"], cfg["to_addrs"], msg.as_string())
        print(f"  [email] 已发往 {cfg['to_addrs']}")
        return True
    except Exception as e:
        print(f"  [email] 失败: {e}")
        return False


# ---------- 飞书 ----------

def _feishu_text_payload(grouped: dict[str, list[dict]], date: str) -> dict:
    lines = [f"📰 每日资讯 · {date}"]
    for kw, picks in grouped.items():
        lines.append(f"\n【{kw}】 {len(picks)} 条")
        if not picks:
            lines.append("  - 无足够重要的资讯")
        for p in picks:
            lines.append(f"  • {p['title']}")
            lines.append(f"    {p['summary']}")
            lines.append(f"    {p['url']}")
    return {"msg_type": "text", "content": {"text": "\n".join(lines)}}


def _feishu_card_payload(grouped: dict[str, list[dict]], date: str) -> dict:
    """交互卡片：标题可点击跳转。"""
    elements = []
    for kw, picks in grouped.items():
        elements.append({
            "tag": "div",
            "text": {"tag": "lark_md", "content": f"**【{kw}】** {len(picks)} 条"},
        })
        if not picks:
            elements.append({
                "tag": "div",
                "text": {"tag": "plain_text", "content": "  无足够重要的资讯"},
            })
            continue
        for p in picks:
            md = (
                f"[**{p['title']}**]({p['url']})\n"
                f"<font color='grey'>{p['source']} · {fmt_local(p['published'])}</font>\n"
                f"{p['summary']}"
            )
            elements.append({"tag": "div", "text": {"tag": "lark_md", "content": md}})
        elements.append({"tag": "hr"})
    return {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": f"📰 每日资讯 · {date}"},
                "template": "blue",
            },
            "elements": elements,
        },
    }


def send_feishu(grouped: dict[str, list[dict]], date: str, cfg: dict) -> bool:
    if not cfg.get("enabled"):
        return False
    webhook = os.getenv("FEISHU_WEBHOOK")
    if not webhook:
        print("  [feishu] 跳过：未设置 FEISHU_WEBHOOK 环境变量")
        return False

    mode = cfg.get("mode", "card")
    payload = _feishu_card_payload(grouped, date) if mode == "card" \
        else _feishu_text_payload(grouped, date)
    try:
        r = httpx.post(webhook, json=payload, timeout=15)
        r.raise_for_status()
        body = r.json()
        if body.get("StatusCode", 0) != 0 and body.get("code", 0) != 0:
            print(f"  [feishu] 服务端返回异常: {body}")
            return False
        print(f"  [feishu] 已推送 ({mode})")
        return True
    except Exception as e:
        print(f"  [feishu] 失败: {e}")
        return False


# ---------- 企业微信群机器人 ----------

# 单条 markdown 消息字节上限（官方文档 4096，留点余量）
_WECOM_MD_LIMIT = 3800


def _wecom_markdown_chunks(grouped: dict[str, list[dict]], date: str) -> list[str]:
    """把每日资讯渲染成多块 markdown，单块不超过 _WECOM_MD_LIMIT 字节。
    按"分类 → 条目"切分；同一条目不拆分。"""
    chunks: list[str] = []
    cur = f"## 📰 每日 AI 存储技术资讯 · {date}\n"

    def flush():
        nonlocal cur
        if cur.strip():
            chunks.append(cur)
        cur = ""

    for cat, picks in grouped.items():
        block = f"\n**【{cat}】** {len(picks)} 条\n"
        if not picks:
            block += "> 无足够重要的资讯\n"
        else:
            for p in picks:
                title = p.get("display_title") or p.get("title") or ""
                summary = (p.get("summary") or "").strip()
                if len(summary) > 280:
                    summary = summary[:280].rstrip() + "…"
                source = p.get("source", "")
                ts = fmt_local(p.get("published"))
                url = p.get("url", "")
                official = "  `[官方]`" if p.get("is_official") else ""
                item_md = (
                    f"\n[**{title}**]({url}){official}\n"
                    f"<font color=\"comment\">{source} · {ts}</font>\n"
                    f"{summary}\n"
                )
                # 单条本身超长就硬截断
                if len(item_md.encode("utf-8")) > _WECOM_MD_LIMIT - 200:
                    item_md = item_md[: _WECOM_MD_LIMIT - 200] + "…\n"
                # 当前块装不下 → 先 flush
                if len((cur + block + item_md).encode("utf-8")) > _WECOM_MD_LIMIT:
                    if block:
                        cur += block
                        block = ""
                    flush()
                    block = f"\n**【{cat} · 续】**\n"
                block += item_md
        if len((cur + block).encode("utf-8")) > _WECOM_MD_LIMIT:
            flush()
        cur += block
    flush()
    return chunks


def _wecom_text_chunks(grouped: dict[str, list[dict]], date: str) -> list[str]:
    """纯文本 fallback —— 不支持 markdown 时使用。"""
    lines = [f"📰 每日 AI 存储技术资讯 · {date}"]
    for cat, picks in grouped.items():
        lines.append(f"\n【{cat}】 {len(picks)} 条")
        if not picks:
            lines.append("  无足够重要的资讯")
        for p in picks:
            title = p.get("display_title") or p.get("title") or ""
            lines.append(f"• {title}")
            lines.append(f"  {p.get('summary', '')[:200]}")
            lines.append(f"  {p.get('url', '')}")
    text = "\n".join(lines)
    # 拆成 ≤ 2000 字节块（text 类型限制 2048 字节）
    chunks, cur = [], ""
    for line in text.split("\n"):
        if len((cur + line + "\n").encode("utf-8")) > 1900:
            chunks.append(cur)
            cur = ""
        cur += line + "\n"
    if cur:
        chunks.append(cur)
    return chunks


def send_wecom(grouped: dict[str, list[dict]], date: str, cfg: dict) -> bool:
    if not cfg.get("enabled"):
        return False
    webhook = os.getenv("WECOM_WEBHOOK")
    if not webhook:
        print("  [wecom] 跳过：未设置 WECOM_WEBHOOK 环境变量")
        return False

    mode = cfg.get("mode", "markdown")
    max_chunks = int(cfg.get("max_chunks", 6))
    if mode == "text":
        chunks = _wecom_text_chunks(grouped, date)
    else:
        chunks = _wecom_markdown_chunks(grouped, date)

    if len(chunks) > max_chunks:
        print(f"  [wecom] 内容拆为 {len(chunks)} 块，超过 max_chunks={max_chunks}，截断尾部")
        chunks = chunks[:max_chunks]

    ok = 0
    for i, chunk in enumerate(chunks, 1):
        if mode == "text":
            payload = {"msgtype": "text", "text": {"content": chunk}}
        else:
            payload = {"msgtype": "markdown", "markdown": {"content": chunk}}
        try:
            r = httpx.post(webhook, json=payload, timeout=15)
            r.raise_for_status()
            body = r.json()
            if body.get("errcode", 0) != 0:
                print(f"  [wecom] 块 {i}/{len(chunks)} 服务端异常: {body}")
                continue
            ok += 1
        except Exception as e:
            print(f"  [wecom] 块 {i}/{len(chunks)} 失败: {e}")
    print(f"  [wecom] 已推送 {ok}/{len(chunks)} 块 ({mode})")
    return ok > 0


# ---------- 统一入口 ----------

def notify_all(
    grouped: dict[str, list[dict]],
    html: str,
    date: str,
    config: dict,
) -> None:
    email_cfg = config.get("email", {}) or {}
    feishu_cfg = config.get("feishu", {}) or {}
    wecom_cfg = config.get("wecom", {}) or {}

    if email_cfg.get("enabled"):
        subject = f"{email_cfg.get('subject_prefix', '[newsCollector]')} {date}"
        send_email(html, subject, email_cfg)
    if feishu_cfg.get("enabled"):
        send_feishu(grouped, date, feishu_cfg)
    if wecom_cfg.get("enabled"):
        send_wecom(grouped, date, wecom_cfg)
