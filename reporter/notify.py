"""邮件 (SMTP) + 飞书 webhook 推送。任一渠道失败不影响其它渠道。"""
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


# ---------- 配置 ----------

def load_notify_config(path: Path) -> dict:
    if not path.exists():
        return {"email": {"enabled": False}, "feishu": {"enabled": False}}
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
                f"<font color='grey'>{p['source']} · {p['published'][:16]}</font>\n"
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


# ---------- 统一入口 ----------

def notify_all(
    grouped: dict[str, list[dict]],
    html: str,
    date: str,
    config: dict,
) -> None:
    email_cfg = config.get("email", {}) or {}
    feishu_cfg = config.get("feishu", {}) or {}

    if email_cfg.get("enabled"):
        subject = f"{email_cfg.get('subject_prefix', '[newsCollector]')} {date}"
        send_email(html, subject, email_cfg)
    if feishu_cfg.get("enabled"):
        send_feishu(grouped, date, feishu_cfg)
