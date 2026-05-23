"""企业微信群机器人 webhook 单测脚本。

用法：
    .venv/bin/python scripts/test_wecom.py
    .venv/bin/python scripts/test_wecom.py --report reports/2026-05-17-0941.html

不传 --report 时发一条占位测试消息验证 webhook 可达；
传 --report 时把指定 HTML 报告里的 grouped picks 反序列化（实际上重新调度最简单：
就解析 reports 同目录的最新 *.html 不可行，因为我们没存中间 json） → 改成发"测试卡片"。

WECOM_WEBHOOK 通过环境变量或 config/secrets.yaml 读取。
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# 让本脚本能从仓库根目录运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.secrets import load_secrets  # noqa: E402
from reporter.notify import load_notify_config, send_wecom  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="测试企业微信群机器人推送是否可达")
    ap.add_argument("--mode", choices=["markdown", "text"], default="markdown")
    args = ap.parse_args()

    load_secrets()
    cfg = load_notify_config(Path(__file__).resolve().parent.parent / "config" / "notify.yaml")
    wecom_cfg = (cfg.get("wecom") or {}).copy()
    wecom_cfg["enabled"] = True   # 测试时强制开
    wecom_cfg["mode"] = args.mode

    now = datetime.now()
    grouped = {
        "存储/基建产品": [
            {
                "display_title": "测试条目 1：企业微信群机器人接入验证",
                "title": "test entry 1",
                "summary": "这是一条测试消息，用于验证 NewsCollector 已经能通过企业微信群机器人推送资讯。"
                           "如果你在群里看到这条消息，说明 WECOM_WEBHOOK 配置正确。",
                "url": "https://example.com/test",
                "source": "test_wecom.py",
                "published": now.isoformat()[:16],
                "is_official": True,
            },
            {
                "display_title": "测试条目 2：markdown 与链接渲染",
                "title": "test entry 2",
                "summary": "标题应当是可点击的蓝色链接；正文支持加粗、行内代码等 markdown 元素。",
                "url": "https://github.com",
                "source": "test_wecom.py",
                "published": now.isoformat()[:16],
            },
        ],
        "模型/框架/算法": [],
        "基准评测": [],
        "学术论文": [],
        "政策导向": [],
        "github趋势": [],
    }

    print(f">>> 发送测试推送 (mode={args.mode}) …")
    ok = send_wecom(grouped, now.strftime("%Y-%m-%d %H:%M"), wecom_cfg)
    if ok:
        print("\n✅ 至少 1 块发送成功。请到企业微信群确认是否收到。")
    else:
        print("\n❌ 推送失败。请检查 WECOM_WEBHOOK 是否正确，或群机器人是否被移除。")
        sys.exit(1)


if __name__ == "__main__":
    main()
