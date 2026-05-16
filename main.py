"""newsCollector v0.3 入口

用法:
  python main.py                              # 单次跑，窗口 = 上次运行时间 ~ 现在
  python main.py --window-hours 24            # 强制 24h 固定窗口（不看 last_run_at）
  python main.py --keywords NVIDIA,vLLM       # CLI 临时覆盖关键词
  python main.py --no-llm                     # 跳过 LLM，按客观分数出列表
  python main.py --all                        # 不限时间窗
  python main.py --top 30                     # --no-llm 时显示条数上限

定时模式（前台守护进程）:
  python main.py --daemon --at 09:00          # 每天 09:00 跑一次
  python main.py --daemon --at 09:00 --run-now  # 启动时先跑一次，再按计划

时间窗逻辑:
  - 默认 "since last_run_at"：每天 9:00 跑则窗口是「昨天 9:00 ~ 今天 9:00」
  - 首次运行无 last_run_at，回退到过去 24h
  - --window-hours N 强制固定 N 小时窗口，不更新 last_run_at

LLM 配置: config/llm.yaml + 环境变量 LLM_API_KEY
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from collectors.rss import fetch_all
from collectors.url_decoder import decode_items_inplace
from core.db import DB
from core.normalizer import normalize
from core.ranker import load_llm_config, rank_all, translate_all, verify_all
from core.secrets import load_secrets
from core.summary import clean_rss_summary

# 启动时加载本地持久化的密钥（TAVILY_API_KEY 等）
load_secrets()
from reporter.notify import load_notify_config, notify_all
from reporter.render import render_html, write_report


ROOT = Path(__file__).parent
DB_PATH = ROOT / "history.sqlite"
REPORTS_DIR = ROOT / "reports"
LAST_RUN_KEY = "last_run_at"


# ---------- 配置加载 ----------

def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_cli_keywords(s: str, config_keywords: list[dict]) -> list[dict]:
    """CLI 覆盖时按 name 在配置里找同名条目，合并继承 official_domains / aliases / case_sensitive。
    配置里没有的 keyword 用裸字段创建。"""
    by_name = {kw["name"].lower(): kw for kw in config_keywords}
    out = []
    for raw in s.split(","):
        n = raw.strip()
        if not n:
            continue
        found = by_name.get(n.lower())
        if found:
            out.append(dict(found))   # 浅拷贝继承全部字段
        else:
            out.append({"name": n, "aliases": []})
    return out


# ---------- 时间窗 ----------

def compute_since(db: DB, args) -> datetime | None:
    """决定时间窗起点。

    优先级：--all > --window-hours > 上次运行时间 > 24h 默认
    返回 None 表示不过滤。
    """
    if args.all:
        return None
    now = datetime.now(timezone.utc)
    if args.window_hours is not None:
        return now - timedelta(hours=args.window_hours)
    last = db.get_meta(LAST_RUN_KEY)
    if last:
        try:
            return datetime.fromisoformat(last)
        except ValueError:
            pass
    return now - timedelta(hours=24)


# ---------- 输出 ----------

def print_no_llm(items: list[dict], top: int):
    items = sorted(items, key=lambda x: x.get("score", 0), reverse=True)[:top]
    print(f"\nTop {len(items)} (按客观分数排序) ↓\n" + "=" * 80)
    for i, it in enumerate(items, 1):
        kws = ",".join(it["matched_keywords"])
        sig = it.get("signals", {})
        sig_str = " ".join(f"{k}={v}" for k, v in sig.items() if v) or "-"
        print(f"\n[{i}] score={it['score']:.2f}  [{kws}]")
        print(f"    {it['title']}")
        print(f"    {it['source']} | {it['published'][:16]} | {sig_str}")
        print(f"    {it['url']}")


def print_llm(grouped: dict[str, list[dict]]):
    total = sum(len(v) for v in grouped.values())
    print("\n" + "=" * 80)
    print(f"LLM 重要性筛选结果 —— 共 {total} 条")
    print("=" * 80)
    for kw, picks in grouped.items():
        print(f"\n【{kw}】 {len(picks)} 条")
        if not picks:
            print("  (无足够重要的资讯)")
            continue
        for j, it in enumerate(picks, 1):
            badge = " [官方]" if it.get("is_official") else ""
            cat = it.get("llm_category", "")
            cat_badge = f" [{cat}]" if cat else ""
            tag = it.get("summary_source", "?")
            if it.get("verified"):
                tag += "+自检改写"
            if it.get("translated"):
                tag += "+翻译"
            disp = it.get("display_title") or it["title"]
            print(f"  [{j}]{badge}{cat_badge} {disp}")
            if disp != it["title"]:
                print(f"      原标题: {it['title']}")
            print(f"      📄 ({tag}) {it.get('summary','')}")
            print(f"      {it['source']} | {it['published'][:16]}")
            print(f"      🔗 {it['url']}")


def flatten_to_render(grouped_raw: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """把 LLM 返回的 item 整理成模板需要的字段。"""
    out: dict[str, list[dict]] = {}
    for kw, picks in grouped_raw.items():
        out[kw] = [
            {
                "title": p["title"],
                "display_title": p.get("display_title") or p["title"],
                "url": p["url"],
                "source": p["source"],
                "published": p["published"],
                "summary": p.get("summary") or p.get("llm_summary") or "",
                "is_official": p.get("is_official", False),
                "summary_source": p.get("summary_source", ""),
                "verified": p.get("verified", False),
                "translated": p.get("translated", False),
                "llm_category": p.get("llm_category", ""),
            }
            for p in picks
        ]
    return out


# ---------- 主流程 ----------

def run_once(args) -> int:
    """跑一次完整流程，返回命中条数。"""
    sources = load_yaml(ROOT / "config" / "sources.yaml")["sources"]
    config_keywords = load_yaml(ROOT / "config" / "keywords.yaml")["keywords"]
    if args.keywords:
        keywords = parse_cli_keywords(args.keywords, config_keywords)
        merged = [k["name"] for k in keywords if k.get("official_domains")]
        extra = f" (从配置合并 official_domains: {merged})" if merged else ""
        print(f"使用 CLI 关键词: {[k['name'] for k in keywords]}{extra}")
    else:
        keywords = config_keywords
        print(f"使用配置文件关键词: {[k['name'] for k in keywords]}")

    db = DB(DB_PATH)
    since = compute_since(db, args)
    now = datetime.now(timezone.utc)
    if since is None:
        print(f"\n时间窗: 不限 (--all)")
    else:
        hours = (now - since).total_seconds() / 3600
        print(f"\n时间窗: {since.isoformat()[:16]} ~ {now.isoformat()[:16]}  ({hours:.1f}h)")

    print(f"\n[1/5] 抓取 {len(sources)} 个源...")
    raw = list(fetch_all(sources, keywords))
    print(f"  共抓到 {len(raw)} 条原始数据")

    print(f"\n[2/5] 去重 + 关键词匹配 + 客观打分（SQLite delta）...")
    items = normalize(raw, keywords, db=db, since=since)
    print(f"  命中关键词且在时窗内: {len(items)} 条")

    if args.no_llm:
        print("\n[3/5] 跳过 LLM 排序 (--no-llm)")
        print_no_llm(items, args.top)
    else:
        print(f"\n[3/5] LLM 按重要性筛选 (top_k 每关键词)...")
        llm_cfg = load_llm_config(ROOT / "config" / "llm.yaml")
        print(f"  模型: {llm_cfg['model']}  base_url: {llm_cfg.get('base_url') or 'OpenAI default'}")
        grouped_raw = rank_all(items, keywords, llm_cfg)

        # 摘要策略：原 RSS 摘要可用 → 直接用；不可用 → LLM 生成 + 二次自检
        rss_used = llm_used = 0
        for picks in grouped_raw.values():
            for p in picks:
                p["raw_rss_summary"] = p.get("summary", "")  # 保留原始供自检参考
                rss = clean_rss_summary(p["raw_rss_summary"])
                if rss:
                    p["summary"] = rss
                    p["summary_source"] = "rss"
                    rss_used += 1
                else:
                    p["summary"] = p.get("llm_summary", "")
                    p["summary_source"] = "llm"
                    llm_used += 1
            decode_items_inplace(picks)
        print(f"  [摘要] {rss_used} 条来自原 RSS，{llm_used} 条 LLM 生成")

        # 二次自检：仅对 LLM 生成的摘要核查推测/扩写
        if llm_used:
            verify_all(grouped_raw, llm_cfg)

        # 翻译：仅对 RSS 原文摘要且非中文的条目走一次 LLM 翻译
        if rss_used:
            translate_all(grouped_raw, llm_cfg)

        grouped = flatten_to_render(grouped_raw)

        print("\n[4/5] 控制台输出")
        print_llm(grouped)

        print("\n[5/5] 生成 HTML 报告 + 推送通知")
        win_from = since.isoformat()[:16] if since else "all"
        win_to = now.isoformat()[:16]
        html = render_html(grouped, win_from, win_to)
        out = write_report(html, REPORTS_DIR)
        print(f"  [HTML] 已保存: {out}")

        notify_cfg = load_notify_config(ROOT / "config" / "notify.yaml")
        date_str = now.strftime("%Y-%m-%d")
        notify_all(grouped, html, date_str, notify_cfg)

    # 仅在使用动态窗口时更新 last_run_at；--window-hours / --all 不更新
    if not args.all and args.window_hours is None:
        db.set_meta(LAST_RUN_KEY, now.isoformat())
    db.close()
    return len(items)


# ---------- daemon ----------

def run_daemon(args):
    import schedule

    def job():
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'#'*80}\n# 定时触发 @ {ts}\n{'#'*80}")
        try:
            run_once(args)
        except Exception as e:
            print(f"!! 本次运行失败: {e}")

    schedule.every().day.at(args.at).do(job)
    print(f"\n>>> daemon 已启动，每天 {args.at} 触发；Ctrl-C 退出")
    if args.run_now:
        job()
    while True:
        schedule.run_pending()
        time.sleep(20)


# ---------- 入口 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keywords", help="逗号分隔的关键词，覆盖配置文件")
    ap.add_argument("--all", action="store_true", help="不限时间窗")
    ap.add_argument("--no-llm", action="store_true", help="跳过 LLM 排序")
    ap.add_argument("--top", type=int, default=20, help="--no-llm 模式下显示前 N 条")
    ap.add_argument("--window-hours", type=float, default=None,
                    help="强制固定窗口（小时），不更新 last_run_at")
    ap.add_argument("--daemon", action="store_true", help="守护进程模式")
    ap.add_argument("--at", default="09:00", help="--daemon 模式下每日触发时间 HH:MM (24h)")
    ap.add_argument("--run-now", action="store_true",
                    help="--daemon 模式下启动时先跑一次")
    args = ap.parse_args()

    if args.daemon:
        run_daemon(args)
    else:
        run_once(args)


if __name__ == "__main__":
    main()
