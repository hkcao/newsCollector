"""newsCollector v0.3 入口

用法:
  python main.py                              # 单次跑，窗口 = 上次运行时间 ~ 现在
  python main.py --window-hours 24            # 强制 24h 固定窗口（不看 last_run_at）
  python main.py --keywords NVIDIA,vLLM       # CLI 临时覆盖关键词
  python main.py --no-llm                     # 跳过 LLM，按客观分数出列表
  python main.py --all                        # 不限时间窗
  python main.py --top 30                     # --no-llm 时显示条数上限
  python main.py --window-hours 24 --debug    # 记录每一步筛选前后内容到 reports/debug/

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
from core.timeutil import fmt_local
from core.ranker import (
    USAGE,
    load_llm_config,
    rank_all,
    reset_usage,
    translate_all,
    verify_all,
    verify_titles,
)
from core.digest import build_digest
from core.debug_trace import TRACE
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
    # 默认固定 24h 窗口：不再以 last_run_at 收窄，避免高频运行时各分类凑不齐数据
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
        print(f"    {it['source']} | {fmt_local(it['published'])} | {sig_str}")
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
            print(f"      {it['source']} | {fmt_local(it['published'])}")
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
                "cross_sources": (p.get("signals", {}) or {}).get("cross_sources")
                                 or ([p["source"]] if p.get("source") else []),
            }
            for p in picks
        ]
    return out


# ---------- 主流程 ----------

def run_once(args) -> int:
    """跑一次完整流程，返回命中条数。"""
    if getattr(args, "debug", False):
        TRACE.enable(REPORTS_DIR / "debug")
        print("  [DEBUG] 调试模式：记录每一步筛选前后内容，结束后落盘到 reports/debug/")

    sources = load_yaml(ROOT / "config" / "sources.yaml")["sources"]
    _kw_data = load_yaml(ROOT / "config" / "keywords.yaml")
    config_keywords = _kw_data["keywords"]
    watchlist = _kw_data.get("watchlist") or []
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
        # 内部存 UTC，显示一律转本地时区
        print(
            f"\n时间窗: {fmt_local(since)} ~ {fmt_local(now)} "
            f"({hours:.1f}h, 本地时区)"
        )

    # 语义主题门用：未命中关键词但来自媒体源的条目按内容召回（仅在用 LLM 时启用）
    themes = load_yaml(ROOT / "config" / "themes.yaml").get("themes", []) if (ROOT / "config" / "themes.yaml").exists() else []
    llm_cfg = None
    if not args.no_llm:
        reset_usage()  # 置于 normalize 之前，让主题门的 LLM 调用一并计入用量
        llm_cfg = load_llm_config(ROOT / "config" / "llm.yaml")

    print(f"\n[1/5] 抓取 {len(sources)} 个源...")
    raw = list(fetch_all(sources, keywords))
    print(f"  共抓到 {len(raw)} 条原始数据")
    TRACE.snapshot("fetch.raw", raw, note="各源抓取的原始条目（去重/时窗/匹配前）")

    print(f"\n[2/5] 去重 + 关键词匹配 + 语义主题门 + 客观打分（SQLite delta）...")
    items = normalize(raw, keywords, db=db, since=since, themes=themes,
                      llm_cfg=llm_cfg, watchlist=watchlist)
    print(f"  命中关键词/主题且在时窗内: {len(items)} 条")

    if args.no_llm:
        print("\n[3/5] 跳过 LLM 排序 (--no-llm)")
        print_no_llm(items, args.top)
    else:
        print(f"\n[3/5] LLM 按重要性筛选 (top_k 每关键词)...")
        print(f"  模型: {llm_cfg['model']}  base_url: {llm_cfg.get('base_url') or 'OpenAI default'}")
        grouped_raw = rank_all(items, keywords, llm_cfg)

        # 摘要策略：原 RSS 摘要可用 → 直接用；不可用 → LLM 生成 + 二次自检
        rss_used = llm_used = 0
        for picks in grouped_raw.values():
            for p in picks:
                p["raw_rss_summary"] = p.get("summary", "")  # 保留原始供自检参考
                rss = clean_rss_summary(p["raw_rss_summary"])
                # 意见领袖发言用 LLM 中文转述（含"为何值得知道"），不回退原始英文推文
                if rss and p.get("llm_category") != "AI意见领袖":
                    p["summary"] = rss
                    p["summary_source"] = "rss"
                    rss_used += 1
                else:
                    p["summary"] = p.get("llm_summary", "")
                    p["summary_source"] = "llm"
                    llm_used += 1
            decode_items_inplace(picks)
        print(f"  [摘要] {rss_used} 条来自原 RSS，{llm_used} 条 LLM 生成")
        TRACE.note("summary.strategy", f"摘要来源：RSS {rss_used} 条 / LLM 生成 {llm_used} 条")

        # 解码后二次过滤：Google News 此时才暴露真实域名，再过一遍黑名单
        from core.normalizer import is_blacklisted_url
        dropped_after_decode = 0
        _dbg_bl: list[tuple] = []
        for cat, picks in list(grouped_raw.items()):
            kept = []
            for p in picks:
                if is_blacklisted_url(p.get("url", "")):
                    dropped_after_decode += 1
                    if TRACE.enabled:
                        _dbg_bl.append((p, "解码后命中域名黑名单"))
                else:
                    kept.append(p)
            grouped_raw[cat] = kept
        if dropped_after_decode:
            print(f"  [二次过滤] 解码后丢弃 {dropped_after_decode} 条命中黑名单域名的 picks")
        TRACE.drops("post_decode.blacklist", _dbg_bl)
        # 去重：同一分类内若有 cross_source_count 大且标题相似的，仅保留客观分最高的
        _dbg_dup: list[tuple] = []
        for cat, picks in list(grouped_raw.items()):
            seen_titles = set()
            kept = []
            for p in sorted(picks, key=lambda x: -float(x.get("score", 0) or 0)):
                key = "".join(c for c in (p.get("display_title") or p.get("title") or "")[:20].lower() if c.isalnum() or "一" <= c <= "鿿")
                if key and key in seen_titles:
                    if TRACE.enabled:
                        _dbg_dup.append((p, "分类内标题近重复"))
                    continue
                seen_titles.add(key)
                kept.append(p)
            grouped_raw[cat] = kept
        TRACE.drops("post_rank.title_dedupe", _dbg_dup)

        # 二次自检：仅对 LLM 生成的摘要核查推测/扩写
        if llm_used:
            verify_all(grouped_raw, llm_cfg)

        # 翻译：仅对 RSS 原文摘要且非中文的条目走一次 LLM 翻译
        if rss_used:
            translate_all(grouped_raw, llm_cfg)

        # 标题反幻觉核查：检查数字 + 主体公司名是否在原文出现过
        print("  [标题核查] 检查所有 picks 的 display_title…")
        verify_titles(grouped_raw, llm_cfg, keywords=keywords)

        # 综合层：在 grouped_raw（带 signals/cross_sources）上做跨条目综合
        print("  [综合层] 生成 今日概览 / 趋势 / 建议…")
        digest = build_digest(grouped_raw, llm_cfg)
        TRACE.note("digest", "综合层输出（今日概览/趋势/建议）", data=digest)
        TRACE.snapshot("final.picks",
                       [p for picks in grouped_raw.values() for p in picks],
                       note="最终展示条目（摘要/翻译/标题核查后）")

        grouped = flatten_to_render(grouped_raw)

        print("\n[4/5] 控制台输出")
        print_llm(grouped)

        print("\n[5/5] 生成 HTML 报告 + 推送通知")
        win_from = fmt_local(since) if since else "all"
        win_to = fmt_local(now)
        html = render_html(grouped, win_from, win_to, digest)
        out = write_report(html, REPORTS_DIR)
        print(f"  [HTML] 已保存: {out}")

        notify_cfg = load_notify_config(ROOT / "config" / "notify.yaml")
        date_str = now.strftime("%Y-%m-%d")
        notify_all(grouped, html, date_str, notify_cfg, digest)

        print("\n" + USAGE.report())

    # 仅在使用动态窗口时更新 last_run_at；--window-hours / --all 不更新
    if not args.all and args.window_hours is None:
        db.set_meta(LAST_RUN_KEY, now.isoformat())
    db.close()

    if TRACE.enabled:
        html_path = TRACE.dump()
        print(f"\n  [DEBUG] 调试追踪已保存（可读 HTML + 完整 JSON）:\n"
              f"    open {html_path}\n    {html_path.with_suffix('.json')}")

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
    ap.add_argument("--debug", action="store_true",
                    help="调试模式：记录每一步筛选前后内容到 reports/debug/（HTML + JSON）")
    args = ap.parse_args()

    if args.daemon:
        run_daemon(args)
    else:
        run_once(args)


if __name__ == "__main__":
    main()
