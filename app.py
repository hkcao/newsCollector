"""newsCollector Streamlit GUI

启动:
    cd newsCollector
    .venv/bin/streamlit run app.py

页面:
    📰 抓取 & 结果    —— 触发一次抓取并查看结果
    ⚙️ 关键词配置     —— 增删关键词、同义词、官方域名
    🤖 LLM 配置       —— Base URL / 模型 / API Key
    🔔 通知设置       —— 邮件 / 飞书
    📚 历史报告       —— 浏览 reports/ 目录里的旧报告
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from core.db import DB                                  # noqa: E402
from core.pipeline import LAST_RUN_KEY, collect_news    # noqa: E402
from core.ranker import load_llm_config                 # noqa: E402
from reporter.notify import load_notify_config, notify_all  # noqa: E402
from reporter.render import render_html, write_report   # noqa: E402


CONFIG_DIR = ROOT / "config"
REPORTS_DIR = ROOT / "reports"
DB_PATH = ROOT / "history.sqlite"


# =========================================================================
# 配置文件读写工具
# =========================================================================

def load_yaml(path: Path, default: dict) -> dict:
    if path.exists():
        return yaml.safe_load(path.read_text(encoding="utf-8")) or default
    return default


def save_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


# =========================================================================
# 页面: 关键词配置
# =========================================================================

def page_keywords():
    st.header("⚙️ 关键词配置")
    st.caption("增删关键词，配置同义词与官方域名。保存后立即生效。")

    cfg = load_yaml(CONFIG_DIR / "keywords.yaml", {"keywords": []})
    keywords = cfg.get("keywords", [])

    if "kw_list" not in st.session_state:
        st.session_state.kw_list = [dict(k) for k in keywords]

    # 编辑现有关键词
    to_delete = None
    for idx, kw in enumerate(st.session_state.kw_list):
        with st.expander(f"🔖  {kw.get('name', '(未命名)')}", expanded=False):
            kw["name"] = st.text_input("名称", value=kw.get("name", ""), key=f"name_{idx}")
            kw["case_sensitive"] = st.checkbox(
                "大小写敏感",
                value=kw.get("case_sensitive", False),
                key=f"case_{idx}",
                help="多词专有名词建议开启（如 VAST Data / vLLM）以避免撞日常短语",
            )
            aliases_text = st.text_input(
                "同义词 (逗号分隔)",
                value=", ".join(kw.get("aliases") or []),
                key=f"alias_{idx}",
            )
            kw["aliases"] = [a.strip() for a in aliases_text.split(",") if a.strip()]

            domains_text = st.text_area(
                "官方域名 (每行一个)",
                value="\n".join(kw.get("official_domains") or []),
                key=f"dom_{idx}",
                height=80,
                help="全局已包含 arxiv.org / github.com，这里只填该关键词专属域名",
            )
            kw["official_domains"] = [d.strip() for d in domains_text.splitlines() if d.strip()]

            if st.button("🗑️ 删除", key=f"del_{idx}"):
                to_delete = idx

    if to_delete is not None:
        st.session_state.kw_list.pop(to_delete)
        st.rerun()

    st.divider()
    if st.button("➕ 添加关键词"):
        st.session_state.kw_list.append({"name": "新关键词", "aliases": [], "official_domains": []})
        st.rerun()

    if st.button("💾 保存所有更改", type="primary"):
        clean = []
        for kw in st.session_state.kw_list:
            if not kw.get("name", "").strip():
                continue
            entry = {"name": kw["name"].strip()}
            if kw.get("case_sensitive"):
                entry["case_sensitive"] = True
            if kw.get("aliases"):
                entry["aliases"] = kw["aliases"]
            if kw.get("official_domains"):
                entry["official_domains"] = kw["official_domains"]
            clean.append(entry)
        save_yaml(CONFIG_DIR / "keywords.yaml", {"keywords": clean})
        st.success(f"已保存 {len(clean)} 个关键词到 config/keywords.yaml")
        del st.session_state["kw_list"]


# =========================================================================
# 页面: LLM 配置
# =========================================================================

def page_llm():
    st.header("🤖 LLM 配置")
    st.caption("OpenAI 兼容接口：DeepSeek / Kimi / OpenAI 通用")

    cfg = load_yaml(CONFIG_DIR / "llm.yaml", {})

    presets = {
        "DeepSeek": ("https://api.deepseek.com/v1", "deepseek-chat"),
        "Kimi (Moonshot)": ("https://api.moonshot.cn/v1", "moonshot-v1-8k"),
        "OpenAI": ("https://api.openai.com/v1", "gpt-4o-mini"),
        "自定义": (cfg.get("base_url", ""), cfg.get("model", "")),
    }
    preset = st.selectbox("快速选择服务商", list(presets.keys()), index=3)
    if preset != "自定义":
        base_url_default, model_default = presets[preset]
    else:
        base_url_default = cfg.get("base_url", "")
        model_default = cfg.get("model", "")

    base_url = st.text_input("Base URL", value=base_url_default)
    model = st.text_input("Model", value=model_default)
    api_key = st.text_input(
        "API Key",
        value=cfg.get("api_key") or os.getenv("LLM_API_KEY", ""),
        type="password",
        help="也可通过环境变量 LLM_API_KEY 提供，环境变量优先级更高",
    )

    col1, col2 = st.columns(2)
    with col1:
        top_k = st.number_input(
            "每关键词最多输出条数 (top_k)",
            min_value=1, max_value=10,
            value=int(cfg.get("top_k_per_keyword", 2)),
        )
    with col2:
        max_cand = st.number_input(
            "送 LLM 的候选池上限",
            min_value=10, max_value=100,
            value=int(cfg.get("max_candidates_per_keyword", 40)),
        )

    if st.button("💾 保存 LLM 配置", type="primary"):
        new_cfg = {
            "base_url": base_url.strip(),
            "model": model.strip(),
            "top_k_per_keyword": int(top_k),
            "max_candidates_per_keyword": int(max_cand),
        }
        if api_key.strip():
            new_cfg["api_key"] = api_key.strip()
        save_yaml(CONFIG_DIR / "llm.yaml", new_cfg)
        st.success("已保存到 config/llm.yaml")

    st.divider()
    st.caption(
        "ℹ️ API Key 会以明文存到 `config/llm.yaml`。如担心泄露，"
        "可不填此处，改用环境变量 `LLM_API_KEY`。"
    )


# =========================================================================
# 页面: 通知设置
# =========================================================================

def page_notify():
    st.header("🔔 通知设置")
    cfg = load_yaml(CONFIG_DIR / "notify.yaml", {"email": {}, "feishu": {}})
    email_cfg = cfg.get("email") or {}
    feishu_cfg = cfg.get("feishu") or {}

    with st.expander("📧 邮件 (SMTP)", expanded=True):
        em_enabled = st.checkbox("启用邮件推送", value=email_cfg.get("enabled", False))
        em_host = st.text_input("SMTP 主机", value=email_cfg.get("smtp_host", "smtp.qq.com"))
        em_port = st.number_input("端口", value=int(email_cfg.get("smtp_port", 465)))
        em_ssl = st.checkbox("使用 SSL", value=email_cfg.get("use_ssl", True))
        em_user = st.text_input("登录用户名", value=email_cfg.get("username", ""))
        em_from = st.text_input("发件地址", value=email_cfg.get("from_addr", ""))
        em_to_raw = st.text_input("收件人 (逗号分隔)",
                                  value=", ".join(email_cfg.get("to_addrs", [])))
        em_subj = st.text_input("邮件主题前缀",
                                value=email_cfg.get("subject_prefix", "[newsCollector] 每日资讯"))
        em_pwd = st.text_input(
            "SMTP 授权码 / 应用密码",
            value=os.getenv("EMAIL_PASSWORD", ""),
            type="password",
            help="多数邮箱需用授权码而非登录密码。也可通过环境变量 EMAIL_PASSWORD 提供。",
        )

    with st.expander("🪶 飞书 webhook"):
        fs_enabled = st.checkbox("启用飞书推送", value=feishu_cfg.get("enabled", False))
        fs_mode = st.radio("消息格式", ["card", "text"],
                           index=0 if feishu_cfg.get("mode", "card") == "card" else 1,
                           horizontal=True)
        fs_webhook = st.text_input(
            "Webhook URL",
            value=os.getenv("FEISHU_WEBHOOK", ""),
            type="password",
            help="群机器人自定义 webhook，也可通过环境变量 FEISHU_WEBHOOK 提供",
        )

    if st.button("💾 保存通知配置", type="primary"):
        new_cfg = {
            "email": {
                "enabled": em_enabled,
                "smtp_host": em_host,
                "smtp_port": int(em_port),
                "use_ssl": em_ssl,
                "username": em_user,
                "from_addr": em_from,
                "to_addrs": [x.strip() for x in em_to_raw.split(",") if x.strip()],
                "subject_prefix": em_subj,
            },
            "feishu": {"enabled": fs_enabled, "mode": fs_mode},
        }
        save_yaml(CONFIG_DIR / "notify.yaml", new_cfg)
        # 密码 / webhook 写到本地 secrets 文件，避免污染主配置
        if em_pwd:
            os.environ["EMAIL_PASSWORD"] = em_pwd
        if fs_webhook:
            os.environ["FEISHU_WEBHOOK"] = fs_webhook
        st.success("通知配置已保存。密码/webhook 仅注入当前进程环境变量；"
                   "如需持久化，请在系统层 export。")


# =========================================================================
# 页面: 抓取 & 结果
# =========================================================================

def render_grouped(grouped: dict[str, list[dict]]):
    if not grouped:
        st.info("还没有结果。点击上方「运行抓取」试试。")
        return
    total = sum(len(v) for v in grouped.values())
    st.metric("LLM 选中条数", total)
    for kw, picks in grouped.items():
        st.subheader(f"【{kw}】 {len(picks)} 条")
        if not picks:
            st.markdown("> *无足够重要的资讯*")
            continue
        for p in picks:
            badges = []
            if p.get("is_official"):
                badges.append("`官方`")
            src = p.get("summary_source", "")
            if src == "rss":
                badges.append("`原文摘要`")
            elif src == "llm":
                badges.append("`LLM 生成`")
            if p.get("verified"):
                badges.append("`自检改写`")
            if p.get("translated"):
                badges.append("`已翻译`")
            badge_str = " ".join(badges)
            disp = p.get("display_title") or p["title"]
            st.markdown(f"#### [{disp}]({p['url']}) {badge_str}")
            if disp != p["title"]:
                st.caption(f"原标题: {p['title']}")
            st.caption(f"{p['source']} · {p['published'][:16]}")
            st.write(p.get("summary", ""))
            st.divider()


def page_run():
    st.header("📰 抓取 & 结果")

    # 显示上次运行时间
    if DB_PATH.exists():
        db = DB(DB_PATH)
        last = db.get_meta(LAST_RUN_KEY)
        db.close()
        if last:
            st.caption(f"上次运行: {last[:16]} (UTC)")

    with st.form("run_form"):
        col1, col2 = st.columns([2, 1])
        with col1:
            kw_override = st.text_input(
                "临时关键词覆盖 (逗号分隔，留空用配置文件)",
                value="",
                help="只影响本次抓取，不修改 keywords.yaml",
            )
        with col2:
            window_mode = st.selectbox(
                "时间窗",
                ["自动 (距上次运行)", "强制 24h", "强制 6h", "不限"],
            )
        use_llm = st.checkbox("启用 LLM 排序 + 自检", value=True)
        send_notify = st.checkbox("跑完后推送通知", value=False)
        submitted = st.form_submit_button("🚀 运行抓取", type="primary")

    if submitted:
        # 准备 keywords
        config_keywords = load_yaml(CONFIG_DIR / "keywords.yaml", {"keywords": []})["keywords"]
        if kw_override.strip():
            by_name = {k["name"].lower(): k for k in config_keywords}
            keywords = []
            for raw in kw_override.split(","):
                n = raw.strip()
                if not n:
                    continue
                keywords.append(dict(by_name.get(n.lower(), {"name": n, "aliases": []})))
        else:
            keywords = config_keywords
        if not keywords:
            st.error("没有可用关键词，请先在「关键词配置」页面添加。")
            return

        sources = load_yaml(CONFIG_DIR / "sources.yaml", {"sources": []})["sources"]

        llm_cfg = None
        if use_llm:
            try:
                llm_cfg = load_llm_config(CONFIG_DIR / "llm.yaml")
            except Exception as e:
                st.error(f"LLM 配置加载失败：{e}")
                return

        # 解析时间窗
        window_hours, use_all = None, False
        if window_mode == "强制 24h":
            window_hours = 24
        elif window_mode == "强制 6h":
            window_hours = 6
        elif window_mode == "不限":
            use_all = True

        log_lines: list[str] = []
        log_box = st.empty()

        def log(msg: str):
            log_lines.append(msg)
            log_box.code("\n".join(log_lines), language="text")

        with st.status("抓取中…", expanded=True) as status:
            try:
                result = collect_news(
                    keywords=keywords,
                    sources=sources,
                    llm_cfg=llm_cfg,
                    db_path=DB_PATH,
                    window_hours=window_hours,
                    use_all=use_all,
                    use_llm=use_llm,
                    update_last_run=(window_hours is None and not use_all),
                    log=log,
                )
                status.update(label="抓取完成 ✓", state="complete")
            except Exception as e:
                status.update(label=f"失败: {e}", state="error")
                st.exception(e)
                return

        st.session_state["last_result"] = result

        # 生成 HTML 报告 + 可选通知
        if result["grouped"]:
            from main import flatten_to_render  # 借用现成的扁平化
            flat = flatten_to_render(result["grouped"])
            html = render_html(flat, result["window_from"] or "all", result["window_to"])
            out = write_report(html, REPORTS_DIR)
            st.session_state["last_report_path"] = str(out)
            st.success(f"HTML 报告: {out}")

            if send_notify:
                cfg = load_notify_config(CONFIG_DIR / "notify.yaml")
                notify_all(flat, html, datetime.now().strftime("%Y-%m-%d"), cfg)
                st.success("通知已发送")

    # 即使没刚跑，也显示最近一次结果
    if "last_result" in st.session_state:
        st.divider()
        st.subheader("📋 本次结果")
        result = st.session_state["last_result"]
        stat = result["stats"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("原始抓取", stat["raw"])
        c2.metric("命中且在时窗内", stat["matched"])
        c3.metric("RSS / LLM 摘要", f"{stat['rss_used']} / {stat['llm_used']}")
        c4.metric("自检改写", stat["verify_fixed"])
        from main import flatten_to_render
        render_grouped(flatten_to_render(result["grouped"]))


# =========================================================================
# 页面: 历史报告
# =========================================================================

def page_history():
    st.header("📚 历史报告")
    if not REPORTS_DIR.exists():
        st.info("还没有任何报告。")
        return
    files = sorted(REPORTS_DIR.glob("*.html"), reverse=True)
    if not files:
        st.info("还没有任何报告。")
        return
    labels = [f.name for f in files]
    pick = st.selectbox("选择报告", labels)
    if pick:
        path = REPORTS_DIR / pick
        st.caption(f"路径: {path}")
        html = path.read_text(encoding="utf-8")
        st.components.v1.html(html, height=900, scrolling=True)


# =========================================================================
# 入口
# =========================================================================

st.set_page_config(page_title="newsCollector", page_icon="📰", layout="wide")
st.sidebar.title("📰 newsCollector")

PAGES = {
    "📰 抓取 & 结果": page_run,
    "⚙️ 关键词配置": page_keywords,
    "🤖 LLM 配置": page_llm,
    "🔔 通知设置": page_notify,
    "📚 历史报告": page_history,
}
choice = st.sidebar.radio("导航", list(PAGES.keys()))
st.sidebar.divider()
st.sidebar.caption("源代码: `/home/hank/Agent/newsCollector`")
PAGES[choice]()
