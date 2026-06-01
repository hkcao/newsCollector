"""newsCollector Streamlit GUI

启动:
    cd newsCollector
    .venv/bin/streamlit run app.py

页面:
    📰 抓取 & 结果    —— 触发一次抓取并查看结果
    ⚙️ 关键词配置     —— 增删关键词、同义词、官方域名
    🌐 数据源        —— 启用/禁用 RSS 源、Tavily Search API Key
    🤖 LLM 配置       —— Base URL / 模型 / API Key
    ✏️ Prompt 设置    —— 编辑 IMPORTANCE_RUBRIC 与各分类定义（写入 prompt_override.yaml）
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
from core.secrets import load_secrets, save_secret      # noqa: E402
from reporter.notify import load_notify_config, notify_all  # noqa: E402
from reporter.render import render_html, write_report   # noqa: E402

# 启动时把 config/secrets.yaml 里的密钥注入环境变量（不覆盖系统层 export）
load_secrets()


CONFIG_DIR = ROOT / "config"
REPORTS_DIR = ROOT / "reports"
DB_PATH = ROOT / "history.sqlite"

MAX_PER_CATEGORY = 5


def cap_per_category(grouped: dict[str, list[dict]], max_per_cat: int = MAX_PER_CATEGORY) -> dict[str, list[dict]]:
    """跨关键词，按 LLM 标记的 category 裁剪：每类最多保留 max_per_cat 条。
    保留优先级：官方源 > 客观分数 > 关键词命中数。"""
    ranked: list[tuple[str, dict]] = []
    for kw, picks in grouped.items():
        for p in picks:
            ranked.append((kw, p))
    ranked.sort(key=lambda x: (
        not x[1].get("is_official", False),
        -float(x[1].get("score", 0) or 0),
    ))
    counts: dict[str, int] = {}
    kept_ids: set[int] = set()
    for _, p in ranked:
        cat = p.get("llm_category") or "(未分类)"
        if counts.get(cat, 0) < max_per_cat:
            counts[cat] = counts.get(cat, 0) + 1
            kept_ids.add(id(p))
    return {kw: [p for p in picks if id(p) in kept_ids] for kw, picks in grouped.items()}


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

def _save_keywords(keywords: list[dict]) -> None:
    """规范化后保存到 keywords.yaml。"""
    clean = []
    for kw in keywords:
        name = (kw.get("name") or "").strip()
        if not name:
            continue
        entry: dict = {"name": name}
        if kw.get("case_sensitive"):
            entry["case_sensitive"] = True
        if kw.get("aliases"):
            entry["aliases"] = list(kw["aliases"])
        if kw.get("official_domains"):
            entry["official_domains"] = list(kw["official_domains"])
        clean.append(entry)
    save_yaml(CONFIG_DIR / "keywords.yaml", {"keywords": clean})


def page_keywords():
    from core.keyword_autocomplete import autocomplete_keyword
    from core.ranker import load_llm_config

    st.header("⚙️ 关键词配置")
    st.caption("输入关键词名称即可，同义词和官方域名由 LLM 自动补全。")

    cfg = load_yaml(CONFIG_DIR / "keywords.yaml", {"keywords": []})
    keywords: list[dict] = list(cfg.get("keywords", []))

    # ---------- 添加新关键词 ----------
    st.subheader("➕ 添加关键词")
    new_input = st.text_area(
        "每行一个关键词（公司 / 产品 / 技术名）",
        placeholder="例如：\nPureStorage\n阿里\n腾讯\n字节跳动\n百度",
        height=110,
        key="new_kw_input",
    )
    auto_complete = st.checkbox("通过 LLM 自动补全同义词 + 官方域名", value=True)
    if st.button("✚ 添加", type="primary"):
        names = [n.strip() for n in new_input.splitlines() if n.strip()]
        existing = {kw.get("name", "").lower() for kw in keywords}
        llm_cfg = None
        if auto_complete:
            try:
                llm_cfg = load_llm_config(CONFIG_DIR / "llm.yaml")
            except Exception as e:
                st.warning(f"LLM 未就绪：{e} —— 将以裸名称保存，不做补全")
                llm_cfg = None
        added, skipped = 0, []
        for name in names:
            if name.lower() in existing:
                skipped.append(name)
                continue
            if llm_cfg:
                with st.spinner(f"LLM 补全「{name}」…"):
                    try:
                        entry = autocomplete_keyword(name, llm_cfg)
                    except Exception as e:
                        st.error(f"「{name}」补全失败：{e}")
                        entry = {"name": name}
            else:
                entry = {"name": name}
            keywords.append(entry)
            existing.add(name.lower())
            added += 1
        if added:
            _save_keywords(keywords)
            msg = f"已添加 {added} 个关键词"
            if skipped:
                msg += f"，跳过已存在：{', '.join(skipped)}"
            st.success(msg)
            st.rerun()
        elif skipped:
            st.info(f"全部已存在：{', '.join(skipped)}")

    st.divider()

    # ---------- 现有关键词列表 ----------
    st.subheader(f"📋 当前 {len(keywords)} 个关键词")
    if not keywords:
        st.info("还没有关键词，先在上面添加。")
        return

    for idx, kw in enumerate(keywords):
        name = kw.get("name", "(未命名)")
        n_alias = len(kw.get("aliases") or [])
        n_dom = len(kw.get("official_domains") or [])
        with st.expander(
            f"**{name}**  ·  同义词 {n_alias}  ·  官方域名 {n_dom}",
            expanded=False,
        ):
            st.caption("**同义词**")
            st.write(", ".join(kw.get("aliases") or []) or "_(无)_")
            st.caption("**官方域名**")
            st.write(", ".join(kw.get("official_domains") or []) or "_(无)_")
            if kw.get("case_sensitive"):
                st.caption("⚠️ 大小写敏感")

            col_a, col_b, col_c = st.columns([1, 1, 4])
            if col_a.button("🔄 重新补全", key=f"refresh_{idx}",
                            help="通过 LLM 重新生成同义词和官方域名（覆盖现有）"):
                try:
                    llm_cfg = load_llm_config(CONFIG_DIR / "llm.yaml")
                    with st.spinner(f"重新补全「{name}」…"):
                        new_entry = autocomplete_keyword(name, llm_cfg)
                    keywords[idx] = new_entry
                    _save_keywords(keywords)
                    st.success("已更新")
                    st.rerun()
                except Exception as e:
                    st.error(f"补全失败：{e}")
            if col_b.button("🗑️ 删除", key=f"del_{idx}"):
                keywords.pop(idx)
                _save_keywords(keywords)
                st.rerun()


# =========================================================================
# 页面: 数据源
# =========================================================================

SOURCE_TYPE_LABEL = {
    "rss": "RSS 直拉",
    "keyword_search": "关键词搜索 (RSS)",
    "tavily": "Tavily Search API",
}


def page_sources():
    st.header("🌐 数据源")
    st.caption("配置 sources.yaml：启用/禁用源、设置 Tavily Search API Key。修改后立即生效。")

    cfg = load_yaml(CONFIG_DIR / "sources.yaml", {"sources": []})
    sources = cfg.get("sources", [])

    # --- Tavily 状态 ---
    with st.expander("🔑 Tavily Search API Key", expanded=True):
        st.markdown(
            "Tavily 提供接近浏览器搜索的覆盖度（含官方/中文/非主流站），"
            "免费档约 1000 次/月。在 https://tavily.com 注册即可拿到 key（以 `tvly-` 开头）。"
        )
        current = os.getenv("TAVILY_API_KEY", "")
        if current:
            st.success(f"已设置：`tvly-...{current[-4:]}`")
        else:
            st.warning("未设置。Tavily 源会被自动跳过，不影响其他源。")
        tk = st.text_input(
            "TAVILY_API_KEY",
            value=current,
            type="password",
            help="保存后写入 config/secrets.yaml（已 gitignore），下次启动自动加载。",
        )
        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("💾 保存并持久化", type="primary"):
                save_secret("TAVILY_API_KEY", tk.strip())
                if tk.strip():
                    st.success("已写入 config/secrets.yaml + 注入当前进程，立即生效")
                else:
                    st.info("已清除")
                st.rerun()
        with col_b:
            if st.button("仅本次有效（不写盘）"):
                if tk.strip():
                    os.environ["TAVILY_API_KEY"] = tk.strip()
                    st.success("已注入当前进程；重启 Streamlit 后失效")
                else:
                    os.environ.pop("TAVILY_API_KEY", None)
                    st.info("已清除")
                st.rerun()

    st.divider()
    if "src_list" not in st.session_state:
        st.session_state.src_list = [dict(s) for s in sources]

    # ---- 添加新源 ----
    with st.expander("➕ 添加新源", expanded=False):
        st.caption("RSS 直接拉一个固定 URL；关键词搜索 URL 含 `{q}` 占位符；Tavily 用 API。")
        new_type = st.selectbox(
            "类型",
            ["rss", "keyword_search", "tavily"],
            format_func=lambda x: SOURCE_TYPE_LABEL.get(x, x),
            key="new_src_type",
        )
        new_name = st.text_input("名称", value="", key="new_src_name",
                                 placeholder="如：HuggingFace Papers / MIT News")
        new_category = st.text_input("分类标签 (可空)", value="",
                                     key="new_src_cat",
                                     placeholder="如：official / news / paper")
        new_url = ""
        if new_type in ("rss", "keyword_search"):
            ph = ("https://example.com/feed.xml" if new_type == "rss"
                  else "https://example.com/search?q={q}")
            new_url = st.text_input("URL", value="", key="new_src_url", placeholder=ph)
        elif new_type == "tavily":
            st.caption("Tavily 源走 TAVILY_API_KEY，无需 URL；下面的参数等会儿可在源列表中再调。")

        if st.button("✚ 添加到列表"):
            if not new_name.strip():
                st.error("请填写名称")
            elif new_type in ("rss", "keyword_search") and not new_url.strip():
                st.error("请填写 URL")
            elif new_type == "keyword_search" and "{q}" not in new_url:
                st.error("关键词搜索 URL 必须包含 `{q}` 占位符")
            else:
                entry = {
                    "name": new_name.strip(),
                    "type": new_type,
                    "enabled": True,
                }
                if new_category.strip():
                    entry["category"] = new_category.strip()
                if new_type in ("rss", "keyword_search"):
                    entry["url"] = new_url.strip()
                elif new_type == "tavily":
                    entry.update({
                        "api_key_env": "TAVILY_API_KEY",
                        "days": 2, "max_results": 10, "topic": "news",
                    })
                st.session_state.src_list.append(entry)
                st.success(f"已加入「{entry['name']}」—— 别忘了点下面的「💾 保存源配置」")
                st.rerun()

    st.subheader("源列表")
    st.caption("勾选 = 启用。Tavily 源即使勾选，没设 Key 也会被跳过。")

    # 按类型分组渲染
    to_delete_src = None
    for idx, s in enumerate(st.session_state.src_list):
        stype = s.get("type", "rss")
        type_label = SOURCE_TYPE_LABEL.get(stype, stype)
        cat = s.get("category", "")
        enabled = s.get("enabled", True)
        title = f"{'✅' if enabled else '⬜'} {s.get('name','(未命名)')}"
        with st.expander(f"{title}  ·  {type_label}  ·  {cat}"):
            s["enabled"] = st.checkbox("启用", value=enabled, key=f"src_en_{idx}")
            s["name"] = st.text_input("名称", value=s.get("name", ""), key=f"src_nm_{idx}")
            if stype in ("rss", "keyword_search"):
                s["url"] = st.text_input(
                    "URL" + (" (含 {q} 占位符)" if stype == "keyword_search" else ""),
                    value=s.get("url", ""),
                    key=f"src_url_{idx}",
                )
            elif stype == "tavily":
                col1, col2, col3 = st.columns(3)
                with col1:
                    s["days"] = st.number_input(
                        "查询近 N 天", min_value=1, max_value=30,
                        value=int(s.get("days", 2)), key=f"src_days_{idx}",
                    )
                with col2:
                    s["max_results"] = st.number_input(
                        "每关键词最多返回", min_value=1, max_value=50,
                        value=int(s.get("max_results", 10)), key=f"src_max_{idx}",
                    )
                with col3:
                    s["topic"] = st.selectbox(
                        "topic", ["news", "general"],
                        index=0 if s.get("topic", "news") == "news" else 1,
                        key=f"src_topic_{idx}",
                    )
                st.caption(
                    f"使用环境变量 `{s.get('api_key_env','TAVILY_API_KEY')}`。"
                    "若未设置，本源会被跳过。"
                )
            if st.button("🗑️ 删除此源", key=f"src_del_{idx}"):
                to_delete_src = idx

    if to_delete_src is not None:
        st.session_state.src_list.pop(to_delete_src)
        st.rerun()

    st.divider()
    if st.button("💾 保存源配置", type="primary"):
        clean = []
        for s in st.session_state.src_list:
            entry = {
                "name": s.get("name", "").strip() or "未命名",
                "type": s.get("type", "rss"),
            }
            if s.get("enabled", True) is False:
                entry["enabled"] = False
            if "url" in s and s["url"]:
                entry["url"] = s["url"].strip()
            if s.get("category"):
                entry["category"] = s["category"]
            if entry["type"] == "tavily":
                entry["api_key_env"] = s.get("api_key_env", "TAVILY_API_KEY")
                entry["days"] = int(s.get("days", 2))
                entry["max_results"] = int(s.get("max_results", 10))
                entry["topic"] = s.get("topic", "news")
            clean.append(entry)
        save_yaml(CONFIG_DIR / "sources.yaml", {"sources": clean})
        st.success(f"已保存 {len(clean)} 个源（启用 {sum(1 for c in clean if c.get('enabled', True) is not False)} 个）")
        del st.session_state["src_list"]


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
            "每个分类最多输出条数 (top_k_per_category)",
            min_value=1, max_value=10,
            value=int(cfg.get("top_k_per_category", cfg.get("top_k_per_keyword", 5))),
            help="排序单元已从「关键词」改为「分类」，每个分类独立挑 top_k 条",
        )
    with col2:
        max_cand = st.number_input(
            "送 LLM 的候选池上限（每分类）",
            min_value=10, max_value=100,
            value=int(cfg.get("max_candidates_per_category",
                              cfg.get("max_candidates_per_keyword", 40))),
        )

    if st.button("💾 保存 LLM 配置", type="primary"):
        new_cfg = {
            "base_url": base_url.strip(),
            "model": model.strip(),
            "top_k_per_category": int(top_k),
            "max_candidates_per_category": int(max_cand),
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
    cfg = load_yaml(CONFIG_DIR / "notify.yaml", {"email": {}, "feishu": {}, "wecom": {}})
    email_cfg = cfg.get("email") or {}
    feishu_cfg = cfg.get("feishu") or {}
    wecom_cfg = cfg.get("wecom") or {}

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
            help="群机器人自定义 webhook。保存后写入 config/secrets.yaml（已 gitignore）",
        )

    with st.expander("💼 企业微信群机器人", expanded=True):
        wc_enabled = st.checkbox(
            "启用企业微信推送",
            value=wecom_cfg.get("enabled", False),
            key="wc_enabled",
        )
        wc_mode = st.radio(
            "消息格式",
            ["markdown", "text"],
            index=0 if wecom_cfg.get("mode", "markdown") == "markdown" else 1,
            horizontal=True,
            help="markdown 支持加粗 / 可点击链接（推荐），text 是纯文本兜底",
        )
        wc_max = st.number_input(
            "最大消息块数",
            min_value=1, max_value=20,
            value=int(wecom_cfg.get("max_chunks", 6)),
            help="单条 markdown 上限 4096 字节，超出会自动分块。机器人 1 分钟限 20 条",
        )
        wc_webhook = st.text_input(
            "Webhook URL",
            value=os.getenv("WECOM_WEBHOOK", ""),
            type="password",
            help="获取：企业微信群 → 群设置 → 群机器人 → 添加 → 自定义机器人 → 复制 URL。"
                 "保存后写入 config/secrets.yaml（已 gitignore），守护进程也能读到。",
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
            "wecom": {
                "enabled": wc_enabled,
                "mode": wc_mode,
                "max_chunks": int(wc_max),
            },
        }
        save_yaml(CONFIG_DIR / "notify.yaml", new_cfg)
        # 凭据持久化到 secrets.yaml（已 gitignore），守护进程 / cron 也能读到
        if em_pwd:
            save_secret("EMAIL_PASSWORD", em_pwd)
        if fs_webhook:
            save_secret("FEISHU_WEBHOOK", fs_webhook)
        if wc_webhook:
            save_secret("WECOM_WEBHOOK", wc_webhook)
        st.success("通知配置已保存到 config/notify.yaml；凭据写入 config/secrets.yaml")


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
        # 用 markdown 普通粗体而非 st.subheader，避免与条目标题层级断崖
        st.markdown(f"**【{kw}】** &nbsp; *{len(picks)} 条*")
        if not picks:
            st.markdown("> *无足够重要的资讯*")
            continue
        for p in picks:
            badges = []
            if p.get("is_official"):
                badges.append("`官方`")
            cat = p.get("llm_category") or ""
            if cat:
                badges.append(f"`{cat}`")
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
            # 不再用 #### 标题（H4 字号过大），改普通粗体链接，整体字号统一
            st.markdown(f"**[{disp}]({p['url']})** &nbsp; {badge_str}")
            if disp != p["title"]:
                st.caption(f"原标题: {p['title']}")
            st.caption(f"{p['source']} · {p['published'][:16]}")
            st.write(p.get("summary", ""))
            st.divider()


from core.ranker import CATEGORIES  # noqa: E402


def render_by_category(grouped: dict[str, list[dict]]):
    """跨关键词，按 LLM 标记的类别聚合呈现。每个分类内按 (官方源, 客观分数) 排序。"""
    bucket: dict[str, list[tuple[str, dict]]] = {c: [] for c in CATEGORIES}
    bucket["(未分类)"] = []
    for kw, picks in grouped.items():
        for p in picks:
            c = p.get("llm_category") or "(未分类)"
            bucket.setdefault(c, []).append((kw, p))
    # 分类内排序：官方源在前，再按客观 score 降序
    for c in bucket:
        bucket[c].sort(key=lambda x: (
            not x[1].get("is_official", False),
            -float(x[1].get("score", 0) or 0),
        ))
    total = sum(len(v) for v in bucket.values())
    if total == 0:
        st.info("还没有结果。")
        return
    st.metric("LLM 选中条数（跨关键词）", total)
    for cat, entries in bucket.items():
        if not entries:
            continue
        st.markdown(f"**🏷️ {cat}** &nbsp; *{len(entries)} 条*")
        for kw, p in entries:
            badges = []
            if p.get("is_official"):
                badges.append("`官方`")
            # 显示关键词命中（matched_keywords）而非分组键（分组键已是分类名）
            mks = p.get("matched_keywords") or ([kw] if kw and kw not in CATEGORIES else [])
            for mk in mks[:3]:
                badges.append(f"`{mk}`")
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
            st.markdown(f"**[{disp}]({p['url']})** &nbsp; {badge_str}")
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

    # 数据源状态条
    all_sources = load_yaml(CONFIG_DIR / "sources.yaml", {"sources": []})["sources"]
    enabled_sources = [s for s in all_sources if s.get("enabled", True) is not False]
    has_tavily = any(s.get("type") == "tavily" for s in enabled_sources)
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    bits = [f"启用源 **{len(enabled_sources)}/{len(all_sources)}**"]
    if has_tavily:
        bits.append("Tavily ✅" if tavily_key else "Tavily ⚠️ 未设 Key")
    st.caption(" · ".join(bits) + "  —— 在「🌐 数据源」页可调整")

    # 加载已保存的个性化偏好
    pref_cfg = load_yaml(CONFIG_DIR / "preference.yaml", {"text": ""})
    saved_pref = pref_cfg.get("text", "")

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
                ["24 小时", "一周"],
            )

        # 个性化偏好（辅助 LLM 重排，可空）
        user_preference = st.text_area(
            "🎯 个性化偏好（可选，辅助 LLM 重排）",
            value=saved_pref,
            placeholder=(
                "示例：\n"
                "- 我更关心 LLM 推理性能优化和量化技术\n"
                "- 只看带 benchmark 数字或具体性能对比的内容\n"
                "- 同等价值时优先选与 vLLM/SGLang 生态相关的"
            ),
            height=120,
            help="留空 = 仅按通用准则。填了会作为高优先级判断依据注入到每个 LLM 调用。",
        )
        save_pref = st.checkbox("💾 保存为默认偏好（写入 config/preference.yaml）",
                                value=False,
                                help="下次进入界面时自动填充")

        use_llm = st.checkbox("启用 LLM 排序 + 自检", value=True)
        send_notify = st.checkbox("跑完后推送通知", value=False)
        submitted = st.form_submit_button("🚀 运行抓取", type="primary")

    if submitted:
        # 个性化偏好持久化（按需）
        if save_pref:
            save_yaml(CONFIG_DIR / "preference.yaml", {"text": user_preference.strip()})

        # 准备 keywords
        _kw_data = load_yaml(CONFIG_DIR / "keywords.yaml", {"keywords": []})
        config_keywords = _kw_data["keywords"]
        watchlist = _kw_data.get("watchlist") or []
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
        themes = load_yaml(CONFIG_DIR / "themes.yaml", {"themes": []}).get("themes", [])

        llm_cfg = None
        if use_llm:
            try:
                llm_cfg = load_llm_config(CONFIG_DIR / "llm.yaml")
            except Exception as e:
                st.error(f"LLM 配置加载失败：{e}")
                return

        # 解析时间窗
        window_hours, use_all = 24, False
        if window_mode == "一周":
            window_hours = 24 * 7

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
                    user_preference=user_preference.strip() or None,
                    themes=themes,
                    watchlist=watchlist,
                    log=log,
                )
                status.update(label="抓取完成 ✓", state="complete")
            except Exception as e:
                status.update(label=f"失败: {e}", state="error")
                st.exception(e)
                return

        # 全局按类别裁剪：每个分类最多 MAX_PER_CATEGORY 条
        if result.get("grouped"):
            before = sum(len(v) for v in result["grouped"].values())
            result["grouped"] = cap_per_category(result["grouped"])
            after = sum(len(v) for v in result["grouped"].values())
            if before != after:
                log(f"[CAP] 按类别裁剪：{before} → {after}（每类 ≤ {MAX_PER_CATEGORY}）")

        st.session_state["last_result"] = result

        # 生成 HTML 报告 + 可选通知
        if result["grouped"]:
            from main import flatten_to_render  # 借用现成的扁平化
            flat = flatten_to_render(result["grouped"])
            html = render_html(flat, result["window_from"] or "all",
                               result["window_to"], result.get("digest"))
            out = write_report(html, REPORTS_DIR)
            st.session_state["last_report_path"] = str(out)
            st.success(f"HTML 报告: {out}")

            if send_notify:
                cfg = load_notify_config(CONFIG_DIR / "notify.yaml")
                notify_all(flat, html, datetime.now().strftime("%Y-%m-%d"),
                           cfg, result.get("digest"))
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

        # 来源分布（命中条目按源）
        items = result.get("items") or []
        if items:
            from collections import Counter
            by_src = Counter(it["source"].split("(")[0].strip() for it in items)
            with st.expander(f"📊 命中按来源分布（共 {sum(by_src.values())} 条）"):
                for src, n in by_src.most_common():
                    st.write(f"- **{n}** · {src}")

        digest = result.get("digest") or {}
        if digest.get("overview"):
            st.markdown("#### 📊 今日概览")
            st.info(digest["overview"])

        from main import flatten_to_render
        flat = flatten_to_render(result["grouped"])
        render_by_category(flat)

        if digest.get("trends"):
            st.markdown("#### 📈 趋势分析")
            for i, t in enumerate(digest["trends"], 1):
                st.markdown(f"{i}. {t}")
        if digest.get("advice"):
            st.markdown("#### 💡 给存储技术规划的建议")
            for i, a in enumerate(digest["advice"], 1):
                st.markdown(f"{i}. {a}")


# =========================================================================
# 页面: 历史报告
# =========================================================================

# =========================================================================
# 页面: Prompt 设置（rubric + 各分类定义的可视化编辑 + override 持久化）
# =========================================================================

def page_prompt():
    from core.ranker import (
        DEFAULT_IMPORTANCE_RUBRIC, DEFAULT_CATEGORY_DEFS,
        get_importance_rubric, get_category_defs,
        save_prompt_overrides, reset_prompt_overrides,
        PROMPT_OVERRIDE_PATH, CATEGORIES,
    )

    st.header("⚙️ Prompt 设置")
    st.caption(
        "LLM 排序时使用的「重要性评判 rubric」+「各分类定义」。修改后写入 "
        "`config/prompt_override.yaml`（已 gitignore），下次抓取自动加载；"
        "**留空 / 恢复默认**则回退到代码内置版。"
    )

    has_override = PROMPT_OVERRIDE_PATH.exists()
    if has_override:
        try:
            meta = yaml.safe_load(PROMPT_OVERRIDE_PATH.read_text(encoding="utf-8")) or {}
            updated = meta.get("updated_at", "未知")
            st.success(f"✅ 当前已启用 override（版本 {meta.get('version','?')}，更新于 {updated}）")
        except Exception:
            st.warning("⚠️ override 文件存在但解析失败，运行时将回退默认")
    else:
        st.info("当前使用代码内置默认 prompt（未启用 override）")

    cur_rubric = get_importance_rubric()
    cur_defs = get_category_defs()

    # ---- 编辑区 ----
    st.subheader("1. 重要性评判 Rubric（IMPORTANCE_RUBRIC）")
    st.caption("LLM 在筛选时遵守的多维评分标准；候选条目都会附在此 rubric 后面送给 LLM。")
    new_rubric = st.text_area(
        "rubric 文本",
        value=cur_rubric,
        height=600,
        key="prompt_rubric_text",
        label_visibility="collapsed",
    )
    rubric_changed = new_rubric.strip() != DEFAULT_IMPORTANCE_RUBRIC.strip()
    st.caption(
        ("🟡 已偏离默认" if rubric_changed else "🟢 与默认一致")
        + f"  ·  字符数: {len(new_rubric)}"
    )

    st.subheader("2. 各分类定义（CATEGORY_DEFS）")
    st.caption("每个分类发给 LLM 的「本分类定义」段落，告诉 LLM 这桶里应该收什么。")
    new_defs: dict[str, str] = {}
    for cat in CATEGORIES:
        cur_text = cur_defs.get(cat, DEFAULT_CATEGORY_DEFS.get(cat, ""))
        default_text = DEFAULT_CATEGORY_DEFS.get(cat, "")
        diverged = cur_text.strip() != default_text.strip()
        with st.expander(
            f"{'🟡' if diverged else '🟢'} {cat}",
            expanded=False,
        ):
            txt = st.text_area(
                f"{cat} 定义",
                value=cur_text,
                height=180,
                key=f"prompt_catdef_{cat}",
                label_visibility="collapsed",
            )
            new_defs[cat] = txt

    st.divider()

    # ---- 操作按钮 ----
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("💾 保存修改", type="primary", use_container_width=True):
            # 若内容与默认完全一致，不写文件（避免无意义的 override）
            rubric_to_save = (
                new_rubric.strip()
                if new_rubric.strip() and new_rubric.strip() != DEFAULT_IMPORTANCE_RUBRIC.strip()
                else None
            )
            defs_to_save = {
                k: v for k, v in new_defs.items()
                if v.strip() and v.strip() != DEFAULT_CATEGORY_DEFS.get(k, "").strip()
            }
            if rubric_to_save is None and not defs_to_save:
                # 所有项都跟默认一样 —— 等价于恢复默认
                reset_prompt_overrides()
                st.success("所有内容已与默认一致，已删除 override 文件。")
            else:
                save_prompt_overrides(rubric_to_save, defs_to_save or None)
                st.success(f"已保存到 `{PROMPT_OVERRIDE_PATH.relative_to(ROOT)}`")
            st.rerun()

    with col2:
        if st.button("↩️ 恢复默认", use_container_width=True, disabled=not has_override):
            removed = reset_prompt_overrides()
            if removed:
                st.success("已删除 override，回到代码内置默认。")
            else:
                st.info("当前无 override 文件，无需恢复。")
            st.rerun()

    with col3:
        if has_override:
            with st.expander("查看原始 override.yaml"):
                st.code(
                    PROMPT_OVERRIDE_PATH.read_text(encoding="utf-8"),
                    language="yaml",
                )


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

# Tavily 风格：亮色底 + 大留白 + 卡片 + 圆角 + 蓝紫色高亮
st.markdown("""
<style>
:root {
    --nc-bg: #fafafa;
    --nc-card: #ffffff;
    --nc-border: #e8eaed;
    --nc-border-strong: #d4d7dc;
    --nc-text: #1a1a1a;
    --nc-text-soft: #4a5160;
    --nc-muted: #8a8f99;
    --nc-accent: #468bff;
    --nc-accent-hover: #2563eb;
    --nc-accent-soft: #eef4ff;
    --nc-chip-bg: #f3f5f9;
    --nc-chip-text: #4a5160;
    --nc-radius: 12px;
    --nc-radius-sm: 8px;
    --nc-shadow: 0 1px 2px rgba(15, 23, 42, 0.04), 0 1px 3px rgba(15, 23, 42, 0.06);
    --nc-shadow-hover: 0 4px 12px rgba(15, 23, 42, 0.08);
    --nc-font: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI",
               "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
    --nc-mono: ui-monospace, SFMono-Regular, "JetBrains Mono", Menlo, Consolas, monospace;
}

/* 全局：白底、Inter 字体、放松行距 */
html, body, [class*="css"], .stApp {
    font-family: var(--nc-font);
    -webkit-font-smoothing: antialiased;
    letter-spacing: -0.005em;
}
.stApp { background: var(--nc-bg); }
section.main { padding: 1.5rem 2rem; }
section.main .block-container { max-width: 1080px; padding-top: 1rem; }

/* 正文 */
section.main p, section.main li, section.main span,
section.main div[data-testid="stMarkdownContainer"] {
    font-size: 0.94rem; line-height: 1.7; color: var(--nc-text-soft);
}
section.main strong { color: var(--nc-text); font-weight: 600; }

/* 标题：清爽、字重克制 */
section.main h1, section.main h2, section.main h3, section.main h4 {
    color: var(--nc-text); font-weight: 600; letter-spacing: -0.015em;
}
section.main h1 { font-size: 1.7rem; margin: 0 0 .35rem; }
section.main h2 { font-size: 1.25rem; margin: 1.2rem 0 .5rem; }
section.main h3 { font-size: 1.05rem; }
section.main h4 { font-size: 0.97rem; }

/* caption */
section.main small, section.main .stCaption,
section.main [data-testid="stCaptionContainer"] {
    font-size: 0.82rem; color: var(--nc-muted); letter-spacing: 0;
}

/* metric：卡片化，蓝色数字 */
[data-testid="stMetric"] {
    background: var(--nc-card);
    border: 1px solid var(--nc-border);
    border-radius: var(--nc-radius);
    padding: 14px 18px;
    box-shadow: var(--nc-shadow);
    transition: box-shadow .15s ease, transform .15s ease;
}
[data-testid="stMetric"]:hover {
    box-shadow: var(--nc-shadow-hover);
}
[data-testid="stMetricValue"] {
    font-size: 1.55rem; font-weight: 600; color: var(--nc-accent);
    letter-spacing: -0.02em;
}
[data-testid="stMetricLabel"] {
    font-size: 0.78rem; color: var(--nc-muted);
    text-transform: uppercase; letter-spacing: 0.06em; font-weight: 500;
}

/* badge：浅灰胶囊 */
section.main code {
    font-family: var(--nc-font); font-size: 0.74rem; font-weight: 500;
    background: var(--nc-chip-bg); color: var(--nc-chip-text);
    border-radius: 999px; padding: 2px 9px; margin: 0 3px;
    border: 1px solid transparent;
    letter-spacing: 0.01em;
}

/* 链接：底蓝下划线 hover */
section.main a {
    color: var(--nc-accent); text-decoration: none; font-weight: 500;
    border-bottom: 1px solid transparent;
    transition: border-color .15s ease, color .15s ease;
}
section.main a:hover { color: var(--nc-accent-hover); border-bottom-color: var(--nc-accent-hover); }

/* expander 卡片化 */
[data-testid="stExpander"] {
    background: var(--nc-card);
    border: 1px solid var(--nc-border) !important;
    border-radius: var(--nc-radius) !important;
    box-shadow: var(--nc-shadow);
    overflow: hidden;
    margin-bottom: .55rem;
}
[data-testid="stExpander"] summary {
    padding: .75rem 1rem; font-weight: 500; color: var(--nc-text);
}
[data-testid="stExpander"] summary:hover { background: var(--nc-accent-soft); }

/* button：圆角 + 蓝色主按钮 */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
    border-radius: var(--nc-radius-sm) !important;
    border: 1px solid var(--nc-border-strong);
    background: var(--nc-card); color: var(--nc-text);
    font-weight: 500; padding: .45rem 1.1rem;
    transition: all .15s ease; box-shadow: var(--nc-shadow);
}
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
    border-color: var(--nc-accent); color: var(--nc-accent);
    box-shadow: var(--nc-shadow-hover);
}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
    background: var(--nc-accent); border-color: var(--nc-accent); color: #fff;
}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {
    background: var(--nc-accent-hover); border-color: var(--nc-accent-hover); color: #fff;
}

/* input / textarea / select */
.stTextInput input, .stTextArea textarea, .stNumberInput input {
    border-radius: var(--nc-radius-sm) !important;
    border: 1px solid var(--nc-border-strong) !important;
    background: var(--nc-card) !important;
    font-family: var(--nc-font) !important;
    transition: border-color .15s ease, box-shadow .15s ease;
}
.stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
    border-color: var(--nc-accent) !important;
    box-shadow: 0 0 0 3px rgba(70, 139, 255, 0.15) !important;
}
.stSelectbox div[data-baseweb="select"] > div {
    border-radius: var(--nc-radius-sm) !important;
    border-color: var(--nc-border-strong) !important;
    background: var(--nc-card) !important;
}

/* form 容器 */
[data-testid="stForm"] {
    background: var(--nc-card);
    border: 1px solid var(--nc-border);
    border-radius: var(--nc-radius);
    padding: 1.25rem 1.4rem;
    box-shadow: var(--nc-shadow);
}

/* sidebar：浅灰底 */
[data-testid="stSidebar"] {
    background: #f7f8fa;
    border-right: 1px solid var(--nc-border);
}
[data-testid="stSidebar"] h1 {
    font-size: 1.15rem; font-weight: 600; color: var(--nc-text);
    letter-spacing: -0.01em;
}
[data-testid="stSidebar"] [data-baseweb="radio"] label {
    padding: .45rem .6rem; border-radius: var(--nc-radius-sm);
    transition: background .15s ease;
}
[data-testid="stSidebar"] [data-baseweb="radio"] label:hover {
    background: var(--nc-accent-soft);
}

/* status widget */
[data-testid="stStatusWidget"] {
    border-radius: var(--nc-radius); border: 1px solid var(--nc-border);
    background: var(--nc-card); box-shadow: var(--nc-shadow);
}

/* divider */
hr { margin: 1rem 0; border-color: var(--nc-border); }

/* radio 横排 */
[data-baseweb="radio"] > div { gap: .25rem; }

/* code block（运行日志） */
section.main pre {
    background: #0f172a; color: #e2e8f0; border-radius: var(--nc-radius-sm);
    font-family: var(--nc-mono); font-size: 0.82rem;
    padding: .9rem 1rem; border: 1px solid #1e293b;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("📰 newsCollector")

PAGES = {
    "📰 抓取 & 结果": page_run,
    "⚙️ 关键词配置": page_keywords,
    "🌐 数据源": page_sources,
    "🤖 LLM 配置": page_llm,
    "✏️ Prompt 设置": page_prompt,
    "🔔 通知设置": page_notify,
    "📚 历史报告": page_history,
}
choice = st.sidebar.radio("导航", list(PAGES.keys()))
st.sidebar.divider()
st.sidebar.caption(f"源代码: `{ROOT}`")
PAGES[choice]()
