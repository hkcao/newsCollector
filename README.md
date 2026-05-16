# newsCollector

按用户指定关键词，每日从多源抓取技术资讯，由 LLM 按"重要性 + 官方源优先"筛选并生成中文摘要。

## 特性

- **多源抓取**：HackerNews / Reddit / arXiv / Google News + 官方博客（vLLM / NVIDIA / VAST Data）+ Tavily Search 兜底
- **关键词匹配**：支持同义词、大小写敏感开关、官方域名白名单
- **LLM 重要性排序**：走 OpenAI 兼容接口（DeepSeek / Kimi / OpenAI 任选），宁缺毋滥
- **摘要忠实性**：原 RSS 摘要优先 → 否则 LLM 生成 → 二次自检剔除推测/扩写
- **英文自动翻译**：RSS 原文是英文时走一次 LLM 翻译，保留专有名词
- **官方源偏好**：避免转载导致的时效失真
- **输出形式**：控制台 / HTML 报告 / 邮件 / 飞书 webhook
- **Streamlit GUI**：配置和结果都在浏览器界面里

## 快速开始

> 支持平台：macOS / Linux / WSL。Python 3.10+（建议 3.12）。

```bash
# 1. 安装依赖
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. 配置 LLM
cp config/llm.yaml.example config/llm.yaml
# 编辑 config/llm.yaml 填入 api_key；或者用环境变量:
export LLM_API_KEY=sk-xxxxxx

# 3. （可选）持久化其它密钥
cp config/secrets.yaml.example config/secrets.yaml
# 编辑 config/secrets.yaml 填入 TAVILY_API_KEY / EMAIL_PASSWORD / FEISHU_WEBHOOK

# 4a. 运行 GUI
.venv/bin/streamlit run app.py
# 打开 http://localhost:8501

# 4b. 或运行 CLI
.venv/bin/python main.py --window-hours 24
```

macOS 上如未安装 Python：`brew install python@3.12`。

## 定时任务

```bash
# 每天 09:00 自动跑
.venv/bin/python main.py --daemon --at 09:00 --run-now
```

或用 cron / Windows 任务计划程序调用 `python main.py`，窗口逻辑会自动绑定到上次运行时间。

## 配置文件

| 文件 | 作用 |
|---|---|
| `config/keywords.yaml` | 关键词 + 同义词 + 官方域名 |
| `config/sources.yaml` | RSS 源列表 |
| `config/llm.yaml` | LLM 服务商 / 模型 / API Key（本地，已 gitignore） |
| `config/notify.yaml` | 邮件 / 飞书 推送配置 |
| `config/secrets.yaml` | 持久化密钥：`TAVILY_API_KEY` 等（本地，已 gitignore） |

环境变量（优先级 > secrets.yaml）：
- `LLM_API_KEY` — 代替 `config/llm.yaml` 里的 api_key
- `TAVILY_API_KEY` — 启用 Tavily Search 源（不设则跳过；免费档 https://tavily.com 注册即得，约 1000 次/月）
- `EMAIL_PASSWORD` / `FEISHU_WEBHOOK` — 推送通道


## 排序逻辑（两层 rerank）

一条原始资讯从抓取到最终被 LLM 选中，要经过 **客观打分预筛 → LLM 语义重排** 两层。
理解这套逻辑能帮你解释「为什么这条没出现 / 那条出现了」。

### 阶段一：抓取 + 客观打分（`core/normalizer.py`）

1. **去重**：按 `id = sha1(url + title)[:16]` 去掉跨源重复
2. **时间窗过滤**：保留 `published ≥ since` 的条目。
   `since` 取自 `meta.last_run_at`（默认）或 `--window-hours N`（强制覆盖）
3. **关键词匹配**：在 `title + summary` 上对每个关键词及其 `aliases` 命中即留下
   - 英文：`(?<!\w)term(?!\w)` 单词边界
   - 中文：直接子串
   - `case_sensitive: true` 的关键词必须大小写完全一致（vLLM / VAST Data 等）
4. **跨源聚合**：在所有命中条目里按归一化标题分组，写入 `cross_source_count = 该标题出现在多少个不同源`（HN / Reddit / Google News / Tavily 任一同时报道都计入）
5. **客观打分**（`compute_score`）：把所有源能给的信号压成 0~2.0 的单值，

   $$\text{score} = 0.25 \log_{10}(\text{points}{+}1) + 0.08 \log_{10}(\text{comments}{+}1)$$
   $$\quad + 0.25 \log_{10}(\Delta_{24h}{+}1) + 0.20 \log_2(\text{cross}{+}1)$$
   $$\quad + 0.15 \cdot \text{tavily\_score} + 0.07 \cdot \text{recency}$$

   | 项 | 权重 | 来源 / 含义 |
   |---|---|---|
   | `points` | 0.25 | HN 主帖点数（绝对热度，log 压尺度） |
   | `comments` | 0.08 | HN 评论数（参与度低于 points） |
   | `delta_24h` | 0.25 | HN points 相对 SQLite 历史基线的增量（"突发性"） |
   | `cross_source_count` | 0.20 | **跨源**同时报道的源数。log2 增长快，被 2 个源同时报权重已显著 |
   | `tavily_score` | 0.15 | Tavily 自带的 0~1 相关性分（topic=news 的"新闻价值"判断） |
   | `recency` | 0.07 | 越接近"现在"越接近 1，>24h 衰减为 0 |

   - **取 log10**：HN 帖子分数从几到上万，防止陈年大热点（5000 分）永远盖过当天 200 分的新事件
   - **`cross_source_count` 用 log2**：被 2 个源同时报道 ≈ +0.32，被 5 个源 ≈ +0.52；强力反映"事件级"重要性
   - **`tavily_score`** 让纯 Tavily 来源的条目也能拿到非零客观分（HN 类信号是 0）
   - 大量 RSS（Reddit/arXiv）没任何信号 → 主要靠 `cross_source_count + tavily_score` 救场，否则得等 LLM 重排

### 阶段二：候选池组装（`core/ranker.py:group_by_keyword`）

每个关键词独立成池，按 `(is_official, score)` **降序排序**，取前 `max_candidates_per_keyword`（默认 40）条进入 LLM。

- 官方源（域名命中 `official_domains` 或全局 `arxiv.org / github.com`）排在最前，保证不会因 score 太低被截掉
- 同为官方源 / 同为非官方时再按 `score` 倒序

> ⚠️ **客观打分只决定"谁进 LLM 候选池"**，不直接决定最终顺序。低分但有信息价值的论文/官方博客只要在前 40 条里，仍有机会被 LLM 选中。

### 阶段三：LLM 语义 rerank（`core/ranker.py:rank_per_keyword`）

把候选池整体喂给 LLM，附上一个长 prompt（`IMPORTANCE_RUBRIC` + `CATEGORY_HINTS`），LLM 综合判断后返回最多 `top_k_per_keyword` 条（默认 2）。

LLM 的"信息价值"判断由 5 个维度综合（rubric 已显式列出）：

1. **新增量**：对未来趋势判断带来的新信息（新产品/版本/数字/合作/评测）；既成事实复盘要降权
2. **首发性**：官方一手 > 二手转载
3. **可验证细节密度**：含具体数字/版本/日期/人名/对比基线 > 通篇形容词
4. **影响范围**：能撬动多少决策（行业标准/开源主流 > 小众教程）
5. **可操作性**：用户能据此采取行动（试用/学习/采购）> 纯舆论/八卦/股价

辅助约束：
- **官方源偏好**：域名命中 `official_domains` 的条目带 `[官方]` 标记；同事件优先官方版本；只有解读没一手时宁可空列表
- **类别多样性**：top_k 条尽量覆盖不同类别（公司新闻 / 政策走向 / 学术论文 / 技术解读）
- **个性化偏好**（界面可填）：用户在「📰 抓取 & 结果」页可填一段自然语言（如"只看带 benchmark 数字的"），叠加到上述 5 维之上作为高优先级判断依据；勾选「保存为默认」会持久化到 `config/preference.yaml`
- **降权**：股价预测、listicle、纯教程、过度推测词

每条选中后由 LLM 同时给出：
- `display_title`：≤30 字中文新闻式短标题
- `summary`：8-15 句 / 500-1000 字客观摘要，**逐项覆盖**事件 / 主体 / 方法细节 / 量化结果 / 价值场景 / 背景 / 局限；目标是让用户不点链接也能完整判断价值。禁止扩写或加推测
- `category`：四选一（公司新闻 / 政策走向 / 学术论文 / 技术解读）

### 阶段四：摘要忠实性兜底（两遍 LLM）

- **优先用原 RSS 摘要**（`core/summary.py:clean_rss_summary`）：清洗 HTML、剥离 HN metadata / Google News 占位符，可用就直接用，不让 LLM 重写
- RSS 不可用才走 LLM 生成 → 再走 **verify pass**（独立 LLM 调用，温度 0.1）检查是否引入推测/扩写，发现就重写
- RSS 原文是英文走一次 **translate pass**，保留专有名词

### 调优建议

| 想要的效果 | 怎么调 |
|---|---|
| 候选池太小，重要条目没进 LLM | 调高 `llm.yaml:max_candidates_per_keyword`（默认 40 → 60） |
| 每个关键词输出太少 | 调高 `top_k_per_keyword`（默认 2 → 3）。LLM 仍会"宁缺毋滥" |
| 摘要太短 / 缺细节 | 已在 prompt 要求 300-500 字；如果还想更长，改 `ranker.py:PROMPT_TEMPLATE` 的字数限制 |
| LLM 选了无价值复盘稿 | 在 `keywords.yaml` 给该关键词加 `official_domains`，或调强 `IMPORTANCE_RUBRIC` 里的反例 |
| 想看为什么某条被刷掉 | 跑 `python main.py --no-llm --top 50`，看客观分数；或在 `rank_per_keyword` 里把 prompt 落盘 |
