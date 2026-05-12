# newsCollector

按用户指定关键词，每日从多源抓取技术资讯，由 LLM 按"重要性 + 官方源优先"筛选并生成中文摘要。

## 特性

- **多源抓取**：HackerNews / Reddit / arXiv / Google News，按关键词搜索
- **关键词匹配**：支持同义词、大小写敏感开关、官方域名白名单
- **LLM 重要性排序**：走 OpenAI 兼容接口（DeepSeek / Kimi / OpenAI 任选），宁缺毋滥
- **摘要忠实性**：原 RSS 摘要优先 → 否则 LLM 生成 → 二次自检剔除推测/扩写
- **英文自动翻译**：RSS 原文是英文时走一次 LLM 翻译，保留专有名词
- **官方源偏好**：避免转载导致的时效失真
- **输出形式**：控制台 / HTML 报告 / 邮件 / 飞书 webhook
- **Streamlit GUI**：配置和结果都在浏览器界面里

## 快速开始

```bash
# 1. 安装依赖
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. 配置 LLM（二选一）
cp config/llm.yaml.example config/llm.yaml
# 编辑 config/llm.yaml 填入 api_key
# 或:
export LLM_API_KEY=sk-xxxxxx

# 3a. 运行 GUI
.venv/bin/streamlit run app.py
# 打开 http://localhost:8501

# 3b. 或运行 CLI
.venv/bin/python main.py --window-hours 24
```

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

环境变量：`LLM_API_KEY` / `EMAIL_PASSWORD` / `FEISHU_WEBHOOK`（均可代替写入配置文件）。
