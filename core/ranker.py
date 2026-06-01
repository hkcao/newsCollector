"""LLM 排序器 —— 走 OpenAI 兼容接口 (DeepSeek / Kimi / OpenAI 通用)。

按关键词分组送给 LLM，让它基于"重要性 + 权威性"挑出 top K，可以宁缺毋滥。
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import yaml
from openai import OpenAI

from core.debug_trace import TRACE


# ---------- Token 统计 + 自适应粗筛 ----------

# 单次 prompt（输入）token 上限，超过则触发基于规则的粗筛
# 注意：这是 prompt 估算上限，不是模型上下文上限；目的是省钱 + 保护 LLM 判断质量
PROMPT_TOKEN_BUDGET = int(os.getenv("LLM_PROMPT_TOKEN_BUDGET", "20000"))


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数：中文按 ~1 字 1.5 token，英文按 ~4 字 1 token。
    混排时用 len/2 作为偏保守的中间值，足以触发粗筛门槛判断。"""
    if not text:
        return 0
    cjk = sum(1 for c in text if "一" <= c <= "鿿" or "぀" <= c <= "ヿ")
    other = len(text) - cjk
    return int(cjk * 1.5 + other / 3.5)


@dataclass
class TokenUsage:
    """累计 LLM token 消耗，按调用类型分桶。"""
    by_kind: dict[str, dict[str, int]] = field(default_factory=dict)

    def add(self, kind: str, prompt: int, completion: int, calls: int = 1) -> None:
        b = self.by_kind.setdefault(kind, {"prompt": 0, "completion": 0, "calls": 0})
        b["prompt"] += prompt
        b["completion"] += completion
        b["calls"] += calls

    @property
    def total_prompt(self) -> int:
        return sum(b["prompt"] for b in self.by_kind.values())

    @property
    def total_completion(self) -> int:
        return sum(b["completion"] for b in self.by_kind.values())

    @property
    def total(self) -> int:
        return self.total_prompt + self.total_completion

    def report(self) -> str:
        if not self.by_kind:
            return "  [LLM Tokens] 未调用 LLM"
        lines = ["  [LLM Tokens]"]
        for k, b in self.by_kind.items():
            lines.append(
                f"    {k:18s}  calls={b['calls']:3d}  "
                f"in={b['prompt']:>8d}  out={b['completion']:>6d}  "
                f"total={b['prompt']+b['completion']:>8d}"
            )
        lines.append(
            f"    {'TOTAL':18s}  calls={sum(b['calls'] for b in self.by_kind.values()):3d}  "
            f"in={self.total_prompt:>8d}  out={self.total_completion:>6d}  "
            f"total={self.total:>8d}"
        )
        return "\n".join(lines)


# 全局单例 —— 每次 run_once 开始时由 main.py 调 reset() 清零
USAGE = TokenUsage()


def reset_usage() -> None:
    USAGE.by_kind.clear()


def _chat(
    client: OpenAI,
    model: str,
    prompt: str,
    *,
    kind: str,
    temperature: float = 0.2,
    response_format: dict | None = None,
    max_tokens: int = 196608,
) -> str:
    """统一的 LLM 调用入口：执行 + 记账。返回 message.content。失败时抛异常。

    max_tokens 默认取 MiniMax-M2.7 的输出上限 196608（模型自身的 max，不是人为限制；
    超过会 400）。推理模型会先吐很长的 <think> 段，若额度太小思考会吃光预算、在输出
    JSON 前就被截断 → 解析得空。给到模型上限让"思考 + 正式答案"都装得下。
    max_tokens 只是上限，模型遇到 EOS 会自然停止，不会因此多花 token。
    """
    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    resp = client.chat.completions.create(**kwargs)
    usage = getattr(resp, "usage", None)
    if usage is not None:
        USAGE.add(
            kind,
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
        )
    else:
        # 兜底：服务端未返 usage（少数 OpenAI 兼容实现），用估算值
        USAGE.add(kind, _estimate_tokens(prompt), _estimate_tokens(resp.choices[0].message.content or ""))
    return _strip_think(resp.choices[0].message.content or "")


# 推理模型（MiniMax-M2.7 等）会在正文前吐 <think>…</think> 推理段。
# 翻译/自检这类"返回纯文本"的 pass 若不剥离，think 会直接污染摘要；rank 这类靠正则
# 抽 JSON 的 pass 也更稳。统一在 _chat 出口剥离闭合的 think 块。
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    cleaned = _THINK_RE.sub("", text).strip()
    # 仅剥离闭合块；若清掉后为空（异常情况）则回退原文，避免丢内容
    return cleaned if cleaned else text


# 任何关键词都视为权威首发源
GLOBAL_OFFICIAL_DOMAINS = {"arxiv.org", "github.com"}

# 用户视角的资讯类型分类（6 类，对存储分析师粒度更友好）
#   存储/基建产品 —— 存储厂商 / SSD / 互联协议 / GPU 数据通路 的产品级动态
#   模型/框架/算法 —— 大模型 release / 训练推理框架 / 推理优化算法
#   基准评测     —— MLPerf / IO500 / SPECstorage 等评测结果（选型硬通货）
#   学术论文     —— arXiv / 顶会 / 期刊原始论文
#   政策导向     —— 出口管制 / 监管 / 法规 / 行业标准联盟
#   github趋势   —— github.com 的项目动态
CATEGORIES = [
    "存储/基建产品",
    "模型/框架/算法",
    "基准评测",
    "学术论文",
    "政策导向",
    "github趋势",
    "AI意见领袖",
]

# 来自 category=influencer 源（Twitter/X 大神）的硬分类目标桶
INFLUENCER_CATEGORY = "AI意见领袖"

# 归类到「存储/基建产品」的关键词集合（命中其一即归此桶）
_STORAGE_INFRA_KEYWORDS = {
    "VAST Data", "WEKA", "DDN", "Pure Storage", "NetApp", "Hammerspace",
    "IBM Storage", "Dell Storage", "HPE Storage", "Huawei Storage",
    "Inspur Storage", "MEGREZ",
    "Lustre", "Ceph", "DAOS", "BeeGFS", "JuiceFS", "Alluxio", "MinIO", "3FS",
    "AgentFS", "parallel file system", "KV file system",
    "GPUDirect", "NVMe-oF", "RDMA", "NIXL",
    "CXL", "HBM", "NVLink", "Ultra Ethernet",
    "BlueField", "Spectrum-X", "SuperPOD",
    "Computational Storage", "FDP", "ZNS", "Pliops", "aiDAPTIV+",
    "焱融", "XSKY", "杉岩", "DapuStor", "Solidigm", "Kioxia", "Micron",
    "AWS Storage", "Azure Storage", "Google Cloud Storage",
    "S3 Express", "Iceberg", "Lance",
    "Vector Database", "Milvus", "Pinecone", "Weaviate", "Qdrant", "Chroma",
    "AI Data Platform", "AI factory",
    "Alibaba Cloud", "Tencent Cloud", "ByteDance", "Baidu Cloud",
    "中国电信", "Neocloud",
    "NVM Express",
}

# 归类到「模型/框架/算法」的关键词集合
_MODEL_FRAMEWORK_KEYWORDS = {
    "NVIDIA", "vLLM", "SGLang", "DeepSeek",
    "OpenAI", "Anthropic", "Google DeepMind", "Meta AI",
    "Mistral", "xAI", "Qwen", "Kimi", "Zhipu", "MiniMax", "Doubao",
    "Hunyuan", "ERNIE",
    "PyTorch", "JAX", "TensorRT-LLM", "Triton", "Dynamo",
    "DeepSpeed", "Megatron", "Ray", "LangChain", "LlamaIndex",
    "MoE", "RLHF", "long context", "agentic", "multimodal",
    "world model", "RAG",
    "speculative decoding", "disaggregated inference",
    "KV Cache", "LMCache", "Mooncake", "checkpoint",
    "AMD AI", "Intel AI", "TPU", "Trainium", "Huawei Ascend",
    "Cambricon", "Cerebras", "Groq", "Hugging Face",
    "MaaS",
}

# 「基准评测」桶的强信号关键词
_BENCHMARK_KEYWORDS = {
    "MLPerf Storage", "IO500", "SPECstorage", "DLIO", "STAC-AI",
}

# 标题/摘要里出现这些词也强制归到「基准评测」（即便没匹配上面 keyword）
_BENCHMARK_HINTS_LC = (
    "mlperf", "io500", "io-500", "specstorage", "specsfs", "spec storage",
    "stac-ai", "stac ai", "dlio benchmark", "tpcx-ai",
    "benchmark results", "benchmark suite", "performance benchmark",
)


DEFAULT_IMPORTANCE_RUBRIC = """用户身份：**资深 AI 存储技术规划与分析师**。日常工作是为大规模 AI 训练/推理集群
做存储栈选型、容量规划、架构演进路线图、TCO 测算、跨厂商对标。同时持续追踪 AI 模型/算法/算力栈
的发展，因为这些直接驱动存储需求的形态变化。

本工具的核心使命：**从全网技术资讯中，挑出"对存储 + AI 基建规划决策真正有用"的最新信号。**

「信息价值」由以下 7 个维度共同决定，请逐项过一遍再综合判断：

1. **存储 / AI 基建相关性**（强但非绝对一票否决）：内容是否影响 AI 存储或 AI 算力栈的规划决策？
   - 直接相关（高）：存储产品/协议/性能、AI 训练或推理的 IO 路径、GPU 内存/带宽、数据平台、
     新模型对存储需求的影响（参数量、上下文、checkpoint、数据集体量）、新训练/推理框架。
   - **间接相关但保留**：GPU 大单 / 模型推理涨价降价 / 国产替代政策 / 算力大单
     —— 这些虽不直接讲存储，但会改变存储采购信号，**作为"市场温度计"保留 1 条**即可。
   - 完全不相关（丢弃）：消费应用八卦、UI 改版、聊天机器人吐槽。

2a. **性能数字**（最关键之一）：内容是否给出"规划者可代入容量/性能模型"的新数字？
    - 带宽 (GB/s)、IOPS、延迟 (µs/ms)、容量密度 (PB/U)、训练吞吐、推理 tok/s、
      MLPerf / IO500 / SPECstorage benchmark 分数、模型参数量、上下文长度 (K/M tokens)、
      checkpoint 大小、KV cache 占用。

2b. **经济数字**（最关键之一）：是否给出"规划者可代入 TCO 模型"的新成本数字？
    - $/GB、$/TB/月、$/GB/s、$/M tokens、tokens per dollar、合同金额、采购大单金额、
      出货量、市场份额、价格调整百分比、租赁报价。

3. **跨厂商对标**（加分项）：是否把两家以上具名厂商的指标做对比？
   - "X 的延迟比 Y 低 N%"、"A 平台 vs B 平台 MLPerf 分数对照" —— **直接加权**，这正是
     分析师工作的核心场景。

4. **首发性**：是该信息的源头，还是 N 手转载？
   - 高：公司官博、官方 GitHub release、arXiv 一作、监管机构原文。
   - 低：综合新闻媒体对官方稿件的二次复述（往往滞后且可能曲解）。

5. **决策影响范围**：撬动多少存储/AI 基建规划场景？
   - 高：行业主流厂商的新 reference architecture、广泛部署开源系统的重大更新、
     行业标准（NVMe-oF 新版、CXL 3.x、Ultra Ethernet 规范）、能改变采购优先级的监管。
   - 低：小众 demo、个人项目、限定地区或非主流厂商的小动作。

6. **可操作性**：分析师看完能立即采取什么动作？
   - 高：能纳入选型评估、跑 PoC、调整路线图、写进采购需求、与现有方案对标的内容。
   - 低：纯舆论、股价预测、八卦、年终回顾、消费应用趣闻。

7. **时效性**：是不是当前真在发生的事？
   - 高：刚发布的产品/版本、最近一周内的官方动作。
   - 低：把几个月前就传开的事再讲一遍的综述、回顾文章。

按上述维度综合后，**优先选信息价值最高的几条；价值不够就少选甚至空列表**（宁缺毋滥）。
间接相关条目每个分类最多保留 1 条作为"市场温度计"，不要堆。

---

「带增量的内容」（视为重要，优先选）：
1. 存储 / AI 基建产品的新 release（带 GB/s、IOPS、PB、$、参数量、上下文长度等具体数字）
2. 新硬件 / 互联协议（CXL / Ultra Ethernet / NVLink 升级 / GPUDirect 新特性 / 新 NIC / BlueField）
3. 新基准结果：MLPerf Storage / Training / Inference、IO500、SPECstorage、reference architecture 性能数据
4. 重大架构演进：分离式推理 / disagg KV cache / megakernel / 新文件系统语义 / S3 协议扩展 / 新 vector index
5. 新基础模型 release（影响存储栈：参数量、上下文、checkpoint、数据规模、token 经济学）
6. 训练 / 推理框架的重大更新：FSDP2 / Megatron-Core / vLLM / SGLang / TensorRT-LLM / Dynamo / Ray / PyTorch / JAX
7. 厂商间的合作 / 认证（如 VAST × NVIDIA SuperPOD / WEKA × DGX Cloud / Pure × OpenAI 等具体大单）
8. 出口管制 / 监管动作对算力 / 存储采购的实际约束（具体条款 + 生效日期 + 受影响产品）

「不带增量的内容」（即使是真事也要降权或丢弃）：
- 既成事实复盘、综述、年终盘点 —— 没有新数据、没有新公告
    × "Chinese A.I. Firms Push Beyond Nvidia as DeepSeek Turns to Huawei"（综述，事实已传几个月）
    × "How OpenAI Won the AI Race"（盘点 / 复盘）
    × "X 公司今年的 5 个重要时刻"（年终回顾）
- 第三方趋势分析 / 行业评论（"据知情人士 / 业内认为 / 据传"无官方一手依据）
- 股价预测、分析师评级、目标价、投资 listicle
- how-to / 教程 / 最佳实践（除非是厂商官方首次发布的部署指引或参考架构）
- 消费类 AI 应用八卦、聊天机器人吐槽、UI 改版
- 二次转载、同质化新闻（多家媒体复述同一通稿时只留一份且优先官方源）

判断"是否带新增量"的简单准则：
- 标题 / 摘要含**具体型号 / 版本号 / 新数字 / 新日期 / 新基准**？没有 → 多半是复盘
- 信源是否在 [官方] 列表？是 → 强烈优先；否 → 谨慎，可能是综述
- 同一事件如果"几个月前就在传"，今天的报道一般是综述

权威性偏好（重要！）：
- **优先 [官方] 条目** —— 公司官网 / 官方博客 / 官方 GitHub release / arXiv 一作
- 同一事件多版本时**优先官方版**（转载常滞后、且可能曲解技术细节，对规划者尤其危险）
- **如果某分类候选里全是复盘 / 综述 / 解读，没有带新增量的一手内容，宁可返回空列表**"""


def _domain(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def is_official_for(item: dict, keyword: dict) -> bool:
    """判断该 item 对于此 keyword 是否来自官方源（域名匹配）。"""
    d = _domain(item.get("url", ""))
    if not d:
        return False
    targets = set(GLOBAL_OFFICIAL_DOMAINS) | set(keyword.get("official_domains") or [])
    for t in targets:
        t = t.lower().lstrip(".")
        if d == t or d.endswith("." + t):
            return True
    return False


def _format_candidate(idx: int, item: dict, official: bool, compact: bool = False) -> str:
    """格式化候选条目。compact=True 时摘要压缩（粗筛触发时省 token）。"""
    sig = item.get("signals", {})
    sig_str = ", ".join(f"{k}={v}" for k, v in sig.items() if v) or "无客观信号"
    summary = (item.get("summary") or "").strip()
    summary = re.sub(r"<[^>]+>", " ", summary)
    limit = 400 if compact else 1600
    summary = re.sub(r"\s+", " ", summary)[:limit]
    tag = "[官方] " if official else ""
    if item.get("watchlisted"):
        tag = "【重点关注】" + tag
    matched = item.get("matched_keywords") or []
    mk_str = ", ".join(matched) if matched else "(无明确匹配)"
    return (
        f"[#{idx}] {tag}源={item['source']} | 域名={_domain(item.get('url', ''))} | 命中关键词: {mk_str} | {sig_str}\n"
        f"  标题: {item['title']}\n"
        f"  摘要: {summary or '(原 RSS 无摘要)'}"
    )


def _parse_response(text: str) -> list[dict]:
    try:
        return json.loads(text).get("selected", [])
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return []
        try:
            return json.loads(m.group(0)).get("selected", [])
        except json.JSONDecodeError:
            return []


def load_llm_config(path: Path) -> dict:
    cfg = {
        "base_url": None,
        "model": "gpt-4o-mini",
        "api_key": None,
        "top_k_per_category": 5,
        "max_candidates_per_category": 40,
    }
    if path.exists():
        cfg.update(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    cfg["base_url"] = os.getenv("LLM_BASE_URL", cfg["base_url"])
    cfg["model"] = os.getenv("LLM_MODEL", cfg["model"])
    cfg["api_key"] = os.getenv("LLM_API_KEY", cfg["api_key"])
    if not cfg["api_key"]:
        raise RuntimeError(
            "未配置 LLM API key —— 请设置环境变量 LLM_API_KEY，或用 --no-llm 跳过"
        )
    return cfg


def group_by_keyword(items: list[dict], keywords: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {kw["name"]: [] for kw in keywords}
    for it in items:
        for kw_name in it.get("matched_keywords", []):
            if kw_name in groups:
                groups[kw_name].append(it)
    # 官方源优先，再按 score 倒序 —— 影响进入 LLM 候选池的优先级
    for k, kw_list in [(kw["name"], kw) for kw in keywords]:
        groups[k].sort(
            key=lambda x: (is_official_for(x, kw_list), x.get("score", 0)),
            reverse=True,
        )
    return groups


def _build_user_preference_block(text: str | None) -> str:
    """把用户自填的个性化偏好包成一个独立段落注入 prompt；为空时不输出该段。"""
    if not text or not text.strip():
        return ""
    return (
        "\n用户的个性化偏好（来自界面输入，作为额外的高优先级判断依据）：\n"
        f"  > {text.strip()}\n"
        "把它叠加到上面的「信息价值」五个维度之上 —— "
        "符合用户偏好的条目应被显著加权；与用户偏好明显不符（但仍有价值）的条目可保留 1-2 条不要全部丢。"
    )


VERIFY_PROMPT = """你是新闻摘要的事实核查员。判断下面这条摘要是否含有"主观推测"或"原文未提及的扩写"。

【原始标题】{title}
【原文摘要 / 可用信息】{original}

【待审核摘要】
{summary}

任务：
1. 检查摘要中是否含有以下问题：
   - 推测词（"可能"、"或将"、"有望"、"预计"、"显示...关注"等），且原文/标题中**没有**出现
   - 评论性词语（"重大意义"、"格局改变"、"标志着"、"凸显"、"反映了"等）
   - 原文/标题里没有的细节、数字、背景或因果解读
2. 如有任一问题：写一个严格基于"原始标题 + 原文摘要"的新版本（中文 8-15 句 500-1000 字），**保留原文已经提到的事件、性能数据、方法细节、影响等所有具体细节**，只剔除推测/评论/超出原文的内容；如果原文本身就只够写两三句，就老实保留短长度，不要编造内容凑字数
3. 若完全忠实：原样返回原摘要。

仅返回 JSON，不要任何其他内容：
{{"clean": <true|false>, "summary": "<最终摘要>"}}"""


def verify_summary(
    title: str,
    original_text: str,
    generated: str,
    client: OpenAI,
    model: str,
) -> tuple[str, bool]:
    """事实自检 LLM 生成的摘要，返回 (摘要文本, 是否被改写)。失败时静默不改。"""
    if not generated or not generated.strip():
        return generated, False
    prompt = VERIFY_PROMPT.format(
        title=title,
        original=original_text or "(原文 RSS 无可用摘要，仅有标题)",
        summary=generated,
    )
    try:
        text = _chat(
            client, model, prompt,
            kind="verify_summary", temperature=0.1,
            response_format={"type": "json_object"},
        )
        data = json.loads(text)
    except Exception as e:
        print(f"    [verify] 失败: {e}")
        return generated, False

    new_summary = (data.get("summary") or generated).strip()
    clean = bool(data.get("clean", True))
    return new_summary, (not clean)


TRANSLATE_PROMPT = """请把下面的新闻/论文摘要翻译为中文。

严格规则：
1. **忠实原文**，不增不减事实，不扩写、不评论、不省略
2. 保留所有专有名词（人名、公司、产品、技术名、版本号、模型名）原样不译
3. 数字、单位、对比基线保留原样
4. 仅输出中文翻译，不要任何前缀、解释或 JSON

原文：
{text}"""


def _is_mostly_chinese(text: str) -> bool:
    if not text:
        return True
    cjk = sum(1 for c in text if "一" <= c <= "鿿")
    return cjk / max(len(text), 1) > 0.20


_NUMERIC_TOKEN_RE = re.compile(
    r"(?:\d+(?:\.\d+)?[%×]?|\d+\.?\d*[KMBkmbx倍]+|v\d+(?:\.\d+)*|"
    r"\d+\s*(?:倍|百分点|tokens?/s|tok/s|GB|MB|TB|FLOPs?|参数|分))",
    re.IGNORECASE,
)


def _extract_numeric_claims(text: str) -> list[str]:
    """从标题里抠出所有"数字 + 单位/倍数"型 token —— 这些是最容易被幻觉的内容。"""
    if not text:
        return []
    out = []
    for m in _NUMERIC_TOKEN_RE.finditer(text):
        tok = m.group(0).strip()
        # 过滤明显是日期/年份 (>= 4 位纯数字) 的情况
        if re.fullmatch(r"\d{4}", tok):
            continue
        out.append(tok)
    return out


def _numeric_claim_supported(claim: str, sources: str) -> bool:
    """检查 claim 是否在 sources 文本里出现过（数字部分必须能对上）。"""
    if not claim:
        return True
    digits = re.findall(r"\d+\.?\d*", claim)
    if not digits:
        return True
    sources_lc = sources.lower()
    # 每个数字片段都必须在源文本里出现
    return all(d in sources_lc for d in digits)


_TITLE_REWRITE_PROMPT = """你是新闻标题事实核查员。

【原标题】{orig_title}
【原文摘要 / 可用信息】{source}

【待审核的中文标题】{candidate_title}

【问题】这个中文标题里出现了下列内容，**原标题和原文摘要里都查不到**：
{bad_claims}

请重写这个中文标题，要求：
- **若问题是"主体"（公司/产品名）幻觉**：抛弃原候选标题，直接基于原标题翻译/改写，**不要硬塞任何具体公司名**。
  例：原标题 "Forget Cerebras IPO and Focus on This Tech Giant Instead" → 直接译成 "评论：与其关注 Cerebras IPO 不如看另一家科技巨头"
- 若问题是"数字"幻觉：去掉所有原文未提及的数字、倍数、百分比、benchmark 分数，用定性词代替
- 保留原意，长度 ≤30 字
- 仅返回新标题文字，不要任何前缀/解释/引号"""


def _extract_subject_entities(text: str, keywords_meta: list[dict] | None) -> list[str]:
    """从中文标题里抠出可能的公司/产品主体名（命中跟踪关键词的）。"""
    if not text or not keywords_meta:
        return []
    out = []
    text_lc = text.lower()
    for kw in keywords_meta:
        names = [kw["name"]] + list(kw.get("aliases") or [])
        for n in names:
            if not n:
                continue
            if n.lower() in text_lc:
                out.append(kw["name"])
                break
    return list(dict.fromkeys(out))  # 去重保序


def verify_titles(
    grouped: dict[str, list[dict]],
    llm_config: dict,
    keywords: list[dict] | None = None,
) -> None:
    """对所有 picks 的 display_title 做幻觉检测：
    1) 标题里出现的数字必须在原文找得到；
    2) 标题里出现的公司/产品名（跟踪关键词）必须出现在原标题或原 RSS 摘要中。
    任一不通过 → 让 LLM 重写或直接回退到翻译后的原标题。"""
    client = OpenAI(api_key=llm_config["api_key"], base_url=llm_config.get("base_url"))
    model = llm_config["model"]
    n_checked = n_fixed = 0
    _chg: list[dict] = []
    for picks in grouped.values():
        for p in picks:
            disp = p.get("display_title") or ""
            if not disp:
                continue
            n_checked += 1
            # **关键**：校验只能基于"原文"（原标题 + 原 RSS 摘要），
            # 不能包括 LLM 自己生成的 summary —— 否则 LLM 标题写"阿里巴巴"，
            # LLM summary 也写"阿里巴巴"，互相印证就坐实了幻觉
            orig_only = ((p.get("title") or "") + " " + (p.get("raw_rss_summary") or ""))
            sources = orig_only
            # 校验 1: 数字幻觉
            claims = _extract_numeric_claims(disp)
            bad_numbers = [c for c in claims if not _numeric_claim_supported(c, sources)]
            # 校验 2: 主体（公司/产品）幻觉
            subjects_in_title = _extract_subject_entities(disp, keywords)
            orig_haystack = orig_only.lower()
            bad_subjects = []
            for s in subjects_in_title:
                # 把 keyword + aliases 全部查一遍，任一在原标题/原摘要里即算找到
                kw_meta = next((k for k in (keywords or []) if k["name"] == s), None)
                if not kw_meta:
                    continue
                terms = [s] + list(kw_meta.get("aliases") or [])
                if not any(t.lower() in orig_haystack for t in terms if t):
                    bad_subjects.append(s)
            if not bad_numbers and not bad_subjects:
                continue
            bad = bad_numbers + [f"主体:{s}（原文未提及）" for s in bad_subjects]
            # 发现编造，让 LLM 重写
            try:
                new_title = _chat(
                    client, model,
                    _TITLE_REWRITE_PROMPT.format(
                        orig_title=p.get("title") or "",
                        source=(p.get("raw_rss_summary") or p.get("summary") or "")[:1200],
                        candidate_title=disp,
                        bad_claims="、".join(bad),
                    ),
                    kind="title_verify", temperature=0.1,
                ).strip().strip('"「」"')
                if new_title and new_title != disp:
                    if TRACE.enabled:
                        _chg.append({"title": p.get("title", ""), "field": "display_title",
                                     "reason": "标题幻觉：" + "、".join(bad),
                                     "before": disp, "after": new_title})
                    p["display_title_original"] = disp
                    p["display_title"] = new_title
                    p["title_fixed"] = True
                    n_fixed += 1
            except Exception as e:
                print(f"    [title-verify] 失败: {e}")
    TRACE.changes("verify_titles", _chg)
    if n_checked:
        print(f"  [标题核查] 检查 {n_checked} 条，修正 {n_fixed} 条编造数字/主体的标题")


def _has_foreign_script(text: str) -> bool:
    """是否含整段非英文非中文文字（韩文 / 日文假名 / 阿拉伯文 / 西里尔文 / 泰文 等）。
    单独的拉丁字母（公司名/型号）不算。"""
    if not text:
        return False
    for c in text:
        # 韩文 Hangul
        if "가" <= c <= "힯" or "ᄀ" <= c <= "ᇿ" or "㄰" <= c <= "㆏":
            return True
        # 日文假名（汉字与中文共享 CJK 区，先不算）
        if "぀" <= c <= "ゟ" or "゠" <= c <= "ヿ":
            return True
        # 阿拉伯文 / 希伯来文 / 西里尔文 / 泰文 / 天城文
        if "؀" <= c <= "ۿ" or "֐" <= c <= "׿":
            return True
        if "Ѐ" <= c <= "ӿ" or "฀" <= c <= "๿" or "ऀ" <= c <= "ॿ":
            return True
    return False


def translate_summary(
    text: str,
    client: OpenAI,
    model: str,
    force: bool = False,
) -> tuple[str, bool]:
    """RSS 原文摘要翻译为中文，返回 (译文, 是否翻译过)。
    force=False（默认）：纯中文跳过；force=True：始终调 LLM 重写（用于清洗残留韩/日/阿拉伯文等）。"""
    if not text or not text.strip():
        return text, False
    if not force and _is_mostly_chinese(text):
        return text, False
    try:
        out = _chat(
            client, model, TRANSLATE_PROMPT.format(text=text),
            kind="translate", temperature=0.1,
        ).strip()
        if out:
            return out, True
    except Exception as e:
        print(f"    [translate] 失败: {e}")
    return text, False


def translate_all(
    grouped: dict[str, list[dict]],
    llm_config: dict,
) -> None:
    """翻译两类摘要：
      1) source=rss 且非中文 → 全量翻译
      2) source=llm 但残留韩/日/阿拉伯等非英非中文字 → 翻译
    原地标 p['translated']=True。"""
    client = OpenAI(api_key=llm_config["api_key"], base_url=llm_config.get("base_url"))
    model = llm_config["model"]
    n_tried = n_done = 0
    _chg: list[dict] = []
    for picks in grouped.values():
        for p in picks:
            src = p.get("summary_source")
            text = p.get("summary", "")
            needs_translate = False
            if src == "rss" and not _is_mostly_chinese(text):
                needs_translate = True
            elif src == "llm" and _has_foreign_script(text):
                needs_translate = True
            if not needs_translate:
                continue
            n_tried += 1
            # LLM 残留外文场景必须强制翻译（即使整体是中文）
            translated, ok = translate_summary(
                text, client, model, force=(src == "llm")
            )
            if ok:
                if TRACE.enabled:
                    _chg.append({"title": p.get("title", ""), "field": "summary",
                                 "reason": f"翻译为中文（src={src}）",
                                 "before": text, "after": translated})
                p["summary_original_en"] = text
                p["summary"] = translated
                p["translated"] = True
                n_done += 1
    TRACE.changes("translate", _chg)
    if n_tried:
        print(f"  [翻译] 检查 {n_tried} 条摘要，翻译 {n_done} 条")


def verify_all(
    grouped: dict[str, list[dict]],
    llm_config: dict,
) -> None:
    """对 LLM 生成的摘要做二次自检，原 RSS 摘要的条目跳过（已是原文）。"""
    from core.summary import clean_rss_summary

    client = OpenAI(api_key=llm_config["api_key"], base_url=llm_config.get("base_url"))
    model = llm_config["model"]
    n_checked = n_fixed = 0
    _chg: list[dict] = []
    for picks in grouped.values():
        for p in picks:
            if p.get("summary_source") != "llm":
                continue
            n_checked += 1
            original = clean_rss_summary(p.get("raw_rss_summary", "")) or ""
            before_text = p.get("summary", "")
            new, modified = verify_summary(
                p["title"], original, before_text, client, model
            )
            if modified:
                if TRACE.enabled:
                    _chg.append({"title": p.get("title", ""), "field": "summary",
                                 "reason": "自检剔除推测/扩写",
                                 "before": before_text, "after": new})
                p["summary"] = new
                p["verified"] = True
                n_fixed += 1
    TRACE.changes("verify_summary", _chg)
    if n_checked:
        print(f"  [自检] 检查 {n_checked} 条 LLM 生成摘要，修正 {n_fixed} 条")



def _is_official_any(item: dict, keywords: list[dict] | None) -> bool:
    """是否对任一跟踪关键词来自官方源（含 github / arxiv 全局白名单）。"""
    for kw in keywords or []:
        if is_official_for(item, kw):
            return True
    return False


def _is_strictly_official(item: dict, keywords: list[dict] | None) -> bool:
    """**严格**官方源：仅检查 keyword.official_domains（不含全局 github / arxiv）。
    用于粗筛分级 —— 厂商官博 (vastdata.com / openai.com 等) > github / arxiv > 其他。
    GitHub Trending 列表里很多是 awesome 合集 / 教程，会污染候选，必须分级处理。"""
    d = _domain(item.get("url", ""))
    if not d:
        return False
    for kw in keywords or []:
        for t in (kw.get("official_domains") or []):
            t = t.lower().lstrip(".")
            if d == t or d.endswith("." + t):
                return True
    return False


def _rule_prefilter(
    pool: list[dict],
    keywords: list[dict] | None,
    keep_n: int,
) -> list[dict]:
    """基于规则的粗筛：把候选压到约 keep_n 条。

    策略：**官方一手源保底 + 关键词多样性轮转**。
      1. 先保底 ~1/4 名额给严格官方一手源（厂商/产品官博，按分数取头部），
         保证 vLLM/NVIDIA 官博这类首发不被稀释；
      2. 其余名额用关键词轮转填——每个主命中关键词轮流贡献最佳一条，
         保证 DeepSeek(Reasonix) / 华为(Tao 定律) / KV file system 等主题都能进 LLM，
         不被某个高产官方源（一天十几篇 NVIDIA 博客）刷屏挤掉真新闻。

    旧策略「严格官方全部保留」会让 28 篇官方博客占满 31 个名额，把只有 Google News
    报道的真新闻（Tao 定律、Reasonix）压到候选池尾部、永远进不了 LLM —— 故改为保底制。
    """
    tier1 = sorted(
        (it for it in pool if _is_strictly_official(it, keywords)),
        key=lambda x: float(x.get("score", 0) or 0), reverse=True,
    )
    reserve = min(len(tier1), max(1, keep_n // 4))   # 官方一手保底 ~1/4
    head = tier1[:reserve]
    head_ids = {id(it) for it in head}
    rest = [it for it in pool if id(it) not in head_ids]
    out = head + _diversify_by_keyword(rest)   # 其余按关键词多样性轮转
    return out[:keep_n]


def _format_pool_block(pool: list[dict], proxy_kw: dict, compact: bool = False) -> str:
    return "\n\n".join(
        _format_candidate(i, it, is_official_for(it, proxy_kw), compact=compact)
        for i, it in enumerate(pool)
    )


# ---------- 按分类独立 LLM 排序 ----------

# 学术论文域名白名单（命中即归类为 学术论文）
_PAPER_DOMAINS = (
    "arxiv.org", "openreview.net", "papers.nips.cc", "papers.neurips.cc",
    "proceedings.mlr.press", "aclanthology.org",
    "dl.acm.org", "ieeexplore.ieee.org", "usenix.org",
    "openaccess.thecvf.com", "biorxiv.org", "direct.mit.edu",
    "link.springer.com", "sciencedirect.com", "nature.com", "science.org",
)

# 政策导向的强信号词（标题/摘要命中任一即归类为 政策导向）
# 选词原则：只收"政府/监管/法律"语境下的明确动作词，避免商业含义（如裸 "ban" 在游戏里也用）
_POLICY_HINTS = (
    # 出口管制 / 制裁 / 关税
    "export control", "export controls", "export ban",
    "sanction", "sanctions", "sanctioned",
    "tariff", "tariffs",
    # 监管 / 反垄断 / 立法
    "antitrust", "anti-trust",
    "executive order", "white house",
    "ai act", "eu ai act", "chips act", "chip act",
    "data act", "ai bill", "ai regulation", "ai regulatory",
    "regulator", "regulators", "regulatory body",
    "national security review", "national security concern",
    # 监管机构动作
    "doj ", "ftc ", "sec sues", "european commission",
    "cyberspace administration", "cac approves", "cac orders",
    # 中文
    "出口管制", "出口许可", "实体清单",
    "反垄断", "行政命令",
    "国家安全审查", "网信办", "网信部门",
    "ai 法案", "ai法案", "数据安全法", "个人信息保护法",
    "工信部发布", "工信部出台",
    "禁售", "制裁", "限制出口",
)

# 政策导向的辅助域名信号（政府 / 监管机构）
_POLICY_DOMAIN_HINTS = (
    ".gov", ".gov.cn", "europa.eu", "bis.doc.gov", "whitehouse.gov",
    "ftc.gov", "sec.gov", "ec.europa.eu", "miit.gov.cn", "cac.gov.cn",
)


def _classify_by_keywords(matched: list[str]) -> str | None:
    """看命中的关键词集合，返回「存储/基建产品」或「模型/框架/算法」，否则 None。

    若同时命中两类，**存储相关性优先**（分析师的本职工作）。
    """
    has_storage = any(k in _STORAGE_INFRA_KEYWORDS for k in matched)
    has_model = any(k in _MODEL_FRAMEWORK_KEYWORDS for k in matched)
    if has_storage:
        return "存储/基建产品"
    if has_model:
        return "模型/框架/算法"
    return None


def heuristic_category(item: dict) -> str:
    """按 source 元数据 + 命中关键词 + URL + 内容信号把候选硬分到 6 档之一。

    优先级（前者命中就 return）：
      1. 命中基准评测关键词 / 标题摘要含 MLPerf/IO500/SPECstorage 等强信号 → 基准评测
      2. source.category == "paper" / "arXiv" / URL 在论文域名 → 学术论文
      3. source 名 "GitHub Trending" / URL 域名是 github.com → github趋势
      4. 标题/摘要含政策信号或政府域名 → 政策导向（早于官方源判断，防止把
         "工信部发布..." 误判为公司新闻）
      5. 命中关键词所属类别（存储/基建产品 或 模型/框架/算法）→ 用 _classify_by_keywords
      6. source.category == "official"：按 source 名启发式：
         vLLM/HF/PyTorch/OpenAI/Anthropic 等 → 模型/框架/算法
         其它 → 存储/基建产品
      7. 兜底 → 存储/基建产品
    """
    src_name = (item.get("source") or "")
    src_name_lc = src_name.lower()
    src_cat = (item.get("category") or "").lower()
    d = _domain(item.get("url", ""))
    title_lc = (item.get("title") or "").lower()
    summary_lc = (item.get("summary") or "").lower()
    text_lc = title_lc + " " + summary_lc
    matched = item.get("matched_keywords") or []

    # 0. AI 意见领袖：来自 influencer 源的发言，无条件进此桶（早于一切关键词判断）
    if src_cat == "influencer":
        return INFLUENCER_CATEGORY

    # 1. 基准评测（最优先 —— 不能让 MLPerf 官博被归到普通厂商博客）
    if any(k in _BENCHMARK_KEYWORDS for k in matched):
        return "基准评测"
    if any(h in text_lc for h in _BENCHMARK_HINTS_LC):
        return "基准评测"

    # 2. 学术论文
    if src_cat == "paper" or src_name.startswith("arXiv") or "arxiv" in src_name_lc:
        return "学术论文"
    if "daily papers" in src_name_lc or "hf papers" in src_name_lc:
        return "学术论文"
    for pd in _PAPER_DOMAINS:
        if d == pd or d.endswith("." + pd):
            return "学术论文"

    # 3. GitHub 趋势
    if src_name.startswith("GitHub Trending") or "github trending" in src_name_lc:
        return "github趋势"
    if d == "github.com" or d.endswith(".github.com"):
        return "github趋势"

    # 4. 政策导向（早于"官方源"判断 —— 工信部发布也是官方但应归政策）
    if any(h in text_lc for h in _POLICY_HINTS):
        return "政策导向"
    if any(h in d for h in _POLICY_DOMAIN_HINTS):
        return "政策导向"

    # 5. 命中关键词分桶（存储 vs 模型）
    by_kw = _classify_by_keywords(matched)
    if by_kw:
        return by_kw

    # 6. 官方源按 source name 启发式分桶
    if src_cat == "official":
        model_source_hints = (
            "vllm", "hugging face", "pytorch", "openai", "anthropic",
            "deepmind", "meta ai", "google research", "microsoft research",
            "meta engineering", "nvidia research", "daily papers", "cerebras",
            "groq", "sambanova",
        )
        if any(h in src_name_lc for h in model_source_hints):
            return "模型/框架/算法"
        return "存储/基建产品"

    # 7. 兜底
    return "存储/基建产品"


CATEGORY_PROMPT_TEMPLATE = """你在为「AI 存储技术规划与分析师」筛选「{category}」分类下今日资讯。

{rubric}

【本分类定义】
{category_def}
{user_preference_block}

【候选条目】共 {n} 条（已硬分类到「{category}」），按「官方源优先 + 客观分数」预排序。
每条都附带 [#编号] / 源 / 域名 / 命中关键词 等字段。

{candidates}

【任务】从中挑选最值得分析师关注的最多 {top_k} 条。**宁缺毋滥**：候选都不够格就少选甚至空列表。

**何时丢弃**（强硬执行）：
- Reddit / HN 上的个人发帖、提问、闲聊（除非确实是产品/项目首发）
- 内容农场 / SEO 模板 / 股评 / 八卦
- 命中关键词但主体不相关（例：标题提一句 NVIDIA 但文章在讲游戏对比 / 消费应用）
- 与 AI / 存储 / 算力 / 模型 / 数据基础设施完全无关
- 同事件多家媒体报道时只留一份（优先官方域名版本）

**何时保留**（按重要度从高到低）：
- 标【重点关注】的条目 —— 用户长期跟踪项，**只要当周有实质进展/新信息就优先纳入**（即使来源单一、客观分低）；仅当确属纯转载、旧闻复述、与该跟踪项无实质关系时才丢
- 来自 [官方] 标记的条目 —— 默认保留，除非确实跑题
- 厂商产品发布 / 版本 release / GA / 收购合作 / 采购大单 / reference architecture / 跨厂商认证
- 新硬件 / 互联协议（CXL / Ultra Ethernet / NVLink / GPUDirect 新特性）
- 性能基准（MLPerf / IO500 / SPECstorage）
- 系统级架构演进（分离式推理 / KV cache / 新文件系统 / S3 协议扩展）
- 新基础模型 release（带参数量、上下文、checkpoint 等量化信息）
- 训练 / 推理框架重大更新（FSDP2 / Megatron-Core / vLLM / SGLang / TensorRT-LLM / Dynamo）
- 监管 / 出口管制 / 政策法规（带具体条款 + 生效日期）

**display_title**: 中文新闻式短标题 (≤30 字)
- 把事件本质说清楚；论文要把核心方法 + 主要成果用日常语言写出来
- **主体一致性**：标题主体必须**和原标题的主体一致**，命中关键词只是匹配，不代表文章主体
- **数字真实性**：所有数字（倍率、百分比、版本、参数量、benchmark 分数）必须**逐字出现在原标题或摘要中**
- 原文没具体数字时**绝对不能**编造"X 倍加速"/"提升 N%"，用定性词代替
- 禁止 "重磅 / 震撼 / 颠覆 / 刷屏" 等夸张词
- 原标题已是简洁中文表达可直接保留

**summary**: 中文摘要，目标 **250-500 字**，信息密度优先于字数。读者是存储 / AI 基建规划分析师，
只看摘要就该掌握：发生了什么 + 关键量化指标 + 对存储 / AI 基建栈的具体含义。

**核心原则（严格优先级）**：**忠实性 > 数字密度 > 字数**。
- 原文有数字 → 全部写进来；原文没有 → 用具体的「版本号 / 产品名 / 动作」代替，**绝不允许编造数字**
- 字数不够也不要靠主观评论凑数

结构（按顺序写，原文没提就跳过，不要编造）：
1. **首句**：谁 + 做了什么 + （原文里有的）关键数字或版本号 / 产品名
2. **量化细节 2-4 句**：原文里所有"分析师能代入决策模型"的数字
   - 性能侧：带宽 / IOPS / 延迟 / 容量 / 参数量 / 上下文长度 / 推理 tok/s / benchmark 分数
   - 经济侧：$/GB / $/TB / $/M tokens / 合同金额 / 价格变动 / 出货量
   - 跨厂商对标：原文如果做了"X vs Y"的对比，**必须保留**对比对象和差距
3. **规划含义 1 句**（仅当原文已明确）：对存储 / AI 基建规划的具体影响。原文没提就跳过

强制规则：
- 首句**优先**含具体数字 / 版本号 / 产品名 / 对比基线 —— 原文若一项都没有，写动作 + 主体即可
- summary 必须 100% 中文（专有名词可保留原拼写）；不允许整句外文
- 禁止填充句：「具有重要意义」「值得关注」「这一举措体现了战略」等
- 禁止评价性词：重磅 / 震撼 / 颠覆 / 标志着 / 凸显
- 禁止推测词：可能 / 或将 / 有望 / 预计 / 据悉
- 候选里的 signals 字段（points / cross_source_count / 命中关键词）**绝对不能**写进摘要
- 若原文确实只有标题，写「原文仅有标题，未提供详细内容，请点击链接查阅。」即可

仅返回 JSON，不要其它任何输出：
{{"selected": [{{"id": <候选编号>, "display_title": "<中文短标题>", "summary": "<中文摘要>"}}]}}"""


# AI 意见领袖（Twitter/X 大神发言）专用 prompt —— 评判标准与存储 rubric 不同：
# 看的是"观点影响力/信息量"，不要求存储相关、不强制数字结构。
INFLUENCER_PROMPT_TEMPLATE = """你在为「AI 存储技术规划与分析师」整理今日「AI 意见领袖」一手发言（来自 Twitter/X）。
读者把这些大神的发言当**行业风向标**：知道他们在想什么、判断什么、做什么动向。

【本分类定义】
{category_def}
{user_preference_block}

【候选发言】共 {n} 条原创推文（已过滤转发/回复），每条附 [#编号] / 作者源 / 发布时间。

{candidates}

【任务】挑选最值得分析师知道的最多 {top_k} 条。**宁缺毋滥**：当天若没有有信息量的发言就少选或空列表。

**优先选**（按价值从高到低）：
- 范式级技术观点 / 新概念（如 Software 3.0、vibe coding、对推理栈/KV cache/存储/硬件的判断）
- 重要职业/机构动向（加入/离开/创办，如"加入 Anthropic"）
- 对某模型/框架/芯片/产品的一手评测或评论
- 新工作 / 新工具 / 论文的作者本人预告或解读

**丢弃**：
- 纯生活/玩梗/情绪、无信息量的转发感想
- 一句话调侃、纯链接无观点、活动寒暄
- 已被同义新闻覆盖且推文本身没有增量的

**display_title**: 中文新闻式短标题（≤30 字），点明"谁 + 说了/做了什么"。例：「Karpathy 宣布加入 Anthropic」。
- 保留原发言里的关键概念词（Software 3.0 / vibe coding 等）原样
- 禁止夸张词（重磅/震撼/颠覆），禁止编造原文没有的数字

**summary**: 中文，**80-300 字**，忠实转述这条发言**说了什么 + 为何值得知道**。
- 100% 中文（专有名词/概念词可保留原拼写，如 Software 3.0、Anthropic）
- 只写推文里真实表达的内容，**不得脑补/扩写/编造数字**
- 若发言含明确技术判断，点出它对 AI 基建/存储规划的潜在含义（仅当能自然带出，不强求）

仅返回 JSON，不要其它任何输出：
{{"selected": [{{"id": <候选编号>, "display_title": "<中文短标题>", "summary": "<中文摘要>"}}]}}"""


DEFAULT_CATEGORY_DEFS = {
    "存储/基建产品": (
        "存储 / AI 基建厂商的产品级动态：产品发布、版本更新、GA、reference architecture、"
        "跨厂商认证、采购大单、收购合作。覆盖 VAST / WEKA / DDN / Pure / NetApp / Hammerspace / "
        "IBM / Dell / HPE / 华为 / 浪潮 / 焱融 / XSKY / 杉岩；并行文件系统（Lustre / Ceph / "
        "DAOS / BeeGFS / JuiceFS / Alluxio / MinIO / 3FS）；GPU IO（GPUDirect / NVMe-oF / RDMA / NIXL / "
        "BlueField / Spectrum-X）；互联协议（CXL / HBM / NVLink / Ultra Ethernet）；SSD/HBM 上游"
        "（Solidigm / Kioxia / Micron / DapuStor / Pliops）；超大规模云存储（AWS / Azure / GCP）；"
        "向量库（Milvus / Pinecone / Weaviate / Qdrant / Lance）；国内云存储（阿里 / 腾讯 / 字节 / 百度 / "
        "运营商）；Neocloud（CoreWeave / Lambda / Crusoe / Nebius / Together / Fireworks）。"
    ),
    "模型/框架/算法": (
        "大模型 release（OpenAI / Anthropic / DeepMind / Meta / Mistral / xAI / Qwen / Kimi / "
        "Zhipu / MiniMax / Doubao / Hunyuan / ERNIE / DeepSeek）、训练推理框架（PyTorch / JAX / "
        "vLLM / SGLang / TensorRT-LLM / Triton / Dynamo / DeepSpeed / Megatron / Ray / LangChain / "
        "LlamaIndex）、推理优化算法（speculative decoding / disaggregated inference / KV Cache / "
        "LMCache / Mooncake）、训练范式（MoE / RLHF / long context / RAG / agentic / multimodal）、"
        "AI 芯片（AMD / Intel / TPU / Trainium / Ascend / Cambricon / Cerebras / Groq）。"
    ),
    "基准评测": (
        "AI 存储/算力的官方权威基准：MLPerf Storage / MLPerf Training / MLPerf Inference / "
        "IO500 / SPECstorage Solution / DLIO / STAC-AI / TPCx-AI；以及厂商发布的可对比性能数据"
        "（带具体数字 + 测试方法）。**纯营销 PR 中的「X 倍加速」不算**，必须是有第三方/标准方法支撑的。"
    ),
    "学术论文": (
        "arXiv（cs.DC / OS / AR / PF / DB / LG / AI 等）/ 会议 / 期刊论文，研究成果、新基准、新数据集"
        "（首发，以论文形式发表）。系统类顶会（FAST / OSDI / SOSP / NSDI / ATC / ISCA / SIGMOD / VLDB）"
        "对存储规划尤其关键。"
    ),
    "政策导向": (
        "政府 / 监管 / 法律层面的政策、法规、行政命令、出口管制（对算力 / 存储采购的实际约束）；"
        "行业标准组织（OCP / SNIA / NVM Express / UEC / UALink / CNCF）与跨厂商联盟动作；"
        "国家算力网络规划（东数西算、算力网络、国产替代）。**不针对单一公司商业动作**。"
    ),
    "github趋势": (
        "github.com 上的项目（trending repo / 新发版 / 高 star 增速），以代码仓库为主体。"
        "对存储分析师特别有价值的子类：并行文件系统、KV cache、向量库、分布式训练推理框架开源项目。"
        "**排除 awesome 列表 / 教程合集 / 个人 dotfiles**。"
    ),
    "AI意见领袖": (
        "AI 领域有影响力人物（Karpathy / 吴恩达 / Tri Dao / Jim Fan / Soumith / Sam Altman 等）"
        "在 Twitter/X 上的一手原创发言：技术判断、范式观点（如 Software 3.0 / vibe coding）、"
        "职业动向（加入/离开某机构）、对模型/框架/硬件/推理栈的评论、新工作/新工具预告。"
        "**这是给存储分析师的「行业风向标」**——不要求每条都直接讲存储，重在谁说了什么、为何值得知道。"
    ),
}

# ---------- Prompt override（持久化用户修改的 rubric 与 category defs）----------
#
# 文件位置: config/prompt_override.yaml（gitignored）
# 文件格式:
#     version: 1
#     updated_at: "2026-05-17T15:00:00"
#     importance_rubric: |
#       （整段 rubric 文本，缺省/空则用 DEFAULT_IMPORTANCE_RUBRIC）
#     category_defs:
#       存储/基建产品: |
#         （单个分类的定义，缺省则用 DEFAULT_CATEGORY_DEFS 对应 key）
#       模型/框架/算法: |
#         ...
#
# 每个字段独立 fallback —— 用户可以只覆盖 rubric，分类定义继续用默认。

PROMPT_OVERRIDE_PATH = Path(__file__).resolve().parent.parent / "config" / "prompt_override.yaml"


def load_prompt_overrides() -> dict:
    """读 prompt_override.yaml；不存在/解析失败返回 {}。每次调用都重读，便于热更新。"""
    try:
        if not PROMPT_OVERRIDE_PATH.exists():
            return {}
        data = yaml.safe_load(PROMPT_OVERRIDE_PATH.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"  ! prompt_override 加载失败，回退默认: {e}")
        return {}


def get_importance_rubric() -> str:
    """返回当前生效的 rubric（覆盖优先 → 默认）。"""
    override = load_prompt_overrides()
    text = (override.get("importance_rubric") or "").strip()
    return text or DEFAULT_IMPORTANCE_RUBRIC


def get_category_defs() -> dict[str, str]:
    """返回当前生效的 category_defs，按 key 逐项 fallback 到默认。"""
    override = load_prompt_overrides()
    user_map = override.get("category_defs") or {}
    out = dict(DEFAULT_CATEGORY_DEFS)
    if isinstance(user_map, dict):
        for k, v in user_map.items():
            if isinstance(v, str) and v.strip():
                out[k] = v.strip()
    return out


def save_prompt_overrides(
    importance_rubric: str | None,
    category_defs: dict[str, str] | None,
) -> None:
    """把用户修改写回 config/prompt_override.yaml。
    传 None 表示该字段恢复默认（不写入覆盖文件）。"""
    data: dict = {
        "version": 1,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if importance_rubric and importance_rubric.strip():
        data["importance_rubric"] = importance_rubric.strip()
    if category_defs:
        kept = {k: v.strip() for k, v in category_defs.items()
                if isinstance(v, str) and v.strip()}
        if kept:
            data["category_defs"] = kept
    PROMPT_OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 自定义 representer：让多行字符串走 `|` 块字面量风格，短字符串保持普通形式
    class _PromptDumper(yaml.SafeDumper):
        pass

    def _str_repr(dumper, value):
        style = "|" if "\n" in value or len(value) > 80 else None
        return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)

    _PromptDumper.add_representer(str, _str_repr)

    with open(PROMPT_OVERRIDE_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, Dumper=_PromptDumper, allow_unicode=True, sort_keys=False)


def reset_prompt_overrides() -> bool:
    """删除覆盖文件，恢复默认。返回是否真的删除了。"""
    if PROMPT_OVERRIDE_PATH.exists():
        PROMPT_OVERRIDE_PATH.unlink()
        return True
    return False


# 各分类的候选池上限（学术论文桶要保护 cs.DC/OS/PF，所以单独放大）
_CATEGORY_MAX_CANDIDATES = {
    "学术论文": 80,
    "存储/基建产品": 40,
    "模型/框架/算法": 40,
    "基准评测": 30,
    "政策导向": 30,
    "github趋势": 30,
}


def _diversify_by_keyword(sorted_pool: list[dict]) -> list[dict]:
    """按主命中关键词做轮转（round-robin），保证候选池被 token 预算截断时，
    每个不同主题都能把自己的"最佳一条"排到前面进 LLM，而不是被某个高产官方源
    （NVIDIA/vLLM 博客）刷屏挤掉真新闻（如华为 Tao 定律、KV file system）。

    入参 sorted_pool 已是 (官方优先, 分数) 降序；同一关键词组内沿用该次序，
    所以每个关键词最先吐出的就是它最权威/最高分的一条。
    """
    from collections import OrderedDict
    groups: "OrderedDict[str, list]" = OrderedDict()
    for it in sorted_pool:
        mks = it.get("matched_keywords") or ["_"]
        groups.setdefault(mks[0], []).append(it)
    out: list[dict] = []
    while any(groups.values()):
        for lst in groups.values():
            if lst:
                out.append(lst.pop(0))
    return out


def _sort_and_prefilter_for_category(
    pool: list[dict],
    keywords: list[dict],
    base_tokens: int,
    category: str,
) -> tuple[list[dict], str]:
    """对单个分类的候选池做排序 + 自适应粗筛，返回 (final_pool, candidates_block)。"""
    sorted_pool = sorted(
        pool,
        key=lambda x: (
            bool(x.get("watchlisted")),   # 重点关注名单：保送到最前，token 粗筛时也优先保留
            _is_strictly_official(x, keywords),
            _is_official_any(x, keywords),
            float(x.get("score", 0) or 0),
        ),
        reverse=True,
    )
    proxy_kw = keywords[0] if keywords else {"name": ""}

    full_block = _format_pool_block(sorted_pool, proxy_kw)
    full_tokens = base_tokens + _estimate_tokens(full_block)
    if full_tokens <= PROMPT_TOKEN_BUDGET:
        return sorted_pool, full_block

    compact_block = _format_pool_block(sorted_pool, proxy_kw, compact=True)
    if base_tokens + _estimate_tokens(compact_block) <= PROMPT_TOKEN_BUDGET:
        print(
            f"    [{category}] {full_tokens} tok > budget → 启用紧凑摘要"
            f"（保 {len(sorted_pool)} 条全集）"
        )
        return sorted_pool, compact_block

    before = len(sorted_pool)
    lo, hi, best = 1, len(sorted_pool), 1
    while lo <= hi:
        mid = (lo + hi) // 2
        trial = _rule_prefilter(sorted_pool, keywords, mid)
        trial_block = _format_pool_block(trial, proxy_kw, compact=True)
        if base_tokens + _estimate_tokens(trial_block) <= PROMPT_TOKEN_BUDGET:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    final = _rule_prefilter(sorted_pool, keywords, best)
    final_block = _format_pool_block(final, proxy_kw, compact=True)
    n_official = sum(1 for it in final if _is_official_any(it, keywords))
    print(
        f"    [{category}] {full_tokens} tok > budget → "
        f"规则粗筛 {before} → {len(final)} 条（含 {n_official} 条官方源）"
    )
    return final, final_block


_FORCED_SUMMARY_PROMPT = """为下面每条「重点关注」新闻写中文短标题和中文摘要，读者是资深 AI 存储技术规划分析师。
严格输出 JSON 对象（不要 markdown、不要多余文字）：
{{"items": [{{"i": 编号, "display_title": "≤30 字中文新闻式短标题", "summary": "2-4 句中文摘要，点明事件本质及对存储/AI 基建规划的意义；只用下面给出的信息，不编造数字"}}]}}

新闻清单：
{catalog}"""


def _summarize_forced(picks: list[dict], client, model) -> None:
    """给「重点关注保底」补回的条目原地补上中文 display_title + llm_summary。
    这些条目没经过排序 LLM（故无现成中文标题/摘要），单独一次 LLM 调用批量补全。"""
    if not picks:
        return
    lines = []
    for i, p in enumerate(picks, 1):
        raw = re.sub(r"<[^>]+>", " ", p.get("summary") or "")
        raw = re.sub(r"\s+", " ", raw).strip()[:300]
        lines.append(f"{i}. 标题：{p.get('title','')}｜原始摘要：{raw or '(无)'}")
    prompt = _FORCED_SUMMARY_PROMPT.format(catalog="\n".join(lines))
    try:
        text = _chat(client, model, prompt, kind="watchlist_summary",
                     temperature=0.3, response_format={"type": "json_object"})
    except Exception as e:
        print(f"  [重点关注] 摘要生成失败，保留原始标题：{e}")
        return
    try:
        data = json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))
    except (json.JSONDecodeError, AttributeError):
        print("  [重点关注] 摘要解析失败，保留原始标题")
        return
    for rec in data.get("items") or []:
        try:
            idx = int(rec.get("i")) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(picks):
            dt = (rec.get("display_title") or "").strip()
            sm = (rec.get("summary") or "").strip()
            if dt:
                picks[idx]["display_title"] = dt
            if sm:
                picks[idx]["llm_summary"] = sm


def rank_by_category(
    items: list[dict],
    keywords: list[dict],
    llm_config: dict,
    user_preference: str | None = None,
) -> dict[str, list[dict]]:
    """按 4 个分类**各自**独立做 LLM 排序：
    - 先用 heuristic_category() 把命中条目硬分到 4 桶
    - 每桶单独走「官方源优先 + 客观分数」排序 → token budget 二分粗筛 → LLM 排序
    - 每桶各得 top_k_per_category 条
    """
    client = OpenAI(api_key=llm_config["api_key"], base_url=llm_config.get("base_url"))
    model = llm_config["model"]
    top_k = int(llm_config.get("top_k_per_category",
                               llm_config.get("top_k_per_keyword", 5)))
    default_max_pool = int(llm_config.get("max_candidates_per_category",
                                          llm_config.get("max_candidates_per_keyword", 40)))
    # 支持按分类配置；llm.yaml 里可写 max_candidates_per_category_overrides: {学术论文: 80}
    overrides: dict = llm_config.get("max_candidates_per_category_overrides", {}) or {}

    # 每次 rank_by_category 启动时重读 prompt_override.yaml（支持热更新，不需要重启）
    active_rubric = get_importance_rubric()
    active_category_defs = get_category_defs()
    if PROMPT_OVERRIDE_PATH.exists():
        print(f"  [Prompt] 已加载 override: {PROMPT_OVERRIDE_PATH.name}")

    if user_preference:
        print(f"  [LLM] 已注入个性化偏好（{len(user_preference)} 字符）")

    # 硬分桶
    buckets: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    for it in items:
        buckets[heuristic_category(it)].append(it)
    print("  [分桶] " + " | ".join(
        f"{c}={len(buckets[c])}" for c in CATEGORIES
    ))

    # 调试 dump
    debug_path = Path("/tmp/news_pool_by_category.log")
    debug_lines = [f"=== bucket dump @ {datetime.now().isoformat()[:19]} ==="]
    for c in CATEGORIES:
        debug_lines.append(f"\n--- {c} ({len(buckets[c])}) ---")
        for i, it in enumerate(buckets[c]):
            mk = ",".join(it.get("matched_keywords") or [])
            off = "[O]" if _is_official_any(it, keywords) else "   "
            debug_lines.append(
                f"  [{i:03d}] {off} score={it.get('score',0):.2f} mk=[{mk}] "
                f"src={it.get('source','')} | {it.get('title','')[:140]}"
            )
    debug_path.write_text("\n".join(debug_lines), encoding="utf-8")
    print(f"  [DEBUG] 候选 dump → {debug_path}")

    if TRACE.enabled:
        for c in CATEGORIES:
            TRACE.snapshot(f"rank.bucket.{c}", buckets[c], note="启发式硬分桶（LLM 排序前）")

    result: dict[str, list[dict]] = {c: [] for c in CATEGORIES}

    for cat in CATEGORIES:
        max_pool = int(overrides.get(cat, _CATEGORY_MAX_CANDIDATES.get(cat, default_max_pool)))
        # 在全桶上选 max_pool 条进 LLM。关键：要在排序后截断，否则 buckets[cat] 是抓取
        # 顺序，[:max_pool] 会把排在桶尾的真新闻（Tao 定律第 700 位）在排序前丢掉。
        # 用「质量头 + 多样性尾」平衡：前半按 (官方,分数) 取最强条目（保质量，避免纯轮转
        # 把多条强存储新闻稀释成 0），后半按关键词轮转补主题覆盖（保 Tao/Reasonix 等真新闻）。
        ordered = sorted(
            buckets[cat],
            key=lambda x: (
                bool(x.get("watchlisted")),   # 重点关注名单：无条件保送到最前，确保进 LLM 候选池
                _is_strictly_official(x, keywords),
                _is_official_any(x, keywords),
                float(x.get("score", 0) or 0),
            ),
            reverse=True,
        )
        head_n = max(1, max_pool // 2)
        head = ordered[:head_n]
        head_ids = {id(x) for x in head}
        tail = _diversify_by_keyword([x for x in ordered if id(x) not in head_ids])
        pool = (head + tail)[:max_pool]
        if not pool:
            print(f"  [LLM] {cat}: 0 条（候选池为空）")
            continue

        tpl = (INFLUENCER_PROMPT_TEMPLATE if cat == INFLUENCER_CATEGORY
               else CATEGORY_PROMPT_TEMPLATE)
        base_prompt = tpl.format(
            rubric=active_rubric,
            category=cat,
            category_def=active_category_defs.get(cat, DEFAULT_CATEGORY_DEFS.get(cat, "")),
            user_preference_block=_build_user_preference_block(user_preference),
            n=0, candidates="", top_k=top_k,
        )
        base_tokens = _estimate_tokens(base_prompt)

        final_pool, candidates_block = _sort_and_prefilter_for_category(
            pool, keywords, base_tokens, cat,
        )
        n_official = sum(1 for it in final_pool if _is_official_any(it, keywords))
        print(f"  [LLM] {cat}: 送 {len(final_pool)} 条候选（{n_official} 官方源）→ LLM 排序…")
        TRACE.snapshot(f"rank.{cat}.sent_to_llm", final_pool,
                       note=f"排序+粗筛后，实际送 LLM 的候选（{n_official} 官方源）")

        prompt = tpl.format(
            rubric=active_rubric,
            category=cat,
            category_def=active_category_defs.get(cat, DEFAULT_CATEGORY_DEFS.get(cat, "")),
            user_preference_block=_build_user_preference_block(user_preference),
            n=len(final_pool), candidates=candidates_block, top_k=top_k,
        )
        try:
            text = _chat(
                client, model, prompt,
                kind=f"rank:{cat}", temperature=0.2,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            print(f"  ! [{cat}] LLM 调用失败: {e}")
            continue

        for entry in _parse_response(text):
            idx = entry.get("id")
            if not isinstance(idx, int) or not (0 <= idx < len(final_pool)):
                continue
            if len(result[cat]) >= top_k:
                break
            src_item = final_pool[idx]
            proxy_kw_for_item = (
                next((kw for kw in keywords
                      if kw["name"] in (src_item.get("matched_keywords") or [])), None)
                or (keywords[0] if keywords else {"name": ""})
            )
            it = dict(src_item)
            it["llm_summary"] = entry.get("summary") or ""
            it["display_title"] = (entry.get("display_title") or "").strip() or it["title"]
            it["llm_category"] = cat
            it["is_official"] = is_official_for(it, proxy_kw_for_item)
            result[cat].append(it)
        print(f"  [LLM] {cat}: 选中 {len(result[cat])} 条")
        if TRACE.enabled:
            sel_ids = {it.get("id") for it in result[cat]}
            TRACE.note(f"rank.{cat}.llm_raw", "LLM 排序原始响应", data=(text or "")[:4000])
            TRACE.snapshot(f"rank.{cat}.selected", result[cat],
                           note=f"送 {len(final_pool)} → 选 {len(result[cat])}")
            TRACE.drops(f"rank.{cat}.not_selected",
                        [(it, "LLM 未选中") for it in final_pool
                         if it.get("id") not in sel_ids])

    # 重点关注保底：被跟踪词（watchlist）的最佳在窗新闻，若 LLM 因"宁缺毋滥"漏选，
    # 强制补回各自分类——保证"只要当周有该跟踪项的新闻就一定展示"。每个词补 1 条（最高分）。
    selected_ids = {it.get("id") for picks in result.values() for it in picks}
    by_term: dict[str, list[dict]] = {}
    for it in items:
        if it.get("watchlisted") and it.get("id") not in selected_ids:
            by_term.setdefault(it.get("watchlist_hit") or "_", []).append(it)
    forced_picks: list[tuple[str, dict]] = []
    for term, lst in by_term.items():
        best = max(lst, key=lambda x: float(x.get("score", 0) or 0))
        cat = heuristic_category(best)
        proxy_kw_for_item = (
            next((kw for kw in keywords
                  if kw["name"] in (best.get("matched_keywords") or [])), None)
            or (keywords[0] if keywords else {"name": ""})
        )
        pick = dict(best)
        pick["display_title"] = best.get("title", "")  # 先占位，下面 LLM 补中文标题
        pick["llm_category"] = cat
        pick["is_official"] = is_official_for(best, proxy_kw_for_item)
        pick["forced_watchlist"] = True
        forced_picks.append((cat, pick))
    if forced_picks:
        # 单次 LLM 调用给这些补回项补中文标题 + 摘要（它们没走排序 LLM）
        _summarize_forced([p for _, p in forced_picks], client, model)
        for cat, pick in forced_picks:
            result[cat].append(pick)
        print(f"  [重点关注] 保底补回 {len(forced_picks)} 条被 LLM 漏选的跟踪项（{', '.join(by_term)}）")
        TRACE.snapshot("rank.watchlist_forced", [p for _, p in forced_picks],
                       note="LLM 漏选、按 watchlist 强制保底补回")

    return result


def rank_all(
    items: list[dict],
    keywords: list[dict],
    llm_config: dict,
    user_preference: str | None = None,
) -> dict[str, list[dict]]:
    """对外接口：按分类独立排序。"""
    return rank_by_category(items, keywords, llm_config, user_preference)
