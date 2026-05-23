"""本地敏感凭据持久化 —— TAVILY_API_KEY 等。

文件 config/secrets.yaml（已 gitignore）。结构：
    TAVILY_API_KEY: tvly-xxx
    EMAIL_PASSWORD: ...
    FEISHU_WEBHOOK: ...

`load_secrets()` 在 app 启动时调用，把文件里有的字段注入 os.environ；
已有环境变量优先，不会被覆盖。
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

SECRETS_PATH_DEFAULT = Path(__file__).parent.parent / "config" / "secrets.yaml"

# 当前支持的字段；不在白名单里的不会注入，避免误覆盖
KNOWN_KEYS = {"TAVILY_API_KEY", "EMAIL_PASSWORD", "FEISHU_WEBHOOK", "WECOM_WEBHOOK", "LLM_API_KEY"}


def load_secrets(path: Path = SECRETS_PATH_DEFAULT) -> dict[str, str]:
    """从 secrets.yaml 加载键值并注入 os.environ（环境变量已存在则不覆盖）。
    返回最终生效的 KNOWN_KEYS 子集（仅文件存在的字段）。"""
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        print(f"  ! secrets.yaml 解析失败: {e}")
        return {}
    loaded: dict[str, str] = {}
    for k, v in data.items():
        if k not in KNOWN_KEYS:
            continue
        if not isinstance(v, str) or not v.strip():
            continue
        loaded[k] = v.strip()
        # 不覆盖已存在的环境变量（系统层 export 优先级更高）
        os.environ.setdefault(k, v.strip())
    return loaded


def save_secret(key: str, value: str, path: Path = SECRETS_PATH_DEFAULT) -> None:
    """更新某个字段；空字符串表示删除。同步注入当前进程 os.environ。"""
    if key not in KNOWN_KEYS:
        raise ValueError(f"不支持的 secret 键: {key}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if path.exists():
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            data = {}
    if value:
        data[key] = value
        os.environ[key] = value
    else:
        data.pop(key, None)
        os.environ.pop(key, None)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
