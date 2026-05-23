"""时区工具 —— 内部存 UTC ISO，对外显示一律转本地时间。

所有时间解析的唯一入口是 parse_dt：接受 ISO / RFC822 字符串、time.struct_time、
datetime，统一归一为 aware UTC datetime；无法解析或为空返回 None。
不要在别处再写 datetime.fromisoformat —— 一处解析，行为一致。
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime


def parse_dt(value) -> datetime | None:
    """统一时间解析。成功返回 aware UTC datetime；空 / 无法解析返回 None。

    支持:
      - time.struct_time（feedparser 的 *_parsed 字段，已是 UTC）
      - datetime（naive 视为 UTC）
      - ISO 8601 字符串（含尾缀 Z）
      - RFC822 字符串（RSS 标准 pubDate，如 "Tue, 19 May 2026 08:30:00 GMT"）
    """
    if value is None:
        return None
    if isinstance(value, time.struct_time):
        return datetime(*value[:6], tzinfo=timezone.utc)
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    s = str(value).strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        dt = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(s)
        if dt is not None:
            dt = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        pass
    return None


def to_local(dt: datetime | str | None) -> datetime | None:
    """把 datetime / ISO 字符串归一为本地时区的 datetime。空 / 解析失败返回 None。"""
    parsed = parse_dt(dt)
    return parsed.astimezone() if parsed else None


def fmt_local(dt: datetime | str | None, fmt: str = "%Y-%m-%d %H:%M") -> str:
    """把 datetime / ISO 字符串格式化为本地时间字符串。空 / 解析失败返回原值。"""
    local = to_local(dt)
    if local is None:
        return str(dt or "")
    return local.strftime(fmt)
