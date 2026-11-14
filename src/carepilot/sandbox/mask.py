"""脱敏工具。输入是不可信的原始数据，任何无法识别的输入都退化为 `****`，不抛异常。"""

from __future__ import annotations

import re

_PHONE_SEP = re.compile(r"[\s\-()]")
_MASKED = "****"


def mask_phone(value: str | None) -> str:
    """`13812345678` → `138****5678`。

    已脱敏的串（含 `*`）原样返回；None、空串、非 11 位数字、带 `+86` 前缀的号码都先归一化，
    归一化后仍不是 11 位数字则整体替换为 `****`。
    """
    if value is None:
        return _MASKED
    text = str(value).strip()
    if "*" in text:
        return text
    normalized = _PHONE_SEP.sub("", text)
    if normalized.startswith("+"):
        normalized = normalized[1:]
    if normalized.startswith("86") and len(normalized) == 13:
        normalized = normalized[2:]
    if len(normalized) != 11 or not normalized.isdigit():
        return _MASKED
    return f"{normalized[:3]}****{normalized[7:]}"


def mask_address(value: str | None) -> str:
    """只保留到第一个「市」或「区」的省市级前缀，剩余部分用 `***` 代替。

    无法识别（None、空串、没有市/区）时返回 `***`；已经脱敏的地址原样返回。
    """
    if value is None:
        return "***"
    text = str(value).strip()
    if not text:
        return "***"
    if text.endswith("***"):
        return text
    markers = [i for i in (text.find("市"), text.find("区")) if i >= 0]
    if not markers:
        return "***"
    return text[: min(markers) + 1] + "***"


def mask_tracking_no(value: str | None) -> str:
    """运单号保留首尾各 4 位，中间打码。"""
    if value is None:
        return _MASKED
    text = str(value)
    if len(text) <= 8:
        return _MASKED
    return f"{text[:4]}****{text[-4:]}"
