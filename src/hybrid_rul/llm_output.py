from __future__ import annotations

import re
from typing import Any

TARGET_TAGS = [
    "risk_summary",
    "maintenance_action",
    "key_evidence",
    "follow_up_checks",
    "confidence_note",
]
TARGET_TAG_SET = set(TARGET_TAGS)
GENERIC_TAG_PATTERN = re.compile(r"</?([A-Za-z][A-Za-z0-9_:-]*)\b[^>]*>")
SENSOR_PATTERN = re.compile(r"\bs\d+\b", re.IGNORECASE)


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _unwrap_outer_tags(text: str) -> tuple[str, list[str]]:
    current = text.strip()
    removed: list[str] = []
    while True:
        match = re.fullmatch(r"<([A-Za-z][A-Za-z0-9_:-]*)>\s*(.*)\s*</\1>", current, flags=re.DOTALL)
        if match is None:
            break
        tag = match.group(1)
        inner = match.group(2).strip()
        if tag in TARGET_TAG_SET:
            break
        if not any(f"<{name}>" in inner for name in TARGET_TAGS):
            break
        removed.append(tag)
        current = inner
    return current, removed


def _extract_tag_contents(text: str) -> tuple[dict[str, str], dict[str, int]]:
    contents: dict[str, str] = {}
    counts: dict[str, int] = {}
    for tag in TARGET_TAGS:
        matches = re.findall(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
        counts[tag] = len(matches)
        if matches:
            contents[tag] = matches[0].strip()
    return contents, counts


def _find_extra_tags(text: str) -> list[str]:
    extras = []
    for tag in GENERIC_TAG_PATTERN.findall(text):
        if tag not in TARGET_TAG_SET and tag not in extras:
            extras.append(tag)
    return extras


def normalize_llm_response(response: str | None) -> dict[str, Any]:
    raw_text = "" if response is None else response
    stripped = _strip_code_fences(raw_text)
    unwrapped, removed_wrappers = _unwrap_outer_tags(stripped)
    contents, counts = _extract_tag_contents(unwrapped)
    missing_tags = [tag for tag in TARGET_TAGS if counts.get(tag, 0) == 0]
    duplicate_tags = [tag for tag in TARGET_TAGS if counts.get(tag, 0) > 1]
    extra_tags = _find_extra_tags(unwrapped)

    cleaned_text = None
    if not missing_tags:
        cleaned_text = "\n\n".join(f"<{tag}>\n{contents[tag]}\n</{tag}>" for tag in TARGET_TAGS)

    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "tags": contents,
        "missing_tags": missing_tags,
        "duplicate_tags": duplicate_tags,
        "extra_tags": extra_tags,
        "removed_wrappers": removed_wrappers,
        "raw_format_ok": not missing_tags and not duplicate_tags and not extra_tags and not removed_wrappers,
        "clean_format_ok": cleaned_text is not None,
    }


def _contains_approx_value(text: str, value: float | None) -> bool:
    if value is None:
        return False
    candidates = set()
    for decimals in range(0, 5):
        candidate = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
        if candidate:
            candidates.add(candidate)
    for candidate in candidates:
        if re.search(rf"(?<![\d.]){re.escape(candidate)}(?![\d.])", text):
            return True
    return False


def _extract_action_urgency(text: str) -> tuple[int | None, str | None]:
    lowered = text.lower()
    if "immediate" in lowered:
        return 3, "immediate"
    if re.search(r"next\s+10\s+cycles", lowered):
        return 2, "next 10 cycles"
    if re.search(r"next\s+20\s+cycles", lowered):
        return 1, "next 20 cycles"
    if "routine monitoring" in lowered:
        return 0, "routine monitoring"
    return None, None


def _expected_action_urgency(prediction: dict[str, Any], thresholds: dict[str, float]) -> int:
    lower = prediction.get("lower_95")
    level = prediction.get("warning", {}).get("level")
    escalated = bool(prediction.get("warning", {}).get("escalated"))

    if lower is None:
        level_map = {"正常": 0, "关注": 1, "预警": 2, "危险": 2 if escalated else 3}
        return level_map.get(level, 0)

    if lower <= thresholds["alert"]:
        return 3
    if lower <= thresholds["watch"]:
        return 2
    if lower <= thresholds["normal"]:
        return 1
    return 0


def evaluate_response_quality(
    report: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or {"normal": 80.0, "watch": 50.0, "alert": 20.0}
    prediction = report["tcn_prediction"]
    summary = report["sensor_summary"]

    audit = normalize_llm_response(report.get("timeomni_response"))
    text = audit["cleaned_text"] or audit["raw_text"]
    mentioned_sensors = sorted(
        {
            sensor.lower()
            for sensor in SENSOR_PATTERN.findall(text)
            if sensor.lower() in {item["feature"].lower() for item in summary.get("top_features", [])}
        }
    )

    grounding = {
        "mentions_predicted_rul": _contains_approx_value(text, prediction.get("predicted_rul")),
        "mentions_lower_bound": _contains_approx_value(text, prediction.get("lower_95")),
        "mentions_warning_level": prediction.get("warning", {}).get("level") in text,
        "mentioned_sensors": mentioned_sensors,
        "sensor_count": len(mentioned_sensors),
    }

    consistency_issues: list[str] = []
    lowered = text.lower()
    lower_bound = prediction.get("lower_95")
    if lower_bound is not None and lower_bound > thresholds["alert"]:
        if "below the 20-cycle threshold" in lowered or "threshold for \"danger\"" in lowered or "threshold for danger" in lowered:
            consistency_issues.append("danger-threshold claim conflicts with the actual lower bound.")
    if lower_bound is not None and lower_bound > thresholds["watch"] and "immediate action" in lowered:
        consistency_issues.append("maintenance urgency appears stronger than the current lower-bound risk level.")

    expected_urgency = _expected_action_urgency(prediction, thresholds)
    actual_urgency, action_label = _extract_action_urgency(text)
    action_assessment = "unknown"
    if actual_urgency is not None:
        if actual_urgency < expected_urgency:
            action_assessment = "too_mild"
        elif actual_urgency > expected_urgency + 1:
            action_assessment = "too_aggressive"
        else:
            action_assessment = "reasonable"

    return {
        "unit_id": report.get("unit_id"),
        "format": {
            "raw_format_ok": audit["raw_format_ok"],
            "clean_format_ok": audit["clean_format_ok"],
            "missing_tags": audit["missing_tags"],
            "duplicate_tags": audit["duplicate_tags"],
            "extra_tags": audit["extra_tags"],
            "removed_wrappers": audit["removed_wrappers"],
        },
        "grounding": {
            **grounding,
            "grounding_ok": (
                grounding["mentions_predicted_rul"]
                and grounding["mentions_lower_bound"]
                and grounding["mentions_warning_level"]
                and grounding["sensor_count"] >= 2
            ),
        },
        "consistency": {
            "ok": not consistency_issues,
            "issues": consistency_issues,
        },
        "action": {
            "label": action_label,
            "actual_urgency": actual_urgency,
            "expected_min_urgency": expected_urgency,
            "assessment": action_assessment,
        },
    }
