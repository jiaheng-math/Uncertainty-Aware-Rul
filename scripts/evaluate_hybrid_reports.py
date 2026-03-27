from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hybrid_rul.llm_output import evaluate_response_quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate formatting and consistency of hybrid LLM reports.")
    parser.add_argument("--input-json", type=str, required=True, help="Path to the hybrid report JSON.")
    parser.add_argument("--config", type=str, default=None, help="Optional TCN or hybrid YAML config for warning thresholds.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output path for the evaluation JSON.")
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_quality.json")


def _resolve_thresholds_from_text(config_path: Path) -> dict[str, float]:
    text = config_path.read_text(encoding="utf-8")
    nested_match = re.search(r"^\s*tcn_config:\s*(.+?)\s*$", text, flags=re.MULTILINE)
    if nested_match and "warning:" not in text:
        nested_path = (config_path.parent / nested_match.group(1).strip()).resolve()
        return _resolve_thresholds_from_text(nested_path)

    thresholds: dict[str, float] = {}
    for key in ("normal", "watch", "alert"):
        match = re.search(rf"^\s*{key}:\s*([0-9]+(?:\.[0-9]+)?)\s*$", text, flags=re.MULTILINE)
        if match is not None:
            thresholds[key] = float(match.group(1))
    if len(thresholds) == 3:
        return thresholds
    return {"normal": 80.0, "watch": 50.0, "alert": 20.0}


def resolve_thresholds(config_path: str | None) -> dict[str, float]:
    if config_path is None:
        return {"normal": 80.0, "watch": 50.0, "alert": 20.0}

    resolved_path = Path(config_path).resolve()
    try:
        from hybrid_rul.paths import load_yaml

        payload = load_yaml(resolved_path)
        if "warning" in payload:
            thresholds = payload["warning"]["thresholds"]
        elif "paths" in payload and "tcn_config" in payload["paths"]:
            nested_payload = load_yaml(resolved_path.parent / payload["paths"]["tcn_config"])
            thresholds = nested_payload["warning"]["thresholds"]
        else:
            thresholds = {"normal": 80.0, "watch": 50.0, "alert": 20.0}
        return {
            "normal": float(thresholds["normal"]),
            "watch": float(thresholds["watch"]),
            "alert": float(thresholds["alert"]),
        }
    except ModuleNotFoundError:
        return _resolve_thresholds_from_text(resolved_path)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json).resolve()
    output_path = Path(args.output_json).resolve() if args.output_json else default_output_path(input_path)
    thresholds = resolve_thresholds(args.config)

    reports = json.loads(input_path.read_text(encoding="utf-8"))
    evaluations = [evaluate_response_quality(report, thresholds=thresholds) for report in reports]

    aggregate = {
        "total_reports": len(evaluations),
        "raw_format_ok": sum(1 for item in evaluations if item["format"]["raw_format_ok"]),
        "clean_format_ok": sum(1 for item in evaluations if item["format"]["clean_format_ok"]),
        "grounding_ok": sum(1 for item in evaluations if item["grounding"]["grounding_ok"]),
        "consistency_ok": sum(1 for item in evaluations if item["consistency"]["ok"]),
        "action_reasonable": sum(1 for item in evaluations if item["action"]["assessment"] == "reasonable"),
    }

    payload = {
        "thresholds": thresholds,
        "aggregate": aggregate,
        "per_unit": evaluations,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Input report: {input_path}")
    print(f"Output evaluation: {output_path}")
    print(
        "Aggregate summary: "
        f"raw_format_ok={aggregate['raw_format_ok']}/{aggregate['total_reports']}, "
        f"clean_format_ok={aggregate['clean_format_ok']}/{aggregate['total_reports']}, "
        f"grounding_ok={aggregate['grounding_ok']}/{aggregate['total_reports']}, "
        f"consistency_ok={aggregate['consistency_ok']}/{aggregate['total_reports']}, "
        f"action_reasonable={aggregate['action_reasonable']}/{aggregate['total_reports']}"
    )

    for item in evaluations:
        issues = item["consistency"]["issues"]
        issue_text = "; ".join(issues) if issues else "none"
        print(
            f"Unit {item['unit_id']}: "
            f"format_raw={item['format']['raw_format_ok']}, "
            f"grounding={item['grounding']['grounding_ok']}, "
            f"consistency={item['consistency']['ok']}, "
            f"action={item['action']['assessment']}, "
            f"issues={issue_text}"
        )


if __name__ == "__main__":
    main()
