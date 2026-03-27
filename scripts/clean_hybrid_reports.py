from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hybrid_rul.llm_output import normalize_llm_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize LLM outputs in hybrid report JSON files.")
    parser.add_argument("--input-json", type=str, required=True, help="Path to the hybrid report JSON.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output path.")
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_cleaned{input_path.suffix}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json).resolve()
    output_path = Path(args.output_json).resolve() if args.output_json else default_output_path(input_path)

    reports = json.loads(input_path.read_text(encoding="utf-8"))

    cleaned_count = 0
    wrapper_count = 0
    for report in reports:
        audit = normalize_llm_response(report.get("timeomni_response"))
        if audit["raw_text"]:
            report["timeomni_raw_response"] = audit["raw_text"]
        report["timeomni_response"] = audit["cleaned_text"] or audit["raw_text"] or None
        report["timeomni_response_audit"] = {
            "missing_tags": audit["missing_tags"],
            "duplicate_tags": audit["duplicate_tags"],
            "extra_tags": audit["extra_tags"],
            "removed_wrappers": audit["removed_wrappers"],
            "raw_format_ok": audit["raw_format_ok"],
            "clean_format_ok": audit["clean_format_ok"],
        }
        if audit["cleaned_text"] is not None:
            cleaned_count += 1
        if audit["removed_wrappers"]:
            wrapper_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Input report: {input_path}")
    print(f"Output report: {output_path}")
    print(f"Responses cleaned: {cleaned_count}/{len(reports)}")
    print(f"Responses with removed wrappers: {wrapper_count}/{len(reports)}")


if __name__ == "__main__":
    main()
