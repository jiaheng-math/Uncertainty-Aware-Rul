from __future__ import annotations

from pathlib import Path

from hybrid_rul.adapters.tcn_adapter import TCNProjectAdapter
from hybrid_rul.adapters.timeomni_adapter import GenerationConfig, TimeOmniAdapter
from hybrid_rul.llm_output import evaluate_response_quality, normalize_llm_response
from hybrid_rul.paths import resolve_path
from hybrid_rul.prompts.builder import build_timeomni_question
from hybrid_rul.summarizers.engine_summary import build_engine_summary


class HybridPipeline:
    def __init__(self, config: dict, config_path: str | Path) -> None:
        self.config = config
        self.config_path = Path(config_path).resolve()
        self.base_dir = self.config_path.parent
        project_root = resolve_path(self.base_dir, self.config["paths"].get("project_root")) or self.base_dir

        device = self.config.get("runtime", {}).get("device", "auto")
        paths_cfg = self.config["paths"]

        tcn_repo = project_root
        tcn_module_root = resolve_path(self.base_dir, paths_cfg.get("tcn_module_root")) or project_root
        tcn_config = resolve_path(self.base_dir, paths_cfg["tcn_config"])
        tcn_checkpoint = resolve_path(self.base_dir, paths_cfg.get("tcn_checkpoint"))
        tcn_predictions_json = resolve_path(self.base_dir, paths_cfg.get("tcn_predictions_json"))

        self.tcn = TCNProjectAdapter(
            repo_root=tcn_repo,
            config_path=tcn_config,
            module_root=tcn_module_root,
            project_root=project_root,
            checkpoint_path=tcn_checkpoint,
            prediction_artifact_path=tcn_predictions_json,
            device=device,
        )

        reasoning_cfg = self.config["reasoning"]
        timeomni_model_dir = resolve_path(self.base_dir, paths_cfg.get("timeomni_model_dir"))
        self.timeomni = TimeOmniAdapter(
            model_dir=str(timeomni_model_dir) if timeomni_model_dir is not None else None,
            generation_config=GenerationConfig(
                max_new_tokens=int(reasoning_cfg.get("max_new_tokens", 768)),
                temperature=float(reasoning_cfg.get("temperature", 0.35)),
                top_p=float(reasoning_cfg.get("top_p", 0.9)),
                repetition_penalty=float(reasoning_cfg.get("repetition_penalty", 1.05)),
            ),
        )

        if reasoning_cfg.get("enable_timeomni", False) and not self.timeomni.enabled:
            raise ValueError(
                "reasoning.enable_timeomni is true, but paths.timeomni_model_dir is not configured."
            )

    def _select_predictions(self, predictions: list[dict], engine_ids: list[int] | None, limit: int | None) -> list[dict]:
        selected = predictions
        if engine_ids:
            selected_ids = set(engine_ids)
            selected = [item for item in selected if item["unit_id"] in selected_ids]
        if limit is None:
            limit = int(self.config.get("runtime", {}).get("default_limit", len(selected)))
        if limit < 0:
            raise ValueError("limit must be greater than or equal to 0.")
        selected = selected[:limit]
        return selected

    def run(self, engine_ids: list[int] | None = None, limit: int | None = None) -> dict:
        predictions = self.tcn.predict_test_set()
        selected_predictions = self._select_predictions(predictions, engine_ids=engine_ids, limit=limit)

        analysis_cfg = self.config["analysis"]
        reasoning_cfg = self.config["reasoning"]
        feature_columns = self.tcn.get_feature_columns()
        system_prompt = reasoning_cfg["system_prompt"]
        should_generate = bool(reasoning_cfg.get("enable_timeomni", False) and self.timeomni.enabled)
        thresholds = self.tcn.config["warning"]["thresholds"]

        reports = []
        prompts = []

        for prediction in selected_predictions:
            unit_frame = self.tcn.get_unit_frame(prediction["unit_id"])
            summary = build_engine_summary(
                unit_frame=unit_frame,
                feature_columns=feature_columns,
                history_cycles=int(analysis_cfg.get("history_cycles", 12)),
                top_k_features=int(analysis_cfg.get("top_k_features", 5)),
            )

            question = build_timeomni_question(prediction, summary, thresholds=thresholds)
            llm_output = self.timeomni.generate(question=question, system_prompt=system_prompt) if should_generate else None
            llm_audit = normalize_llm_response(llm_output) if llm_output is not None else None

            prompt_record = {
                "unit_id": prediction["unit_id"],
                "system_prompt": system_prompt,
                "question": question,
            }
            prompts.append(prompt_record)

            report_payload = {
                "unit_id": prediction["unit_id"],
                "tcn_prediction": prediction,
                "sensor_summary": summary,
                "timeomni_question": question,
                "timeomni_response": llm_audit["cleaned_text"] if llm_audit is not None else None,
                "timeomni_raw_response": llm_output,
                "timeomni_response_audit": (
                    {
                        "missing_tags": llm_audit["missing_tags"],
                        "duplicate_tags": llm_audit["duplicate_tags"],
                        "extra_tags": llm_audit["extra_tags"],
                        "removed_wrappers": llm_audit["removed_wrappers"],
                        "raw_format_ok": llm_audit["raw_format_ok"],
                        "clean_format_ok": llm_audit["clean_format_ok"],
                    }
                    if llm_audit is not None
                    else None
                ),
            }
            if llm_output is not None:
                report_payload["timeomni_quality"] = evaluate_response_quality(report_payload, thresholds=thresholds)

            reports.append(
                report_payload
            )

        return {
            "reports": reports,
            "prompts": prompts,
        }
