from __future__ import annotations


def build_timeomni_question(prediction: dict, summary: dict, thresholds: dict | None = None) -> str:
    sigma = prediction.get("sigma")
    sigma_text = "unavailable" if sigma is None else f"{sigma:.4f}"
    lower = prediction.get("lower_95")
    lower_text = f"{lower:.4f}" if lower is not None else "unavailable"
    thresholds = thresholds or {"normal": 80, "watch": 50, "alert": 20}

    trend_block = "\n".join(f"- {line}" for line in summary["trend_lines"])
    return (
        "Assess the maintenance risk for the following engine.\n\n"
        f"Engine ID: {prediction['unit_id']}\n"
        f"Observed cycle: {prediction['observed_cycle']}\n"
        f"Model backbone: {prediction.get('model_backbone', 'tcn')}\n"
        f"Model type: {prediction['model_type']}\n"
        f"Predicted RUL: {prediction['predicted_rul']:.4f}\n"
        f"Predictive sigma: {sigma_text}\n"
        f"Lower 95% bound: {lower_text}\n"
        f"Warning level: {prediction['warning']['level']}\n"
        f"Escalated by uncertainty: {prediction['warning']['escalated']}\n"
        "Warning thresholds:\n"
        f"- 正常: lower bound > {thresholds['normal']}\n"
        f"- 关注: {thresholds['watch']} < lower bound <= {thresholds['normal']}\n"
        f"- 预警: {thresholds['alert']} < lower bound <= {thresholds['watch']}\n"
        f"- 危险: lower bound <= {thresholds['alert']}\n"
        "Recommended action windows:\n"
        "- 正常: routine monitoring\n"
        "- 关注: next 20 cycles\n"
        "- 预警: next 10 cycles\n"
        "- 危险: immediate\n"
        f"Summary window: cycles {summary['summary_window']['start_cycle']} to {summary['summary_window']['end_cycle']}\n"
        "Recent feature trends:\n"
        f"{trend_block}\n\n"
        "Explain the risk, recommend a maintenance action, identify the key evidence, "
        "list follow-up checks, and state how much confidence the operator should have."
    )
