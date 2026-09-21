"""Build and write the per-case answer JSON exactly as the README specifies."""
from __future__ import annotations

import json
from pathlib import Path

STATUS_OPEN = "open"


def build_answer(case: dict, a, status: str, initial: list, final: list, evidence_requests: list,
                 stop_reason: str, explain: dict, written: bool, graph_case_id: str,
                 activity_dates: list, metrics: dict) -> dict:
    names = [x["action"] for x in final]
    file_report = "FILE_REPORT" in names
    subjects = []
    if file_report:
        subjects = list(dict.fromkeys([case["customer_id"], case["card_id"], *a.connected_card_ids,
                                       *a.connected_device_profiles]))
    sar = {
        "file": file_report,
        "reason": explain.get("sar_reason", ""),
        "narrative": explain.get("sar_narrative", "") if file_report else "",
        "subjects": subjects,
        "total_amount_usd": round(a.exposure, 2) if file_report else 0,
        "activity_dates": activity_dates if file_report else [],
    }
    return {
        "case_id": case["case_id"],
        "case": {
            "status": status,
            "verdict": a.verdict,
            "fraud_probability": round(a.p, 2),
            "pattern": a.pattern,
            "pattern_description": explain.get("pattern_description", "") if a.pattern == "undocumented" else "",
            "affected_txn_ids": a.affected_txn_ids,
            "first_suspicious_txn_id": a.first_suspicious_txn_id,
            "connected_card_ids": a.connected_card_ids,
            "connected_device_profiles": a.connected_device_profiles,
            "exposure_usd": round(a.exposure, 2),
            "evidence": [e.as_dict() for e in a.evidence],
            "similar_prior_cases": a.similar_prior_cases,
            "summary": explain.get("summary", ""),
            "written_to_graph": bool(written),
            "graph_case_id": graph_case_id if written else "",
        },
        "evidence_requests": evidence_requests,
        "next_best_actions": {
            "initial": initial,
            "final": final,
            "what_changed": explain.get("what_changed", "nothing"),
        },
        "sar": sar,
        "stop_reason": stop_reason,
        "tool_calls": int(metrics["tool_calls"]),
        "tokens": int(metrics["tokens"]),
        "latency_s": round(float(metrics["latency_s"]), 2),
    }


def write_answer(answer: dict, out_dir: str = "cases") -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    f = p / f"{answer['case_id']}.json"
    f.write_text(json.dumps(answer, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return f