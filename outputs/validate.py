"""Validate answer files before submitting. Missing fields score zero, so run this often.

    python -m outputs.validate cases/                 # format + policy consistency
    python -m outputs.validate cases/ --data          # also check every ID exists in the dataset
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from agent.policy import ALL_ACTIONS, route

STATUS = {"open", "closed_fraud", "closed_legitimate", "escalated"}
VERDICT = {"fraud", "legitimate", "uncertain"}
PATTERN = {"card_testing", "card_not_present_fraud", "card_not_present_new_device", "out_of_region_use",
           "account_takeover", "undocumented", "none"}
SOURCE = {"graph", "document", "customer", "external"}
EVREQ = {"customer_validation", "step_up_auth", "analyst_info"}
ROUTES = {"auto", "L1", "L2"}


def _n_sent(t: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'$])", t.strip()) if s])


def _actions(errs, label, lst, exposure):
    if not isinstance(lst, list) or not lst:
        errs.append(f"next_best_actions.{label} must be a non-empty list")
        return
    for i, a in enumerate(lst):
        if not isinstance(a, dict) or not {"action", "route", "reason"} <= set(a):
            errs.append(f"{label}[{i}] needs action, route, reason")
            continue
        if a["action"] not in ALL_ACTIONS:
            errs.append(f"{label}[{i}] unknown action {a['action']}")
        elif a["route"] != route(a["action"], exposure):
            errs.append(f"{label}[{i}] {a['action']} route {a['route']} should be {route(a['action'], exposure)} at exposure {exposure}")
        if a["route"] not in ROUTES:
            errs.append(f"{label}[{i}] bad route {a['route']}")
        if not a.get("reason"):
            errs.append(f"{label}[{i}] empty reason")


def validate_answer(ans: dict, ids: dict | None = None, pack_ids: set | None = None) -> list[str]:
    e: list[str] = []
    for k in ("case_id", "case", "evidence_requests", "next_best_actions", "sar", "stop_reason",
              "tool_calls", "tokens", "latency_s"):
        if k not in ans:
            e.append(f"missing top-level field {k}")
    if e:
        return e
    if pack_ids is not None and ans["case_id"] not in pack_ids:
        e.append(f"case_id {ans['case_id']} not in case_pack")
    c, sar, nba = ans["case"], ans["sar"], ans["next_best_actions"]

    for k in ("status", "verdict", "fraud_probability", "pattern", "pattern_description", "affected_txn_ids",
              "first_suspicious_txn_id", "connected_card_ids", "connected_device_profiles", "exposure_usd",
              "evidence", "similar_prior_cases", "summary", "written_to_graph", "graph_case_id"):
        if k not in c:
            e.append(f"case.{k} missing")
    if e:
        return e
    if c["status"] not in STATUS: e.append(f"bad status {c['status']}")
    if c["verdict"] not in VERDICT: e.append(f"bad verdict {c['verdict']}")
    if c["pattern"] not in PATTERN: e.append(f"bad pattern {c['pattern']}")
    if not (isinstance(c["fraud_probability"], (int, float)) and 0 <= c["fraud_probability"] <= 1):
        e.append("fraud_probability must be 0..1")
    if c["pattern"] == "undocumented" and not c["pattern_description"].strip():
        e.append("pattern_description required for undocumented")
    if c["pattern"] != "undocumented" and c["pattern_description"] != "":
        e.append("pattern_description must be '' unless pattern is undocumented")
    if c["verdict"] == "legitimate":
        if c["affected_txn_ids"] or c["exposure_usd"] != 0 or sar["file"]:
            e.append("legitimate verdict needs empty affected_txn_ids, exposure 0 and sar.file false")
    if c["affected_txn_ids"] and c["first_suspicious_txn_id"] not in c["affected_txn_ids"]:
        e.append("first_suspicious_txn_id must be one of affected_txn_ids")
    n = _n_sent(c["summary"])
    if not 2 <= n <= 6:
        e.append(f"summary should be 2-6 sentences (has {n})")
    for i, ev in enumerate(c["evidence"]):
        if not {"claim", "source", "ref", "entity_ids"} <= set(ev):
            e.append(f"evidence[{i}] missing fields")
        elif ev["source"] not in SOURCE:
            e.append(f"evidence[{i}] bad source {ev['source']}")
    if not c["evidence"]:
        e.append("evidence must not be empty")
    if c["written_to_graph"] and not c["graph_case_id"]:
        e.append("graph_case_id required when written_to_graph")

    for i, r in enumerate(ans["evidence_requests"]):
        if not {"type", "asked_after_step", "assumed_response"} <= set(r):
            e.append(f"evidence_requests[{i}] missing fields")
        elif r["type"] not in EVREQ:
            e.append(f"evidence_requests[{i}] bad type")

    exp = c["exposure_usd"]
    _actions(e, "initial", nba.get("initial"), exp)
    _actions(e, "final", nba.get("final"), exp)
    if "what_changed" not in nba:
        e.append("what_changed missing")
    else:
        same = nba.get("initial") == nba.get("final")
        if not ans["evidence_requests"] and not same:
            e.append("no evidence requested, so final must equal initial")
        if same and nba["what_changed"] != "nothing":
            e.append("final equals initial, so what_changed must be 'nothing'")
        if not same and nba["what_changed"] == "nothing":
            e.append("final differs from initial, so what_changed must explain why")

    final_names = [a.get("action") for a in nba.get("final", [])]
    if sar["file"] != ("FILE_REPORT" in final_names):
        e.append("sar.file must agree with FILE_REPORT in final actions")
    if "FILE_REPORT" in final_names and "CREATE_CASE" not in final_names:
        e.append("FILE_REPORT needs CREATE_CASE (a report always has a case behind it)")
    if "BLOCK_ALL_CARDS" in final_names and c["verdict"] != "fraud":
        e.append("BLOCK_ALL_CARDS only for a fraud verdict (R10)")
    if "CLOSE_NO_FRAUD" in final_names and any(x in final_names for x in ("BLOCK_CARD", "FILE_REPORT", "BLOCK_ALL_CARDS")):
        e.append("CLOSE_NO_FRAUD contradicts a block or report")
    if not str(ans["stop_reason"]).strip():
        e.append("stop_reason empty")

    if sar["file"]:
        if not (6 <= _n_sent(sar["narrative"]) <= 12):
            e.append(f"sar.narrative should be 6-12 sentences (has {_n_sent(sar['narrative'])})")
        if not sar["subjects"]: e.append("sar.subjects empty")
        if abs(sar["total_amount_usd"] - exp) > 0.01: e.append("sar.total_amount_usd should equal exposure_usd")
        d = sar["activity_dates"]
        if len(d) != 2 or not all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", x) for x in d):
            e.append("sar.activity_dates must be two YYYY-MM-DD strings")
        if not sar["reason"]: e.append("sar.reason empty")
    else:
        if sar["narrative"] != "" or sar["subjects"] != [] or sar["total_amount_usd"] != 0 or sar["activity_dates"] != []:
            e.append("sar.file false: narrative '', subjects [], total 0, activity_dates []")
        if not sar["reason"]: e.append("sar.reason empty")

    for k in ("tool_calls", "tokens"):
        if not isinstance(ans[k], int): e.append(f"{k} must be int")
    if not isinstance(ans["latency_s"], (int, float)): e.append("latency_s must be a number")

    if ids:
        for t in c["affected_txn_ids"] + ([c["first_suspicious_txn_id"]] if c["first_suspicious_txn_id"] else []):
            if t not in ids["txns"]: e.append(f"txn id {t} not in dataset")
        for k in c["connected_card_ids"]:
            if k not in ids["cards"]: e.append(f"card id {k} not in dataset")
        for k in c["similar_prior_cases"]:
            if k not in ids["cases"]: e.append(f"closed case {k} not in closed_cases_history")
        if ids.get("amounts") and c["affected_txn_ids"]:
            tot = sum(abs(ids["amounts"].get(t, 0.0)) for t in c["affected_txn_ids"])
            if abs(tot - exp) > 0.02: e.append(f"exposure_usd {exp} != sum of amounts {tot:.2f}")
    return e


def load_ids(tx_path="data/tx_slim.parquet", cases_path="data/closed_cases_history.csv") -> dict | None:
    try:
        import pandas as pd
        tx = pd.read_parquet(tx_path, columns=["txn_id", "card_id", "amount"])
        cases = pd.read_csv(cases_path, dtype=str, usecols=["case_id"])
    except Exception as ex:
        print(f"[validate] cannot load dataset ids: {ex}")
        return None
    return dict(txns=set(tx["txn_id"].astype(str)), cards=set(tx["card_id"]), cases=set(cases["case_id"]),
                amounts=dict(zip(tx["txn_id"].astype(str), tx["amount"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default="cases")
    ap.add_argument("--data", action="store_true", help="check IDs against data/")
    ap.add_argument("--pack", default="data/case_pack.csv")
    args = ap.parse_args()

    ids = load_ids() if args.data else None
    pack_ids = None
    if Path(args.pack).exists():
        import pandas as pd
        pack_ids = set(pd.read_csv(args.pack, dtype=str)["case_id"])
    files = sorted(p for p in Path(args.folder).glob("*.json") if not p.name.endswith(".trace.json"))
    bad = 0
    for f in files:
        errs = validate_answer(json.loads(f.read_text()), ids, pack_ids)
        if errs:
            bad += 1
            print(f"FAIL {f.name}")
            for x in errs: print("   -", x)
        else:
            print(f"ok   {f.name}")
    if pack_ids:
        missing = pack_ids - {f.stem for f in files}
        if missing:
            bad += 1
            print("MISSING answer files:", ", ".join(sorted(missing)))
    print(f"\n{len(files)} files, {bad} problem(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()