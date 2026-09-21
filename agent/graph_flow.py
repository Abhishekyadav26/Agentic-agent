"""The investigation workflow (LangGraph) and the CLI.

trigger -> open_case -> gather -> assess -> initial_actions -> decide
                                     ^                            |-- more evidence needed --> request_evidence -> simulate
                                     |____________________________|                                          |
                                                                   |-- enough to act --> final_actions -> explain -> write_memory

Graph facts, probabilities and actions come from code (assess.py, policy.py).
The LLM only writes explanations (summary, SAR narrative, what changed), and every
ID it mentions is checked against the evidence packet before it is used.

Usage:
    python -m agent.graph_flow --case HHG-003
    python -m agent.graph_flow --all
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from langgraph.graph import END, StateGraph

from outputs.writer import build_answer, write_answer

from . import assess as A
from .memory import remember
from .policy import evidence_request_type, recommend, stop_reason as policy_stop
from .simulator import simulate
from .state import CaseState, Evidence
from .tools import Tools

PATTERN_WORDS = {
    "card_testing": "card testing small online authorizations then larger purchase",
    "card_not_present_fraud": "card not present online purchase unusual amount product burst",
    "card_not_present_new_device": "card not present new device proxy online purchase",
    "out_of_region_use": "out of region card present billing region trip clone",
    "account_takeover": "account takeover mixed channel device match anomalies credentials",
    "undocumented": "shared device ring coordinated multiple customers",
    "none": "false alarm cleared legitimate recurring subscription monthly charge",
}


# ================================================================== LLM (explanations only)
class LLM:
    def __init__(self):
        self.tokens = 0
        self.enabled = os.getenv("USE_LLM", "1") == "1" and bool(os.getenv("ANTHROPIC_API_KEY"))
        self.model = os.getenv("LLM_MODEL", "claude-sonnet-5")
        self.client = None
        if self.enabled:
            import anthropic
            self.client = anthropic.Anthropic()

    def complete(self, system: str, user: str, max_tokens: int = 1400) -> str | None:
        if not self.enabled:
            return None
        try:
            r = self.client.messages.create(model=self.model, max_tokens=max_tokens, system=system,
                                            messages=[{"role": "user", "content": user}])
            self.tokens += r.usage.input_tokens + r.usage.output_tokens
            return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        except Exception as e:                      # explanations must never break an investigation
            print(f"[llm] {e}")
            return None


SYSTEM = (
    "You are a fraud investigation writer at a bank. You receive a JSON evidence packet produced by graph queries and a "
    "policy engine. Write clear, factual text using ONLY facts, numbers and IDs in the packet. Never invent IDs, amounts, "
    "dates or names. Do not change any verdict, probability or action. Reply with a single JSON object and nothing else."
)


def _sentences(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'$])", text.strip()) if s])


def _allowed_ids(packet: dict) -> set[str]:
    return set(re.findall(r"\b(?:C\d{5}(?:-K\d)?|CC-\d+|CASE-\d{4}-\d+|HHG-\d+|\d{7})\b", json.dumps(packet)))


def _ids_ok(text: str, allowed: set[str]) -> bool:
    return all(i in allowed for i in re.findall(r"\b(?:C\d{5}(?:-K\d)?|CC-\d+|CASE-\d{4}-\d+)\b", text))


# ================================================================== template explanations
def _names(actions: list) -> list[str]:
    return [x["action"] for x in actions]


def template_explain(case, a, initial, final, ereqs, dates, txns) -> dict:
    cust, card = case["customer_id"], case["card_id"]
    top = [e.claim for e in a.evidence if e.source == "graph"][2:5] or [e.claim for e in a.evidence][:2]
    resp = ereqs[-1]["assumed_response"] if ereqs else ""
    s = [f"Verdict {a.verdict} at probability {a.p:.2f}; pattern: {a.pattern.replace('_', ' ')}."]
    s += [t if t.endswith(".") else t + "." for t in top[:2]]
    if resp:
        s.append(f"Evidence was requested and the assumed response was: {resp}")
    s.append("Recommended actions: " + ", ".join(_names(final)) + ".")
    summary = " ".join(s[:6])

    ok_report = "FILE_REPORT" in _names(final)
    reason = next((x["reason"] for x in final if x["action"] == "FILE_REPORT"),
                  "No report: the policy's section 3a conditions are not met (exposure not above $1,000, no shared device/region, not undocumented).")
    if ok_report:
        d0, d1 = (dates[0], dates[-1]) if dates else ("", "")
        n = len(a.affected_txn_ids)
        sar = [
            f"Customer {cust} holds card {card}, which showed suspected unauthorized card activity.",
            f"Between {d0} and {d1}, {n} transaction(s) totalling ${a.exposure:,.2f} were identified as part of the suspected episode, beginning with transaction {a.first_suspicious_txn_id}.",
            f"The activity was classified as {a.pattern.replace('_', ' ')} on the basis of graph analysis of the card's transaction history, device and billing-region records.",
        ]
        sar += [e.claim if e.claim.endswith(".") else e.claim + "." for e in a.evidence[2:6]]
        if a.connected_card_ids:
            sar.append(f"The activity is linked to other cards ({', '.join(a.connected_card_ids[:4])}) through {a.shared_element_desc or 'shared entities'}.")
        if resp:
            sar.append(resp if resp.endswith(".") else resp + ".")
        sar.append(f"The bank recommends {', '.join(_names(final))}, subject to the approval routes in its fraud policy.")
        sar_text = " ".join(sar[:12])
        while _sentences(sar_text) < 6:
            sar_text += " The activity is inconsistent with the cardholder's established behavior."
    else:
        sar_text = ""

    ini, fin = set(_names(initial)), set(_names(final))
    if not ereqs or ini == fin:
        changed = "nothing"
    else:
        add, rem = sorted(fin - ini), sorted(ini - fin)
        changed = (f"After the assumed response ({ereqs[-1]['type']}), probability moved to {a.p:.2f}. "
                   + (f"Added {', '.join(add)}. " if add else "") + (f"Dropped {', '.join(rem)}." if rem else "")).strip()
    return dict(summary=summary, sar_narrative=sar_text, sar_reason=reason, what_changed=changed,
                pattern_description=a.pattern_description)


def llm_explain(llm: LLM, case, a, initial, final, ereqs, dates, base: dict) -> dict:
    """Let the LLM improve the wording; keep the template when its output fails the checks."""
    if not llm.enabled:
        return base
    packet = dict(
        case=dict(id=case["case_id"], trigger=case.get("trigger_text", ""), customer=case["customer_id"], card=case["card_id"]),
        verdict=a.verdict, probability=a.p, pattern=a.pattern, exposure_usd=a.exposure, activity_dates=dates,
        affected_txn_ids=a.affected_txn_ids, connected_cards=a.connected_card_ids, devices=a.connected_device_profiles,
        evidence=[e.claim for e in a.evidence], evidence_requests=ereqs, initial_actions=initial, final_actions=final,
        file_report="FILE_REPORT" in _names(final), shared_element=a.shared_element_desc,
    )
    user = ("Write JSON with keys: summary (2-6 sentences an analyst can read), "
            "sar_narrative (6-12 sentences covering who, what, when, where, how, why suspicious; empty string if file_report is false), "
            "what_changed (1-2 sentences on why final differs from initial, or \"nothing\"), "
            "pattern_description (2-3 sentences, only if pattern is undocumented, else empty).\n\nPACKET:\n" + json.dumps(packet, default=str))
    raw = llm.complete(SYSTEM, user)
    if not raw:
        return base
    try:
        m = re.search(r"\{.*\}", raw, re.S)
        out = json.loads(m.group(0))
        allowed = _allowed_ids(packet)
        good = dict(base)
        if 2 <= _sentences(out.get("summary", "")) <= 6 and _ids_ok(out["summary"], allowed):
            good["summary"] = out["summary"]
        if packet["file_report"] and 6 <= _sentences(out.get("sar_narrative", "")) <= 12 and _ids_ok(out["sar_narrative"], allowed):
            good["sar_narrative"] = out["sar_narrative"]
        if out.get("what_changed") and _ids_ok(out["what_changed"], allowed):
            good["what_changed"] = out["what_changed"] if base["what_changed"] != "nothing" else "nothing"
        if a.pattern == "undocumented" and out.get("pattern_description") and _ids_ok(out["pattern_description"], allowed):
            good["pattern_description"] = out["pattern_description"]
        return good
    except Exception as e:
        print(f"[llm] could not parse output: {e}")
        return base


# ================================================================== workflow
def _status(a, final) -> str:
    names = _names(final)
    if "VERIFY_WITH_CUSTOMER" in names and a.customer_response is None:
        return "open"                       # a verification is pending (e.g. R7)
    if a.verdict == "legitimate" or a.customer_response == "confirmed":
        return "closed_legitimate"
    if "ESCALATE_TO_ANALYST" in names:
        return "escalated"
    if a.verdict == "fraud":
        return "closed_fraud"
    return "open"


def build_app(tools: Tools, llm: LLM):
    def log(state, node, text):
        state["step_no"] = state.get("step_no", 0) + 1
        state.setdefault("steps", []).append(dict(step=state["step_no"], node=node, text=text))

    def open_case(state):
        c = state["case"]
        log(state, "open_case", f"Trigger {c['trigger_type']}: {c.get('trigger_text', '')}")
        return {"evidence_requests": [], "rounds": 0, "steps": state["steps"], "step_no": state["step_no"]}

    def gather(state):
        F = A.gather(tools, state["case"])
        log(state, "gather", f"Queried graph: {len(F['win'])} txns in window, {len(F['devices'])} device profile(s), "
                             f"{len(F['own_cases'])} prior case(s) on the card.")
        return {"features": F, "steps": state["steps"], "step_no": state["step_no"]}

    def assess(state):
        F, case = state["features"], state["case"]
        resp = state.get("customer_response")
        a = A.score(F, resp)
        if "retrieved" not in F:
            q = PATTERN_WORDS.get(a.pattern, "") + " " + " ".join(k.replace("_", " ") for k in a.signals)
            F["retrieved"] = tools.retrieve(q, k=6, before=F["opened"], exclude=[case["case_id"]])
        for r in F["retrieved"]:
            if r["kind"] == "case" and r["id"] not in a.similar_prior_cases and len(a.similar_prior_cases) < 5:
                a.similar_prior_cases.append(r["id"])
        docs = [r for r in F["retrieved"] if r["kind"] == "doc"][:2]
        for r in docs:
            a.evidence.append(Evidence(f"Policy/pattern guidance retrieved for this pattern: {r['id']}.", "document", r["id"], []))
        if resp:
            n = len(state["evidence_requests"])
            a.evidence.append(Evidence(state["evidence_requests"][-1]["assumed_response"], "customer", f"evidence_request:{n}", []))
        log(state, "assess", f"p={a.p:.2f} verdict={a.verdict} pattern={a.pattern} independent_evidence={a.n_independent} signals={a.signals}")
        return {"assessment": a, "steps": state["steps"], "step_no": state["step_no"]}

    def initial_actions(state):
        if state.get("initial_actions") is not None:
            return {}
        a = state["assessment"]
        will_ask = policy_stop(a, state["rounds"]) is None
        acts = recommend(a, evidence_requested=will_ask)
        log(state, "recommend", "Initial recommendation: " + ", ".join(f"{x['action']}[{x['route']}]" for x in acts))
        return {"initial_actions": acts, "p_initial": a.p, "steps": state["steps"], "step_no": state["step_no"]}

    def decide(state):
        return "final" if policy_stop(state["assessment"], state["rounds"]) else "request"

    def request_evidence(state):
        a = state["assessment"]
        typ = evidence_request_type(a)
        reqs = list(state["evidence_requests"])
        reqs.append(dict(type=typ, asked_after_step=state["step_no"], assumed_response="pending"))
        log(state, "request_evidence", f"Uncertain (p={a.p:.2f}, {a.n_independent} independent evidence): requesting {typ}")
        return {"evidence_requests": reqs, "rounds": state["rounds"] + 1, "steps": state["steps"], "step_no": state["step_no"]}

    def sim(state):
        a = state["assessment"]
        reqs = list(state["evidence_requests"])
        resp, text = simulate(a, reqs[-1]["type"])
        reqs[-1] = {**reqs[-1], "assumed_response": text}
        log(state, "simulate", f"Assumed response: {resp} ({text})")
        return {"customer_response": resp, "evidence_requests": reqs, "steps": state["steps"], "step_no": state["step_no"]}

    def final_actions(state):
        a = state["assessment"]
        acts = recommend(a, evidence_requested=bool(state["evidence_requests"]))
        log(state, "recommend", "Final recommendation: " + ", ".join(f"{x['action']}[{x['route']}]" for x in acts))
        why = policy_stop(a, state["rounds"]) or "Investigation stopped."
        return {"final_actions": acts, "stop_reason": why, "steps": state["steps"], "step_no": state["step_no"]}

    def explain(state):
        a, case, F = state["assessment"], state["case"], state["features"]
        aff = F["win"][F["win"]["txn_id"].isin(a.affected_txn_ids)]
        dates = [f"{aff['ts'].min():%Y-%m-%d}", f"{aff['ts'].max():%Y-%m-%d}"] if len(aff) else []
        base = template_explain(case, a, state["initial_actions"], state["final_actions"], state["evidence_requests"], dates, aff)
        out = llm_explain(llm, case, a, state["initial_actions"], state["final_actions"], state["evidence_requests"], dates, base)
        out["dates"] = dates
        log(state, "explain", out["summary"])
        return {"explain": out, "steps": state["steps"], "step_no": state["step_no"]}

    def write_memory(state):
        a, case = state["assessment"], state["case"]
        status = _status(a, state["final_actions"])
        written, gid = remember(tools, case, a, status, state["final_actions"], state["explain"]["summary"])
        log(state, "write_memory", f"Case stored as {gid} (written_to_graph={written})")
        return {"written_to_graph": written, "graph_case_id": gid, "steps": state["steps"], "step_no": state["step_no"]}

    g = StateGraph(CaseState)
    for name, fn in [("open_case", open_case), ("gather", gather), ("assess", assess), ("initial", initial_actions),
                     ("request", request_evidence), ("simulate", sim), ("final", final_actions),
                     ("explain", explain), ("write_memory", write_memory)]:
        g.add_node(name, fn)
    g.set_entry_point("open_case")
    g.add_edge("open_case", "gather")
    g.add_edge("gather", "assess")
    g.add_edge("assess", "initial")
    g.add_conditional_edges("initial", decide, {"request": "request", "final": "final"})
    g.add_edge("request", "simulate")
    g.add_edge("simulate", "assess")
    g.add_edge("final", "explain")
    g.add_edge("explain", "write_memory")
    g.add_edge("write_memory", END)
    return g.compile()


class Agent:
    def __init__(self, tools: Tools | None = None, llm: LLM | None = None):
        self.tools = tools or Tools()
        self.llm = llm or LLM()
        self.app = build_app(self.tools, self.llm)

    def investigate(self, case: dict) -> dict:
        t0, calls0, tok0 = time.time(), self.tools.calls, self.llm.tokens
        st = self.app.invoke({"case": case, "step_no": 0, "steps": []}, config={"recursion_limit": 40})
        a = st["assessment"]
        status = _status(a, st["final_actions"])
        metrics = dict(tool_calls=self.tools.calls - calls0, tokens=self.llm.tokens - tok0, latency_s=time.time() - t0)
        ans = build_answer(case, a, status, st["initial_actions"], st["final_actions"], st["evidence_requests"],
                           st["stop_reason"], st["explain"], st["written_to_graph"], st["graph_case_id"],
                           st["explain"]["dates"], metrics)
        ans["_trace"] = st["steps"]          # for the UI; stripped before writing the answer file
        return ans


# ================================================================== CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="case id, e.g. HHG-003")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--pack", default="data/case_pack.csv")
    ap.add_argument("--out", default="cases")
    args = ap.parse_args()

    pack = pd.read_csv(args.pack, dtype=str)
    rows = pack if args.all else pack[pack["case_id"] == args.case]
    if rows.empty:
        raise SystemExit("no matching case; use --all or --case HHG-001")
    agent = Agent()
    for _, r in rows.iterrows():
        ans = agent.investigate({k: (None if pd.isna(v) else v) for k, v in r.to_dict().items()})
        trace = ans.pop("_trace")
        path = write_answer(ans, args.out)
        Path(args.out, f"{ans['case_id']}.trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
        c = ans["case"]
        print(f"{ans['case_id']}: {c['verdict']:<10} p={c['fraud_probability']:.2f} {c['pattern']:<28} "
              f"-> {[x['action'] for x in ans['next_best_actions']['final']]}")


if __name__ == "__main__":
    main()