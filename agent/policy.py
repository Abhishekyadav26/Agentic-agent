"""Fraud Policy v1.0 as code.

The LLM never chooses actions or approval routes. It produces evidence and
explanations; this module maps an Assessment to policy-compliant actions.
Action names and routes are the exact identifiers from the README.
"""
from __future__ import annotations

from .state import Assessment

AUTO_ACTIONS = {
    "ALLOW_TRANSACTION", "MONITOR_CARD", "MONITOR_CONNECTED_CARDS", "WARN_CUSTOMER",
    "VERIFY_WITH_CUSTOMER", "STEP_UP_AUTH", "GENERATE_REPORT", "CREATE_CASE",
    "ESCALATE_TO_ANALYST", "CLOSE_NO_FRAUD",
}
L1_ACTIONS = {"DECLINE_TRANSACTION"}
L2_ACTIONS = {"BLOCK_ALL_CARDS", "FILE_REPORT"}
ALL_ACTIONS = AUTO_ACTIONS | L1_ACTIONS | L2_ACTIONS | {"BLOCK_CARD"}

# "Order them by what happens first."
ORDER = [
    "DECLINE_TRANSACTION", "STEP_UP_AUTH", "VERIFY_WITH_CUSTOMER", "BLOCK_CARD",
    "BLOCK_ALL_CARDS", "MONITOR_CARD", "MONITOR_CONNECTED_CARDS", "WARN_CUSTOMER",
    "CREATE_CASE", "ESCALATE_TO_ANALYST", "FILE_REPORT", "GENERATE_REPORT",
    "ALLOW_TRANSACTION", "CLOSE_NO_FRAUD",
]

MAX_EVIDENCE_ROUNDS = 1


def route(action: str, exposure: float) -> str:
    """Section 2: approval routing."""
    if action in AUTO_ACTIONS:
        return "auto"
    if action in L1_ACTIONS:
        return "L1"
    if action == "BLOCK_CARD":
        return "L1" if exposure <= 2500 else "L2"
    if action in L2_ACTIONS:
        return "L2"
    raise ValueError(f"unknown action {action}")


def can_execute(action: str, exposure: float) -> bool:
    """The agent may execute only `auto` actions; everything else waits for a human."""
    return route(action, exposure) == "auto"


def effective_denial(a: Assessment) -> bool:
    """A customer report is itself a denial (R2) unless the charge matches the
    customer's own recurring pattern (R7) or the graph does not corroborate it."""
    if a.customer_response == "denied":
        return True
    return (
        a.customer_response is None
        and a.customer_disputes
        and not a.recurring_match
        and a.p >= 0.50
    )


def report_warranted(a: Assessment) -> tuple[bool, str]:
    """Section 3a: a SAR needs (confirmed or strongly suspected) AND one trigger."""
    strong = a.p >= 0.70 or a.customer_response == "denied"
    if not strong or a.verdict == "legitimate":
        return False, ""
    if a.exposure > 1000:
        return True, f"exposure ${a.exposure:,.2f} exceeds $1,000"
    if a.shared_element:
        return True, f"activity connects to a shared element ({a.shared_element_desc})"
    if a.pattern == "undocumented" and a.coordinated:
        return True, "R9: coordinated undocumented pattern"
    return False, ""


def recommend(a: Assessment, evidence_requested: bool = False) -> list[dict]:
    """Assessment -> ordered list of {action, route, reason}."""
    acts: dict[str, str] = {}

    def add(action: str, reason: str) -> None:
        acts.setdefault(action, reason)

    resp = a.customer_response
    strong = a.p >= 0.70
    high_conf = a.p >= 0.85 and a.n_independent >= 2

    # ---- R3: customer confirms -> close, nothing else
    if resp == "confirmed":
        add("CLOSE_NO_FRAUD", "R3: customer confirmed the transaction; confirmation noted in the case file")
        if a.customer_disputes or evidence_requested:
            add("CREATE_CASE", "3a: a case is opened when evidence is requested or a charge is disputed; closed as legitimate")
        return _finish(a, acts)

    denied = effective_denial(a)

    if denied:
        who = "customer denies the transaction" if resp == "denied" else "customer reports they did not make the transaction and the graph corroborates it"
        add("BLOCK_CARD", f"R2: {who}; exposure ${a.exposure:,.2f}")
        add("CREATE_CASE", f"R2: {who}")
    elif resp == "no_reply":
        add("MONITOR_CARD", "R4: no customer reply within 24 hours")
        add("DECLINE_TRANSACTION", "R4: decline pending authorizations while unverified")
        add("CREATE_CASE", "3a: evidence was requested")
        if a.exposure > 500:
            add("ESCALATE_TO_ANALYST", f"R4: no reply and exposure ${a.exposure:,.2f} exceeds $500")
    elif a.recurring_match and a.customer_disputes:
        add("CREATE_CASE", "R7: disputed charge matches the customer's own recurring pattern")
        add("VERIFY_WITH_CUSTOMER", "R7: confirm with the customer; do not block")
        add("WARN_CUSTOMER", "R7: remind the customer of the recurring charge")
    elif a.card_testing:
        add("DECLINE_TRANSACTION", "R5: card-testing sequence followed by a larger purchase")
        add("STEP_UP_AUTH", "R5: require step-up authentication before further activity")
        if a.cleared_over_100:
            if a.single_signal and a.p < 0.70:
                add("VERIFY_WITH_CUSTOMER", "R1: weak single signal, verify before blocking despite R5")
            else:
                add("BLOCK_CARD", "R5: a purchase over $100 already cleared")
    elif high_conf and a.verdict == "fraud":
        add("BLOCK_CARD", f"R1 satisfied: p={a.p:.2f} with {a.n_independent} independent pieces of evidence")
    elif strong and a.verdict == "fraud":
        add("DECLINE_TRANSACTION", f"p={a.p:.2f}: stop the flagged authorization now")
        add("VERIFY_WITH_CUSTOMER", "verify before blocking the card; not yet at the 0.85 threshold")
    elif a.verdict == "legitimate" and not a.customer_disputes:
        add("ALLOW_TRANSACTION", f"p={a.p:.2f}: activity is consistent with the cardholder's history")
        add("CLOSE_NO_FRAUD", f"p={a.p:.2f}: the alert is a false alarm")
    else:
        # uncertain, or legitimate-looking but disputed
        if a.channel == "online" and a.pattern == "card_not_present_new_device":
            add("STEP_UP_AUTH", f"R1: p={a.p:.2f} on limited signals; verify before any block")
        else:
            add("VERIFY_WITH_CUSTOMER", f"R1: p={a.p:.2f} on limited signals; verify before any block")

    # ---- R8: uncertain and exposed, or conflicting evidence
    if (a.verdict == "uncertain" and a.exposure > 500 and resp != "confirmed") or a.conflicting:
        add("ESCALATE_TO_ANALYST", "R8: verdict uncertain with exposure over $500, or evidence conflicts")

    # ---- R6: shared origin
    if a.shared_element and (denied or strong):
        add("CREATE_CASE", "R6: shared origin across cards")
        add("FILE_REPORT", f"R6: shared {a.shared_element_desc}")
        if a.connected_card_ids:
            add("MONITOR_CONNECTED_CARDS", f"R6: cards sharing {a.shared_element_desc}")

    # ---- R9: undocumented, coordinated
    if a.pattern == "undocumented" and a.coordinated and a.p >= 0.50:
        add("CREATE_CASE", "R9: undocumented coordinated pattern")
        add("FILE_REPORT", "R9: undocumented coordinated pattern across customers")
        add("ESCALATE_TO_ANALYST", "R9: undocumented pattern needs an analyst")
        if a.connected_card_ids:
            add("MONITOR_CONNECTED_CARDS", "R9: cards caught in the same pattern")

    # ---- R10: BLOCK_ALL_CARDS only with real justification
    if (denied or strong) and (a.confirmed_fraud_cards >= 2 or a.credentials_compromised):
        add("BLOCK_ALL_CARDS", "R10: at least two cards show confirmed fraud or credentials are compromised")

    # ---- 3a: SAR criteria
    ok, why = report_warranted(a)
    if ok:
        add("FILE_REPORT", f"3a: fraud strongly suspected and {why}")

    # ---- 3a: open a case
    if a.p >= 0.30 or evidence_requested or a.customer_disputes or "FILE_REPORT" in acts:
        add("CREATE_CASE", "3a: probability reached 0.30, evidence requested, or a charge was disputed")

    return _finish(a, acts)


def _finish(a: Assessment, acts: dict[str, str]) -> list[dict]:
    if "FILE_REPORT" in acts:
        acts.setdefault("CREATE_CASE", "3a: a report always has a case behind it")
    return [
        {"action": k, "route": route(k, a.exposure), "reason": acts[k]}
        for k in ORDER if k in acts
    ]


# ------------------------------------------------------------------ stopping / evidence

def stop_reason(a: Assessment, rounds: int) -> str | None:
    """Section 6. Returns the reason to stop, or None to keep investigating."""
    if a.customer_response in ("denied", "confirmed"):
        return "A verification response settled the question (policy section 6)."
    if a.customer_response == "no_reply":
        return "No reply within 24 hours; R4 actions apply and further steps would not change them (policy section 6)."
    if effective_denial(a) and a.p >= 0.5 and not a.conflicting:
        return "The customer's own report is a denial (R2) and the graph corroborates it; more evidence would not change the actions (policy section 6)."
    if (a.p >= 0.85 or a.p <= 0.15) and a.n_independent >= 2:
        return f"Fraud probability {a.p:.2f} is supported by {a.n_independent} independent pieces of evidence (policy section 6)."
    if a.p < 0.25 and not a.customer_disputes:
        return f"Probability {a.p:.2f} on weak signals; further steps are unlikely to change the decision (policy section 6)."
    if a.recurring_match and a.customer_disputes and rounds >= 1:
        return "Recurring-charge dispute handled under R7; further steps are unlikely to change the decision."
    if rounds >= MAX_EVIDENCE_ROUNDS:
        return "Requested evidence has been assessed; further steps are unlikely to change the decision (policy section 6)."
    return None


def evidence_request_type(a: Assessment) -> str:
    """Which controlled, policy-approved request to make (section 5)."""
    if a.conflicting:
        return "analyst_info"
    if a.channel == "online" and not a.customer_disputes and a.trigger_type == "risk_score":
        return "step_up_auth"
    return "customer_validation"