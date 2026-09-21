"""Simulated customer / analyst responses.

The dataset provides no replies (README section 5), so the agent simulates them and
must record the assumption in `evidence_requests.assumed_response`. Responses are
derived from the evidence, never from any hidden label:

* charge matches the customer's own recurring pattern      -> customer recognises it
* strong fraud evidence (testing, shared fraud device, p high) -> customer denies it
* clearly benign                                            -> customer confirms it
* the ambiguous middle                                      -> no reply within 24h (R4)
"""
from __future__ import annotations

from .state import Assessment


def simulate(a: Assessment, request_type: str) -> tuple[str, str]:
    """Return (response, assumed_response_text). response in confirmed | denied | no_reply."""
    channel = "step-up authentication" if request_type == "step_up_auth" else "verification request"

    if request_type == "analyst_info":
        # An analyst reviewing conflicting evidence: assume they side with the stronger side.
        if a.p >= 0.5:
            return "denied", "Analyst reviews the conflicting evidence and judges the activity unauthorised."
        return "confirmed", "Analyst reviews the conflicting evidence and judges the activity legitimate."

    if a.recurring_match:
        return "confirmed", (f"Customer answers the {channel} and recognises the charge as their own recurring payment "
                             "(same amount and product roughly every month).")
    if a.card_testing or "shared_fraud" in a.signals or a.p >= 0.55:
        if request_type == "step_up_auth":
            return "denied", "Step-up authentication is not completed and the customer states they did not make the purchases."
        return "denied", "Customer states they did not make these purchases and still has the card."
    if a.p <= 0.35:
        if request_type == "step_up_auth":
            return "confirmed", "Customer completes step-up authentication and confirms the purchase."
        return "confirmed", "Customer confirms they made the purchase."
    return "no_reply", f"No response to the {channel} within 24 hours."