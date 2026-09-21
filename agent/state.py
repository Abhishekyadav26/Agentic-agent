"""Shared data structures for the fraud investigation agent."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional, TypedDict


@dataclass
class Evidence:
    """One claim the agent rests on. Maps 1:1 to the answer-file `evidence` items."""
    claim: str
    source: str            # graph | document | customer | external
    ref: str               # query name, document section or request id
    entity_ids: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Assessment:
    """Everything the policy engine and the answer writer need."""
    # --- conclusion
    p: float = 0.0
    verdict: str = "uncertain"            # fraud | legitimate | uncertain
    pattern: str = "none"
    pattern_description: str = ""
    affected_txn_ids: list = field(default_factory=list)
    first_suspicious_txn_id: str = ""
    connected_card_ids: list = field(default_factory=list)
    connected_device_profiles: list = field(default_factory=list)
    exposure: float = 0.0
    evidence: list = field(default_factory=list)          # list[Evidence]
    similar_prior_cases: list = field(default_factory=list)
    signals: dict = field(default_factory=dict)           # name -> log-odds contribution
    n_independent: int = 0                                # independent evidence categories

    # --- flags consumed by the policy engine
    channel: str = "online"
    customer_response: Optional[str] = None   # None | confirmed | denied | no_reply
    customer_disputes: bool = False           # trigger was a customer report
    recurring_match: bool = False
    card_testing: bool = False
    cleared_over_100: bool = False
    single_signal: bool = True
    shared_element: bool = False              # device / region / email shared with other cards' fraud
    shared_element_desc: str = ""
    coordinated: bool = False                 # several customers linked through a shared element
    confirmed_fraud_cards: int = 0
    credentials_compromised: bool = False
    conflicting: bool = False                 # evidence points in opposite directions
    trigger_type: str = "risk_score"


class CaseState(TypedDict, total=False):
    """State carried through the LangGraph workflow."""
    case: dict                     # row from case_pack.csv
    features: dict                 # raw graph evidence (cached; gathered once)
    assessment: Any                # Assessment
    initial_actions: list
    final_actions: list
    evidence_requests: list
    rounds: int
    customer_response: str        # confirmed | denied | no_reply (simulated)
    p_initial: float
    stop_reason: str
    steps: list                    # human-readable trace, drives the UI timeline
    step_no: int
    explain: dict                  # summary / narrative / what_changed ...
    written_to_graph: bool
    graph_case_id: str
    metrics: dict                  # tool_calls, tokens, latency_s
    answer: dict