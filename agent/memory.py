"""Case memory and retrieval.

* Retriever  - TF-IDF retrieval over (a) closed-case narratives and (b) policy /
               pattern documents in docs/. This is the local baseline for the
               GraphRAG text side. To satisfy "TigerGraph vector store", embed the same
               chunks and store them as vector attributes; keep this class as the
               interface (see README "Next steps").
* remember() - writes the finished case back so later investigations can retrieve it.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def _chunks(path: Path) -> list[tuple[str, str]]:
    """Split a markdown file on headings so each chunk is one rule/section."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    parts = re.split(r"\n(?=#{1,4} )", text)
    out = []
    for p in parts:
        p = p.strip()
        if len(p) > 40:
            title = p.splitlines()[0].lstrip("# ").strip()
            out.append((f"{path.stem}:{title[:60]}", p))
    return out


class Retriever:
    def __init__(self, cases: pd.DataFrame, docs_dir: str = "docs"):
        self.docs_dir = docs_dir
        self.refresh(cases)

    def refresh(self, cases: pd.DataFrame) -> None:
        rows = []
        for r in cases.itertuples():
            text = " ".join(str(x) for x in (getattr(r, "pattern", ""), getattr(r, "outcome", ""),
                                            getattr(r, "analyst_notes", ""), getattr(r, "actions_taken", "")))
            ref = r.closed_at if pd.notna(r.closed_at) else r.opened_at
            rows.append(dict(kind="case", id=r.case_id, text=text, when=ref))
        d = Path(self.docs_dir)
        if d.exists():
            for f in sorted(d.glob("*.md")):
                for cid, txt in _chunks(f):
                    rows.append(dict(kind="doc", id=cid, text=txt, when=pd.NaT))
        self.items = pd.DataFrame(rows, columns=["kind", "id", "text", "when"])
        if self.items.empty:
            self.vec, self.M = None, None
            return
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.M = self.vec.fit_transform(self.items["text"])

    def search(self, query: str, k: int = 5, kinds=("case", "doc"), before=None, exclude=None) -> list[dict]:
        if self.vec is None or not query.strip():
            return []
        sims = linear_kernel(self.vec.transform([query]), self.M).ravel()
        mask = self.items["kind"].isin(kinds).to_numpy().copy()
        if before is not None:
            when = pd.to_datetime(self.items["when"])
            mask = mask & (when.isna() | (when < pd.Timestamp(before))).to_numpy()
        if exclude:
            mask = mask & ~self.items["id"].isin(set(exclude)).to_numpy()
        sims = np.where(mask, sims, -1.0)
        out = []
        for i in np.argsort(-sims)[:k]:
            if sims[i] > 0.02:
                r = self.items.iloc[i]
                out.append(dict(kind=r["kind"], id=r["id"], score=float(sims[i]), text=r["text"]))
        return out


# ------------------------------------------------------------------ write-back
def graph_case_id(case_id: str) -> str:
    digits = re.findall(r"\d+", case_id)
    n = 1000 + int(digits[-1]) if digits else abs(hash(case_id)) % 9000 + 1000
    return f"CASE-2016-{n}"


def build_case_record(case: dict, a, status: str, final_actions: list, summary: str) -> dict:
    outcome = {"closed_fraud": "agent_fraud", "closed_legitimate": "agent_legitimate"}.get(status, "agent_open")
    return {
        "case_id": graph_case_id(case["case_id"]),
        "customer_id": case["customer_id"],
        "card_id": case["card_id"],
        "opened_at": str(case["opened_at"]),
        "closed_at": "",
        "outcome": outcome,
        "pattern": a.pattern,
        "first_fraud_txn_id": a.first_suspicious_txn_id,
        "txn_ids": "|".join(a.affected_txn_ids),
        "n_txns": len(a.affected_txn_ids),
        "exposure_usd": a.exposure,
        "connected_card_ids": "|".join(a.connected_card_ids),
        "actions_taken": "|".join(x["action"] for x in final_actions),
        "report_filed": any(x["action"] == "FILE_REPORT" for x in final_actions),
        "analyst_notes": summary,
        "source": "agent",
    }


def remember(tools, case: dict, a, status: str, final_actions: list, summary: str) -> tuple[bool, str]:
    rec = build_case_record(case, a, status, final_actions, summary)
    try:
        written = bool(tools.write_case(rec))
    except Exception as e:                       # memory is best-effort; never crash the investigation
        print(f"[memory] write failed: {e}")
        written = False
    return written, rec["case_id"]