"""Data access for the agent.

Two interchangeable backends behind one facade (`Tools`):

* PandasBackend      - reads data/tx_slim.parquet. Fast local development, tests,
                       and eval. Not a graph, so answers report written_to_graph=False.
* TigerGraphBackend  - runs the installed GSQL queries in graph/queries and writes
                       cases back into the graph (this is what you submit).

Every method call through `Tools` increments `tools.calls`, which feeds the
`tool_calls` field of the answer file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

DATA_START = pd.Timestamp("2016-07-01")
DATA_END = pd.Timestamp("2017-01-02")
TS_FMT = "%Y-%m-%d %H:%M:%S"

TXN_COLS = [
    "txn_id", "card_id", "customer_id", "ts", "amount", "product_cd", "channel", "risk_score",
    "addr1", "addr2", "device_id", "id_15", "proxy", "m_fails", "p_email",
]


def _split(v) -> list[str]:
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return []
    return [x for x in str(v).split("|") if x]


def load_cases(src) -> pd.DataFrame:
    """Normalise closed_cases_history.csv (or a DataFrame of it)."""
    df = pd.read_csv(src, dtype=str) if isinstance(src, (str, Path)) else src.copy()
    for c in ("opened_at", "closed_at"):
        df[c] = pd.to_datetime(df.get(c), errors="coerce")
    df["exposure_usd"] = pd.to_numeric(df.get("exposure_usd"), errors="coerce").fillna(0.0)
    df["txn_list"] = df["txn_ids"].map(_split)
    df["connected_list"] = df["connected_card_ids"].map(_split)
    if "source" not in df:
        df["source"] = "bank"
    return df


def _visible(df: pd.DataFrame, before, exclude) -> pd.DataFrame:
    """Only cases that were already known at `before` (prevents leakage in eval)."""
    if df.empty:
        return df
    if before is not None:
        ref = df["closed_at"].fillna(df["opened_at"])
        df = df[ref < pd.Timestamp(before)]
    if exclude:
        df = df[~df["case_id"].isin(set(exclude))]
    return df


# =============================================================================== pandas
class PandasBackend:
    is_graph = False

    def __init__(self, tx: pd.DataFrame | None = None, cases=None,
                 tx_path="data/tx_slim.parquet", cases_path="data/closed_cases_history.csv",
                 agent_cases_path="data/agent_cases.jsonl"):
        tx = pd.read_parquet(tx_path) if tx is None else tx
        self.tx = tx.sort_values("ts").reset_index(drop=True)
        self.tx["txn_id"] = self.tx["txn_id"].astype(str)
        self.tx_idx = self.tx.set_index("txn_id", drop=False)
        self.by_card = {k: g for k, g in self.tx.groupby("card_id")}
        dev = self.tx[self.tx["device_id"].notna()]
        self.by_device = {k: g for k, g in dev.groupby("device_id")}
        reg = self.tx[self.tx["addr1"].notna()]
        self.by_region = {k: g for k, g in reg.groupby("addr1")}
        self.cust_cards = self.tx.groupby("customer_id")["card_id"].unique().to_dict()
        self.agent_cases_path = Path(agent_cases_path)
        base = load_cases(cases if cases is not None else cases_path)
        self.cases = base
        if self.agent_cases_path.exists():
            rows = [json.loads(l) for l in self.agent_cases_path.read_text().splitlines() if l.strip()]
            if rows:
                self.cases = pd.concat([base, load_cases(pd.DataFrame(rows))], ignore_index=True)
        self._ent = None

    # ---- transactions
    def txn(self, txn_id: str):
        txn_id = str(txn_id)
        return self.tx_idx.loc[txn_id] if txn_id in self.tx_idx.index else None

    def card_window(self, card_id, t0, t1) -> pd.DataFrame:
        g = self.by_card.get(card_id)
        if g is None:
            return self.tx.iloc[0:0]
        return g[(g["ts"] >= t0) & (g["ts"] <= t1)].sort_values("ts")

    def customer_cards(self, customer_id) -> list[str]:
        return list(self.cust_cards.get(customer_id, []))

    # ---- neighbours
    @staticmethod
    def _agg(g: pd.DataFrame, t0, t1, exclude_card) -> pd.DataFrame:
        g = g[(g["ts"] >= t0) & (g["ts"] <= t1)]
        if exclude_card is not None:
            g = g[g["card_id"] != exclude_card]
        if g.empty:
            return pd.DataFrame(columns=["card_id", "customer_id", "n_txn", "first_ts", "last_ts"])
        out = g.groupby(["card_id", "customer_id"]).agg(
            n_txn=("txn_id", "size"), first_ts=("ts", "min"), last_ts=("ts", "max")).reset_index()
        return out.sort_values("n_txn", ascending=False)

    def device_neighbors(self, device_id, t0, t1, exclude_card=None) -> pd.DataFrame:
        return self._agg(self.by_device.get(device_id, self.tx.iloc[0:0]), t0, t1, exclude_card)

    def region_neighbors(self, region, t0, t1, exclude_card=None) -> pd.DataFrame:
        return self._agg(self.by_region.get(region, self.tx.iloc[0:0]), t0, t1, exclude_card)

    def device_str(self, device_id) -> str:
        g = self.by_device.get(device_id)
        if g is None or "device_str" not in g:
            return str(device_id)
        return str(g["device_str"].iloc[0])

    # ---- closed cases
    def _entities(self) -> pd.DataFrame:
        if self._ent is None:
            ce = self.cases[["case_id", "txn_list"]].explode("txn_list").rename(columns={"txn_list": "txn_id"})
            ce = ce.dropna(subset=["txn_id"])
            self._ent = ce.merge(self.tx[["txn_id", "card_id", "device_id", "addr1"]], on="txn_id", how="left")
        return self._ent

    def _cases_from(self, ids) -> pd.DataFrame:
        return self.cases[self.cases["case_id"].isin(set(ids))]

    def cases_by_card(self, card_id, before=None, exclude=None) -> pd.DataFrame:
        c = self.cases
        m = (c["card_id"] == card_id) | c["connected_list"].map(lambda l: card_id in l)
        return _visible(c[m], before, exclude)

    def cases_by_device(self, device_id, before=None, exclude=None) -> pd.DataFrame:
        e = self._entities()
        return _visible(self._cases_from(e[e["device_id"] == device_id]["case_id"]), before, exclude)

    def cases_by_region(self, region, before=None, exclude=None) -> pd.DataFrame:
        e = self._entities()
        return _visible(self._cases_from(e[e["addr1"] == region]["case_id"]), before, exclude)

    def all_cases(self) -> pd.DataFrame:
        return self.cases

    # ---- memory write (local stand-in for the graph write)
    def write_case(self, rec: dict) -> bool:
        self.agent_cases_path.parent.mkdir(parents=True, exist_ok=True)
        with self.agent_cases_path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        self.cases = pd.concat([self.cases, load_cases(pd.DataFrame([rec]))], ignore_index=True)
        self._ent = None
        return False   # not written to a graph


# =============================================================================== TigerGraph
def get_conn():
    """Connection details differ between Savanna and Community Edition; adjust here."""
    import pyTigerGraph as tg
    kw = dict(host=os.environ["TG_HOST"], graphname=os.environ.get("TG_GRAPH", "FraudGraph"),
              username=os.environ.get("TG_USER", "tigergraph"), password=os.environ.get("TG_PASS", ""))
    if os.environ.get("TG_CLOUD", "0") == "1":
        kw["tgCloud"] = True
    conn = tg.TigerGraphConnection(**kw)
    if os.environ.get("TG_TOKEN"):
        conn.apiToken = os.environ["TG_TOKEN"]
    else:
        try:
            conn.getToken(conn.createSecret())
        except Exception:      # older / community setups authenticate with user+password
            pass
    return conn


def _fmt(ts) -> str:
    return pd.Timestamp(ts).strftime(TS_FMT)


def _verts_to_df(verts: list[dict], id_col: str) -> pd.DataFrame:
    rows = [{id_col: v["v_id"], **v["attributes"]} for v in verts]
    return pd.DataFrame(rows)


class TigerGraphBackend:
    """Runs the GSQL queries from graph/queries. Untested against a live instance in this
    repo's CI: expect to fix small GSQL / pyTigerGraph differences on first run."""
    is_graph = True

    def __init__(self, cases_path="data/closed_cases_history.csv", conn=None):
        self.conn = conn or get_conn()
        self.cases = load_cases(cases_path)     # 5.5k rows: keep the narratives locally too
        self._dev_str: dict[str, str] = {}

    def _run(self, name: str, params: dict):
        return self.conn.runInstalledQuery(name, params)

    def txn(self, txn_id):
        res = self.conn.getVerticesById("Transaction", str(txn_id))
        if not res:
            return None
        v = res[0] if isinstance(res, list) else res
        row = {"txn_id": v["v_id"], **v["attributes"]}
        row["ts"] = pd.Timestamp(row["ts"])
        return pd.Series(row)

    def card_window(self, card_id, t0, t1) -> pd.DataFrame:
        res = self._run("card_window", {"c": card_id, "c.type": "Card", "t0": _fmt(t0), "t1": _fmt(t1)})
        df = _verts_to_df(res[0].get("Txns", []), "txn_id")
        if df.empty:
            return pd.DataFrame(columns=TXN_COLS)
        df["ts"] = pd.to_datetime(df["ts"])
        return df.sort_values("ts")

    def customer_cards(self, customer_id) -> list[str]:
        edges = self.conn.getEdges("Customer", customer_id, "OWNS")
        return [e["to_id"] for e in edges]

    def _neighbors(self, query, key, value, vtype, t0, t1, exclude_card):
        res = self._run(query, {key: value, f"{key}.type": vtype, "t0": _fmt(t0), "t1": _fmt(t1)})
        df = pd.DataFrame(res[0].get("Cards", []))
        if df.empty:
            return pd.DataFrame(columns=["card_id", "customer_id", "n_txn", "first_ts", "last_ts"])
        for c in ("first_ts", "last_ts"):
            df[c] = pd.to_datetime(df[c])
        if exclude_card is not None:
            df = df[df["card_id"] != exclude_card]
        return df.sort_values("n_txn", ascending=False)

    def device_neighbors(self, device_id, t0, t1, exclude_card=None):
        return self._neighbors("device_neighbors", "d", device_id, "DeviceProfile", t0, t1, exclude_card)

    def region_neighbors(self, region, t0, t1, exclude_card=None):
        return self._neighbors("region_neighbors", "r", region, "BillingRegion", t0, t1, exclude_card)

    def device_str(self, device_id) -> str:
        if device_id not in self._dev_str:
            res = self.conn.getVerticesById("DeviceProfile", device_id)
            self._dev_str[device_id] = res[0]["attributes"].get("device_str", device_id) if res else device_id
        return self._dev_str[device_id]

    def _cases_via(self, query, key, value, vtype, before, exclude):
        res = self._run(query, {key: value, f"{key}.type": vtype})
        ids = [v["v_id"] for v in res[0].get("Cases", [])]
        return _visible(self.cases[self.cases["case_id"].isin(ids)], before, exclude)

    def cases_by_card(self, card_id, before=None, exclude=None):
        return self._cases_via("closed_cases_by_card", "c", card_id, "Card", before, exclude)

    def cases_by_device(self, device_id, before=None, exclude=None):
        return self._cases_via("closed_cases_by_device", "d", device_id, "DeviceProfile", before, exclude)

    def cases_by_region(self, region, before=None, exclude=None):
        return self._cases_via("closed_cases_by_region", "r", region, "BillingRegion", before, exclude)

    def all_cases(self):
        return self.cases

    def write_case(self, rec: dict) -> bool:
        """Case memory: the case becomes a ClosedCase vertex (source='agent') linked to
        the transactions and cards it involves, so the next investigation can find it."""
        cid = rec["case_id"]
        self.conn.upsertVertex("ClosedCase", cid, {
            "outcome": rec["outcome"], "pattern": rec["pattern"], "exposure": float(rec["exposure_usd"]),
            "notes": rec["analyst_notes"], "source": "agent",
            "opened_at": _fmt(rec["opened_at"]), "closed_at": _fmt(rec.get("closed_at") or rec["opened_at"]),
        })
        self.conn.upsertEdge("ClosedCase", cid, "ON_CARD", "Card", rec["card_id"])
        for t in _split(rec["txn_ids"]):
            self.conn.upsertEdge("ClosedCase", cid, "INVOLVES", "Transaction", t)
        for c in _split(rec["connected_card_ids"]):
            self.conn.upsertEdge("ClosedCase", cid, "CONNECTED_TO", "Card", c)
        self.cases = pd.concat([self.cases, load_cases(pd.DataFrame([rec]))], ignore_index=True)
        return True


# =============================================================================== facade
def make_backend(kind: str | None = None):
    kind = (kind or os.environ.get("BACKEND", "pandas")).lower()
    return TigerGraphBackend() if kind == "tigergraph" else PandasBackend()


class Tools:
    """What the agent calls. Counts calls; hides which backend is behind it."""

    def __init__(self, backend=None, docs_dir="docs"):
        self.b = backend or make_backend()
        self.calls = 0
        self.trace: list[str] = []
        self.docs_dir = docs_dir
        self._retriever = None

    @property
    def is_graph(self) -> bool:
        return self.b.is_graph

    def _c(self, name, *a, **k):
        self.calls += 1
        self.trace.append(name)
        return getattr(self.b, name)(*a, **k)

    def txn(self, txn_id): return self._c("txn", txn_id)
    def card_window(self, card_id, t0, t1): return self._c("card_window", card_id, t0, t1)
    def customer_cards(self, customer_id): return self._c("customer_cards", customer_id)
    def device_neighbors(self, d, t0, t1, exclude_card=None): return self._c("device_neighbors", d, t0, t1, exclude_card)
    def region_neighbors(self, r, t0, t1, exclude_card=None): return self._c("region_neighbors", r, t0, t1, exclude_card)
    def device_str(self, d): return self._c("device_str", d)
    def cases_by_card(self, c, before=None, exclude=None): return self._c("cases_by_card", c, before, exclude)
    def cases_by_device(self, d, before=None, exclude=None): return self._c("cases_by_device", d, before, exclude)
    def cases_by_region(self, r, before=None, exclude=None): return self._c("cases_by_region", r, before, exclude)
    def write_case(self, rec):
        ok = self._c("write_case", rec)
        if self._retriever is not None:            # the new case becomes retrievable memory
            self._retriever.refresh(self.b.all_cases())
        return ok

    def retrieve(self, query, k=5, kinds=("case", "doc"), before=None, exclude=None):
        """GraphRAG text side: closed-case narratives + policy/pattern documents."""
        if self._retriever is None:
            from .memory import Retriever
            self._retriever = Retriever(self.b.all_cases(), self.docs_dir)
        self.calls += 1
        self.trace.append("retrieve")
        return self._retriever.search(query, k=k, kinds=kinds, before=before, exclude=exclude)

    def all_cases(self): return self.b.all_cases()     # bulk read for the retriever; not counted