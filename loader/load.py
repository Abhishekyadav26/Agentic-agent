"""Data loading.

    python -m loader.inspect_data            # 1) find how card IDs (C01234-K1) are derived
    python -m loader.load slim               # 2) transactions.csv + identity.csv -> data/tx_slim.parquet
    python -m loader.load graph              # 3) slim parquet + closed cases -> TigerGraph
    python -m loader.load graph --only closed_cases   # re-load just one part

Why a slim file: transactions.csv has 393 columns (~700 MB). The graph and the agent
need ~25 of them; the V/C/D columns stay in the CSV and can be joined by TransactionID
when a case needs them.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
SLIM = DATA / "tx_slim.parquet"

TX_COLS = (["TransactionID", "customer_id", "ts", "channel", "risk_score", "TransactionAmt", "ProductCD",
            "card2", "card3", "card4", "card5", "card6", "addr1", "addr2", "dist1", "P_emaildomain", "R_emaildomain"]
           + [f"M{i}" for i in range(1, 10)])
ID_COLS = ["TransactionID", "DeviceType", "DeviceInfo", "id_15", "id_23", "id_30", "id_31", "id_33"]

# how a card number K1/K2 is derived inside a customer; `inspect_data` finds the one that reproduces case_pack.csv
CARD_KEYS = {
    "c4c6": ["card4", "card6"],
    "c2": ["card2"],
    "c3": ["card3"],
    "c5": ["card5"],
    "c2c4c6": ["card2", "card4", "card6"],
    "c2_6": ["card2", "card3", "card4", "card5", "card6"],
}
ORDERS = ("first_seen", "frequency", "sorted")


def derive_card_ids(df: pd.DataFrame, key: str = "c4c6", order: str = "first_seen") -> np.ndarray:
    """customer_id + '-K' + rank of the card key inside the customer."""
    cols = CARD_KEYS[key]
    k = df[cols[0]].astype(str)
    for c in cols[1:]:
        k = k + "|" + df[c].astype(str)
    d = pd.DataFrame({"customer_id": df["customer_id"].values, "key": k.values, "ts": df["ts"].values})
    t = d.groupby(["customer_id", "key"]).agg(first=("ts", "min"), n=("ts", "size")).reset_index()
    if order == "first_seen":
        t = t.sort_values(["customer_id", "first", "key"])
    elif order == "frequency":
        t = t.sort_values(["customer_id", "n", "first"], ascending=[True, False, True])
    else:
        t = t.sort_values(["customer_id", "key"])
    t["rank"] = t.groupby("customer_id").cumcount() + 1
    m = d.merge(t[["customer_id", "key", "rank"]], on=["customer_id", "key"], how="left")
    return (m["customer_id"] + "-K" + m["rank"].astype(str)).values


def _region(v) -> str | None:
    return None if pd.isna(v) else str(float(v))


def build_slim(strategy: str | None = None) -> pd.DataFrame:
    strategy = strategy or os.environ.get("CARD_STRATEGY") or "c4c6:first_seen"
    key, order = strategy.split(":")
    print(f"reading transactions.csv (only {len(TX_COLS)} of 393 columns) ...")
    tx = pd.read_csv(DATA / "transactions.csv", usecols=TX_COLS, dtype={"TransactionID": str}, parse_dates=["ts"])
    idn = pd.read_csv(DATA / "identity.csv", usecols=ID_COLS, dtype={"TransactionID": str})
    print(f"{len(tx):,} transactions, {len(idn):,} identity records")

    dev = idn[["DeviceInfo", "id_30", "id_31", "id_33"]].fillna("?").astype(str)
    idn["device_str"] = dev["DeviceInfo"] + " | " + dev["id_30"] + " | " + dev["id_31"] + " | " + dev["id_33"]
    idn.loc[idn[["DeviceInfo", "id_30", "id_31", "id_33"]].isna().all(axis=1), "device_str"] = None
    idn["device_id"] = idn["device_str"].map(lambda s: None if pd.isna(s) else "D" + hashlib.md5(s.encode()).hexdigest()[:10])
    idn["proxy"] = idn["id_23"].notna()
    tx = tx.merge(idn[["TransactionID", "device_id", "device_str", "id_15", "proxy"]], on="TransactionID", how="left")
    tx["proxy"] = tx["proxy"].fillna(False).astype(bool)

    tx["card_id"] = derive_card_ids(tx, key, order)
    mcols = [f"M{i}" for i in range(1, 10)]
    tx["m_fails"] = (tx[mcols] == "F").sum(axis=1).astype("int8")

    slim = pd.DataFrame({
        "txn_id": tx["TransactionID"].astype(str), "customer_id": tx["customer_id"], "card_id": tx["card_id"],
        "ts": tx["ts"], "amount": tx["TransactionAmt"].astype(float), "product_cd": tx["ProductCD"],
        "channel": tx["channel"], "risk_score": tx["risk_score"].astype(float),
        "addr1": tx["addr1"].map(_region), "addr2": tx["addr2"].map(_region), "dist1": tx["dist1"],
        "p_email": tx["P_emaildomain"], "r_email": tx["R_emaildomain"],
        "device_id": tx["device_id"], "device_str": tx["device_str"], "id_15": tx["id_15"],
        "proxy": tx["proxy"], "m_fails": tx["m_fails"], "card4": tx["card4"], "card6": tx["card6"],
    }).sort_values("ts").reset_index(drop=True)
    slim.to_parquet(SLIM, index=False)
    print(f"wrote {SLIM} ({len(slim):,} rows, {slim['card_id'].nunique():,} cards, {slim['customer_id'].nunique():,} customers)")
    return slim


# ================================================================== TigerGraph
def _batches(df: pd.DataFrame, n: int = 20000):
    for i in range(0, len(df), n):
        yield df.iloc[i:i + n]


def _s(x) -> str:
    return x.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(x) else "1970-01-01 00:00:00"


def load_graph(only: str | None = None) -> None:
    from agent.tools import get_conn
    conn = get_conn()
    slim = pd.read_parquet(SLIM)
    slim["ts_s"] = slim["ts"].map(_s)

    def up_v(df, vtype, vid, attrs):
        for b in _batches(df):
            conn.upsertVertexDataFrame(b, vtype, v_id=vid, attributes=attrs)
        print(f"  {vtype}: {len(df):,}")

    def up_e(df, s_type, e_type, t_type, f, t):
        for b in _batches(df):
            conn.upsertEdgeDataFrame(b, s_type, e_type, t_type, from_id=f, to_id=t)
        print(f"  {e_type}: {len(df):,}")

    def want(part):
        return only is None or only == part

    if want("core"):
        print("customers, cards, transactions ...")
        cust = slim[["customer_id"]].drop_duplicates()
        up_v(cust, "Customer", "customer_id", {})
        cards = slim.groupby("card_id").agg(customer_id=("customer_id", "first"), card4=("card4", "first"),
                                            card6=("card6", "first")).reset_index().fillna("")
        up_v(cards, "Card", "card_id", {"customer_id": "customer_id", "card4": "card4", "card6": "card6"})
        t = slim.assign(risk_score=slim["risk_score"].fillna(0.0), addr1=slim["addr1"].fillna(""),
                        addr2=slim["addr2"].fillna(""), p_email=slim["p_email"].fillna(""),
                        device_id=slim["device_id"].fillna(""), id_15=slim["id_15"].fillna(""),
                        proxy=slim["proxy"].astype(bool), m_fails=slim["m_fails"].astype(int))
        up_v(t, "Transaction", "txn_id", {
            "ts": "ts_s", "amount": "amount", "product_cd": "product_cd", "channel": "channel", "risk_score": "risk_score",
            "addr1": "addr1", "addr2": "addr2", "p_email": "p_email", "device_id": "device_id", "id_15": "id_15",
            "proxy": "proxy", "m_fails": "m_fails", "card_id": "card_id", "customer_id": "customer_id"})
        up_e(cards[["customer_id", "card_id"]], "Customer", "OWNS", "Card", "customer_id", "card_id")
        up_e(slim[["card_id", "txn_id"]], "Card", "MADE", "Transaction", "card_id", "txn_id")
        srt = slim.sort_values(["card_id", "ts"])
        nxt = pd.DataFrame({"a": srt["txn_id"].values[:-1], "b": srt["txn_id"].values[1:],
                            "same": srt["card_id"].values[:-1] == srt["card_id"].values[1:]})
        up_e(nxt[nxt["same"]][["a", "b"]], "Transaction", "NEXT", "Transaction", "a", "b")

    if want("entities"):
        print("devices, regions, email domains ...")
        d = slim[slim["device_id"].notna()]
        dv = d.groupby("device_id").agg(device_str=("device_str", "first")).reset_index()
        up_v(dv, "DeviceProfile", "device_id", {"device_str": "device_str"})
        up_e(d[["txn_id", "device_id"]], "Transaction", "FROM_DEVICE", "DeviceProfile", "txn_id", "device_id")
        r = slim[slim["addr1"].notna()]
        up_v(r[["addr1"]].drop_duplicates(), "BillingRegion", "addr1", {})
        up_e(r[["txn_id", "addr1"]], "Transaction", "BILLED_IN", "BillingRegion", "txn_id", "addr1")
        e = slim[slim["p_email"].notna()]
        up_v(e[["p_email"]].drop_duplicates(), "EmailDomain", "p_email", {})
        up_e(e[["txn_id", "p_email"]], "Transaction", "PURCHASER_EMAIL", "EmailDomain", "txn_id", "p_email")

    if want("closed_cases"):
        print("closed cases ...")
        cc = pd.read_csv(DATA / "closed_cases_history.csv", dtype=str)
        cc["opened_at"] = pd.to_datetime(cc["opened_at"]).map(_s)
        cc["closed_at"] = pd.to_datetime(cc["closed_at"]).map(_s)
        cc["exposure"] = pd.to_numeric(cc["exposure_usd"], errors="coerce").fillna(0.0)
        cc["notes"] = cc["analyst_notes"].fillna("")
        cc["pattern"] = cc["pattern"].fillna("none")
        cc["source"] = "bank"
        up_v(cc, "ClosedCase", "case_id", {"outcome": "outcome", "pattern": "pattern", "exposure": "exposure",
                                            "notes": "notes", "source": "source", "opened_at": "opened_at",
                                            "closed_at": "closed_at"})
        up_e(cc[["case_id", "card_id"]].dropna(), "ClosedCase", "ON_CARD", "Card", "case_id", "card_id")
        inv = cc[["case_id", "txn_ids"]].dropna().assign(txn=lambda d: d["txn_ids"].str.split("|")).explode("txn")
        up_e(inv[inv["txn"] != ""][["case_id", "txn"]], "ClosedCase", "INVOLVES", "Transaction", "case_id", "txn")
        con = cc[["case_id", "connected_card_ids"]].dropna().assign(c=lambda d: d["connected_card_ids"].str.split("|")).explode("c")
        up_e(con[con["c"] != ""][["case_id", "c"]], "ClosedCase", "CONNECTED_TO", "Card", "case_id", "c")
    print("done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["slim", "graph"])
    ap.add_argument("--strategy", help="card id strategy 'key:order', e.g. c4c6:first_seen")
    ap.add_argument("--only", choices=["core", "entities", "closed_cases"])
    a = ap.parse_args()
    if a.cmd == "slim":
        build_slim(a.strategy)
    else:
        load_graph(a.only)


if __name__ == "__main__":
    main()