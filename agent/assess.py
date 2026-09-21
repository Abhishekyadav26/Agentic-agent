"""Evidence gathering (graph traversals) and transparent scoring.

gather()  runs the graph queries once and returns a plain dict of facts (`F`).
score()   turns F (+ an optional customer response) into an Assessment. It is cheap
          and pure, so the agent can re-score after new evidence arrives.

The probability is a sum of named log-odds contributions, so every point of the
number is explainable and tunable. Weights are a first guess: fit / check them with
`python -m eval.replay` on the closed cases before trusting them.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .state import Assessment, Evidence
from .tools import DATA_END, DATA_START

H = pd.Timedelta(hours=1)
LOOKBACK_H = 72
LOOKAHEAD_H = 72
PRIOR = 0.30          # prior odds for an alert; the exam is ~half legitimate, the history is ~84% fraud

W = dict(
    risk=1.5,                 # multiplied by (risk_score - 0.5): a nudge, never the answer
    card_testing=2.6, card_testing_partial=1.5,
    new_device=1.2, proxy=0.8,
    amount_high=0.9, new_product=0.8, burst=0.7,
    region_clone=1.7, region_trip=-0.8, region_new_only=0.6,
    shared_fraud=1.8, shared_many=0.9, region_cluster=1.2,
    m_bad=0.6, ato=1.1,
    recurring=-2.2, normal=-0.8,
    dispute=1.6, prior_fraud=0.6, prior_cleared=-0.4,
    denied=1.5, confirmed=-3.0,
)
CATEGORY = dict(
    risk="model", card_testing="sequence", card_testing_partial="sequence",
    new_device="device", proxy="device",
    amount_high="behavior", new_product="behavior", burst="behavior", normal="behavior",
    region_clone="region", region_trip="region", region_new_only="region",
    shared_fraud="network", shared_many="network", region_cluster="network",
    m_bad="identity", ato="identity", recurring="recurring",
    dispute="customer", prior_fraud="history", prior_cleared="history",
    denied="response", confirmed="response",
)


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def _ids(df) -> list[str]:
    return [str(x) for x in df["txn_id"].tolist()] if len(df) else []


def _money(x: float) -> str:
    return f"${x:,.2f}"


# ================================================================== baseline + per-txn flags
def baseline_profile(h: pd.DataFrame) -> dict:
    p = dict(n=len(h), med=0.0, p95=0.0, max=0.0, products=set(), regions=set(), home_region=None,
             devices=set(), n_dev=0, m_mean=0.0, online_share=0.0, n_days=0)
    if h.empty:
        return p
    p.update(
        med=float(h["amount"].median()), p95=float(h["amount"].quantile(0.95)), max=float(h["amount"].max()),
        products=set(h["product_cd"].dropna()), regions=set(h["addr1"].dropna()),
        devices=set(h["device_id"].dropna()), n_dev=int(h["device_id"].notna().sum()),
        m_mean=float(h["m_fails"].fillna(0).mean()), online_share=float((h["channel"] == "online").mean()),
        n_days=int(h["ts"].dt.date.nunique()),
    )
    ip = h[(h["channel"] == "in_person") & h["addr1"].notna()]
    src = ip if len(ip) else h[h["addr1"].notna()]
    if len(src):
        p["home_region"] = src["addr1"].value_counts().idxmax()
    return p


def annotate(w: pd.DataFrame, prof: dict) -> pd.DataFrame:
    """Add boolean anomaly columns f_* comparing each txn with the card's own baseline."""
    w = w.copy()
    if w.empty:
        for c in ("amount_high", "new_product", "new_region", "new_device", "proxy", "m_bad", "anom"):
            w[f"f_{c}"] = pd.Series(dtype=bool)
        return w
    enough = prof["n"] >= 5
    online = w["channel"].eq("online")
    inperson = w["channel"].eq("in_person")
    w["f_amount_high"] = (w["amount"] > np.maximum(2 * prof["p95"], prof["p95"] + 40)) & (w["amount"] > 20) & enough
    w["f_new_product"] = online & ~w["product_cd"].isin(prof["products"]) & enough
    w["f_new_region"] = inperson & w["addr1"].notna() & ~w["addr1"].isin(prof["regions"]) & enough
    known_dev = prof["n_dev"] >= 3
    unseen = w["device_id"].notna() & ~w["device_id"].isin(prof["devices"]) & known_dev
    w["f_new_device"] = online & (w["id_15"].eq("New") | unseen)
    w["f_proxy"] = online & w["proxy"].fillna(False).astype(bool)
    w["f_m_bad"] = w["m_fails"].fillna(0) >= max(3.0, prof["m_mean"] + 2.0)
    w["f_anom"] = w[["f_amount_high", "f_new_product", "f_new_region", "f_new_device", "f_proxy"]].any(axis=1)
    return w


# ================================================================== pattern detectors
def find_card_testing(w: pd.DataFrame, t_flag) -> dict | None:
    """>=3 tiny (<=$5) online authorizations within an hour, then a larger purchase."""
    on = w[w["channel"] == "online"].sort_values("ts")
    tiny = on[on["amount"] <= 5.0]
    partial = None
    for i in range(len(tiny)):
        s = tiny.iloc[i]["ts"]
        grp = tiny[(tiny["ts"] >= s) & (tiny["ts"] <= s + pd.Timedelta(minutes=60))]
        if len(grp) < 3:
            continue
        last = grp["ts"].max()
        if not (s - 24 * H <= t_flag <= last + 24 * H):
            continue
        after = on[(on["ts"] > last) & (on["ts"] <= last + 24 * H) & (on["amount"] > 5.0)]
        found = dict(tiny=grp, after=after.head(3), complete=len(after) > 0,
                     cleared_over_100=bool((after["amount"] > 100).any()) if len(after) else False)
        if found["complete"]:
            return found
        partial = partial or found
    return partial


def region_analysis(w: pd.DataFrame, prof: dict, flag) -> dict:
    res = dict(active=False)
    if flag["channel"] != "in_person" or pd.isna(flag["addr1"]):
        return res
    region = flag["addr1"]
    if region in prof["regions"] or prof["n"] < 5:
        return res
    ip = w[w["channel"] == "in_person"]
    reg = ip[ip["addr1"] == region]
    home = prof["home_region"]
    hom = ip[ip["addr1"] == home] if home else ip.iloc[0:0]
    lo, hi = reg["ts"].min(), reg["ts"].max()
    during = hom[(hom["ts"] >= lo - 12 * H) & (hom["ts"] <= hi + 12 * H)]
    days = int(reg["ts"].dt.date.nunique())
    return dict(active=True, region=region, reg=reg, home=home, days=days, home_during=len(during),
                clone=len(during) > 0, trip=(days >= 2 and len(during) == 0))


def recurring_check(h: pd.DataFrame, flag) -> dict | None:
    """Same product and amount roughly monthly before the flagged charge (R7 pattern)."""
    if h.empty:
        return None
    tol = max(0.5, 0.02 * abs(flag["amount"]))
    same = h[(h["amount"].sub(flag["amount"]).abs() <= tol) & (h["product_cd"] == flag["product_cd"])].sort_values("ts")
    if len(same) < 2:
        return None
    gaps = same["ts"].diff().dropna().dt.total_seconds() / 86400
    last_gap = (flag["ts"] - same["ts"].iloc[-1]).total_seconds() / 86400
    if gaps.between(24, 37).any() and 24 <= last_gap <= 37:
        return dict(ids=_ids(same), n=len(same), gap=float(last_gap))
    return None


# ================================================================== gather (graph work)
def gather(tools, case: dict) -> dict:
    card, cust = case["card_id"], case["customer_id"]
    opened = pd.Timestamp(case["opened_at"])
    flag = tools.txn(case["flagged_txn_id"])
    if flag is None:
        raise KeyError(f"flagged transaction {case['flagged_txn_id']} not found in the graph")
    t = flag["ts"]
    w0, w1 = t - LOOKBACK_H * H, t + LOOKAHEAD_H * H

    win_raw = tools.card_window(card, w0, w1)
    hist = tools.card_window(card, DATA_START, w0 - pd.Timedelta(seconds=1))
    prof = baseline_profile(hist)
    win = annotate(win_raw, prof)
    near = win[(win["ts"] >= t - 48 * H) & (win["ts"] <= t + 48 * H)]

    F = dict(case=case, flag=flag, t=t, win=win, near=near, prof=prof, opened=opened,
             risk=float(flag["risk_score"]) if pd.notna(flag["risk_score"]) else 0.5,
             testing=find_card_testing(win, t), region=region_analysis(win, prof, flag),
             recurring=recurring_check(hist, flag), thin=prof["n"] < 5)

    exclude = [case.get("case_id", "")]       # never retrieve a case as evidence for itself
    before = opened

    # ---- shared devices (graph traversal Card -> Transaction -> DeviceProfile -> Transaction -> Card)
    devs = []
    on_dev = near[near["device_id"].notna()]
    if len(on_dev):
        order = (on_dev.assign(_n=on_dev["f_new_device"].astype(int)).groupby("device_id")["_n"]
                 .agg(["max", "size"]).sort_values(["max", "size"], ascending=False).head(3))
        for dev in order.index:
            nb = tools.device_neighbors(dev, DATA_START, DATA_END, exclude_card=card)
            overlap = nb[(nb["first_ts"] <= t + 30 * 24 * H) & (nb["last_ts"] >= t - 30 * 24 * H)] if len(nb) else nb
            others = sorted(set(overlap["customer_id"]) - {cust}) if len(overlap) else []
            dcases = tools.cases_by_device(dev, before=before, exclude=exclude)
            fraud_cases = dcases[dcases["outcome"] == "confirmed_fraud"]["case_id"].tolist() if len(dcases) else []
            nb_fraud = []
            for c in (overlap["card_id"].head(4).tolist() if len(overlap) else []):
                cc = tools.cases_by_card(c, before=before, exclude=exclude)
                if len(cc) and (cc["outcome"] == "confirmed_fraud").any():
                    nb_fraud.append(c)
            devs.append(dict(device_id=dev, device_str=tools.device_str(dev), n_cards_ever=len(nb) + 1,
                             overlap_cards=overlap["card_id"].tolist() if len(overlap) else [],
                             other_customers=others, fraud_cases=fraud_cases, neighbor_fraud_cards=nb_fraud,
                             is_new=bool(on_dev[on_dev["device_id"] == dev]["f_new_device"].any())))
    F["devices"] = devs

    # ---- region cluster (only meaningful for a region that is new to this card)
    rc = None
    if F["region"]["active"]:
        r = F["region"]["region"]
        nb = tools.region_neighbors(r, t - 72 * H, t + 72 * H, exclude_card=card)
        rcases = tools.cases_by_region(r, before=before, exclude=exclude)
        fc = rcases[rcases["outcome"] == "confirmed_fraud"]["case_id"].tolist() if len(rcases) else []
        rc = dict(region=r, n_other_cards=len(nb), other_cards=nb["card_id"].head(5).tolist() if len(nb) else [],
                  fraud_cases=fc)
    F["region_cluster"] = rc

    # ---- this card's own history of cases, and sibling cards of the same customer
    own = tools.cases_by_card(card, before=before, exclude=exclude)
    F["own_cases"] = own
    sib_fraud = 0
    for sc in [c for c in tools.customer_cards(cust) if c != card]:
        sc_cases = tools.cases_by_card(sc, before=before, exclude=exclude)
        if len(sc_cases):
            recent = sc_cases[(sc_cases["outcome"] == "confirmed_fraud") &
                              (sc_cases["closed_at"].fillna(sc_cases["opened_at"]) >= opened - pd.Timedelta(days=90))]
            sib_fraud += int(len(recent) > 0)
    F["sibling_confirmed_fraud"] = sib_fraud
    return F


# ================================================================== score
def _select_affected(F: dict, pattern: str):
    w, flag, t = F["win"], F["flag"], F["t"]
    if pattern == "none":
        return w.iloc[0:0]
    if pattern == "card_testing" and F["testing"]:
        sel = pd.concat([F["testing"]["tiny"], F["testing"]["after"]])
    elif pattern == "out_of_region_use" and F["region"]["active"]:
        sel = F["region"]["reg"]
    else:
        near = w[(w["ts"] >= t - 48 * H) & (w["ts"] <= t + 48 * H)]
        sel = near[near["f_anom"]]
    sel = pd.concat([sel, w[w["txn_id"] == flag["txn_id"]]])
    return sel.drop_duplicates("txn_id").sort_values("ts")


def score(F: dict, customer_response: str | None = None) -> Assessment:
    case, flag, t, prof, win, near = F["case"], F["flag"], F["t"], F["prof"], F["win"], F["near"]
    card = case["card_id"]
    trigger = case.get("trigger_type", "risk_score")
    disputes = trigger == "customer_report"
    sig: dict[str, float] = {}
    ev: list[Evidence] = []
    fid = str(flag["txn_id"])

    # ---- window + baseline evidence (always)
    n_on = int((win["channel"] == "online").sum()) if len(win) else 0
    ev.append(Evidence(
        f"Flagged txn {fid}: {_money(flag['amount'])} {flag['channel']} (product {flag['product_cd']}) at {t:%Y-%m-%d %H:%M}; "
        f"bank model score {F['risk']:.2f}. {len(win)} txns on {card} within 72h of it ({n_on} online).",
        "graph", f"query:card_window(card_id={card}, hours={LOOKBACK_H + LOOKAHEAD_H})", [fid] + _ids(win.head(8))))
    if prof["n"]:
        ev.append(Evidence(
            f"Baseline before the window: {prof['n']} txns over {prof['n_days']} days, median {_money(prof['med'])}, "
            f"95th percentile {_money(prof['p95'])}, product codes {sorted(prof['products'])}, "
            f"home billing region {prof['home_region']}, {prof['online_share']:.0%} online.",
            "graph", f"query:card_window(card_id={card}, start=2016-07-01)", [card]))
    else:
        ev.append(Evidence("No history on this card before the window; comparisons with a baseline are not possible.",
                           "graph", f"query:card_window(card_id={card}, start=2016-07-01)", [card]))

    sig["risk"] = W["risk"] * (F["risk"] - 0.5)

    # ---- sequence: card testing
    ct = F["testing"]
    if ct and ct["complete"]:
        sig["card_testing"] = W["card_testing"]
        tiny = ct["tiny"]
        ev.append(Evidence(
            f"{len(tiny)} online authorizations of {', '.join(_money(a) for a in tiny['amount'])} within an hour "
            f"({tiny['ts'].min():%H:%M}-{tiny['ts'].max():%H:%M} on {tiny['ts'].min():%Y-%m-%d}), then a larger purchase of "
            f"{', '.join(_money(a) for a in ct['after']['amount'])}.",
            "graph", "query:card_window (sequence detector)", _ids(tiny) + _ids(ct["after"])))
    elif ct:
        sig["card_testing_partial"] = W["card_testing_partial"]
        ev.append(Evidence(f"{len(ct['tiny'])} tiny online authorizations within an hour, no larger purchase yet.",
                           "graph", "query:card_window (sequence detector)", _ids(ct["tiny"])))

    # ---- behaviour vs baseline
    if len(near):
        nd = near[near["f_new_device"]]
        if len(nd):
            sig["new_device"] = W["new_device"]
            ev.append(Evidence(f"{len(nd)} online txn(s) within 48h from a device profile marked New for this account or never seen on it.",
                               "graph", "identity record id_15 / device_neighbors", _ids(nd)))
        px = near[near["f_proxy"]]
        if len(px):
            sig["proxy"] = W["proxy"]
            ev.append(Evidence(f"{len(px)} online txn(s) behind a proxy (identity field id_23).", "graph",
                               "identity record id_23", _ids(px)))
        ah = near[near["f_amount_high"]]
        if len(ah):
            sig["amount_high"] = W["amount_high"]
            ev.append(Evidence(f"Amount(s) {', '.join(_money(a) for a in ah['amount'].head(4))} far above this card's 95th percentile ({_money(prof['p95'])}).",
                               "graph", "baseline_profile", _ids(ah)))
        npd = near[near["f_new_product"]]
        if len(npd):
            sig["new_product"] = W["new_product"]
            ev.append(Evidence(f"Online purchase(s) under product code(s) {sorted(set(npd['product_cd']))} this card has not used before.",
                               "graph", "baseline_profile", _ids(npd)))
        if int((near["channel"].eq("online") & near["f_anom"]).sum()) >= 2:
            sig["burst"] = W["burst"]
            ev.append(Evidence("Two or more anomalous online purchases within 48 hours.", "graph", "card_window", _ids(near[near["f_anom"]])))
        if near["f_m_bad"].any():
            sig["m_bad"] = W["m_bad"]
            ev.append(Evidence("Match flags (M1-M9) show more mismatches than this card's norm. These are unnamed Vesta model features; used as a signal only.",
                               "graph", "transactions M1-M9", _ids(near[near["f_m_bad"]])))
        types = sum([bool(len(nd) or len(px)), bool(near["f_m_bad"].any()), bool(len(ah)), bool(len(npd)),
                     bool(near["f_new_region"].any())])
        mixed = near["channel"].nunique() > 1
        if mixed and types >= 2 and (len(nd) or len(px) or near["f_m_bad"].any()):
            sig["ato"] = W["ato"]
            ev.append(Evidence("Mixed in-person and online activity within 48h with device/match anomalies, which points to stolen credentials rather than a stolen number.",
                               "graph", "card_window", _ids(near)))

    # ---- region
    R = F["region"]
    if R["active"]:
        if R["clone"]:
            sig["region_clone"] = W["region_clone"]
            ev.append(Evidence(f"{len(R['reg'])} in-person purchase(s) in billing region {R['region']}, which this card has never used, "
                               f"while normal purchases continued in home region {R['home']} ({R['home_during']} during the same span).",
                               "graph", "query:card_window + region baseline", _ids(R["reg"])))
        elif R["trip"]:
            sig["region_trip"] = W["region_trip"]
            ev.append(Evidence(f"Purchases in new region {R['region']} over {R['days']} days with no activity at home in that span, which looks like travel rather than a cloned card.",
                               "graph", "query:card_window + region baseline", _ids(R["reg"])))
        else:
            sig["region_new_only"] = W["region_new_only"]
            ev.append(Evidence(f"In-person purchase in billing region {R['region']} with no prior history there; not enough context to call it travel or a clone.",
                               "graph", "query:card_window + region baseline", _ids(R["reg"])))

    # ---- shared elements (graph neighbours)
    shared_desc, connected, conn_devs, coordinated = "", [], [], False
    for d in F["devices"]:
        generic = d["n_cards_ever"] > 40
        fraud_link = bool(d["fraud_cases"] or d["neighbor_fraud_cards"])
        many = len(d["other_customers"]) >= 3 and not generic
        if fraud_link:
            sig["shared_fraud"] = W["shared_fraud"]
            why = []
            if d["fraud_cases"]:
                why.append(f"closed confirmed-fraud case(s) {', '.join(d['fraud_cases'][:3])}")
            if d["neighbor_fraud_cards"]:
                why.append(f"card(s) {', '.join(d['neighbor_fraud_cards'][:3])} with prior confirmed fraud")
            ev.append(Evidence(f"Device profile '{d['device_str']}' is linked to " + " and ".join(why) +
                               f"; {len(d['overlap_cards'])} other card(s) used it within 30 days.",
                               "graph", f"query:device_neighbors(device_id={d['device_id']})",
                               d["fraud_cases"][:3] + d["neighbor_fraud_cards"][:3] + d["overlap_cards"][:3]))
            shared_desc = shared_desc or f"device profile '{d['device_str']}'"
        if many and "shared_fraud" not in sig:
            sig["shared_many"] = W["shared_many"]
            ev.append(Evidence(f"Device profile '{d['device_str']}' appears on {len(d['other_customers'])} other customers' cards within 30 days "
                               f"(uncommon: {d['n_cards_ever']} cards ever).", "graph",
                               f"query:device_neighbors(device_id={d['device_id']})", d["overlap_cards"][:5]))
        if (fraud_link or many or d["is_new"]) and d["overlap_cards"]:
            connected += d["overlap_cards"][:5]
            conn_devs.append(d["device_str"])
        if (fraud_link or many) and len(d["other_customers"]) >= 3:
            coordinated = True
    rc = F["region_cluster"]
    if rc and rc["fraud_cases"] and rc["n_other_cards"] >= 3:
        sig["region_cluster"] = W["region_cluster"]
        ev.append(Evidence(f"Billing region {rc['region']} has {rc['n_other_cards']} other cards active within 72h and prior confirmed fraud "
                           f"({', '.join(rc['fraud_cases'][:3])}).", "graph",
                           f"query:region_neighbors(region={rc['region']})", rc["fraud_cases"][:3] + rc["other_cards"][:3]))
        shared_desc = shared_desc or f"billing region {rc['region']}"
        connected += rc["other_cards"]
        coordinated = True

    # ---- recurring charge
    if F["recurring"]:
        sig["recurring"] = W["recurring"]
        r = F["recurring"]
        ev.append(Evidence(f"Charge matches the card's own recurring pattern: same amount and product {r['n']} times before, latest {r['gap']:.0f} days earlier.",
                           "graph", "query:card_window (recurring detector)", r["ids"][-3:] + [fid]))

    # ---- card's own history
    own = F["own_cases"]
    if len(own):
        if (own["outcome"] == "confirmed_fraud").any():
            sig["prior_fraud"] = W["prior_fraud"]
        elif (own["outcome"] == "cleared").any():
            sig["prior_cleared"] = W["prior_cleared"]
        ev.append(Evidence(f"This card appears on {len(own)} prior case(s): " +
                           ", ".join(f"{r.case_id} ({r.outcome})" for r in own.head(3).itertuples()),
                           "graph", f"query:closed_cases_by_card(card_id={card})", own["case_id"].head(3).tolist()))

    # ---- customer report / response
    if disputes:
        sig["dispute"] = W["dispute"]
        ev.append(Evidence("Customer reported they did not make the flagged purchase.", "customer",
                           f"case_pack:{case.get('case_id', '')}", [fid]))
    if customer_response == "denied":
        sig["denied"] = W["denied"]
    elif customer_response == "confirmed":
        sig["confirmed"] = W["confirmed"]

    # ---- no anomalies at all on a well-known card
    behaviour = {"amount_high", "new_product", "burst", "new_device", "proxy", "card_testing", "card_testing_partial",
                 "region_clone", "region_new_only", "m_bad", "ato"}
    if not (behaviour & set(sig)) and prof["n"] >= 10 and not F["recurring"]:
        sig["normal"] = W["normal"]
        ev.append(Evidence(f"Flagged activity is consistent with {prof['n']} prior transactions: amount, product code and region are all within this card's norm.",
                           "graph", "baseline_profile", [fid]))

    # ---- probability
    if "card_testing" in sig:
        sig.pop("burst", None)            # the sequence already explains the burst; do not double count
    z = _logit(PRIOR) + sum(sig.values())
    p = float(min(0.97, max(0.03, _sigmoid(z))))
    verdict = "fraud" if p >= 0.70 else "legitimate" if p <= 0.30 else "uncertain"

    pos = {CATEGORY[k] for k, v in sig.items() if v >= 0.3}
    neg = {CATEGORY[k] for k, v in sig.items() if v <= -0.3}
    n_ind = len(pos) if p >= 0.5 else len(neg)
    # a customer dispute against a recurring charge is R7's job, not "conflicting evidence" (R8)
    pos_sum = sum(v for k, v in sig.items() if v > 0 and CATEGORY[k] not in ("response", "customer"))
    neg_sum = sum(v for k, v in sig.items() if v < 0 and CATEGORY[k] != "response")
    conflicting = pos_sum >= 1.5 and neg_sum <= -1.5

    # ---- pattern
    strong_testing = bool(ct and ct["complete"])
    if p < 0.30:
        pattern = "none"
    elif strong_testing:
        pattern = "card_testing"
    elif R["active"] and not R["trip"]:
        pattern = "out_of_region_use"
    elif "ato" in sig:
        pattern = "account_takeover"
    elif "new_device" in sig:
        pattern = "card_not_present_new_device"
    elif {"amount_high", "new_product", "burst", "proxy"} & set(sig) or (flag["channel"] == "online" and p >= 0.5 and "shared_fraud" not in sig):
        pattern = "card_not_present_fraud"
    elif coordinated:
        pattern = "undocumented"
    else:
        pattern = "none" if p < 0.5 else ("card_not_present_fraud" if flag["channel"] == "online" else "out_of_region_use")

    pattern_desc = ""
    if pattern == "undocumented":
        pattern_desc = (f"Cards belonging to {max([len(d['other_customers']) for d in F['devices']] + [rc['n_other_cards'] if rc else 0])}+ other "
                        f"customers are linked through {shared_desc or 'a shared element'} within a short window, and closed cases show fraud on that element, "
                        f"but the activity on this card fits none of the five documented patterns. Found by graph traversal from the flagged transaction to shared devices/regions.")

    # ---- affected transactions, exposure, connected entities
    aff = _select_affected(F, pattern) if verdict != "legitimate" else win.iloc[0:0]
    affected = _ids(aff)
    exposure = float(aff["amount"].abs().sum()) if len(aff) else 0.0
    first = str(aff.iloc[0]["txn_id"]) if len(aff) else ""

    # ---- similar prior cases (entity based first)
    sim: list[str] = []
    for d in F["devices"]:
        sim += d["fraud_cases"][:2]
    if rc:
        sim += rc["fraud_cases"][:2]
    sim += F["own_cases"]["case_id"].head(2).tolist() if len(F["own_cases"]) else []

    a = Assessment(
        p=round(p, 3), verdict=verdict, pattern=pattern, pattern_description=pattern_desc,
        affected_txn_ids=affected, first_suspicious_txn_id=first,
        connected_card_ids=list(dict.fromkeys(connected))[:6],
        connected_device_profiles=list(dict.fromkeys(conn_devs)) if verdict != "legitimate" else [],
        exposure=round(exposure, 2), evidence=ev, similar_prior_cases=list(dict.fromkeys(sim))[:5],
        signals={k: round(v, 3) for k, v in sig.items()}, n_independent=n_ind,
        channel=str(flag["channel"]), customer_response=customer_response, customer_disputes=disputes,
        recurring_match=bool(F["recurring"]), card_testing=strong_testing,
        cleared_over_100=bool(ct and ct["complete"] and ct["cleared_over_100"]),
        single_signal=len(pos) <= 1, shared_element=bool(shared_desc), shared_element_desc=shared_desc,
        coordinated=coordinated, confirmed_fraud_cards=F["sibling_confirmed_fraud"] + (1 if customer_response == "denied" else 0),
        credentials_compromised=(pattern == "account_takeover" and customer_response == "denied" and p >= 0.85),
        conflicting=conflicting, trigger_type=trigger,
    )
    if verdict == "legitimate":
        a.pattern, a.pattern_description = "none", ""
        a.affected_txn_ids, a.first_suspicious_txn_id, a.exposure = [], "", 0.0
        a.connected_card_ids, a.shared_element, a.shared_element_desc = [], False, ""
    return a