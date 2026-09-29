"""Synthetic policy-decision cases for a policy-following service agent.

A case is a policy (numbered rules; the first rule that applies decides), a case
description, and a typed Choice question. The rule engine computes the gold decision
and the truth value of every predicate the policy mentions, so intermediate
judgements ("was the request within 30 days?") are program-verifiable.

If a fact is withheld from the case and the decision depends on it, the gold decision
is ASK (the agent should ask for the missing information). Dependence is decided by
enumerating the withheld fact over values that straddle every threshold.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import random
from typing import Any, Callable

ASK = "ask"
ASK_TEXT = "Ask the customer for the missing information"

TRAIN_DOMAINS = ("returns", "subscription", "expense")
HELDOUT_DOMAINS = ("travel", "insurance")


@dataclasses.dataclass(frozen=True)
class Predicate:
    pid: str
    question: str                  # yes/no question about the case
    fn: Callable[[dict], bool]
    needs: tuple[str, ...]         # facts it reads
    kind: str = "lookup"           # "lookup" (read a fact) or "arith" (dates/thresholds)


@dataclasses.dataclass(frozen=True)
class Rule:
    rid: str
    text: str
    conds: tuple[tuple[Predicate, bool], ...]   # conjunction of (predicate, required value)
    outcome: str


@dataclasses.dataclass
class Case:
    item_id: str
    domain: str
    title: str
    rules: list[Rule]
    default_text: str
    default_outcome: str
    facts: dict[str, Any]
    fact_space: dict[str, list[Any]]
    hidden: str | None
    options: dict[str, str]                     # outcome key -> option text
    render_facts: Callable[[dict, str | None], list[str]]

    # ---- rule engine ----
    def _evaluate(self, facts: dict) -> tuple[str, list[tuple[Predicate, bool]], int]:
        """Return (outcome, predicates evaluated in order, 1-based index of firing rule)."""
        path: list[tuple[Predicate, bool]] = []
        for i, rule in enumerate(self.rules, start=1):
            fired = True
            for pred, want in rule.conds:
                val = bool(pred.fn(facts))
                path.append((pred, val))
                if val != want:
                    fired = False
                    break
            if fired:
                return rule.outcome, path, i
        return self.default_outcome, path, len(self.rules) + 1

    def gold(self) -> str:
        if self.hidden is None:
            return self._evaluate(self.facts)[0]
        outcomes = {self._evaluate({**self.facts, self.hidden: v})[0] for v in self.fact_space[self.hidden]}
        return outcomes.pop() if len(outcomes) == 1 else ASK

    def path(self) -> list[dict]:
        """Distinct predicates on the decision path (true fact values), in first-checked order.

        Truth is None for predicates that read the hidden fact.
        """
        _, path, _ = self._evaluate(self.facts)
        seen, out = set(), []
        for p, v in path:
            if p.pid not in seen:
                seen.add(p.pid)
                out.append(self._node(p, None if self.hidden in p.needs else v))
        return out

    def depth(self) -> int:
        """Number of distinct predicate checks needed to reach the decision."""
        return len(self.path())

    def firing_rule(self) -> int:
        return self._evaluate(self.facts)[2]

    def all_predicates(self) -> list[dict]:
        seen, out = set(), []
        for rule in self.rules:
            for pred, _ in rule.conds:
                if pred.pid in seen:
                    continue
                seen.add(pred.pid)
                val = None if self.hidden in pred.needs else bool(pred.fn(self.facts))
                out.append(self._node(pred, val))
        return out

    @staticmethod
    def _node(p: Predicate, v: bool | None) -> dict:
        return {"pid": p.pid, "question": p.question, "truth": v, "kind": p.kind}

    # ---- rendering ----
    def policy_text(self) -> str:
        lines = [f"Policy ({self.title}). Rules are checked in order; the first rule that applies decides the outcome."]
        lines += [f"{i}. {r.text}" for i, r in enumerate(self.rules, start=1)]
        lines.append(f"Otherwise: {self.default_text}")
        return "\n".join(lines)

    def case_text(self) -> str:
        return "Case:\n" + "\n".join(f"- {s}" for s in self.render_facts(self.facts, self.hidden))

    def prompt(self, label_order: list[str]) -> str:
        """Policy + case + lettered options. `label_order` lists outcome keys in option order."""
        opts = "\n".join(f"{chr(65 + i)}) {self.options[k]}" for i, k in enumerate(label_order))
        return (f"You are a customer-service agent. Apply the policy to the case.\n\n{self.policy_text()}\n\n"
                f"{self.case_text()}\n\nQuestion: What should the agent do?\nOptions:\n{opts}")

    def to_record(self, label_order: list[str]) -> dict:
        return {
            "item_id": self.item_id, "domain": self.domain, "hidden": self.hidden,
            "gold": self.gold(), "label_order": label_order,
            "gold_label": chr(65 + label_order.index(self.gold())),
            "depth": self.depth(), "firing_rule": self.firing_rule(), "n_rules": len(self.rules),
            "path": self.path(), "predicates": self.all_predicates(),
            "prompt": self.prompt(label_order), "options": dict(self.options),
        }


# ---------------------------------------------------------------- helpers
def _date(rng: random.Random) -> dt.date:
    return dt.date(2026, 1, 1) + dt.timedelta(days=rng.randrange(0, 240))


def _near(rng: random.Random, thresholds: list[int], lo: int, hi: int) -> int:
    """Sample an integer, often close to a threshold (so decisions are not trivial)."""
    if thresholds and rng.random() < 0.6:
        t = rng.choice(thresholds)
        return max(lo, min(hi, t + rng.choice([-3, -2, -1, 0, 1, 2, 3])))
    return rng.randint(lo, hi)


def _days(facts: dict, a: str, b: str) -> int:
    return (facts[b] - facts[a]).days


def _yes(flag: bool, yes: str, no: str) -> str:
    return yes if flag else no


# ---------------------------------------------------------------- domains
def _returns(rng: random.Random) -> dict:
    d_u = rng.choice([14, 30, 45]); d_o = rng.choice([7, 10]) if d_u == 14 else rng.choice([7, 14])
    d_def = rng.choice([60, 90]); p_high = rng.choice([150, 200, 300, 500])
    ext = rng.choice([0, 15])

    def win(limit: int) -> Callable[[dict], bool]:
        return lambda f: _days(f, "purchase_date", "request_date") <= limit + (ext if f["member"] == "gold" else 0)
    ext_txt = f" ({ext} extra days for gold members)" if ext else ""
    P = {
        "defective": Predicate("defective", "Does the customer report a defect?", lambda f: f["defective"], ("defective",)),
        "win_def": Predicate("win_def", f"Was the return requested within {d_def} days of purchase?",
                             lambda f: _days(f, "purchase_date", "request_date") <= d_def, ("purchase_date",), "arith"),
        "clearance": Predicate("clearance", "Is the item a clearance item?", lambda f: f["category"] == "clearance", ("category",)),
        "high": Predicate("high", f"Is the price above ${p_high}?", lambda f: f["price"] > p_high, ("price",), "arith"),
        "receipt": Predicate("receipt", "Does the customer have the receipt?", lambda f: f["receipt"], ("receipt",)),
        "opened": Predicate("opened", "Has the item been opened?", lambda f: f["opened"], ("opened",)),
        "win_u": Predicate("win_u", f"Is the request within the unopened-item window of {d_u} days{ext_txt}?",
                           win(d_u), ("purchase_date", "member"), "arith"),
        "win_o": Predicate("win_o", f"Is the request within the opened-item window of {d_o} days{ext_txt}?",
                           win(d_o), ("purchase_date", "member"), "arith"),
    }
    rules = []
    if rng.random() < 0.6:
        rules.append(Rule("R_def", f"Defective items can be returned for a full refund within {d_def} days of purchase.",
                          ((P["defective"], True), (P["win_def"], True)), "refund"))
    if rng.random() < 0.8:
        rules.append(Rule("R_clear", "Clearance items are final sale and cannot be returned.", ((P["clearance"], True),), "deny"))
    if rng.random() < 0.6:
        rules.append(Rule("R_high", f"Returns of items priced above ${p_high} need a manager's approval.", ((P["high"], True),), "escalate"))
    if rng.random() < 0.6:
        rules.append(Rule("R_rcpt", "Without a receipt, only an exchange or store credit is possible.", ((P["receipt"], False),), "credit"))
    rules.append(Rule("R_unop", f"Unopened items can be refunded within {d_u} days of purchase{ext_txt}.",
                      ((P["opened"], False), (P["win_u"], True)), "refund"))
    rules.append(Rule("R_op", f"Opened items can be exchanged or credited within {d_o} days of purchase{ext_txt}.",
                      ((P["opened"], True), (P["win_o"], True)), "credit"))

    purchase = _date(rng)
    days = _near(rng, [d_o, d_u, d_u + ext, d_def], 0, 150)
    facts = {"purchase_date": purchase, "request_date": purchase + dt.timedelta(days=days),
             "opened": rng.random() < 0.5, "category": rng.choice(["regular", "regular", "clearance", "electronics"]),
             "price": _near(rng, [p_high], 5, 900), "receipt": rng.random() < 0.8,
             "member": rng.choice(["standard", "standard", "gold"]), "defective": rng.random() < 0.2}
    day_space = sorted({0, 3, d_o, d_o + 1, d_u, d_u + 1, d_u + ext, d_u + ext + 1, d_def, d_def + 1, 300})
    space = {"opened": [True, False], "receipt": [True, False], "defective": [True, False],
             "category": ["regular", "clearance", "electronics"], "price": [5, p_high, p_high + 1, 5000],
             "member": ["standard", "gold"],
             "purchase_date": [facts["request_date"] - dt.timedelta(days=d) for d in day_space]}

    def render(f: dict, hidden: str | None) -> list[str]:
        def show(k: str, text: str) -> str:
            return f"{text.split(':')[0]}: unknown (not stated)" if k == hidden else text
        return [show("purchase_date", f"Purchase date: {f['purchase_date'].isoformat()}"),
                f"Return requested on: {f['request_date'].isoformat()}",
                show("member", f"Membership: {f['member']}"),
                show("category", f"Item category: {f['category']}"),
                show("price", f"Price: ${f['price']}"),
                show("opened", f"Opened: {_yes(f['opened'], 'yes', 'no')}"),
                show("receipt", f"Receipt: {_yes(f['receipt'], 'has receipt', 'no receipt')}"),
                show("defective", f"Defect reported: {_yes(f['defective'], 'yes', 'no')}")]
    return dict(title="retail returns", rules=rules, default_text="deny the return.", default_outcome="deny",
                facts=facts, fact_space=space, render=render,
                options={"refund": "Give a full refund", "credit": "Offer an exchange or store credit only",
                         "deny": "Deny the return", "escalate": "Escalate to a manager for approval"})


def _subscription(rng: random.Random) -> dict:
    cool = 14; ann = rng.choice([30, 45]); cap = rng.choice([10, 20, 30]); notice = rng.choice([7, 10])
    p_esc = rng.choice([300, 500])
    P = {
        "promo": Predicate("promo", "Was the plan bought under a promotional offer?", lambda f: f["promo"], ("promo",)),
        "eu": Predicate("eu", "Is the customer in the EU?", lambda f: f["region"] == "EU", ("region",)),
        "win_cool": Predicate("win_cool", f"Is the request within {cool} days of the renewal?",
                              lambda f: _days(f, "renewal_date", "request_date") <= cool, ("renewal_date",), "arith"),
        "reminder": Predicate("reminder", "Was a renewal reminder sent before the renewal?", lambda f: f["reminder"], ("reminder",)),
        "win_notice": Predicate("win_notice", f"Is the request within {notice} days of the renewal?",
                                lambda f: _days(f, "renewal_date", "request_date") <= notice, ("renewal_date",), "arith"),
        "annual": Predicate("annual", "Is it an annual plan?", lambda f: f["plan"] == "annual", ("plan",)),
        "pricey": Predicate("pricey", f"Is the plan price above ${p_esc}?", lambda f: f["price"] > p_esc, ("price",), "arith"),
        "win_ann": Predicate("win_ann", f"Is the request within {ann} days of the renewal?",
                             lambda f: _days(f, "renewal_date", "request_date") <= ann, ("renewal_date",), "arith"),
        "low_use": Predicate("low_use", f"Has the customer used less than {cap}% of the service this period?",
                             lambda f: f["usage"] < cap, ("usage",), "arith"),
    }
    rules = []
    if rng.random() < 0.7:
        rules.append(Rule("S_promo", "Promotional plans are non-refundable.", ((P["promo"], True),), "deny"))
    if rng.random() < 0.7:
        rules.append(Rule("S_eu", f"EU customers get a full refund if they cancel within {cool} days of a renewal.",
                          ((P["eu"], True), (P["win_cool"], True)), "refund"))
    if rng.random() < 0.6:
        rules.append(Rule("S_notice", f"If no renewal reminder was sent, cancelling within {notice} days of the renewal gets a full refund.",
                          ((P["reminder"], False), (P["win_notice"], True)), "refund"))
    if rng.random() < 0.5:
        rules.append(Rule("S_esc", f"Refund requests on annual plans priced above ${p_esc} go to a supervisor.",
                          ((P["annual"], True), (P["pricey"], True)), "escalate"))
    rules.append(Rule("S_pro", f"Annual plans cancelled within {ann} days of renewal with less than {cap}% usage get a prorated refund.",
                      ((P["annual"], True), (P["win_ann"], True), (P["low_use"], True)), "prorate"))

    renewal = _date(rng); plan = rng.choice(["monthly", "annual"])
    days = _near(rng, [notice, cool, ann], 0, 90)
    facts = {"renewal_date": renewal, "request_date": renewal + dt.timedelta(days=days),
             "region": rng.choice(["EU", "US"]), "plan": plan, "promo": rng.random() < 0.2,
             "reminder": rng.random() < 0.75, "usage": _near(rng, [cap], 0, 100),
             "price": _near(rng, [p_esc], 50, 900) if plan == "annual" else rng.randint(5, 60)}
    day_space = sorted({0, notice, notice + 1, cool, cool + 1, ann, ann + 1, 200})
    space = {"region": ["EU", "US"], "plan": ["monthly", "annual"], "promo": [True, False], "reminder": [True, False],
             "usage": [0, cap - 1, cap, 100], "price": [5, p_esc, p_esc + 1, 5000],
             "renewal_date": [facts["request_date"] - dt.timedelta(days=d) for d in day_space]}

    def render(f: dict, hidden: str | None) -> list[str]:
        def show(k: str, text: str) -> str:
            return f"{text.split(':')[0]}: unknown (not stated)" if k == hidden else text
        return [show("plan", f"Plan: {f['plan']}"), show("price", f"Plan price: ${f['price']}"),
                show("renewal_date", f"Last renewal date: {f['renewal_date'].isoformat()}"),
                f"Cancellation requested on: {f['request_date'].isoformat()}",
                show("region", f"Customer region: {f['region']}"),
                show("promo", f"Promotional offer: {_yes(f['promo'], 'yes', 'no')}"),
                show("reminder", f"Renewal reminder sent: {_yes(f['reminder'], 'yes', 'no')}"),
                show("usage", f"Usage this period: {f['usage']}%")]
    return dict(title="subscription cancellations", rules=rules,
                default_text="cancel at the end of the current period with no refund.", default_outcome="deny",
                facts=facts, fact_space=space, render=render,
                options={"refund": "Give a full refund", "prorate": "Give a prorated refund",
                         "deny": "Cancel at period end with no refund", "escalate": "Escalate to a supervisor"})


def _expense(rng: random.Random) -> dict:
    cap = rng.choice([50, 75, 100]); rmin = rng.choice([25, 50]); late = rng.choice([30, 60, 90])
    pre = rng.choice([500, 1000])
    P = {
        "late": Predicate("late", f"Was the claim submitted more than {late} days after the expense?",
                          lambda f: _days(f, "expense_date", "submit_date") > late, ("expense_date",), "arith"),
        "ent": Predicate("ent", "Is it an entertainment expense?", lambda f: f["category"] == "entertainment", ("category",)),
        "client": Predicate("client", "Was the expense for a client meeting?", lambda f: f["client"], ("client",)),
        "receipt": Predicate("receipt", "Is there a receipt?", lambda f: f["receipt"], ("receipt",)),
        "above_min": Predicate("above_min", f"Is the amount above ${rmin}?", lambda f: f["amount"] > rmin, ("amount",), "arith"),
        "big": Predicate("big", f"Is the amount above ${pre}?", lambda f: f["amount"] > pre, ("amount",), "arith"),
        "preapp": Predicate("preapp", "Was the expense approved in advance?", lambda f: f["preapproved"], ("preapproved",)),
        "meal": Predicate("meal", "Is it a meal expense?", lambda f: f["category"] == "meal", ("category",)),
        "over_cap": Predicate("over_cap", f"Is the amount above the ${cap} meal cap?", lambda f: f["amount"] > cap, ("amount",), "arith"),
    }
    rules = [Rule("E_late", f"Claims submitted more than {late} days after the expense are rejected.", ((P["late"], True),), "deny")]
    if rng.random() < 0.7:
        rules.append(Rule("E_ent", "Entertainment is reimbursed only for client meetings.",
                          ((P["ent"], True), (P["client"], False)), "deny"))
    if rng.random() < 0.7:
        rules.append(Rule("E_rcpt", f"Expenses above ${rmin} need a receipt.", ((P["receipt"], False), (P["above_min"], True)), "deny"))
    if rng.random() < 0.6:
        rules.append(Rule("E_big", f"Expenses above ${pre} without advance approval go to the finance manager.",
                          ((P["big"], True), (P["preapp"], False)), "escalate"))
    rules.append(Rule("E_cap", f"Meals are reimbursed up to ${cap}.", ((P["meal"], True), (P["over_cap"], True)), "partial"))

    exp = _date(rng); cat = rng.choice(["meal", "meal", "travel", "equipment", "entertainment"])
    days = _near(rng, [late], 0, 150)
    facts = {"expense_date": exp, "submit_date": exp + dt.timedelta(days=days), "category": cat,
             "client": rng.random() < 0.5, "receipt": rng.random() < 0.8, "preapproved": rng.random() < 0.3,
             "amount": _near(rng, [cap, rmin, pre], 5, 2500)}
    space = {"category": ["meal", "travel", "equipment", "entertainment"], "client": [True, False],
             "receipt": [True, False], "preapproved": [True, False],
             "amount": sorted({1, rmin, rmin + 1, cap, cap + 1, pre, pre + 1, 10000}),
             "expense_date": [facts["submit_date"] - dt.timedelta(days=d) for d in (0, late, late + 1, 400)]}

    def render(f: dict, hidden: str | None) -> list[str]:
        def show(k: str, text: str) -> str:
            return f"{text.split(':')[0]}: unknown (not stated)" if k == hidden else text
        return [show("category", f"Category: {f['category']}"), show("amount", f"Amount: ${f['amount']}"),
                show("expense_date", f"Expense date: {f['expense_date'].isoformat()}"),
                f"Claim submitted on: {f['submit_date'].isoformat()}",
                show("receipt", f"Receipt attached: {_yes(f['receipt'], 'yes', 'no')}"),
                show("client", f"Client meeting: {_yes(f['client'], 'yes', 'no')}"),
                show("preapproved", f"Approved in advance: {_yes(f['preapproved'], 'yes', 'no')}")]
    return dict(title="expense reimbursement", rules=rules, default_text="approve the full amount.", default_outcome="approve",
                facts=facts, fact_space=space, render=render,
                options={"approve": "Approve the full amount", "partial": "Approve up to the cap only",
                         "deny": "Reject the claim", "escalate": "Send to the finance manager"})


def _travel(rng: random.Random) -> dict:
    free_h = rng.choice([24, 48]); fee = rng.choice([50, 75, 100])
    P = {
        "group": Predicate("group", "Is the booking for more than 9 passengers?", lambda f: f["group"] > 9, ("group",), "arith"),
        "medical": Predicate("medical", "Did the passenger provide a medical certificate?", lambda f: f["medical"], ("medical",)),
        "basic": Predicate("basic", "Is it a basic fare?", lambda f: f["fare"] == "basic", ("fare",)),
        "flex": Predicate("flex", "Is it a flex fare?", lambda f: f["fare"] == "flex", ("fare",)),
        "elite": Predicate("elite", "Does the passenger have elite status?", lambda f: f["elite"], ("elite",)),
        "early": Predicate("early", f"Is the change requested more than {free_h} hours before departure?",
                           lambda f: f["hours"] > free_h, ("hours",), "arith"),
    }
    rules = []
    if rng.random() < 0.4:
        rules.append(Rule("T_group", "Changes to bookings of more than 9 passengers go to the group desk.", ((P["group"], True),), "escalate"))
    if rng.random() < 0.6:
        rules.append(Rule("T_med", "Passengers with a medical certificate can change for free.", ((P["medical"], True),), "free"))
    rules.append(Rule("T_basic", "Basic fares cannot be changed.", ((P["basic"], True),), "deny"))
    rules.append(Rule("T_flex", "Flex fares can be changed for free.", ((P["flex"], True),), "free"))
    if rng.random() < 0.5:
        rules.append(Rule("T_elite", "Elite members can change any non-basic fare for free.", ((P["elite"], True),), "free"))
    rules.append(Rule("T_fee", f"Standard fares can be changed for a ${fee} fee if requested more than {free_h} hours before departure.",
                      ((P["early"], True),), "fee"))
    facts = {"fare": rng.choice(["basic", "standard", "standard", "flex"]), "hours": _near(rng, [free_h], 1, 240),
             "medical": rng.random() < 0.15, "elite": rng.random() < 0.3,
             "group": rng.choice([1, 1, 2, 3, 4, 10, 12])}
    space = {"fare": ["basic", "standard", "flex"], "hours": [1, free_h, free_h + 1, 500], "medical": [True, False],
             "elite": [True, False], "group": [1, 9, 10]}

    def render(f: dict, hidden: str | None) -> list[str]:
        def show(k: str, text: str) -> str:
            return f"{text.split(':')[0]}: unknown (not stated)" if k == hidden else text
        return [show("fare", f"Fare type: {f['fare']}"), show("hours", f"Hours before departure: {f['hours']}"),
                show("group", f"Passengers on booking: {f['group']}"),
                show("medical", f"Medical certificate: {_yes(f['medical'], 'provided', 'none')}"),
                show("elite", f"Elite status: {_yes(f['elite'], 'yes', 'no')}")]
    return dict(title="flight changes", rules=rules, default_text="refuse the change.", default_outcome="deny",
                facts=facts, fact_space=space, render=render,
                options={"free": "Change the flight for free", "fee": "Change the flight for a fee",
                         "deny": "Refuse the change", "escalate": "Escalate to the group desk"})


def _insurance(rng: random.Random) -> dict:
    wait = rng.choice([30, 60]); report = rng.choice([30, 60]); esc = rng.choice([5000, 10000])
    P = {
        "waiting": Predicate("waiting", f"Did the incident happen within the first {wait} days of the policy?",
                             lambda f: _days(f, "policy_start", "incident_date") < wait, ("policy_start",), "arith"),
        "late": Predicate("late", f"Was the claim reported more than {report} days after the incident?",
                          lambda f: _days(f, "incident_date", "report_date") > report, ("report_date",), "arith"),
        "theft": Predicate("theft", "Is it a theft claim?", lambda f: f["type"] == "theft", ("type",)),
        "police": Predicate("police", "Is there a police report?", lambda f: f["police"], ("police",)),
        "water": Predicate("water", "Is it a water-damage claim?", lambda f: f["type"] == "water", ("type",)),
        "gradual": Predicate("gradual", "Was the damage caused by a gradual leak?", lambda f: f["gradual"], ("gradual",)),
        "big": Predicate("big", f"Is the claimed amount above ${esc}?", lambda f: f["amount"] > esc, ("amount",), "arith"),
    }
    rules = [Rule("I_wait", f"Incidents in the first {wait} days of a policy are not covered.", ((P["waiting"], True),), "deny")]
    if rng.random() < 0.7:
        rules.append(Rule("I_late", f"Claims reported more than {report} days after the incident are rejected.", ((P["late"], True),), "deny"))
    rules.append(Rule("I_theft", "Theft claims need a police report.", ((P["theft"], True), (P["police"], False)), "deny"))
    if rng.random() < 0.7:
        rules.append(Rule("I_grad", "Water damage from a gradual leak is excluded.", ((P["water"], True), (P["gradual"], True)), "deny"))
    rules.append(Rule("I_big", f"Claims above ${esc} go to a senior adjuster.", ((P["big"], True),), "escalate"))
    start = _date(rng); inc = start + dt.timedelta(days=_near(rng, [wait], 0, 300))
    facts = {"policy_start": start, "incident_date": inc, "report_date": inc + dt.timedelta(days=_near(rng, [report], 0, 120)),
             "type": rng.choice(["water", "fire", "theft"]), "police": rng.random() < 0.7, "gradual": rng.random() < 0.3,
             "amount": _near(rng, [esc], 100, 20000)}
    space = {"type": ["water", "fire", "theft"], "police": [True, False], "gradual": [True, False],
             "amount": [100, esc, esc + 1, 50000],
             "policy_start": [facts["incident_date"] - dt.timedelta(days=d) for d in (0, wait - 1, wait, 1000)],
             "report_date": [facts["incident_date"] + dt.timedelta(days=d) for d in (0, report, report + 1, 400)]}

    def render(f: dict, hidden: str | None) -> list[str]:
        def show(k: str, text: str) -> str:
            return f"{text.split(':')[0]}: unknown (not stated)" if k == hidden else text
        return [show("policy_start", f"Policy start date: {f['policy_start'].isoformat()}"),
                f"Incident date: {f['incident_date'].isoformat()}",
                show("report_date", f"Claim reported on: {f['report_date'].isoformat()}"),
                show("type", f"Claim type: {f['type']}"), show("amount", f"Claimed amount: ${f['amount']}"),
                show("police", f"Police report: {_yes(f['police'], 'yes', 'no')}"),
                show("gradual", f"Cause: {_yes(f['gradual'], 'gradual leak', 'sudden event')}")]
    return dict(title="home insurance claims", rules=rules, default_text="approve the claim minus the deductible.",
                default_outcome="approve", facts=facts, fact_space=space, render=render,
                options={"approve": "Approve the claim minus the deductible", "deny": "Deny the claim",
                         "escalate": "Send to a senior adjuster"})


DOMAINS: dict[str, Callable[[random.Random], dict]] = {
    "returns": _returns, "subscription": _subscription, "expense": _expense,
    "travel": _travel, "insurance": _insurance,
}


def make_case(domain: str, rng: random.Random, item_id: str, p_hidden: float = 0.25) -> Case:
    spec = DOMAINS[domain](rng)
    used = {fact for r in spec["rules"] for p, _ in r.conds for fact in p.needs}
    hidden = rng.choice(sorted(used)) if (used and rng.random() < p_hidden) else None
    options = dict(spec["options"]); options[ASK] = ASK_TEXT
    return Case(item_id=item_id, domain=domain, title=spec["title"], rules=spec["rules"],
                default_text=spec["default_text"], default_outcome=spec["default_outcome"],
                facts=spec["facts"], fact_space=spec["fact_space"], hidden=hidden, options=options,
                render_facts=spec["render"])


def generate(n: int, domains: tuple[str, ...] = TRAIN_DOMAINS, seed: int = 0,
             balance: float = 1.6, max_tries: int = 200) -> list[dict]:
    """Generate `n` case records, roughly balancing gold outcomes within each domain.

    `balance` caps each outcome at balance * (n_domain / n_outcomes) cases per domain.
    Option order is shuffled per case; records carry the prompt and verifiable path.
    """
    rng = random.Random(seed)
    per_dom = {d: n // len(domains) + (1 if i < n % len(domains) else 0) for i, d in enumerate(domains)}
    out: list[dict] = []
    for d in domains:
        counts: dict[str, int] = {}
        made, tries = 0, 0
        while made < per_dom[d] and tries < per_dom[d] * max_tries:
            tries += 1
            case = make_case(d, rng, item_id=f"{d}-{seed}-{made:05d}")
            g = case.gold()
            limit = max(1, int(balance * per_dom[d] / len(case.options)))
            if counts.get(g, 0) >= limit:
                continue
            counts[g] = counts.get(g, 0) + 1
            order = list(case.options); rng.shuffle(order)
            out.append(case.to_record(order))
            made += 1
    rng.shuffle(out)
    return out


def render_prompt(rec: dict, order: list[str]) -> str:
    """Re-render a record's prompt with a different option order (outcome keys in `order`)."""
    head = rec["prompt"].split("\nOptions:\n", 1)[0]
    opts = "\n".join(f"{chr(65 + i)}) {rec['options'][k]}" for i, k in enumerate(order))
    return f"{head}\nOptions:\n{opts}"
