"""Rule-decision cases with exact intermediate variables and counterfactual pairs (policygen2).

A case is a policy of ordered rules, a case description and a lettered question. Each rule is a conjunction of one
to three conditions on case facts; the first rule whose conditions all hold decides, otherwise a default applies.
Rules are sampled from a domain schema (most have two or three conditions), and the engine exposes every
intermediate variable:

  p:<pid>   a predicate on the facts, e.g. p:cost>250 ("is the estimated repair cost above $250?")
  a:<k>     rule k (1-based position) applies: each of its conditions holds
  decision  the outcome of the first applying rule, else the default

`run(case, overrides)` evaluates the policy with any of these variables set by hand and recomputes what lies
downstream, which gives program-exact counterfactual pairs:

  edit      one fact changed so that exactly one predicate flips (sensitive: the decision changes; masked: it does not)
  distract  a fact that no rule reads is changed (the decision cannot change)
  rule      the case is kept and the policy changed (one threshold moved, one condition negated, two adjacent rules
            swapped, or one rule's outcome replaced)
  third     base and source share the policy, and setting one variable in the base to its source value gives an
            outcome different from both the base and the source outcome (interchange-intervention targets)

Domains come as a training set and a held-out set with separate vocabularies. Renderers "list" and "bullets" are for
training and "prose" is held out. Nothing here calls a teacher: tree nodes are the program's own variables.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import random
from typing import Any


@dataclasses.dataclass(frozen=True)
class Fact:
    name: str
    kind: str                       # bool | cat | num | days
    label: str                      # case-line label
    pos: str = ""                   # bool: clause when true
    neg: str = ""                   # bool: clause when false
    q: str = ""                     # bool: question
    say_yes: str = ""               # bool: prose sentence when true
    say_no: str = ""
    p_true: float = 0.5
    noun: str = ""                  # cat/num: "the repair cost"; days: the anchor event, "the purchase"
    values: tuple = ()              # cat values
    lo: int = 0
    hi: int = 0
    cuts: tuple = ()                # num/days thresholds the policy may use
    money: bool = False


def B(name, label, pos, neg, q, say_yes, say_no, p=0.5) -> Fact:
    return Fact(name, "bool", label, pos=pos, neg=neg, q=q, say_yes=say_yes, say_no=say_no, p_true=p)


def C(name, label, noun, values) -> Fact:
    return Fact(name, "cat", label, noun=noun, values=tuple(values))


def N(name, label, noun, lo, hi, cuts, money=False) -> Fact:
    return Fact(name, "num", label, noun=noun, lo=lo, hi=hi, cuts=tuple(cuts), money=money)


def D(name, label, noun, cuts) -> Fact:
    return Fact(name, "days", label, noun=noun, lo=0, hi=3 * max(cuts), cuts=tuple(cuts))


@dataclasses.dataclass(frozen=True)
class Domain:
    name: str
    title: str
    role: str                       # "a warranty-service agent"
    ask: str                        # final question
    event: str                      # what the shared request date dates: "the claim"
    request: str                    # case-line label of the request date
    request_say: str                # prose sentence, "{d}" is the date
    facts: tuple
    outcomes: tuple                 # (key, option text, imperative)
    default: str

    def fact(self, name: str) -> Fact:
        return next(f for f in self.facts if f.name == name)

    def option(self, key: str) -> str:
        return next(o[1] for o in self.outcomes if o[0] == key)

    def imperative(self, key: str) -> str:
        return next(o[2] for o in self.outcomes if o[0] == key)


# ---------------------------------------------------------------- domains
TRAIN = [
    Domain("library", "library loan renewals", "a library circulation assistant", "What should the assistant do?",
           "the renewal request", "Renewal requested on", "The renewal was requested on {d}.",
           (C("member", "Borrower type", "the borrower type", ["student", "staff", "public"]),
            B("hold", "Hold by another reader", "another reader has placed a hold on the item",
              "no other reader has a hold on the item", "Has another reader placed a hold on the item?",
              "Another reader has placed a hold on the item.", "No other reader has a hold on the item.", .35),
            N("fines", "Unpaid fines", "the unpaid fine balance", 0, 60, (5, 10, 20), money=True),
            C("item", "Item type", "the item type", ["book", "DVD", "laptop", "journal"]),
            N("renewals", "Previous renewals", "the number of previous renewals", 0, 6, (1, 2, 3)),
            D("checkout", "Checkout date", "the checkout", (14, 21, 28)),
            B("damaged", "Damage noted", "the item has been reported damaged", "the item has no reported damage",
              "Has the item been reported damaged?", "The item has been reported damaged.",
              "The item has no reported damage.", .2),
            B("card", "Library card valid", "the library card is valid", "the library card has expired",
              "Is the library card valid?", "The library card is valid.", "The library card has expired.", .8)),
           (("renew", "Renew the loan", "renew the loan"), ("short", "Renew for one week only", "renew for one week only"),
            ("refuse", "Refuse the renewal", "refuse the renewal"),
            ("refer", "Refer to the head librarian", "refer the request to the head librarian")), "renew"),
    Domain("gym", "gym membership freezes", "a fitness-club front-desk assistant", "What should the assistant do?",
           "the freeze request", "Freeze requested on", "The freeze was requested on {d}.",
           (C("plan", "Plan", "the membership plan", ["monthly", "annual", "student"]),
            B("doctor", "Doctor's note", "the member has a doctor's note", "the member has no doctor's note",
              "Does the member have a doctor's note?", "The member has a doctor's note.",
              "The member has no doctor's note.", .3),
            N("tenure", "Months as a member", "the membership length in months", 1, 48, (3, 6, 12)),
            N("freezes", "Freezes used this year", "the number of freezes used this year", 0, 4, (0, 1, 2)),
            N("balance", "Outstanding balance", "the outstanding balance", 0, 200, (0, 20, 50), money=True),
            D("lastfreeze", "Last freeze ended on", "the end of the last freeze", (30, 60, 90)),
            B("corporate", "Employer-paid membership", "the membership is paid by an employer",
              "the membership is paid by the member", "Is the membership paid by an employer?",
              "The membership is paid by an employer.", "The membership is paid by the member.", .3),
            B("trainer", "Personal-training package", "the member has a personal-training package",
              "the member has no personal-training package", "Does the member have a personal-training package?",
              "The member has a personal-training package.", "The member has no personal-training package.", .3)),
           (("freeze", "Approve the freeze", "approve the freeze"),
            ("fee", "Approve the freeze for a fee", "approve the freeze for a fee"),
            ("refuse", "Refuse the freeze", "refuse the freeze"),
            ("manager", "Refer to the club manager", "refer the request to the club manager")), "refuse"),
    Domain("parking", "residential parking permits", "a city parking-office clerk", "What should the clerk do?",
           "the application", "Application received on", "The application was received on {d}.",
           (B("resident", "Proof of residence", "the applicant has proof of residence",
              "the applicant has no proof of residence", "Does the applicant have proof of residence?",
              "The applicant has proof of residence.", "The applicant has no proof of residence.", .75),
            C("zone", "Zone", "the parking zone", ["A", "B", "C"]),
            C("vehicle", "Vehicle type", "the vehicle type", ["car", "van", "motorcycle"]),
            N("tickets", "Unpaid parking tickets", "the number of unpaid parking tickets", 0, 6, (0, 1, 2)),
            N("emissions", "CO2 emissions (g/km)", "the vehicle's CO2 emissions in g/km", 60, 250, (100, 120, 150)),
            B("badge", "Disability badge", "the applicant holds a disability badge",
              "the applicant holds no disability badge", "Does the applicant hold a disability badge?",
              "The applicant holds a disability badge.", "The applicant holds no disability badge.", .15),
            N("months", "Months at the address", "the time at the address in months", 0, 60, (3, 6, 12)),
            D("expiry", "Previous permit expired on", "the expiry of the previous permit", (30, 90))),
           (("issue", "Issue the permit", "issue the permit"), ("temp", "Issue a temporary permit", "issue a temporary permit"),
            ("refuse", "Refuse the application", "refuse the application"),
            ("review", "Send for manual review", "send the application for manual review")), "refuse"),
    Domain("hotel", "hotel booking cancellations", "a hotel reservations agent", "What should the agent do?",
           "the cancellation", "Cancellation requested on", "The cancellation was requested on {d}.",
           (C("rate", "Rate type", "the rate type", ["flexible", "saver", "member"]),
            N("nights", "Nights booked", "the number of nights booked", 1, 14, (1, 3, 7)),
            N("hours", "Hours before check-in", "the time before check-in in hours", 1, 300, (24, 48, 72)),
            C("loyalty", "Loyalty level", "the loyalty level", ["none", "silver", "gold"]),
            N("rooms", "Rooms booked", "the number of rooms booked", 1, 20, (1, 4, 9)),
            B("event", "Event weekend", "the stay falls on an event weekend",
              "the stay does not fall on an event weekend", "Does the stay fall on an event weekend?",
              "The stay falls on an event weekend.", "The stay does not fall on an event weekend.", .25),
            N("deposit", "Deposit paid", "the deposit paid", 0, 900, (100, 300), money=True),
            D("booked", "Booking made on", "the booking", (7, 30))),
           (("full", "Give a full refund", "give a full refund"),
            ("partial", "Refund minus one night's charge", "refund the stay minus one night's charge"),
            ("keep", "Keep the full payment", "keep the full payment"),
            ("manager", "Refer to the reservations manager", "refer the request to the reservations manager")), "keep"),
    Domain("warranty", "product warranty claims", "a warranty-service agent", "What should the agent do?",
           "the claim", "Claim filed on", "The claim was filed on {d}.",
           (D("purchase", "Purchase date", "the purchase", (90, 365, 730)),
            B("extended", "Extended protection plan", "the customer bought the extended protection plan",
              "the customer did not buy the extended protection plan",
              "Did the customer buy the extended protection plan?", "The customer bought the extended protection plan.",
              "The customer did not buy the extended protection plan.", .3),
            C("cause", "Cause of the fault", "the cause of the fault", ["wear", "drop", "water", "manufacturing"]),
            B("registered", "Product registered", "the product is registered", "the product is not registered",
              "Is the product registered?", "The product is registered.", "The product is not registered.", .6),
            N("cost", "Estimated repair cost", "the estimated repair cost", 20, 900, (100, 250, 500), money=True),
            N("repairs", "Earlier repairs", "the number of earlier repairs", 0, 5, (0, 1, 2)),
            C("product", "Product type", "the product type", ["phone", "laptop", "tablet", "headphones"]),
            B("receipt", "Proof of purchase", "the customer has proof of purchase",
              "the customer has no proof of purchase", "Does the customer have proof of purchase?",
              "The customer has proof of purchase.", "The customer has no proof of purchase.", .8)),
           (("free", "Repair free of charge", "repair the product free of charge"),
            ("paid", "Repair for a fee", "repair the product for a fee"),
            ("replace", "Replace the product", "replace the product"), ("deny", "Deny the claim", "deny the claim")), "deny"),
    Domain("course", "late course registrations", "a university registrar's assistant", "What should the assistant do?",
           "the registration request", "Registration requested on", "The registration was requested on {d}.",
           (N("credits", "Credits already enrolled", "the number of credits already enrolled", 0, 30, (12, 18, 21)),
            N("average", "Grade average (out of 100)", "the grade average", 40, 100, (60, 70, 85)),
            D("termstart", "Term start date", "the start of term", (7, 14, 21)),
            B("advisor", "Advisor approval", "the advisor has approved", "the advisor has not approved",
              "Has the advisor approved?", "The advisor has approved.", "The advisor has not approved.", .5),
            B("prereq", "Prerequisites completed", "the prerequisites are completed",
              "the prerequisites are not completed", "Are the prerequisites completed?",
              "The prerequisites are completed.", "The prerequisites are not completed.", .7),
            C("year", "Year of study", "the year of study", ["first-year", "second-year", "final-year"]),
            B("hold", "Account hold", "the student's account has a hold", "the student's account has no hold",
              "Does the student's account have a hold?", "The student's account has a hold.",
              "The student's account has no hold.", .2),
            B("full", "Course full", "the course is full", "the course has free places", "Is the course full?",
              "The course is full.", "The course has free places.", .4)),
           (("enroll", "Enroll the student", "enroll the student"),
            ("wait", "Put the student on the waitlist", "put the student on the waitlist"),
            ("refuse", "Refuse the registration", "refuse the registration"),
            ("dean", "Refer to the dean's office", "refer the case to the dean's office")), "enroll"),
    Domain("deposit", "rental deposit returns", "a letting-agency assistant", "What should the assistant do?",
           "the deposit request", "Deposit return requested on", "The deposit return was requested on {d}.",
           (N("damage", "Assessed damage", "the assessed damage", 0, 1500, (50, 200, 500), money=True),
            D("moveout", "Move-out date", "the move-out", (14, 30)),
            B("cleaned", "Professional cleaning done", "professional cleaning was done",
              "professional cleaning was not done", "Was professional cleaning done?",
              "Professional cleaning was done.", "Professional cleaning was not done.", .6),
            N("tenancy", "Tenancy length in months", "the tenancy length in months", 1, 60, (6, 12, 24)),
            N("arrears", "Rent arrears", "the rent arrears", 0, 1200, (0, 100, 500), money=True),
            B("keys", "All keys returned", "all keys were returned", "not all keys were returned",
              "Were all keys returned?", "All keys were returned.", "Not all keys were returned.", .8),
            B("pets", "Pets kept", "the tenant kept pets", "the tenant kept no pets", "Did the tenant keep pets?",
              "The tenant kept pets.", "The tenant kept no pets.", .35),
            B("signed", "Check-out inspection signed", "the check-out inspection was signed",
              "the check-out inspection was not signed", "Was the check-out inspection signed?",
              "The check-out inspection was signed.", "The check-out inspection was not signed.", .7)),
           (("full", "Return the full deposit", "return the full deposit"),
            ("partial", "Return the deposit minus deductions", "return the deposit minus deductions"),
            ("withhold", "Withhold the deposit", "withhold the deposit"),
            ("dispute", "Send to the dispute service", "send the case to the dispute service")), "full"),
    Domain("carrental", "car-rental return charges", "a car-rental branch agent", "What should the agent do?",
           "the return", "Car returned on", "The car was returned on {d}.",
           (N("late", "Hours late", "the delay in hours", 0, 72, (1, 3, 24)),
            N("fuel", "Fuel level on return (%)", "the fuel level on return in percent", 0, 100, (25, 50, 90)),
            C("cover", "Insurance cover", "the insurance cover", ["none", "basic", "full"]),
            B("damage", "New damage found", "new damage was found", "no new damage was found", "Was new damage found?",
              "New damage was found.", "No new damage was found.", .3),
            N("age", "Driver age", "the driver's age", 18, 80, (21, 25, 70)),
            N("km", "Kilometres over the limit", "the distance over the mileage limit in km", 0, 900, (0, 100, 500)),
            B("corporate", "Corporate account", "the rental is on a corporate account",
              "the rental is on a personal account", "Is the rental on a corporate account?",
              "The rental is on a corporate account.", "The rental is on a personal account.", .3),
            D("pickup", "Pick-up date", "the pick-up", (3, 7, 14))),
           (("close", "Close the rental with no extra charge", "close the rental with no extra charge"),
            ("fee", "Charge a late-return fee", "charge a late-return fee"),
            ("excess", "Charge the damage excess", "charge the damage excess"),
            ("manager", "Refer to the branch manager", "refer the case to the branch manager")), "close"),
]

HELDOUT = [
    Domain("petboarding", "pet boarding bookings", "a pet-boarding receptionist", "What should the receptionist do?",
           "the booking request", "Booking requested on", "The booking was requested on {d}.",
           (B("vaccinated", "Vaccinations up to date", "the pet's vaccinations are up to date",
              "the pet's vaccinations are not up to date", "Are the pet's vaccinations up to date?",
              "The pet's vaccinations are up to date.", "The pet's vaccinations are not up to date.", .75),
            C("size", "Dog size", "the dog's size", ["small", "medium", "large"]),
            N("nights", "Nights requested", "the number of nights requested", 1, 30, (3, 7, 14)),
            D("laststay", "Last stay ended on", "the end of the last stay", (30, 180)),
            B("meds", "Daily medication", "the pet needs daily medication", "the pet needs no medication",
              "Does the pet need daily medication?", "The pet needs daily medication.", "The pet needs no medication.", .25),
            B("incident", "Previous incident report", "the pet has a previous incident report",
              "the pet has no incident report", "Does the pet have a previous incident report?",
              "The pet has a previous incident report.", "The pet has no incident report.", .15),
            C("owner", "Owner membership", "the owner's membership", ["none", "regular", "premium"]),
            N("prepaid", "Prepaid amount", "the prepaid amount", 0, 200, (20, 50), money=True)),
           (("accept", "Accept the booking", "accept the booking"),
            ("surcharge", "Accept with a supervision surcharge", "accept the booking with a supervision surcharge"),
            ("decline", "Decline the booking", "decline the booking"),
            ("vet", "Ask for a vet assessment first", "ask for a vet assessment first")), "accept"),
    Domain("coworking", "coworking room requests", "a coworking-space community assistant", "What should the assistant do?",
           "the request", "Request made on", "The request was made on {d}.",
           (C("membership", "Membership", "the membership type", ["day pass", "part-time", "full-time"]),
            N("hours", "Hours requested", "the number of hours requested", 1, 60, (4, 8, 40)),
            N("guests", "Guests", "the number of guests", 0, 12, (1, 3, 6)),
            D("joined", "Join date", "joining", (30, 90)),
            N("invoices", "Overdue invoices", "the number of overdue invoices", 0, 5, (0, 1, 2)),
            B("office", "Private office holder", "the member rents a private office",
              "the member does not rent a private office", "Does the member rent a private office?",
              "The member rents a private office.", "The member does not rent a private office.", .25),
            B("afterhours", "After-hours use", "the room is needed after hours",
              "the room is needed during staffed hours", "Is the room needed after hours?",
              "The room is needed after hours.", "The room is needed during staffed hours.", .35),
            N("complaints", "Noise complaints this year", "the number of noise complaints this year", 0, 5, (0, 1, 2))),
           (("approve", "Approve the request", "approve the request"),
            ("fee", "Approve for an extra fee", "approve the request for an extra fee"),
            ("decline", "Decline the request", "decline the request"),
            ("manager", "Refer to the community manager", "refer the request to the community manager")), "decline"),
    Domain("museum", "museum membership refunds", "a museum membership officer", "What should the officer do?",
           "the refund request", "Refund requested on", "The refund was requested on {d}.",
           (C("tier", "Membership tier", "the membership tier", ["individual", "family", "patron"]),
            N("visits", "Visits this year", "the number of visits this year", 0, 30, (2, 5, 10)),
            D("bought", "Membership purchase date", "the membership purchase", (14, 30, 60)),
            B("exhibition", "Special-exhibition tickets used", "special-exhibition tickets were used",
              "no special-exhibition tickets were used", "Were special-exhibition tickets used?",
              "Special-exhibition tickets were used.", "No special-exhibition tickets were used.", .4),
            B("student", "Student card", "the member has a student card", "the member has no student card",
              "Does the member have a student card?", "The member has a student card.",
              "The member has no student card.", .25),
            N("passes", "Guest passes used", "the number of guest passes used", 0, 10, (1, 2, 4)),
            N("donation", "Donation included", "the included donation", 0, 2000, (100, 500), money=True),
            B("online", "Bought online", "the membership was bought online", "the membership was bought at the desk",
              "Was the membership bought online?", "The membership was bought online.",
              "The membership was bought at the desk.", .5)),
           (("refund", "Give a full refund", "give a full refund"), ("prorate", "Give a prorated refund", "give a prorated refund"),
            ("refuse", "Refuse the refund", "refuse the refund"),
            ("office", "Refer to the development office", "refer the request to the development office")), "refuse"),
    Domain("bikeshare", "bike-share penalty appeals", "a bike-share customer-care agent", "What should the agent do?",
           "the appeal", "Appeal submitted on", "The appeal was submitted on {d}.",
           (N("over", "Minutes over the time limit", "the time over the limit in minutes", 1, 180, (10, 30, 60)),
            B("docked", "Bike docked correctly", "the bike was docked correctly", "the bike was not docked correctly",
              "Was the bike docked correctly?", "The bike was docked correctly.", "The bike was not docked correctly.", .6),
            B("reported", "Damage reported by the rider", "the rider reported damage", "the rider reported no damage",
              "Did the rider report damage?", "The rider reported damage.", "The rider reported no damage.", .25),
            C("plan", "Plan", "the rider's plan", ["pay-as-you-go", "monthly", "annual"]),
            N("violations", "Earlier violations", "the number of earlier violations", 0, 8, (0, 1, 3)),
            D("trip", "Trip date", "the trip", (7, 30)),
            B("ebike", "E-bike", "the trip was on an e-bike", "the trip was on a standard bike",
              "Was the trip on an e-bike?", "The trip was on an e-bike.", "The trip was on a standard bike.", .4),
            B("outage", "App outage reported", "an app outage was reported during the trip",
              "no app outage was reported during the trip", "Was an app outage reported during the trip?",
              "An app outage was reported during the trip.", "No app outage was reported during the trip.", .2)),
           (("waive", "Waive the penalty", "waive the penalty"), ("halve", "Reduce the penalty by half", "reduce the penalty by half"),
            ("uphold", "Uphold the full penalty", "uphold the full penalty"),
            ("specialist", "Send to a care specialist", "send the appeal to a care specialist")), "uphold"),
]

DOMAINS = {d.name: d for d in TRAIN + HELDOUT}
TRAIN_DOMAINS = tuple(d.name for d in TRAIN)
HELDOUT_DOMAINS = tuple(d.name for d in HELDOUT)
RENDERERS = ("list", "bullets", "prose")
TRAIN_RENDERERS = ("list", "bullets")


# ---------------------------------------------------------------- predicates and rules
@dataclasses.dataclass(frozen=True)
class Pred:
    fact: str
    arg: Any = None                 # cat: value; num/days: threshold; bool: None

    @property
    def pid(self) -> str:
        return self.fact if self.arg is None else f"{self.fact}{'=' if isinstance(self.arg, str) else '>'}{self.arg}"

    @staticmethod
    def parse(pid: str) -> "Pred":
        if "=" in pid:
            f, v = pid.split("=", 1)
            return Pred(f, v)
        if ">" in pid:
            f, t = pid.split(">", 1)
            return Pred(f, int(t))
        return Pred(pid)


@dataclasses.dataclass(frozen=True)
class Rule:
    conds: tuple                    # ((Pred, want), ...)
    outcome: str


@dataclasses.dataclass
class Case:
    item_id: str
    domain: Domain
    rules: list
    facts: dict                     # fact name -> value; days facts hold the anchor date; "_request" the request date
    label_order: list
    renderer: str = "list"

    def evaluate(self, overrides: dict | None = None) -> dict:
        return run(self.domain, self.rules, self.facts, overrides)

    def gold(self) -> str:
        return self.evaluate()["decision"]


def fact_value(dom: Domain, facts: dict, name: str) -> Any:
    """The value a predicate reads: days facts give the number of days from the anchor date to the request."""
    f = dom.fact(name)
    if f.kind == "days":
        return (facts["_request"] - facts[name]).days
    return facts[name]


def pred_value(dom: Domain, pred: Pred, facts: dict) -> bool:
    f = dom.fact(pred.fact)
    v = fact_value(dom, facts, pred.fact)
    if f.kind == "bool":
        return bool(v)
    if f.kind == "cat":
        return v == pred.arg
    return v > pred.arg


def run(dom: Domain, rules: list, facts: dict, overrides: dict | None = None) -> dict:
    """Evaluate the policy. `overrides` maps variable names (p:<pid>, a:<k>) to forced values."""
    ov = overrides or {}
    p: dict[str, bool] = {}
    for rule in rules:
        for pred, _ in rule.conds:
            if pred.pid not in p:
                p[pred.pid] = ov.get(f"p:{pred.pid}", pred_value(dom, pred, facts))
    a = [ov.get(f"a:{k}", all(p[pr.pid] == want for pr, want in rule.conds)) for k, rule in enumerate(rules, 1)]
    fire = next((k for k, v in enumerate(a, 1) if v), None)
    return {"p": p, "a": a, "fire": fire, "decision": rules[fire - 1].outcome if fire else dom.default}


def variables(res: dict) -> dict[str, bool]:
    return {**{f"p:{k}": v for k, v in res["p"].items()}, **{f"a:{k}": v for k, v in enumerate(res["a"], 1)}}


# ---------------------------------------------------------------- text
def fmt(f: Fact, x: int) -> str:
    return f"${x}" if f.money else str(x)


def clause(dom: Domain, pred: Pred, want: bool) -> str:
    f = dom.fact(pred.fact)
    if f.kind == "bool":
        return f.pos if want else f.neg
    if f.kind == "cat":
        return f"{f.noun} is {pred.arg}" if want else f"{f.noun} is not {pred.arg}"
    if f.kind == "num":
        return f"{f.noun} is above {fmt(f, pred.arg)}" if want else f"{f.noun} is {fmt(f, pred.arg)} or less"
    return (f"{dom.event} comes more than {pred.arg} days after {f.noun}" if want
            else f"{dom.event} comes within {pred.arg} days of {f.noun}")


def question(dom: Domain, pred: Pred) -> str:
    f = dom.fact(pred.fact)
    if f.kind == "bool":
        return f.q
    if f.kind == "cat":
        return f"Is {f.noun} {pred.arg}?"
    if f.kind == "num":
        return f"Is {f.noun} above {fmt(f, pred.arg)}?"
    return f"Does {dom.event} come more than {pred.arg} days after {f.noun}?"


def conds_text(dom: Domain, rule: Rule) -> str:
    cs = [clause(dom, p, w) for p, w in rule.conds]
    return cs[0] if len(cs) == 1 else ", ".join(cs[:-1]) + " and " + cs[-1]


def rule_question(dom: Domain, rule: Rule) -> str:
    if len(rule.conds) == 1:
        return f"Is it true that {conds_text(dom, rule)}?"
    return f"Do all of these hold: {conds_text(dom, rule)}?"


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def case_lines(dom: Domain, facts: dict) -> list[tuple[str, str]]:
    out = []
    for f in dom.facts:
        v = facts[f.name]
        if f.kind == "bool":
            out.append((f.label, "yes" if v else "no"))
        elif f.kind == "days":
            out.append((f.label, v.isoformat()))
        elif f.kind == "num":
            out.append((f.label, fmt(f, v)))
        else:
            out.append((f.label, v))
    out.append((dom.request, facts["_request"].isoformat()))
    return out


def case_sentences(dom: Domain, facts: dict) -> list[str]:
    out = []
    for f in dom.facts:
        v = facts[f.name]
        if f.kind == "bool":
            out.append(f.say_yes if v else f.say_no)
        elif f.kind == "days":
            out.append(f"{_cap(f.noun)} was on {v.isoformat()}.")
        elif f.kind == "num":
            out.append(f"{_cap(f.noun)} is {fmt(f, v)}.")
        else:
            out.append(f"{_cap(f.noun)} is {v}.")
    out.append(dom.request_say.format(d=facts["_request"].isoformat()))
    return out


def render(case: Case, renderer: str | None = None) -> str:
    dom, r = case.domain, renderer or case.renderer
    opts = "\n".join(f"{chr(65 + i)}) {dom.option(k)}" for i, k in enumerate(case.label_order))
    if r == "list":
        rules = [f"{i}. If {conds_text(dom, x)}, {dom.imperative(x.outcome)}." for i, x in enumerate(case.rules, 1)]
        lines = [f"- {a}: {b}" for a, b in case_lines(dom, case.facts)]
        return (f"You are {dom.role}. Apply the policy to the case.\n\n"
                f"Policy ({dom.title}). Rules are checked in order; the first rule whose conditions all hold decides "
                f"the outcome.\n" + "\n".join(rules) + f"\nOtherwise: {dom.imperative(dom.default)}.\n\n"
                f"Case:\n" + "\n".join(lines) + f"\n\nQuestion: {dom.ask}\nOptions:\n{opts}")
    if r == "bullets":
        rules = [f"* Rule {i}: when {conds_text(dom, x)} -> {dom.imperative(x.outcome)}"
                 for i, x in enumerate(case.rules, 1)]
        facts = "; ".join(f"{a} = {b}" for a, b in case_lines(dom, case.facts))
        return (f"Role: {dom.role}.\n{_cap(dom.title)} policy (the earliest matching rule wins):\n" + "\n".join(rules)
                + f"\n* No rule matches -> {dom.imperative(dom.default)}\n\nRequest details: {facts}\n\n"
                f"{dom.ask}\nOptions:\n{opts}")
    if r == "prose":
        parts = []
        for i, x in enumerate(case.rules):
            lead = "First, if" if i == 0 else "Otherwise, if"
            parts.append(f"{lead} {conds_text(dom, x)}, you {dom.imperative(x.outcome)}.")
        parts.append(f"In every other case, you {dom.imperative(dom.default)}.")
        return (f"You work as {dom.role}. Here is how {dom.title} are handled. " + " ".join(parts) + "\n\n"
                + " ".join(case_sentences(dom, case.facts)) + f"\n\n{dom.ask}\nOptions:\n{opts}")
    raise ValueError(r)


# ---------------------------------------------------------------- sampling
def _near(rng: random.Random, cuts: list[int], lo: int, hi: int) -> int:
    """An integer, often within a few steps of a threshold (so decisions are not trivial)."""
    if cuts and rng.random() < 0.6:
        t = rng.choice(cuts)
        return max(lo, min(hi, t + max(1, t // 25) * rng.choice([-3, -2, -1, 0, 1, 1, 2, 3])))
    return rng.randint(lo, hi)


def policy_cuts(rules: list) -> dict[str, list[int]]:
    cuts: dict[str, list[int]] = {}
    for rule in rules:
        for p, _ in rule.conds:
            if isinstance(p.arg, int):
                cuts.setdefault(p.fact, []).append(p.arg)
    return cuts


def sample_facts(dom: Domain, rng: random.Random, cuts: dict[str, list[int]]) -> dict:
    request = dt.date(2026, 1, 1) + dt.timedelta(days=rng.randrange(0, 300))
    facts: dict[str, Any] = {"_request": request}
    for f in dom.facts:
        if f.kind == "bool":
            facts[f.name] = rng.random() < f.p_true
        elif f.kind == "cat":
            facts[f.name] = rng.choice(f.values)
        elif f.kind == "num":
            facts[f.name] = _near(rng, cuts.get(f.name, list(f.cuts)), f.lo, f.hi)
        else:
            facts[f.name] = request - dt.timedelta(days=_near(rng, cuts.get(f.name, list(f.cuts)), f.lo, f.hi))
    return facts


def _fact_preds(f: Fact, rng: random.Random) -> list[Pred]:
    if f.kind == "bool":
        return [Pred(f.name)]
    pool = list(f.values if f.kind == "cat" else f.cuts)
    return [Pred(f.name, a) for a in rng.sample(pool, 1 if rng.random() < 0.7 or len(pool) < 2 else 2)]


def sample_policy(dom: Domain, rng: random.Random, tries: int = 200) -> list:
    """Three to five rules of one to three conditions (mostly two or three), each able to fire."""
    for _ in range(tries):
        active = rng.sample(list(dom.facts), rng.randint(4, 5))
        preds = {f.name: _fact_preds(f, rng) for f in active}
        rules = []
        for _ in range(rng.choice([3, 4, 4, 5])):
            m = rng.choices([1, 2, 3], weights=[0.2, 0.5, 0.3])[0]
            conds = []
            for f in rng.sample(active, m):
                p = rng.choice(preds[f.name])
                want = rng.random() < (0.65 if f.kind == "cat" else 0.5)
                conds.append((p, want))
            outcome = rng.choice([o[0] for o in dom.outcomes if o[0] != dom.default])
            rules.append(Rule(tuple(sorted(conds, key=lambda c: [x.name for x in dom.facts].index(c[0].fact))), outcome))
        if valid_policy(dom, rules, rng):
            return rules
    raise RuntimeError(f"no valid policy for {dom.name}")


def valid_policy(dom: Domain, rules: list, rng: random.Random, n: int = 300) -> bool:
    if len({r.conds for r in rules}) < len(rules) or len({r.outcome for r in rules} | {dom.default}) < 3:
        return False
    cuts = policy_cuts(rules)
    fired = set()
    for _ in range(n):
        fired.add(run(dom, rules, sample_facts(dom, rng, cuts))["fire"])
        if len(fired) == len(rules) + 1:
            return True
    return all(k in fired for k in range(1, len(rules) + 1))


def make_case(dom: Domain, rng: random.Random, item_id: str, renderer: str = "list") -> Case:
    rules = sample_policy(dom, rng)
    facts = sample_facts(dom, rng, policy_cuts(rules))
    order = [o[0] for o in dom.outcomes]
    rng.shuffle(order)
    return Case(item_id, dom, rules, facts, order, renderer)


# ---------------------------------------------------------------- tree nodes and control questions
def tree_nodes(case: Case) -> list[dict]:
    """Program tree of the rules checked up to the deciding rule: one node per predicate those rules read, then one
    node per rule ("do all of its conditions hold?") that depends on its predicate nodes. Every node's truth is the
    program's value; `var` names the variable (p:<pid> or a:<k>)."""
    dom, res = case.domain, case.evaluate()
    checked = case.rules[:res["fire"]] if res["fire"] else case.rules
    nodes, ids = [], {}
    for k, rule in enumerate(checked, 1):
        deps = []
        for pred, _ in rule.conds:
            if pred.pid not in ids:
                ids[pred.pid] = f"n{len(nodes) + 1}"
                v = res["p"][pred.pid]
                nodes.append({"id": ids[pred.pid], "question": question(dom, pred), "truth": v, "p_cot": float(v),
                              "p_fresh": None, "type": "verify", "depends_on": [], "var": f"p:{pred.pid}",
                              "kind": "arith" if dom.fact(pred.fact).kind in ("num", "days") else "lookup"})
            deps.append(ids[pred.pid])
        v = res["a"][k - 1]
        nodes.append({"id": f"n{len(nodes) + 1}", "question": rule_question(dom, rule), "truth": v, "p_cot": float(v),
                      "p_fresh": None, "type": "derive", "depends_on": deps, "var": f"a:{k}", "kind": "rule"})
    return nodes


def control_questions(case: Case, rng: random.Random, n: int) -> list[dict]:
    """Questions of the same forms about facts that no rule reads (causally irrelevant to the decision), with the
    share of Yes answers matched to the tree nodes as far as the pool allows."""
    dom = case.domain
    used = {p.fact for r in case.rules for p, _ in r.conds}
    pool = []
    for f in dom.facts:
        if f.name in used:
            continue
        args = [None] if f.kind == "bool" else list(f.values if f.kind == "cat" else f.cuts)
        for a in args:
            p = Pred(f.name, a)
            v = pred_value(dom, p, case.facts)
            pool.append({"question": question(dom, p), "truth": v, "p_fresh": float(v), "var": f"ctl:{p.pid}",
                         "kind": "arith" if f.kind in ("num", "days") else "lookup"})
    rng.shuffle(pool)
    nodes = tree_nodes(case)
    want_yes = round(n * sum(x["truth"] for x in nodes) / max(len(nodes), 1))
    yes = [x for x in pool if x["truth"]]
    no = [x for x in pool if not x["truth"]]
    k_yes = min(want_yes, len(yes))
    out = yes[:k_yes] + no[:n - k_yes]
    out += yes[k_yes:][:n - len(out)]
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------- counterfactual pairs
def _all_preds(case: Case) -> list[Pred]:
    seen, out = set(), []
    for r in case.rules:
        for p, _ in r.conds:
            if p.pid not in seen:
                seen.add(p.pid)
                out.append(p)
    return out


def _with(case: Case, item_id: str, facts: dict | None = None, rules: list | None = None) -> Case:
    return Case(item_id, case.domain, list(rules if rules is not None else case.rules),
                dict(facts if facts is not None else case.facts), list(case.label_order), case.renderer)


def _set_value(dom: Domain, facts: dict, name: str, value: Any) -> dict:
    """Facts with one fact set; for days facts `value` is the number of days and the anchor date moves."""
    out = dict(facts)
    out[name] = facts["_request"] - dt.timedelta(days=value) if dom.fact(name).kind == "days" else value
    return out


def flip_one(case: Case, pred: Pred, rng: random.Random) -> dict | None:
    """Facts where `pred` flips and every other predicate of the policy keeps its value (None if impossible)."""
    dom, f = case.domain, case.domain.fact(pred.fact)
    preds = _all_preds(case)
    before = {p.pid: pred_value(dom, p, case.facts) for p in preds}
    x = fact_value(dom, case.facts, f.name)
    if f.kind == "bool":
        cands = [not x]
    elif f.kind == "cat":
        cands = [v for v in f.values if v != x]
        rng.shuffle(cands)
    else:
        t, step = pred.arg, max(1, pred.arg // 25)
        cands = [t - i * step for i in range(0, 6)] if x > t else [t + 1 + i * step for i in range(0, 6)]
        cands = [c for c in cands if f.lo <= c <= f.hi and c != x]
        same = [c for c in cands if len(str(c)) == len(str(x))]
        cands = same + [c for c in cands if c not in same]
    for c in cands:
        facts = _set_value(dom, case.facts, f.name, c)
        after = {p.pid: pred_value(dom, p, facts) for p in preds}
        if {k for k in before if before[k] != after[k]} == {pred.pid}:
            return facts
    return None


def edit_pairs(case: Case, rng: random.Random) -> list[tuple[dict, Case]]:
    """One-fact edits that flip exactly one predicate, one per predicate where possible."""
    base = case.gold()
    out = []
    for i, pred in enumerate(_all_preds(case)):
        facts = flip_one(case, pred, rng)
        if facts is None:
            continue
        cf = _with(case, f"{case.item_id}-e{i}", facts=facts)
        out.append(({"type": "edit", "sub": "sensitive" if cf.gold() != base else "masked", "var": f"p:{pred.pid}",
                     "fact": pred.fact}, cf))
    return out


def distractor_pairs(case: Case, rng: random.Random, n: int = 1) -> list[tuple[dict, Case]]:
    dom = case.domain
    used = {p.fact for r in case.rules for p, _ in r.conds}
    free = [f for f in dom.facts if f.name not in used]
    out = []
    for i, f in enumerate(rng.sample(free, min(n, len(free)))):
        x = fact_value(dom, case.facts, f.name)
        if f.kind == "bool":
            v = not x
        elif f.kind == "cat":
            v = rng.choice([c for c in f.values if c != x])
        else:
            v = rng.choice([c for c in range(f.lo, f.hi + 1) if c != x and len(str(c)) == len(str(x))] or
                           [c for c in (f.lo, f.hi) if c != x])
        cf = _with(case, f"{case.item_id}-d{i}", facts=_set_value(dom, case.facts, f.name, v))
        assert cf.gold() == case.gold()
        out.append(({"type": "distract", "sub": "invariant", "var": None, "fact": f.name}, cf))
    return out


def rule_twins(case: Case, rng: random.Random) -> list[tuple[dict, Case]]:
    """The same case under a minimally changed policy: a moved threshold (the predicate flips for this case), a
    negated condition, two adjacent rules swapped, or one rule's outcome replaced."""
    dom, base = case.domain, case.gold()
    out = []

    def add(kind: str, rules: list, k: int, j: int = 0):
        cf = _with(case, f"{case.item_id}-r{len(out)}", rules=rules)
        out.append(({"type": "rule", "sub": kind, "rule": k, "gold_change": cf.gold() != base}, cf))

    k = rng.randrange(len(case.rules))
    rule = case.rules[k]
    num = [(j, p, w) for j, (p, w) in enumerate(rule.conds) if isinstance(p.arg, int)]
    if num:
        j, p, w = rng.choice(num)
        f, x = dom.fact(p.fact), fact_value(dom, case.facts, p.fact)
        cands = sorted({c for c in f.cuts + tuple(range(max(f.lo, 1), f.hi + 1, max(1, (f.hi - f.lo) // 12)))
                        if (x > c) != (x > p.arg)}, key=lambda c: abs(c - p.arg))
        if cands:
            conds = list(rule.conds)
            conds[j] = (Pred(p.fact, cands[0]), w)
            rules = list(case.rules)
            rules[k] = Rule(tuple(conds), rule.outcome)
            add("threshold", rules, k + 1)
    j = rng.randrange(len(rule.conds))
    conds = list(rule.conds)
    conds[j] = (conds[j][0], not conds[j][1])
    rules = list(case.rules)
    rules[k] = Rule(tuple(conds), rule.outcome)
    add("negate", rules, k + 1)
    if len(case.rules) > 1:
        i = rng.randrange(len(case.rules) - 1)
        rules = list(case.rules)
        rules[i], rules[i + 1] = rules[i + 1], rules[i]
        add("swap", rules, i + 1)
    other = [o[0] for o in dom.outcomes if o[0] != rule.outcome]
    rules = list(case.rules)
    rules[k] = Rule(rule.conds, rng.choice(other))
    add("outcome", rules, k + 1)
    return out


def third_outcome_pairs(case: Case, rng: random.Random, tries: int = 40, per_case: int = 4) -> list[tuple[dict, Case]]:
    """Sources with the same policy where setting one variable of the base to its source value gives an outcome that
    differs from both the base and the source outcome. Sources are one- or two-fact edits of the base (aligned text)
    or fresh fact draws; at most one pair per variable, rule variables of multi-condition rules preferred."""
    dom = case.domain
    b = case.evaluate()
    bv = variables(b)
    cuts = policy_cuts(case.rules)
    preds = _all_preds(case)
    out, seen = [], set()
    for t in range(tries):
        if rng.random() < 0.5:
            facts = dict(case.facts)
            for p in rng.sample(preds, min(len(preds), rng.choice([1, 2]))):
                facts = flip_one(_with(case, "", facts=facts), p, rng) or facts
            kind = "edit"
        else:
            facts = sample_facts(dom, rng, cuts)
            facts["_request"] = case.facts["_request"]
            for f in dom.facts:
                if f.kind == "days":
                    days = (case.facts["_request"] - case.facts[f.name]).days if rng.random() < 0.3 else \
                        _near(rng, cuts.get(f.name, list(f.cuts)), f.lo, f.hi)
                    facts[f.name] = facts["_request"] - dt.timedelta(days=days)
            kind = "fresh"
        s = run(dom, case.rules, facts)
        sv = variables(s)
        order = rng.sample(sorted(bv), len(bv))
        order.sort(key=lambda v: not (v.startswith("a:") and len(case.rules[int(v[2:]) - 1].conds) > 1))
        for var in order:                           # rule variables of multi-condition rules first
            if var in seen or bv[var] == sv[var]:
                continue
            target = run(dom, case.rules, case.facts, {var: sv[var]})["decision"]
            if target != b["decision"] and target != s["decision"]:
                seen.add(var)
                src = _with(case, f"{case.item_id}-s{len(out)}", facts=facts)
                out.append(({"type": "third", "sub": kind, "var": var, "base_out": b["decision"],
                             "source_out": s["decision"], "target_out": target}, src))
                break
        if len(out) >= per_case:
            break
    return out


# ---------------------------------------------------------------- records
def _facts_json(facts: dict) -> dict:
    return {k: (v.isoformat() if isinstance(v, dt.date) else v) for k, v in facts.items()}


def facts_from_json(dom: Domain, js: dict) -> dict:
    out = {}
    for k, v in js.items():
        out[k] = dt.date.fromisoformat(v) if k == "_request" or dom.fact(k).kind == "days" else v
    return out


def rules_json(rules: list) -> list[dict]:
    return [{"conds": [[p.pid, w] for p, w in r.conds], "outcome": r.outcome} for r in rules]


def rules_from_json(js: list[dict]) -> list:
    return [Rule(tuple((Pred.parse(p), w) for p, w in r["conds"]), r["outcome"]) for r in js]


def case_from_record(rec: dict) -> Case:
    dom = DOMAINS[rec["domain"]]
    return Case(rec["item_id"], dom, rules_from_json(rec["rules"]), facts_from_json(dom, rec["facts"]),
                list(rec["label_order"]), rec["renderer"])


def eval_record(case: Case) -> dict:
    """Evaluation record (same fields as data/eval/policy_heldout.jsonl plus the program state)."""
    res = case.evaluate()
    labels = [chr(65 + i) for i in range(len(case.label_order))]
    n_fire = len(case.rules[res["fire"] - 1].conds) if res["fire"] else 0
    return {"item_id": case.item_id, "source": "rule", "group": f"rule/{case.domain.name}/c{n_fire}", "type": "choice",
            "prompt": render(case), "labels": labels, "label_order": case.label_order,
            "gold_label": labels[case.label_order.index(res["decision"])], "gold": res["decision"],
            "label_names": [case.domain.option(k) for k in case.label_order],
            "depth": len(tree_nodes(case)), "domain": case.domain.name, "renderer": case.renderer,
            "fire": res["fire"], "n_conds_fire": n_fire, "vars": variables(res),
            "rules": rules_json(case.rules), "facts": _facts_json(case.facts)}


def train_record(case: Case, rng: random.Random) -> dict:
    """Student training record: one-hot final target, program tree nodes as `subqs` (truth-valued) and matched
    questions about unread facts as `random_subqs` (the control arm's views)."""
    ev = eval_record(case)
    nodes = tree_nodes(case)
    return {"item_id": case.item_id, "source": "rule", "domain": case.domain.name, "renderer": case.renderer,
            "prompt": ev["prompt"], "labels": ev["labels"], "gold_label": ev["gold_label"],
            "teacher": {L: (1.0 if L == ev["gold_label"] else 0.0) for L in ev["labels"]},
            "depth": len(nodes), "subqs": nodes, "random_subqs": control_questions(case, rng, len(nodes)),
            "rationale": "", "augment": None, "has_tree": True,
            "fire": ev["fire"], "n_conds_fire": ev["n_conds_fire"], "rules": ev["rules"], "facts": ev["facts"],
            "label_order": case.label_order}


def generate(n: int, domains: tuple, seed: int, renderers: tuple = ("list",), balance: float = 1.5,
             prefix: str = "rg2") -> list[Case]:
    """`n` cases spread evenly over `domains`, gold outcomes capped at `balance` x their fair share per domain."""
    out = []
    per = {d: n // len(domains) + (1 if i < n % len(domains) else 0) for i, d in enumerate(domains)}
    for d in domains:
        dom, rng = DOMAINS[d], random.Random(f"{seed}-{d}")
        cap = max(-(-per[d] // len(dom.outcomes)), int(balance * per[d] / len(dom.outcomes)))
        counts: dict[str, int] = {}
        made = 0
        while made < per[d]:
            case = make_case(dom, rng, f"{prefix}-{d}-{seed}-{made:05d}", rng.choice(renderers))
            g = case.gold()
            if counts.get(g, 0) >= cap:
                continue
            counts[g] = counts.get(g, 0) + 1
            out.append(case)
            made += 1
    return out
