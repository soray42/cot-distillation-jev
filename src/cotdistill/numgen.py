"""Own numeric-chain generator (plan data_expansion §1.1 row M1, gsm8k.md §§3-5, numeric_nodes §5.1): GSM-like word
problems written from programs, with every intermediate value exact. No teacher and no LLM: the surface text is
templated here, and the program (quantities, operations, constants, values, names) is kept with each item so a later
rewriting step can lock the numbers.

A problem is a chain of k operations (k = depth). It starts from a stated count ("On Monday, Maria baked 24
muffins.") and each operation creates one new named quantity from the latest one (so every operation is needed) and,
for fan-in operations, one earlier quantity or a freshly stated count. Kinds and operations:

  count    add / sub (n more / fewer), mul / div (k times / half, a third, a quarter as many), pct (p% as many),
           sum2 (as many as two earlier batches combined: fan-in), dozen (a stated "n dozen" batch: unit conversion,
           first step only); conversions to money (sell / paid / buy at $p each), to containers (pack r to a
           container) and to minutes (t minutes each); sell2 (two batches sold at their own prices, then put
           together: fan-in); total (final question only: two batches altogether, fan-in)
  money    earn frames: spend, bonus, share (split equally among k), spend_each (after share), save (p% into
           savings), rel (a relative makes k times as much); buy frames: coupon, fee, split, discount (p% off), rel
  box      sell_box (each container sells / costs $p), give (n containers given away)
  minutes  setup (n more minutes), reltime (a relative takes k times as long), hours (final question only, / 60)

Every value is a positive integer <= 10,000 (divisions exact), operands are 2-99 except prices and container sizes.
Surface: 47 scenario frames (FRAMES; the 6 HELD_OUT are kept out of training for the OOD file), paraphrase variants per
sentence, premises shuffled in half the items (explicit references, no quantity pronouns), and 1-2 irrelevant numeric
sentences in every item (frame facts, generic facts, or another person's batch of the same item).

Options (5, letters shuffled), recipes:
  qo2    QO-v2 as in plan §3.2: 2 partial results (one binary operation on two stated numbers, or the gold combined
         with one stated number; uniform over candidates != gold, positive, <= 10x gold), then 2 perturbations
         (+-d, x2, /2) of an anchor drawn uniformly from {gold, partial 1, partial 2}. A local implementation of the
         recipe text, not the frozen copy build_math_evals.py registers.
  inter  up to 2 program intermediates, sometimes 1 partial, the rest perturbations anchored uniformly on the options
         so far (the gold included at the same rate)
  slip   up to 2 planted-slip results (one operation swapped, a constant off by one, a step skipped, the wrong fan-in
         input), the rest perturbations as in inter
  mix    per item 40% qo2 / 30% inter / 30% slip (plan §3.2.4, the training mixture)

Nodes (types follow gsm8k.md §1.3): subqs are yes/no questions on every intermediate except the final operation and
any value equal to the gold: N1 "Is <name> equal to x?" (x the value or a slip/perturbation, 50/50), and one of N2
"Is <name> more than X?" (X = value x r, r in .5/.67/1.5/2) or N3 "Is the last digit of <name> d?" (50/50).
depends_on links an intermediate's nodes to the N1 nodes of its intermediate parents. Standalone 5-option node rows
(node_rows): N1 "<story> <question for the intermediate>" with mix-recipe options, and N4 "which of the numbers that
come up answers the question".
"""
from __future__ import annotations

import dataclasses
import itertools
import random
import re
from fractions import Fraction

LETTERS = ["A", "B", "C", "D", "E"]
MAXV = 10_000

NAMES = [("Maria", "f"), ("James", "m"), ("Aisha", "f"), ("Carlos", "m"), ("Mei", "f"), ("Tom", "m"), ("Priya", "f"),
         ("Lucas", "m"), ("Fatima", "f"), ("Noah", "m"), ("Elena", "f"), ("Omar", "m"), ("Grace", "f"), ("Ivan", "m"),
         ("Sofia", "f"), ("Daniel", "m"), ("Hana", "f"), ("Kwame", "m"), ("Chloe", "f"), ("Ravi", "m"),
         ("Nadia", "f"), ("Felix", "m"), ("Leah", "f"), ("Mateo", "m"), ("Yuki", "f"), ("Samuel", "m"),
         ("Amara", "f"), ("Diego", "m"), ("Ingrid", "f"), ("Hassan", "m"), ("Rosa", "f"), ("Peter", "m"),
         ("Zara", "f"), ("Kenji", "m"), ("Olivia", "f"), ("Tariq", "m"), ("Clara", "f"), ("Victor", "m"),
         ("Lina", "f"), ("Marcus", "m"), ("Ada", "f"), ("Joel", "m"), ("Beatriz", "f"), ("Arjun", "m"),
         ("Freya", "f"), ("Emeka", "m"), ("Julia", "f"), ("Andre", "m"), ("Nora", "f"), ("Hugo", "m")]
RELS = ["sister", "brother", "cousin", "neighbor", "friend", "aunt", "uncle"]
GROUPS = ["helpers", "friends", "cousins", "teammates", "classmates", "neighbors"]
SPEND = ["a new pair of shoes", "a birthday present", "groceries", "a concert ticket", "a bus pass", "art supplies",
         "a board game", "a winter coat", "books", "a phone case"]
BONUS = ["tip", "bonus", "gift"]
RECIPIENTS = ["a food bank", "a neighbor", "the school", "a local shelter", "the community center"]
MODSETS = {
    "days": ["on Monday", "on Tuesday", "on Wednesday", "on Thursday", "on Friday", "on Saturday", "on Sunday"],
    "months": ["in January", "in February", "in March", "in April", "in May", "in June", "in July", "in August"],
    "weeks": ["in the first week", "in the second week", "in the third week", "in the fourth week",
              "in the fifth week", "in the sixth week", "in the seventh week", "in the eighth week"],
}
GENERIC_NOISE = ["{N} is {n} years old.", "{N} lives {n} miles from the town center.", "{N} has {n} cousins.",
                 "{N}'s street has {n} houses on it.", "{N} has been doing this for {n} months.",
                 "{N}'s phone has {n} saved contacts."]


def F(fid, intro, item, verb, mode, price, mods="days", container=None, tverb=None, dozen=False, noise=(),
      spend=()):
    return {"id": fid, "intro": intro, "item": item, "verb": verb, "mode": mode, "price": price, "mods": mods,
            "container": container, "tverb": tverb, "dozen": dozen, "noise": list(noise), "spend": list(spend)}


# id, intro, (sg, pl), (base, past), money mode, price range, modifier set, container (sg, pl, lo, hi),
# time verb (base, -ing, lo, hi), dozen allowed, frame noise, frame spending
FRAMES = [
    F("bakery", "{N} runs a small bakery.", ("muffin", "muffins"), ("bake", "baked"), "sell", (2, 6), "days",
      ("tray", "trays", 4, 24), ("decorate", "decorating", 2, 9), True,
      ["The bakery has {n} chairs for customers.", "{N}'s oven can reach {n}0 degrees."], ["flour", "a new mixer"]),
    F("orchard", "{N} works at an apple orchard.", ("apple", "apples"), ("pick", "picked"), "sell", (1, 3), "weeks",
      ("basket", "baskets", 5, 40), ("wash", "washing", 1, 3), True,
      ["The orchard has {n} rows of trees.", "The orchard is {n} miles from the city."], ["a ladder", "new gloves"]),
    F("knitting", "{N} knits winter clothes to sell.", ("scarf", "scarves"), ("knit", "knitted"), "sell", (8, 25),
      "months", ("box", "boxes", 2, 8), ("label", "labeling", 2, 6), False,
      ["{N} owns {n} balls of spare yarn.", "{N}'s knitting club has {n} members."], ["yarn", "new needles"]),
    F("candles", "{N} makes candles in a garage workshop.", ("candle", "candles"), ("make", "made"), "sell", (3, 12),
      "weeks", ("crate", "crates", 4, 20), ("wrap", "wrapping", 2, 8), True,
      ["The workshop has {n} shelves.", "Each candle is {n} centimeters tall."], ["wax", "wicks"]),
    F("jam", "{N} makes jam from garden fruit.", ("jar of jam", "jars of jam"), ("fill", "filled"), "sell", (3, 9),
      "weeks", ("case", "cases", 4, 24), ("label", "labeling", 1, 4), True,
      ["{N}'s garden has {n} berry bushes.", "The jam stall is {n} steps from the gate."], ["sugar", "new lids"]),
    F("fishing", "{N} is a fisher on a small boat.", ("fish", "fish"), ("catch", "caught"), "sell", (4, 10), "days",
      ("cooler", "coolers", 3, 15), ("clean", "cleaning", 2, 6), False,
      ["The boat is {n} feet long.", "The harbor has {n} docks."], ["bait", "a new net"]),
    F("lemonade", "{N} runs a lemonade stand.", ("cup of lemonade", "cups of lemonade"), ("pour", "poured"), "sell",
      (1, 3), "days", ("pitcher", "pitchers", 6, 16), None, False,
      ["The stand is {n} feet from the corner.", "{N} has {n} lemon trees at home."], ["lemons", "paper cups"]),
    F("bikes", "{N} repairs bikes in a small shop.", ("bike", "bikes"), ("repair", "repaired"), "paid", (10, 30),
      "weeks", ("rack", "racks", 2, 10), ("test", "testing", 5, 20), False,
      ["The shop has {n} tools on the wall.", "{N} rides {n} miles every weekend."], ["spare tires", "chain oil"]),
    F("carwash", "{N} washes cars to earn money.", ("car", "cars"), ("wash", "washed"), "paid", (5, 20), "days",
      None, ("dry", "drying", 5, 15), False,
      ["The car wash is on a street with {n} shops.", "{N}'s hose is {n} feet long."], ["soap", "sponges"]),
    F("posters", "{N} runs a print shop.", ("poster", "posters"), ("print", "printed"), "sell", (1, 5), "days",
      ("bundle", "bundles", 5, 50), ("trim", "trimming", 1, 3), False,
      ["The printer can hold {n} sheets of paper.", "The print shop is open {n} hours a week."], ["ink", "paper"]),
    F("bricks", "{N} is a bricklayer.", ("brick", "bricks"), ("lay", "laid"), "paid", (1, 3), "days",
      ("pallet", "pallets", 10, 50), None, False,
      ["The wall is {n} feet long.", "{N}'s truck is {n} years old."], ["a new trowel", "work boots"]),
    F("delivery", "{N} delivers packages around town.", ("package", "packages"), ("deliver", "delivered"), "paid",
      (2, 5), "days", ("bag", "bags", 4, 20), ("scan", "scanning", 1, 3), False,
      ["The delivery van has {n} seats.", "The town has {n} streets."], ["gas", "a new phone charger"]),
    F("sewing", "{N} sews dresses for a boutique.", ("dress", "dresses"), ("sew", "sewed"), "sell", (15, 40),
      "months", ("box", "boxes", 2, 6), ("iron", "ironing", 3, 10), False,
      ["The boutique has {n} mirrors.", "{N}'s sewing machine is {n} years old."], ["fabric", "thread"]),
    F("origami", "{N} folds paper cranes for a craft fair.", ("paper crane", "paper cranes"), ("fold", "folded"),
      "sell", (1, 3), "days", ("string", "strings", 5, 25), ("fold", "folding", 1, 4), False,
      ["The craft fair has {n} tables.", "{N} has {n} sheets of colored paper left."], ["paper", "string"]),
    F("spoons", "{N} carves wooden spoons.", ("wooden spoon", "wooden spoons"), ("carve", "carved"), "sell", (4, 12),
      "months", ("crate", "crates", 4, 12), ("sand", "sanding", 5, 15), False,
      ["{N}'s workshop has {n} chisels.", "The wood comes from a forest {n} miles away."], ["wood", "oil"]),
    F("pumpkins", "{N} grows pumpkins on a farm.", ("pumpkin", "pumpkins"), ("harvest", "harvested"), "sell", (3, 8),
      "weeks", ("wagon", "wagons", 5, 30), ("clean", "cleaning", 1, 4), False,
      ["The farm has {n} acres.", "The farm's barn is {n} feet tall."], ["seeds", "fertilizer"]),
    F("cards", "{N} writes greeting cards by hand.", ("greeting card", "greeting cards"), ("write", "wrote"), "sell",
      (2, 5), "days", ("box", "boxes", 5, 25), ("decorate", "decorating", 2, 8), False,
      ["{N} owns {n} colored pens.", "The card shop is {n} blocks away."], ["envelopes", "stamps"]),
    F("cookies", "{N} bakes cookies for the school fair.", ("cookie", "cookies"), ("bake", "baked"), "sell", (1, 2),
      "days", ("tin", "tins", 6, 30), ("ice", "icing", 1, 3), True,
      ["The school fair has {n} stalls.", "{N}'s class has {n} students."], ["chocolate chips", "butter"]),
    F("tomatoes", "{N} grows tomatoes in a greenhouse.", ("tomato", "tomatoes"), ("pick", "picked"), "sell", (1, 2),
      "weeks", ("crate", "crates", 10, 30), None, True,
      ["The greenhouse is {n} meters long.", "{N} waters the plants {n} times a week."], ["seedlings", "a hose"]),
    F("eggs", "{N} keeps chickens on a farm.", ("egg", "eggs"), ("collect", "collected"), "sell", (1, 2), "days",
      ("carton", "cartons", 6, 12), None, True,
      ["The farm has {n} chickens.", "The chicken coop is {n} feet wide."], ["chicken feed", "straw"]),
    F("library", "{N} volunteers at the library and is paid for shelving.", ("book", "books"), ("shelve", "shelved"),
      "paid", (1, 2), "days", ("cart", "carts", 10, 40), None, False,
      ["The library has {n} reading tables.", "The library opened {n} years ago."], ["a book bag", "lunch"]),
    F("sandwiches", "{N} runs a sandwich cart.", ("sandwich", "sandwiches"), ("make", "made"), "sell", (3, 8), "days",
      ("platter", "platters", 4, 12), ("wrap", "wrapping", 1, 3), False,
      ["The cart has {n} kinds of bread.", "The cart parks {n} meters from the station."], ["bread", "cheese"]),
    F("pottery", "{N} makes pottery bowls.", ("bowl", "bowls"), ("shape", "shaped"), "sell", (6, 20), "weeks",
      ("box", "boxes", 2, 10), ("glaze", "glazing", 4, 12), False,
      ["The kiln holds {n} pieces at a time.", "{N}'s studio has {n} windows."], ["clay", "glaze"]),
    F("flowers", "{N} arranges flowers at a florist.", ("bouquet", "bouquets"), ("arrange", "arranged"), "sell",
      (8, 20), "days", ("bucket", "buckets", 3, 12), ("wrap", "wrapping", 2, 6), False,
      ["The florist has {n} kinds of roses.", "The shop is {n} years old."], ["ribbon", "vases"]),
    F("soap", "{N} makes soap at home.", ("bar of soap", "bars of soap"), ("cut", "cut"), "sell", (3, 7), "weeks",
      ("box", "boxes", 4, 12), ("wrap", "wrapping", 1, 3), False,
      ["{N} owns {n} soap molds.", "Each bar weighs {n}0 grams."], ["lye", "essential oils"]),
    F("honey", "{N} keeps bees.", ("jar of honey", "jars of honey"), ("fill", "filled"), "sell", (5, 12), "months",
      ("crate", "crates", 6, 24), ("label", "labeling", 1, 3), True,
      ["{N} has {n} beehives.", "The hives are {n} meters from the house."], ["new jars", "a bee suit"]),
    F("bracelets", "{N} makes beaded bracelets.", ("bracelet", "bracelets"), ("make", "made"), "sell", (3, 10),
      "days", ("bag", "bags", 3, 10), ("pack", "packing", 1, 3), False,
      ["{N} has {n} colors of beads.", "{N}'s best friend is {n} years old."], ["beads", "wire"]),
    F("kites", "{N} builds kites.", ("kite", "kites"), ("build", "built"), "sell", (6, 15), "weeks",
      ("box", "boxes", 2, 8), ("test", "testing", 5, 15), False,
      ["The kite string is {n} meters long.", "The beach is {n} miles away."], ["string", "fabric"]),
    F("pies", "{N} bakes pies for a diner.", ("pie", "pies"), ("bake", "baked"), "sell", (8, 15), "days",
      ("rack", "racks", 4, 12), ("slice", "slicing", 2, 5), False,
      ["The diner has {n} booths.", "The diner has {n} regular customers."], ["apples", "pie dishes"]),
    F("frames", "{N} assembles picture frames.", ("picture frame", "picture frames"), ("assemble", "assembled"),
      "sell", (7, 20), "weeks", ("carton", "cartons", 4, 12), ("polish", "polishing", 2, 6), False,
      ["The workshop has {n} clamps.", "Each frame holds a photo {n} inches wide."], ["glass", "glue"]),
    F("shirts", "{N} prints T-shirts.", ("T-shirt", "T-shirts"), ("print", "printed"), "sell", (6, 15), "days",
      ("box", "boxes", 10, 30), ("fold", "folding", 1, 3), False,
      ["The shop has {n} shirt designs.", "The printer is {n} years old."], ["blank shirts", "ink"]),
    F("birdhouses", "{N} builds birdhouses.", ("birdhouse", "birdhouses"), ("build", "built"), "sell", (10, 25),
      "months", ("crate", "crates", 2, 6), ("paint", "painting", 10, 30), False,
      ["{N}'s yard has {n} trees.", "{N} has seen {n} kinds of birds."], ["wood", "nails"]),
    F("dogs", "{N} walks dogs for neighbors.", ("dog", "dogs"), ("walk", "walked"), "paid", (5, 15), "days",
      None, ("brush", "brushing", 5, 15), False,
      ["The park is {n} blocks away.", "{N}'s own dog is {n} years old."], ["leashes", "dog treats"]),
    F("lawns", "{N} mows lawns in the summer.", ("lawn", "lawns"), ("mow", "mowed"), "paid", (15, 40), "weeks",
      None, ("edge", "edging", 10, 30), False,
      ["The mower is {n} years old.", "{N}'s street has {n} houses."], ["gas", "a new blade"]),
    F("tickets", "{N} sells raffle tickets for a club.", ("raffle ticket", "raffle tickets"), ("sell", "sold"),
      "sell", (1, 5), "days", ("book", "books", 10, 50), None, False,
      ["The club has {n} members.", "The raffle prize is a bike with {n} gears."], ["snacks", "a poster"]),
    F("notebooks", "{N} is shopping for school supplies.", ("notebook", "notebooks"), ("buy", "bought"), "buy",
      (2, 6), "weeks", ("pack", "packs", 2, 10), ("label", "labeling", 1, 3), False,
      ["The store has {n} aisles.", "{N}'s school is {n} blocks away."]),
    F("seedlings", "{N} is planting a vegetable garden.", ("seedling", "seedlings"), ("buy", "bought"), "buy", (2, 6),
      "weeks", ("tray", "trays", 6, 24), ("plant", "planting", 2, 8), False,
      ["The garden bed is {n} feet long.", "The garden center is {n} miles away."]),
    F("balloons", "{N} is getting ready for a party.", ("balloon", "balloons"), ("buy", "bought"), "buy", (1, 3),
      "days", ("bag", "bags", 10, 50), ("blow up", "blowing up", 1, 2), False,
      ["The party is for {n} guests.", "The party room is {n} feet wide."]),
    F("comics", "{N} collects comic books.", ("comic book", "comic books"), ("buy", "bought"), "buy", (3, 9),
      "months", ("box", "boxes", 10, 30), ("read", "reading", 10, 30), False,
      ["{N}'s shelf has {n} sections.", "The comic shop is {n} blocks away."]),
    F("soup", "{N} stocks a food pantry.", ("can of soup", "cans of soup"), ("buy", "bought"), "buy", (1, 3), "weeks",
      ("case", "cases", 12, 24), None, True,
      ["The pantry serves {n} families.", "The pantry has {n} shelves."]),
    F("coffee", "{N} runs a coffee cart.", ("cup of coffee", "cups of coffee"), ("brew", "brewed"), "sell", (2, 5),
      "days", ("carrier", "carriers", 2, 8), ("pour", "pouring", 1, 3), False,
      ["The cart has {n} kinds of beans.", "The cart has been in business for {n} months."], ["milk", "coffee beans"]),
    F("bottles", "{N} collects bottles for recycling.", ("bottle", "bottles"), ("collect", "collected"), "paid",
      (1, 2), "weeks", ("bag", "bags", 10, 50), None, False,
      ["The recycling center is {n} miles away.", "{N}'s bike has {n} gears."], ["a snack", "a bus ticket"]),
    F("paintings", "{N} paints landscapes.", ("painting", "paintings"), ("paint", "painted"), "sell", (20, 60),
      "months", ("crate", "crates", 2, 6), ("frame", "framing", 10, 30), False,
      ["{N} owns {n} brushes.", "The gallery is {n} miles away."], ["canvas", "paint"]),
    F("cupcakes", "{N} frosts cupcakes at a cafe.", ("cupcake", "cupcakes"), ("frost", "frosted"), "sell", (2, 4),
      "days", ("box", "boxes", 4, 12), ("box", "boxing", 1, 2), True,
      ["The cafe has {n} tables.", "The cafe's sign is {n} feet wide."], ["sprinkles", "frosting"]),
    F("portraits", "{N} draws portraits at a market.", ("portrait", "portraits"), ("draw", "drew"), "sell", (10, 30),
      "weeks", ("folder", "folders", 3, 10), ("sign", "signing", 1, 3), False,
      ["The market has {n} stalls.", "{N} has {n} pencils."], ["sketchbooks", "charcoal"]),
    F("sushi", "{N} makes sushi at a restaurant.", ("sushi roll", "sushi rolls"), ("roll", "rolled"), "sell", (5, 10),
      "days", ("platter", "platters", 4, 12), ("slice", "slicing", 1, 3), False,
      ["The restaurant has {n} tables.", "The fish market is {n} miles away."], ["rice", "seaweed"]),
    F("hats", "{N} crochets hats.", ("hat", "hats"), ("crochet", "crocheted"), "sell", (8, 18), "months",
      ("bag", "bags", 3, 8), ("block", "blocking", 5, 15), False,
      ["{N} has {n} crochet hooks.", "{N}'s cat is {n} years old."], ["yarn", "buttons"]),
]
HELD_OUT = ("sushi", "kites", "lawns", "seedlings", "comics", "paintings")
FRAME_BY_ID = {f["id"]: f for f in FRAMES}
assert len(FRAME_BY_ID) == len(FRAMES) and all(h in FRAME_BY_ID for h in HELD_OUT)

# operation classes: how each named operation computes its value
OPCLS = {"add": "ADD", "bonus": "ADD", "fee": "ADD", "setup": "ADD",
         "sub": "SUB", "spend": "SUB", "spend_each": "SUB", "coupon": "SUB", "give": "SUB",
         "mul": "MUL", "sell": "MUL", "time": "MUL", "rel": "MUL", "sell_box": "MUL", "reltime": "MUL",
         "div": "DIV", "pack": "DIV", "share": "DIV", "split": "DIV", "hours": "DIV",
         "pct": "PCT", "save": "PCT", "discount": "PCTOFF", "sum2": "SUM", "total": "SUM", "jar": "SUM",
         "dozen": "DOZEN"}
FANIN = {"sum2", "total", "jar"}


def apply(cls: str, vals: list, c) -> Fraction | None:
    v = [Fraction(x) for x in vals]
    if cls == "ADD":
        return v[0] + c
    if cls == "SUB":
        return v[0] - c
    if cls == "MUL":
        return v[0] * c
    if cls == "DIV":
        return v[0] / c if c else None
    if cls == "PCT":
        return v[0] * c / 100
    if cls == "PCTOFF":
        return v[0] * (100 - c) / 100
    if cls == "SUM":
        return v[0] + v[1]
    if cls == "DOZEN":
        return Fraction(12 * c)
    raise ValueError(cls)


def as_int(x) -> int | None:
    if x is None or x.denominator != 1 or not 1 <= x <= MAXV:
        return None
    return int(x)


@dataclasses.dataclass
class Qty:
    id: str
    kind: str                      # count | box | money | minutes
    value: int
    np: str                        # noun phrase ("the number of muffins Maria baked on Tuesday")
    ask: str                       # question asking for it
    ref: str = ""                  # reference inside later sentences
    stated: bool = False           # a premise fact, not an operation
    op: str | None = None
    inputs: list = dataclasses.field(default_factory=list)
    const: int | None = None
    sentence: str | None = None    # None for implicit (final-question) operations
    mod: str | None = None
    ask_hours: str | None = None

    def fmt(self, x: int) -> str:
        return f"${x}" if self.kind == "money" else (f"{x} minutes" if self.kind == "minutes" else str(x))


def cap(s: str) -> str:
    return s[0].upper() + s[1:]


def kword(k: int) -> str:
    return {2: "twice", 3: "three times", 4: "four times", 5: "five times"}.get(k, f"{k} times")


class Story:
    """One problem under construction: quantities in creation order, premise sentences, used names."""

    def __init__(self, frame: dict, rng: random.Random):
        self.f, self.rng = frame, rng
        self.name, g = rng.choice(NAMES)
        self.pron, self.poss = ("she", "her") if g == "f" else ("he", "his")
        self.qs: list[Qty] = []
        self.mods = list(MODSETS[frame["mods"]])
        self.next_mod = 0
        self.used: set = set()
        self.extra_numbers: list[int] = []     # numbers written as words (twice, half, dozen, ...)

    # ---- helpers
    def r(self, a: int, b: int) -> int:
        return self.rng.randint(a, b)

    def take_mod(self) -> str | None:
        if self.next_mod >= len(self.mods):
            return None
        self.next_mod += 1
        return self.mods[self.next_mod - 1]

    def mods_left(self) -> int:
        return len(self.mods) - self.next_mod

    def new(self, **kw) -> Qty:
        q = Qty(id=f"q{len(self.qs) + 1}", **kw)
        self.qs.append(q)
        return q

    def by(self, i: str) -> Qty:
        return next(q for q in self.qs if q.id == i)

    def ops(self) -> list[Qty]:
        return [q for q in self.qs if q.op]

    def count_names(self, mod: str) -> dict:
        N, (sg, pl), (base, past) = self.name, self.f["item"], self.f["verb"]
        return {"np": f"the number of {pl} {N} {past} {mod}", "ask": f"How many {pl} did {N} {base} {mod}?",
                "ref": f"the {pl} {self.pron} {past} {mod}", "mod": mod}

    # ---- starting quantity
    def start(self, dozen: bool) -> Qty:
        N, (sg, pl), (base, past) = self.name, self.f["item"], self.f["verb"]
        mod = self.take_mod()
        if dozen:
            n = self.r(2, 9)
            s = self.rng.choice([f"{cap(mod)}, {N} {past} {n} dozen {pl}.", f"{N} {past} {n} dozen {pl} {mod}."])
            self.extra_numbers.append(12)
            return self.new(kind="count", value=12 * n, op="dozen", const=n, sentence=s, **self.count_names(mod))
        return self.stated_count(mod)

    def stated_count(self, mod: str) -> Qty:
        N, (sg, pl), (base, past) = self.name, self.f["item"], self.f["verb"]
        n = self.r(5, 99)
        s = self.rng.choice([f"{cap(mod)}, {N} {past} {n} {pl}.", f"{N} {past} {n} {pl} {mod}."])
        return self.new(kind="count", value=n, stated=True, sentence=s, **self.count_names(mod))

    # ---- count operations
    def count_op(self, q: Qty, op: str) -> Qty | None:
        N, (sg, pl), (base, past) = self.name, self.f["item"], self.f["verb"]
        if self.mods_left() < 1:
            return None
        m1, rng = q.mod, self.rng
        if op == "add":
            n = self.r(2, 60)
            v, c = q.value + n, n
            words = f"{n} more {pl} than {m1}"
        elif op == "sub":
            if q.value < 4:
                return None
            n = self.r(2, min(60, q.value - 1))
            v, c = q.value - n, n
            words = f"{n} fewer {pl} than {m1}"
        elif op == "mul":
            c = self.r(2, 5)
            v = q.value * c
            words = f"{kword(c)} as many {pl} as {m1}"
            if c == 2:
                self.extra_numbers.append(2)
        elif op == "div":
            ks = [k for k in (2, 3, 4) if q.value % k == 0 and q.value // k >= 2]
            if not ks:
                return None
            c = rng.choice(ks)
            v = q.value // c
            words = f"{ {2: 'half', 3: 'a third', 4: 'a quarter'}[c]} as many {pl} as {m1}"
            self.extra_numbers.append(c)
        elif op == "pct":
            ps = [p for p in (20, 25, 40, 50, 60, 75, 80) if q.value * p % 100 == 0 and q.value * p // 100 >= 2]
            if not ps:
                return None
            c = rng.choice(ps)
            v = q.value * c // 100
            words = f"{c}% as many {pl} as {m1}"
        else:
            raise ValueError(op)
        if not 1 <= v <= MAXV:
            return None
        m2 = self.take_mod()
        cut = " than " if " than " in words else " as "
        head, _, tail = words.rpartition(cut)
        s = rng.choice([f"{cap(m2)}, {N} {past} {words}.", f"{N} {past} {head} {m2}{cut}{tail}."])
        return self.new(kind="count", value=v, op=op, inputs=[q.id], const=c, sentence=s, **self.count_names(m2))

    def sum2(self, q: Qty) -> Qty | None:
        N, (sg, pl), (base, past) = self.name, self.f["item"], self.f["verb"]
        earlier = [x for x in self.qs if x.kind == "count" and x is not q]
        fresh = not earlier or self.rng.random() < 0.4
        if self.mods_left() < (2 if fresh else 1):
            return None
        other = self.stated_count(self.take_mod()) if fresh else self.rng.choice(earlier)
        v = q.value + other.value
        if v > MAXV:
            return None
        a, b = sorted([q, other], key=lambda x: self.mods.index(x.mod))
        m3 = self.take_mod()
        s = self.rng.choice([f"{cap(m3)}, {N} {past} as many {pl} as {a.mod} and {b.mod} combined.",
                             f"{N} {past} as many {pl} {m3} as {self.pron} did {a.mod} and {b.mod} put together."])
        return self.new(kind="count", value=v, op="sum2", inputs=[q.id, other.id], sentence=s, **self.count_names(m3))

    def total(self, q: Qty) -> Qty | None:
        N, (sg, pl), (base, past) = self.name, self.f["item"], self.f["verb"]
        earlier = [x for x in self.qs if x.kind == "count" and x is not q]
        if not earlier:
            return None
        other = self.rng.choice(earlier)
        if q.value + other.value > MAXV:
            return None
        a, b = sorted([q, other], key=lambda x: self.mods.index(x.mod))
        return self.new(kind="count", value=q.value + other.value, op="total", inputs=[q.id, other.id],
                        np=f"the total number of {pl} {N} {past} {a.mod} and {b.mod}",
                        ask=f"How many {pl} did {N} {base} {a.mod} and {b.mod} altogether?")

    # ---- conversions from a count
    def money_names(self, q: Qty, via: str = "") -> dict:
        N, (sg, pl), (base, past) = self.name, self.f["item"], self.f["verb"]
        what = via or f"the {pl} {self.pron} {past} {q.mod}"
        if self.f["mode"] == "buy":
            return {"np": f"the money {N} spent on {what}", "ask": f"How much money did {N} spend on {what}?",
                    "ref": f"the money {N} spent on {what}"}
        verb = "makes" if self.f["mode"] == "sell" else "earns"
        return {"np": f"the money {N} {verb} from {what}",
                "ask": f"How much money does {N} {verb[:-1]} from {what}?",
                "ref": f"the money {N} {verb} from {what}"}

    def sell(self, q: Qty) -> Qty | None:
        N, (sg, pl), (base, past) = self.name, self.f["item"], self.f["verb"]
        p = self.r(*self.f["price"])
        v = q.value * p
        if v > MAXV or p < 1:
            return None
        if p == 1 and self.rng.random() < 0.7:          # $1 prices make many trivial items
            return None
        mode, th = self.f["mode"], f"the {pl} {self.pron} {past} {q.mod}"
        s = self.rng.choice({"sell": [f"{N} sells each of {th} for ${p}.", f"Each of {th} sells for ${p}."],
                             "paid": [f"{N} is paid ${p} for each of {th}.", f"For each of {th}, {N} earns ${p}."],
                             "buy": [f"Each of {th} cost ${p}.", f"{N} paid ${p} for each of {th}."]}[mode])
        return self.new(kind="money", value=v, op="sell", inputs=[q.id], const=p, sentence=s, **self.money_names(q))

    def pack(self, q: Qty) -> Qty | None:
        c = self.f["container"]
        if not c:
            return None
        N, (sg, pl), (base, past) = self.name, self.f["item"], self.f["verb"]
        csg, cpl, lo, hi = c
        rs = [r for r in range(max(2, lo), hi + 1) if q.value % r == 0 and q.value // r >= 2]
        if not rs:
            return None
        r = self.rng.choice(rs)
        th = f"the {pl} {self.pron} {past} {q.mod}"
        if self.f["mode"] == "buy":
            s = self.rng.choice([f"The {pl} {N} {past} {q.mod} came in {cpl} of {r}.",
                                 f"{cap(th.replace(self.pron, N, 1))} were sold in {cpl} of {r} each."])
            names = {"np": f"the number of {cpl} of {pl} {N} {past} {q.mod}",
                     "ask": f"How many {cpl} of {pl} did {N} {base} {q.mod}?",
                     "ref": f"the {cpl} of {pl} {N} {past} {q.mod}"}
        else:
            s = self.rng.choice([f"{N} packs {th} into {cpl} of {r}.",
                                 f"{N} puts {th} into {cpl} that hold {r} each."])
            names = {"np": f"the number of {cpl} {N} fills with {th}",
                     "ask": f"How many {cpl} does {N} fill with {th}?",
                     "ref": f"the {cpl} {N} filled with {th}"}
        return self.new(kind="box", value=q.value // r, op="pack", inputs=[q.id], const=r, sentence=s, **names)

    def time(self, q: Qty) -> Qty | None:
        tv = self.f["tverb"]
        if not tv:
            return None
        N, (sg, pl), (base, past) = self.name, self.f["item"], self.f["verb"]
        tb, ting, lo, hi = tv
        t = self.r(max(lo, 2), max(hi, 3))
        v = q.value * t
        if v > MAXV:
            return None
        th = f"the {pl} {self.pron} {past} {q.mod}"
        s = self.rng.choice([f"It takes {N} {t} minutes to {tb} each of {th}.",
                             f"{N} spends {t} minutes {ting} each of {th}."])
        return self.new(kind="minutes", value=v, op="time", inputs=[q.id], const=t, sentence=s,
                        np=f"the number of minutes {N} spends {ting} {th}",
                        ask=f"How many minutes does {N} spend {ting} {th}?",
                        ref=f"the time {N} spends {ting} {th}",
                        ask_hours=f"How many hours does {N} spend {ting} {th}?")

    def sell2(self, q: Qty) -> list[Qty] | None:
        """Two batches sold at their own prices, then put together (3 operations, the last a fan-in)."""
        earlier = [x for x in self.qs if x.kind == "count" and x is not q]
        if not earlier:
            return None
        other = self.rng.choice(earlier)
        a = self.sell(q)
        if a is None:
            return None
        b = self.sell(other)
        if b is None:
            self.qs.remove(a)
            return None
        v = a.value + b.value
        if v > MAXV:
            self.qs.remove(a); self.qs.remove(b)
            return None
        N = self.name
        if self.f["mode"] == "buy":
            s = f"{N} adds {self.own(b.ref)} and {self.own(a.ref)} to get {self.poss} total bill."
            names = {"np": f"{N}'s total bill", "ask": f"How much is {N}'s total bill?", "ref": f"{N}'s total bill"}
        else:
            s = self.rng.choice([f"{N} puts {self.own(b.ref)} and {self.own(a.ref)} into one jar.",
                                 f"{N} keeps {self.own(b.ref)} and {self.own(a.ref)} together in one jar."])
            names = {"np": f"the money in {N}'s jar", "ask": f"How much money is in {N}'s jar?",
                     "ref": f"the money in {N}'s jar"}
        return [a, b, self.new(kind="money", value=v, op="jar", inputs=[a.id, b.id], sentence=s, **names)]

    # ---- money operations
    def pick_unused(self, pool: list) -> str | None:
        free = [x for x in pool if x not in self.used]
        if not free:
            return None
        x = self.rng.choice(free)
        self.used.add(x)
        return x

    def own(self, ref: str) -> str:
        """A reference inside a sentence whose subject is the protagonist: "the money Maria has left" -> "the money
        she has left" (possessives such as "Maria's jar" stay)."""
        return re.sub(rf"\b{self.name}\b(?!'s)", self.pron, ref, count=1)

    def money_op(self, q: Qty, op: str) -> Qty | None:
        N, R, rng = self.name, self.own(q.ref), self.rng
        if op in ("spend", "coupon", "spend_each"):
            if q.value < 6:
                return None
            n = self.r(2, min(99, q.value - 2))
            v = q.value - n
            if op == "spend":
                thing = self.pick_unused(self.f["spend"] + SPEND)
                s = rng.choice([f"{N} spends ${n} of {R} on {thing}.", f"{N} uses ${n} of {R} to buy {thing}."])
                names = {"np": f"the money {N} has left after buying {thing}",
                         "ask": f"How much money does {N} have left after buying {thing}?",
                         "ref": f"the money {N} has left after buying {thing}"}
            elif op == "spend_each":
                grp = self.group
                thing = self.pick_unused(SPEND)
                s = f"Each of the {grp} spends ${n} of {R} on {thing}."
                names = {"np": f"the money each of the {grp} has left", "ask": f"How much money does each of the {grp} have left?",
                         "ref": f"the money each of the {grp} has left"}
            else:
                s = rng.choice([f"{N} uses a ${n} coupon on {R}.", f"A ${n} coupon is taken off {R}."])
                names = {"np": f"the amount {N} pays after the coupon", "ask": f"How much does {N} pay after using the coupon?",
                         "ref": f"the amount {N} pays after the coupon"}
            return self.new(kind="money", value=v, op=op, inputs=[q.id], const=n, sentence=s, **names)
        if op in ("bonus", "fee"):
            n = self.r(2, 60)
            v = q.value + n
            if op == "bonus":
                w = self.pick_unused(BONUS)
                s = rng.choice([f"{N} also gets a ${n} {w}, which {self.pron} adds to {R}.",
                                f"{N} adds a ${n} {w} to {R}."])
                names = {"np": f"the money {N} has after adding the {w}", "ask": f"How much money does {N} have after adding the {w}?",
                         "ref": f"the money {N} has after adding the {w}"}
            else:
                s = rng.choice([f"{N} also pays a ${n} delivery fee on top of {R}.", f"A ${n} delivery fee is added to {R}."])
                names = {"np": f"the total {N} pays including the delivery fee",
                         "ask": f"How much does {N} pay in total including the delivery fee?",
                         "ref": f"the total {N} pays including the delivery fee"}
            return self.new(kind="money", value=v, op=op, inputs=[q.id], const=n, sentence=s, **names)
        if op in ("share", "split"):
            ks = [k for k in range(2, 7) if q.value % k == 0 and q.value // k >= 2]
            if not ks:
                return None
            k = rng.choice(ks)
            if op == "share":
                self.group = self.pick_unused(GROUPS)
                s = f"{N} splits {R} equally among {k} {self.group}."
                names = {"np": f"the amount each of the {self.group} gets", "ask": f"How much money does each of the {self.group} get?",
                         "ref": f"the share each of the {self.group} gets"}
            else:
                who = "a friend" if k == 2 else f"{k - 1} friends"
                s = f"{N} and {who} split {R} equally."
                names = {"np": "the amount each person pays", "ask": "How much does each person pay?",
                         "ref": "the amount each person pays"}
            return self.new(kind="money", value=q.value // k, op=op, inputs=[q.id], const=k, sentence=s, **names)
        if op in ("save", "discount"):
            ps = [p for p in (10, 20, 25, 30, 40, 50, 75) if q.value * p % 100 == 0 and q.value * p // 100 >= 2
                  and q.value * (100 - p) // 100 >= 2]
            if not ps:
                return None
            p = rng.choice(ps)
            if op == "save":
                s = rng.choice([f"{N} puts {p}% of {R} into savings.", f"{N} saves {p}% of {R}."])
                names = {"np": f"the money {N} puts into savings", "ask": f"How much money does {N} put into savings?",
                         "ref": f"the money {N} saved"}
                v = q.value * p // 100
            else:
                s = f"{N} gets a {p}% discount on {R}."
                names = {"np": f"the amount {N} pays after the discount", "ask": f"How much does {N} pay after the discount?",
                         "ref": f"the amount {N} pays after the discount"}
                v = q.value * (100 - p) // 100
            return self.new(kind="money", value=v, op=op, inputs=[q.id], const=p, sentence=s, **names)
        if op == "rel":
            k = self.r(2, 4)
            if q.value * k > MAXV:
                return None
            rel = self.pick_unused(RELS)
            if k == 2:
                self.extra_numbers.append(2)
            verb = {"sell": "makes", "paid": "earns", "buy": "spends"}[self.f["mode"]]
            s = f"{N}'s {rel} {verb} {kword(k)} as much money as {R}."
            names = {"np": f"the money {N}'s {rel} {verb}", "ask": f"How much money does {N}'s {rel} {verb[:-1]}?",
                     "ref": f"the money {N}'s {rel} {verb}"}
            return self.new(kind="money", value=q.value * k, op="rel", inputs=[q.id], const=k, sentence=s, **names)
        raise ValueError(op)

    # ---- containers and minutes
    def box_op(self, q: Qty, op: str) -> Qty | None:
        N, R = self.name, self.own(q.ref)
        csg, cpl = self.f["container"][:2]
        if op == "sell_box":
            lo, hi = self.f["price"]
            p = self.r(max(3, lo * 2), max(8, hi * 4))
            v = q.value * p
            if v > MAXV:
                return None
            if self.f["mode"] == "buy":
                s = f"Each of {R} cost ${p}."
                names = {"np": f"the money {N} spent on the {cpl}", "ask": f"How much money did {N} spend on the {cpl}?",
                         "ref": f"the money {N} spent on the {cpl}"}
            else:
                s = self.rng.choice([f"{N} sells each of {R} for ${p}.", f"Each of {R} sells for ${p}."])
                names = {"np": f"the money {N} makes from selling the {cpl}", "ask": f"How much money does {N} make from selling the {cpl}?",
                         "ref": f"the money {N} makes from selling the {cpl}"}
            return self.new(kind="money", value=v, op="sell_box", inputs=[q.id], const=p, sentence=s, **names)
        if op == "give":
            if q.value < 3:
                return None
            n = self.r(1, q.value - 2) if q.value < 12 else self.r(2, min(40, q.value - 2))
            who = self.pick_unused(RECIPIENTS)
            verb = "gives" if self.f["mode"] != "buy" else "donates"
            s = f"{N} {verb} {n} of {R} to {who}."
            names = {"np": f"the number of {cpl} {N} has left after giving some to {who}",
                     "ask": f"How many {cpl} does {N} have left after giving some to {who}?",
                     "ref": f"the {cpl} {N} has left"}
            return self.new(kind="box", value=q.value - n, op="give", inputs=[q.id], const=n, sentence=s, **names)
        raise ValueError(op)

    def minutes_op(self, q: Qty, op: str) -> Qty | None:
        N, R = self.name, self.own(q.ref)
        if op == "setup":
            n = self.r(5, 60)
            if self.rng.random() < 0.5 and 5 <= 60 - q.value % 60 <= 60:     # often rounds to whole hours
                n = 60 - q.value % 60
            s = self.rng.choice([f"{N} also spends {n} minutes cleaning up afterwards, on top of {R}.",
                                 f"Cleaning up adds {n} minutes to {R}."])
            v = q.value + n
            ting = self.f["tverb"][1]
            return self.new(kind="minutes", value=v, op="setup", inputs=[q.id], const=n, sentence=s,
                            np=f"the total number of minutes {N} spends {ting} and cleaning up",
                            ask=f"How many minutes does {N} spend {ting} and cleaning up in total?",
                            ref=f"the total time {N} spends {ting} and cleaning up",
                            ask_hours=f"How many hours does {N} spend {ting} and cleaning up in total?")
        if op == "reltime":
            k = self.r(2, 4)
            if q.value * k > MAXV:
                return None
            rel = self.pick_unused(RELS)
            if k == 2:
                self.extra_numbers.append(2)
            s = f"{N}'s {rel} would need {kword(k)} as long as {R}."
            return self.new(kind="minutes", value=q.value * k, op="reltime", inputs=[q.id], const=k, sentence=s,
                            np=f"the number of minutes {N}'s {rel} would need", ask=f"How many minutes would {N}'s {rel} need?",
                            ref=f"the time {N}'s {rel} would need", ask_hours=f"How many hours would {N}'s {rel} need?")
        if op == "hours":
            if q.value % 60 or q.value < 120 or not q.ask_hours:
                return None
            self.extra_numbers.append(60)
            return self.new(kind="hours", value=q.value // 60, op="hours", inputs=[q.id], const=60,
                            np=q.np.replace("the number of minutes", "the number of hours").replace(
                                "the total number of minutes", "the total number of hours"), ask=q.ask_hours)
        raise ValueError(op)


EARN_OPS = {"spend": 1.2, "bonus": 0.8, "share": 0.8, "save": 0.6, "rel": 0.6}
BUY_OPS = {"coupon": 1.0, "fee": 1.0, "split": 0.8, "discount": 0.8, "rel": 0.6}
ONCE = {"share", "save", "rel", "split", "discount", "give", "setup", "reltime", "bonus", "fee", "coupon",
        "spend_each", "sell_box"}


def candidate_ops(st: Story, q: Qty, remaining: int, done: set) -> dict:
    """Operation weights given the latest quantity and the steps left."""
    w: dict = {}
    if q.kind == "count":
        if st.mods_left() >= 1:
            w.update({"add": 1.0, "sub": 1.0, "mul": 1.0, "div": 0.7, "pct": 0.5})
            if st.ops():
                w["sum2"] = 3.0
        w["sell"] = 1.2
        if st.f["container"]:
            w["pack"] = 0.6
        if st.f["tverb"]:
            w["time"] = 0.6
        if remaining >= 3 and any(x.kind == "count" for x in st.qs if x is not q):
            w["sell2"] = 1.2
        if remaining == 1 and any(x.kind == "count" for x in st.qs if x is not q):
            w["total"] = 1.0
    elif q.kind == "money":
        if q.op == "share":
            w = {"spend_each": 1.0}
        else:
            w = dict(EARN_OPS if st.f["mode"] != "buy" else BUY_OPS)
            if st.f["mode"] != "buy" and len(set(st.f["spend"] + SPEND) - st.used) == 0:
                w.pop("spend")
    elif q.kind == "box":
        w = {"sell_box": 1.0, "give": 0.8}
    elif q.kind == "minutes":
        w = {"setup": 0.8, "reltime": 0.6}
        if remaining == 1:
            w["hours"] = 1.2
    if remaining > 1:                     # a relative's money or time ends the story (nothing more happens to it)
        w.pop("rel", None)
        w.pop("reltime", None)
    return {k: v for k, v in w.items() if not (k in ONCE and k in done)}


def run_op(st: Story, q: Qty, op: str) -> list[Qty] | None:
    if op in ("add", "sub", "mul", "div", "pct"):
        r = st.count_op(q, op)
    elif op == "sum2":
        r = st.sum2(q)
    elif op == "total":
        r = st.total(q)
    elif op in ("sell", "pack", "time"):
        r = getattr(st, op)(q)
    elif op == "sell2":
        return st.sell2(q)
    elif op in ("spend", "bonus", "share", "save", "rel", "coupon", "fee", "split", "discount", "spend_each"):
        r = st.money_op(q, op)
    elif op in ("sell_box", "give"):
        r = st.box_op(q, op)
    else:
        r = st.minutes_op(q, op)
    return None if r is None else [r]


@dataclasses.dataclass
class Problem:
    frame: str
    depth: int
    story: Story
    shuffled: bool
    noise: list
    body: str            # intro + premises + noise, no question
    question: str        # the final question
    stated: list         # numbers a reader can see (digits and number words)

    @property
    def answer(self) -> Qty:
        return self.story.qs[-1]


def generate(frame: dict, depth: int, rng: random.Random, shuffled: bool | None = None, tries: int = 400) -> Problem:
    for _ in range(tries):
        st = Story(frame, rng)
        q = st.start(dozen=frame["dozen"] and rng.random() < 0.15)
        done: set = set()
        ok = True
        while len(st.ops()) < depth:
            remaining = depth - len(st.ops())
            w = candidate_ops(st, q, remaining, done)
            if q.kind == "count" and remaining == 1 and "total" in w and rng.random() < 0.7:
                w = {"total": 1.0}
            res = None
            for _ in range(8):
                if not w:
                    break
                op = rng.choices(list(w), weights=list(w.values()))[0]
                if op == "sell2" and remaining < 3:
                    w.pop(op); continue
                res = run_op(st, q, op)
                if res:
                    done.add(op)
                    break
                w.pop(op)
            if not res:
                ok = False
                break
            q = res[-1]
            if q.op in ("total", "hours") and len(st.ops()) < depth:       # implicit ops end a problem
                ok = False
                break
        if not ok or len(st.ops()) != depth:
            continue
        return render(st, depth, rng, shuffled)
    raise RuntimeError(f"could not generate {frame['id']} depth {depth}")


NUM_WORDS = {"twice": 2, "half": 2, "three": 3, "four": 4, "five": 5, "third": 3, "quarter": 4, "dozen": 12,
             "hours": 60}
NUM_RE = re.compile(r"\d+")


def stated_numbers(text: str) -> list[int]:
    """Numbers a shallow reader can take from the text: digits plus number words (twice, half, a third, a quarter,
    dozen; 60 when hours are asked)."""
    nums = [int(x) for x in NUM_RE.findall(text)]
    low = text.lower()
    nums += [v for w, v in NUM_WORDS.items() if re.search(rf"\b{w}\b", low)]
    return sorted(set(n for n in nums if n > 0))


def render(st: Story, depth: int, rng: random.Random, shuffled: bool | None) -> Problem:
    f = st.f
    intro = f["intro"].format(N=st.name, poss=st.poss)
    prem = [q.sentence for q in st.qs if q.sentence]
    if shuffled is None:
        shuffled = rng.random() < 0.5
    if shuffled:
        order = list(range(len(prem)))
        while len(prem) > 1 and order == sorted(order):
            rng.shuffle(order)
        prem = [prem[i] for i in order]
    others = [n for n, _ in NAMES if n != st.name]
    noise_pool = [t for t in f["noise"]] + GENERIC_NOISE
    n_noise = rng.choice([1, 2])
    noise = []
    for t in rng.sample(noise_pool, k=n_noise):
        noise.append(t.format(N=st.name, n=rng.randint(2, 99)))
    if rng.random() < 0.35:            # a confusable batch made by someone else
        mod = rng.choice(st.mods)
        noise[-1] = f"{rng.choice(others)} {f['verb'][1]} {rng.randint(5, 99)} {f['item'][1]} {mod}."
    for s in noise:
        prem.insert(rng.randint(0, len(prem)), s)
    body = " ".join([intro] + prem)
    question = st.qs[-1].ask
    return Problem(frame=f["id"], depth=depth, story=st, shuffled=shuffled, noise=noise, body=body, question=question,
                   stated=stated_numbers(body + " " + question))


# ---------------------------------------------------------------------------------------------------- execution
def execute(st: Story, slip: tuple | None = None) -> dict | None:
    """Values of every quantity, recomputed from the program; slip = (qid, kind[, arg]) plants one error:
    swap (the inverse operation), off1 (constant +- 1), skip (the step is left out), wrong (another fan-in input)."""
    vals: dict = {}
    for q in st.qs:
        if q.stated:
            vals[q.id] = Fraction(q.value)
            continue
        cls, c, ins = OPCLS[q.op], q.const, [vals[i] for i in q.inputs]
        if slip and slip[0] == q.id:
            kind = slip[1]
            if kind == "swap":
                cls = {"ADD": "SUB", "SUB": "ADD", "MUL": "DIV", "DIV": "MUL", "PCT": "PCTOFF", "PCTOFF": "PCT",
                       "SUM": "DIFF", "DOZEN": "SKIPD"}[cls]
            elif kind == "off1":
                if c is None or cls in ("SUM",):
                    return None
                c = c + slip[2]
            elif kind == "skip":
                cls = "SKIP" if cls != "DOZEN" else "SKIPD"
            elif kind == "wrong":
                ins = [ins[0], Fraction(slip[2])]
        if cls == "DIFF":
            v = abs(ins[0] - ins[1])
        elif cls == "SKIP":
            v = ins[0]
        elif cls == "SKIPD":
            v = Fraction(c)
        else:
            v = apply(cls, ins, c)
        if v is None or v <= 0:
            return None
        vals[q.id] = v
    return vals


def verify(st: Story) -> None:
    vals = execute(st)
    for q in st.qs:
        assert vals[q.id] == q.value and isinstance(q.value, int) and 1 <= q.value <= MAXV, (q, vals[q.id])
        if q.op in FANIN:
            assert len(q.inputs) == 2


def slips(st: Story, target: str) -> list[int]:
    """Values of `target` under one planted slip at it or any of its ancestors (integers only)."""
    anc, todo = set(), [target]
    while todo:
        i = todo.pop()
        if i in anc:
            continue
        anc.add(i)
        todo += st.by(i).inputs
    true = st.by(target).value
    out = set()
    for q in st.qs:
        if q.id not in anc or q.stated:
            continue
        plans = [(q.id, "swap"), (q.id, "skip")]
        if q.const is not None and OPCLS[q.op] != "DOZEN":
            plans += [(q.id, "off1", 1), (q.id, "off1", -1)]
        if q.op in FANIN:
            plans += [(q.id, "wrong", x.value) for x in st.qs if x.id not in q.inputs and x.kind == st.by(q.inputs[1]).kind]
        for p in plans:
            vals = execute(st, p)
            if vals is None:
                continue
            v = vals[target]
            if v.denominator == 1 and 1 <= v <= MAXV and v != true:
                out.add(int(v))
    return sorted(out)


# ------------------------------------------------------------------------------------------------------ options
def perturb(a: int, rng: random.Random) -> int | None:
    ops = ["+", "-", "x2"] + (["/2"] if a % 2 == 0 else [])
    o = rng.choice(ops)
    d = rng.choice([1, 2] if a < 20 else [1, 2, 5, 10])
    v = {"+": a + d, "-": a - d, "x2": a * 2, "/2": a // 2}[o]
    return v if v > 0 else None


def partials(target: int, stated: list[int]) -> list[int]:
    out = set()
    for a, b in itertools.permutations(stated, 2):
        out.update([a + b, a - b, a * b])
        if b and a % b == 0:
            out.add(a // b)
    for n in stated:
        out.update([target + n, target - n, target * n])
        if n and target % n == 0:
            out.add(target // n)
    return sorted(x for x in out if x > 0 and x != target and x <= 10 * max(target, 2))


def decoy(target: int, rng: random.Random) -> int:
    lo, hi = max(1, target // 2), target * 2 + 5
    v = rng.randint(lo, hi)
    if target >= 100 and target % 10 == 0:
        v = max(10, round(v, -1))
    return v


def build_options(target: int, stated: list[int], inter: list[int], slip_vals: list[int], recipe: str,
                  rng: random.Random) -> tuple[list[int], dict]:
    """Five distinct positive options (gold included) and the kind of each; recipe qo2 / inter / slip / mix."""
    if recipe == "mix":
        recipe = rng.choices(["qo2", "inter", "slip"], weights=[0.4, 0.3, 0.3])[0]
    opts: dict = {target: "gold"}

    def add(v, kind):
        if v is not None and v > 0 and v not in opts and len(opts) < 5:
            opts[v] = kind
            return True
        return False

    part = partials(target, stated)
    if recipe == "qo2":
        for v in rng.sample(part, k=min(2, len(part))):
            add(v, "partial")
        anchors = list(opts)
        for _ in range(60):
            if len(opts) >= 5:
                break
            add(perturb(rng.choice(anchors), rng), "perturb")
    else:
        pool = [v for v in (inter if recipe == "inter" else slip_vals) if v != target]
        for v in rng.sample(pool, k=min(2, len(pool))):
            add(v, "intermediate" if recipe == "inter" else "slip")
        if part and rng.random() < 0.5:
            add(rng.choice(part), "partial")
        for _ in range(60):
            if len(opts) >= 5:
                break
            add(perturb(rng.choice(list(opts)), rng), "perturb")
    for _ in range(200):
        if len(opts) >= 5:
            break
        add(decoy(target, rng), "decoy")
    vals = list(opts)
    rng.shuffle(vals)
    assert len(vals) == 5 and len(set(map(str, vals))) == 5
    return vals, {"recipe": recipe, "kinds": [opts[v] for v in vals]}


def mc_prompt(text: str, vals: list) -> str:
    return f"Question: {text}\nOptions:\n" + "\n".join(f"{L}) {v}" for L, v in zip(LETTERS, vals))


# -------------------------------------------------------------------------------------------------------- items
def program(p: Problem) -> dict:
    st = p.story
    return {"frame": p.frame, "name": st.name, "answer": st.qs[-1].id, "noise": p.noise, "shuffled": p.shuffled,
            "quantities": [{"id": q.id, "kind": q.kind, "value": q.value, "stated": q.stated, "op": q.op,
                            "inputs": q.inputs, "const": q.const, "name": q.np, "question": q.ask,
                            "sentence": q.sentence} for q in st.qs]}


def signature(p: Problem) -> str:
    st = p.story
    return p.frame + "|" + ";".join(f"{q.op}:{q.const}:{','.join(q.inputs)}:{q.value}" for q in st.qs)


def node_subqs(p: Problem, rng: random.Random) -> list[dict]:
    st, gold = p.story, p.answer.value
    nodes = [q for q in st.ops()[:-1] if q.value != gold]
    keep = {q.id for q in nodes}
    out = []
    for q in nodes:
        v = q.value
        parents = []
        stack = list(q.inputs)
        while stack:                                   # nearest intermediate ancestors that are nodes
            i = stack.pop()
            x = st.by(i)
            if x.id in keep:
                parents.append(f"{x.id}v")
            elif x.op and not x.stated:
                stack += x.inputs
        yes = rng.random() < 0.5
        if yes:
            x = v
        else:
            wrong = [s for s in slips(st, q.id) if s != v]
            x = rng.choice(wrong) if wrong and rng.random() < 0.6 else None
            for _ in range(20):
                if x is not None and x != v:
                    break
                x = perturb(v, rng)
        out.append({"id": f"{q.id}v", "type": "N1", "question": f"Is {q.np} equal to {q.fmt(x)}?", "truth": x == v,
                    "depends_on": sorted(set(parents)), "value": v})
        if rng.random() < 0.5:
            rs = [0.5, 0.67] if rng.random() < 0.5 else [1.5, 2.0]
            for r in rng.sample(rs, 2):
                X = max(1, round(v * r))
                if X != v:
                    break
            out.append({"id": f"{q.id}m", "type": "N2", "question": f"Is {q.np} more than {q.fmt(X)}?",
                        "truth": v > X, "depends_on": sorted(set(parents)), "value": v})
        else:
            d = v % 10 if rng.random() < 0.5 else rng.choice([x for x in range(10) if x != v % 10])
            out.append({"id": f"{q.id}d", "type": "N3", "question": f"Is the last digit of {q.np} {d}?",
                        "truth": v % 10 == d, "depends_on": sorted(set(parents)), "value": v})
    from .treeplan import node_depths
    depth = node_depths(out)
    return [{"question": n["question"], "p_cot": 1.0 if n["truth"] else 0.0, "p_fresh": None, "truth": n["truth"],
             "type": n["type"], "depends_on": n["depends_on"], "id": n["id"], "initial": None, "status": "direct",
             "node_depth": depth[n["id"]], "value": n["value"]} for n in out]


def final_item(p: Problem, item_id: str, recipe: str, rng: random.Random, pick=None, n_cand: int = 48) -> dict:
    """The 5-option final. With `pick` (recipe "bal"), n_cand option sets are drawn from the mix recipe and
    pick(problem, [(vals, meta), ...]) returns the index of the one to keep (scripts/build_numgen.py: the set whose
    gold rank under frozen shallow cue models matches a uniformly drawn target rank)."""
    st, gold = p.story, p.answer.value
    inter = [q.value for q in st.ops()[:-1]]
    sv = slips(st, p.answer.id)
    if pick is None:
        vals, meta = build_options(gold, p.stated, inter, sv, recipe, rng)
    else:
        cands = [build_options(gold, p.stated, inter, sv, "mix", rng) for _ in range(n_cand)]
        vals, meta = cands[pick(p, cands)]
        meta = {"recipe": "bal/" + meta["recipe"], "kinds": meta["kinds"]}
    gl = LETTERS[vals.index(gold)]
    return {"item_id": item_id, "source": "numgen", "prompt": mc_prompt(f"{p.body} {p.question}", vals),
            "labels": list(LETTERS), "gold_label": gl, "depth": p.depth, "level": level(st), "frame": p.frame,
            "shuffled": p.shuffled, "fanin": sum(q.op in FANIN for q in st.ops()), "n_irrelevant": len(p.noise),
            "recipe": meta["recipe"], "option_kinds": dict(zip(LETTERS, meta["kinds"])),
            "options": dict(zip(LETTERS, vals)), "stated_numbers": p.stated, "program": program(p)}


def level(st: Story) -> int:
    d: dict = {}
    for q in st.qs:
        d[q.id] = 0 if q.stated else 1 + max([d[i] for i in q.inputs] or [0])
    return d[st.qs[-1].id]


def node_rows(p: Problem, item_id: str, rng: random.Random) -> list[dict]:
    """Standalone 5-option node rows: N1 per intermediate (the story asked for that quantity) and N4 (which of the
    numbers that come up answers the question). Weight 1/rows per problem."""
    st, gold = p.story, p.answer.value
    rows = []
    for q in st.ops()[:-1]:
        if q.value == gold:
            continue
        inter = [x.value for x in st.ops() if x is not q]
        vals, meta = build_options(q.value, p.stated, inter, slips(st, q.id), "mix", rng)
        rows.append({"item_id": f"{item_id}-{q.id}", "source": "numgen_node", "node_type": "N1", "node_of": item_id,
                     "prompt": mc_prompt(f"{p.body} {q.ask}", vals), "labels": list(LETTERS),
                     "gold_label": LETTERS[vals.index(q.value)], "option_kinds": dict(zip(LETTERS, meta["kinds"])),
                     "recipe": meta["recipe"], "node_depth": level_of(st, q.id)})
    chain = sorted(set(x.value for x in st.ops()))
    if len(chain) >= 2:
        others = [v for v in chain if v != gold]
        vals = rng.sample(others, k=min(4, len(others))) + [gold]
        rng.shuffle(vals)
        labs = LETTERS[:len(vals)]
        text = (f"{p.body} {p.question}\nWhile working it out, these numbers come up: "
                f"{', '.join(map(str, chain))}. Which of them answers the question?")
        rows.append({"item_id": f"{item_id}-pick", "source": "numgen_node", "node_type": "N4", "node_of": item_id,
                     "prompt": f"Question: {text}\nOptions:\n" + "\n".join(f"{L}) {v}" for L, v in zip(labs, vals)),
                     "labels": labs, "gold_label": labs[vals.index(gold)], "option_kinds": None, "recipe": "chain"})
    for r in rows:
        r["teacher"] = {L: float(L == r["gold_label"]) for L in r["labels"]}
        r["weight"] = round(1.0 / len(rows), 6)
    return rows


def level_of(st: Story, qid: str) -> int:
    d: dict = {}
    for q in st.qs:
        d[q.id] = 0 if q.stated else 1 + max([d[i] for i in q.inputs] or [0])
    return d[qid]
