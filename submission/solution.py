"""
Temporal knowledge system for the credit-union snapshots (MVP, standard library only).

Pipeline
--------
ingest(snapshot)
    * every page is typed by its URL (rate sheet, product page, landing page, calculator,
      PDF, newsroom, blog) which gives it an *authority* weight
    * markdown tables are split into one unit per row, keeping the table heading and the
      column headers, so a row like "2024-27 Model Year | 60 Months | 5.39% APR" is a fact
    * prose is split into short chunks
    * each snapshot gets its own BM25 index, so `as_of` questions can be answered from the
      website exactly as it stood on that date

answer(question, as_of)
    1. pick the snapshot (newest one taken on/before `as_of`, else the latest)
    2. parse qualifiers from the question: term (60-month / 30-year), model year, dollar
       amounts (balance / loan tiers), jumbo / IRA / business / refinance / intro flags
    3. score units = BM25 x page authority x qualifier fit; rows whose qualifiers clash
       with the question (wrong term, wrong model-year range, wrong balance tier) are dropped
    4. answer with the best row (specific question) or the rows of the best table(s)
       (general question), or with a prose chunk for non-rate questions
    5. refuse ("isn't listed, please contact us") when the question's distinguishing words
       are not covered by any authoritative unit

conflicts()
    Table rows are keyed by (product family, normalised row label, column). The same key with
    different values -- across pages of the latest snapshot or across snapshots of the same
    page -- is a conflict; the winner is the value on the most authoritative, newest page.
"""
from __future__ import annotations

import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm  # noqa: E402  (optional LLM composer; inert without credentials)

# ---------------------------------------------------------------------------------------------
# Contract types: use the harness's classes when available, otherwise identical local copies
# (the harness accepts any object with the same fields).
try:  # pragma: no cover - depends on where the folder is mounted
    _h = next((d / "harness" for d in Path(__file__).resolve().parents
               if (d / "harness" / "contract.py").exists()), None)
    if _h is not None and str(_h) not in sys.path:
        sys.path.insert(0, str(_h))
    from contract import Answer, Conflict, Snapshot, Source  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover
    @dataclass
    class Source:  # type: ignore[no-redef]
        url: str
        snapshot_id: Optional[str] = None
        quote: Optional[str] = None

    @dataclass
    class Answer:  # type: ignore[no-redef]
        text: str
        sources: list = field(default_factory=list)
        confident: bool = True
        conflict_note: Optional[str] = None

    @dataclass
    class Conflict:  # type: ignore[no-redef]
        topic: str
        values: list
        sources: list
        resolution: Optional[str] = None
        reason: Optional[str] = None

    Snapshot = object  # type: ignore

# ---------------------------------------------------------------------------------------------
# Text utilities

TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
STOP = set("""a an the of to for and or in on at by with from is are was were be been do does did
what whats what's which who whom how when where why can could would should will shall may might
i me my we our you your it its this that these those there their they them as about any all
tell give get show provide like know need want please much many current currently right now
today dupaco dupacos credit union s vs than then also just only still going new have has had pay
charge charged earn get got am im i'm looking would like some any one there here want using use
repeat again remind confirm say said tell if keep main primary general it my million thousand balance balances
january february march april june july august september october november december jan feb mar apr jun
jul aug sep sept oct nov dec as of""".split())
# Words that describe the *kind* of fact asked for, not the product; they never decide coverage.
GENERIC = set("""rate rates interest apr apy yield dividend percentage annual price cost fee fees
number info information detail details link url page website site offer offers available option
options account accounts address located location where tier tiers program between highest lowest best
top balance standard purchase regular minimum maximum over under more less much new term""".split())

# Answer-type detection: what the answer must *look like* for a given kind of question.
ANSWER_TYPES = [
    (re.compile(r"\b(hours?|open|close[sd]?|opening|closing)\b", re.I),
     re.compile(r"\b\d{1,2}(:\d\d)?\s*(a\.?m\.?|p\.?m\.?)|\bnoon\b", re.I)),
    (re.compile(r"\b(address|located|location of|where is)\b", re.I),
     re.compile(r"\b\d{2,6}\s+[A-Z][A-Za-z.]*(\s+[A-Z][A-Za-z.]*)*\s+(St|Ave|Rd|Ln|Dr|Lane|Street|Avenue|Road|Drive|"
                r"Blvd|Pkwy|Hwy|Way|Ct|Pl)\b")),
    (re.compile(r"\b(phone|call|telephone)\b", re.I), re.compile(r"\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}")),
    (re.compile(r"\b(routing|aba)\b", re.I), re.compile(r"\b\d{9}\b")),
    (re.compile(r"\b(fee|fees|cost|costs|charge|charged)\b|\bhow much\b", re.I), re.compile(r"\$\s?\d")),
    (re.compile(r"\b(link|url|website|web address)\b", re.I), re.compile(r"https?://|\b[a-z]+\.dupaco\.com")),
]

MONTHS = {m: i for i, m in enumerate(
    "january february march april may june july august september october november december".split(), 1)}
DATE_RE = re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|"
                     r"November|December) (\d{1,2}), (20\d\d)\b")
TERM_RE = re.compile(r"\b(\d{1,3})\s*[- ]?\s*(months?|mos?\.?|years?|yrs?)\b", re.I)
MODEL_YEAR_RE = re.compile(r"\b(?:model year\s*)?(20\d\d|19\d\d)\s*[-–]\s*(\d{2,4})(?:\s*model year)?(?!\d)|"
                           r"\b(20\d\d|19\d\d)\s*model year(\s*\(and older\))?", re.I)
MONEY_RE = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s*(k|m|million|thousand)?", re.I)
PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
FOOTNOTE_RE = re.compile(r"(\d(?:\.\d+)?%)\d\b")      # "5.39% APR1" / "4.20%1" -> drop footnote digit

FLAGS = {
    "jumbo": re.compile(r"\bjumbo\b", re.I),
    "ira": re.compile(r"\b(ira|roth|traditional)\b", re.I),
    "business": re.compile(r"\bbusiness\b", re.I),
    "refinance": re.compile(r"\brefinanc", re.I),
    "intro": re.compile(r"\b(intro|introductory|balance transfer)\b", re.I),
    "special": re.compile(r"\bspecial\b", re.I),
    "health": re.compile(r"\b(hsa|health savings)\b", re.I),
    "minimax": re.compile(r"\bmini-?max\b", re.I),
}
VALUE_CELL_RE = re.compile(r"[\s$\d.,%*\-]*(apr|apy)?\s*\d?|-+|n/?a")
TIER_CELL_RE = re.compile(r"\d\s*[-]\s*\$?\d|\+\s*$")      # "$10,000 - $24,999.99", "$25,000+"
METRIC_WORD_RE = re.compile(r"\b(apy|apr|annual percentage|dividend rate|interest rate|rate)\b", re.I)
LINK_Q_RE = re.compile(r"\b(link|url|website|web ?page|where can i (see|find))\b", re.I)
RATE_Q_RE = re.compile(r"\b(rates?|aprs?|apys?|yields?|dividends?|interest|percent)\b|%", re.I)
LIST_Q_RE = re.compile(r"\b(rates|what .* (do you|does dupaco) offer|all|list|options|cards|"
                       r"certificates|accounts|loans)\b", re.I)
# Prose claims about the organisation that drift over time (checked for conflicts).
COUNT_CLAIM_RE = re.compile(r"\b(nearly|more than|over|about|approximately|almost)?\s*(\d{1,3}(?:,\d{3})*|\d+)\+?\s+"
                            r"(branch office locations|branch offices|branches|employees|members|counties)\b", re.I)
COUNT_NOUNS = {"branch": (5, 200), "employee": (100, 100_000), "member": (10_000, 10_000_000),
               "count": (10, 1000)}      # noun prefix -> plausible organisation-level range
COUNT_LABELS = {"branch": "branch office locations (branches)", "employee": "employees",
                "member": "members", "count": "charter-area counties"}
ORDINALS = {w: i for i, w in enumerate("first second third fourth fifth sixth seventh eighth ninth tenth".split(), 1)}
ORDINAL_CLAIM_RE = re.compile(r"[^.\n]*\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+"
                              r"(?:time|consecutive year|year in a row|straight year)\b[^.\n]*", re.I)
RECOGNIZERS = ("Forbes", "Newsweek", "Money", "Bankrate", "Iowa Business Record", "Employ Humanity")

REFUSAL_TEXT = ("I'm not able to confirm that from the Dupaco website - it isn't listed there. "
                "Please contact Dupaco directly for this information.")


NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
             "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "eighteen": 18, "twenty": 20,
             "twenty-four": 24, "thirty": 30, "thirty-six": 36, "forty-eight": 48, "sixty": 60,
             "seventy-two": 72, "eighty-four": 84}
NUM_WORD_RE = re.compile(r"\b(" + "|".join(sorted(NUM_WORDS, key=len, reverse=True)) +
                         r")(?=[\s-]+(?:months?|years?|yr|mo)\b)", re.I)


def norm_question(q: str) -> str:
    """Member phrasing -> site phrasing: 'six month' -> '6 month', 'five-year' -> '5-year'."""
    return NUM_WORD_RE.sub(lambda m: str(NUM_WORDS[m.group(1).lower()]), q)


def norm_text(s: str) -> str:
    s = s.replace("–", "-").replace("—", "-").replace("\\r\\n", " ").replace("\\n", " ")
    s = FOOTNOTE_RE.sub(r"\1", s)
    s = re.sub(r"\\([*_$#|])", r"\1", s)
    return s


def stem(t: str) -> str:
    if t in ("months", "mos", "mo"):
        return "month"
    if t in ("years", "yrs", "yr"):
        return "year"
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        t = t[:-1]
    if len(t) > 7 and not t[0].isdigit():
        t = t[:6]          # crude prefix stemming: recognition/recognized, eligible/eligibility
    return t


def tokens(s: str) -> list[str]:
    return [stem(t) for t in TOKEN_RE.findall(s.lower())]


# Member vocabulary -> website vocabulary. A query term counts as covered if any alternative is.
_SYN_RAW = {
    "mortgage": "mortgage home", "mortgages": "mortgage home", "house": "home mortgage",
    "cd": "certificate cd", "cds": "certificate cd", "certificate": "certificate cd",
    "car": "auto vehicle car", "vehicle": "auto vehicle car", "truck": "auto vehicle",
    "auto": "auto vehicle car", "motorcycle": "motorcycle auto",
    "card": "card visa", "visa": "visa card",
    "branch": "branch location office", "branches": "branch location office",
    "atm": "atm", "heloc": "heloc home equity line",
    "savings": "savings saving", "checking": "checking",
    "join": "join member membership", "eligible": "eligible join membership",
    "contact": "contact phone call", "university": "college student",
    "share": "share certificate savings",
    "wallet": "wallet pay apple google samsung mobile", "wallets": "wallet pay apple google samsung mobile",
}
SYN = {stem(k): {stem(x) for x in v.split()} | {stem(k)} for k, v in _SYN_RAW.items()}


def expand(terms: list[str]) -> list[str]:
    out = list(terms)
    for t in terms:
        out.extend(x for x in SYN.get(t, ()) if x not in out)
    return out


STOP_S = None   # filled lazily (stem() is defined below the word lists)
GENERIC_S = None


def content_terms(s: str) -> list[str]:
    global STOP_S, GENERIC_S
    if STOP_S is None:
        STOP_S = {stem(w) for w in STOP} | STOP
        GENERIC_S = {stem(w) for w in GENERIC} | GENERIC
    return [t for t in tokens(s) if t not in STOP_S and not t.replace(".", "").isdigit()]


def money_value(m: re.Match) -> float:
    v = float(m.group(1).replace(",", "") + ("." + m.group(2) if m.group(2) else ""))
    mult = (m.group(3) or "").lower()
    if mult in ("k", "thousand"):
        v *= 1_000
    elif mult in ("m", "million"):
        v *= 1_000_000
    return v


def terms_months(s: str) -> set[int]:
    out = set()
    for m in TERM_RE.finditer(s):
        n, unit = int(m.group(1)), m.group(2).lower()
        out.add(n * 12 if unit.startswith("y") else n)
    return out


def model_year_range(s: str):
    m = MODEL_YEAR_RE.search(s)
    if not m:
        return None
    if m.group(1):
        if "model" not in m.group(0).lower():
            return None
        a, b = int(m.group(1)), m.group(2)
        b = int(b) if len(b) == 4 else int(str(a)[:2] + b)
        return (min(a, b), max(a, b))       # "2024-27" and "2026-23" both work
    a = int(m.group(3))
    return (1900, a) if m.group(4) else (a, a)


def money_range(s: str):
    """Balance / loan-amount tier of a row: ($lo, $hi) or None."""
    vals = [money_value(m) for m in MONEY_RE.finditer(s)]
    if not vals:
        return None
    low = s.lower()
    if re.search(r"or less|and under|up to|or below", low):
        return (0.0, vals[0])
    if re.search(r"\+|and (up|over|above)|or more", low) and len(vals) == 1:
        return (vals[0], float("inf"))
    if len(vals) >= 2:
        return (min(vals[0], vals[1]), max(vals[0], vals[1]))
    return None


def url_key(u: str) -> str:
    p = urlparse(u)
    return (p.netloc.lower().removeprefix("www.") + p.path.rstrip("/").lower())


# ---------------------------------------------------------------------------------------------
# Page typing / authority

def page_type(url: str, title: str) -> tuple[str, float]:
    p = urlparse(url)
    path = p.path.lower().rstrip("/")
    if p.query and "privacy-policy" in path:
        return "junk", 0.05
    if path in ("/rates", "/business-rates") or re.match(r"^/(business-)?rates(/|$)", path):
        return "rates", 1.0
    if "calculator" in path:
        return "calculator", 0.25
    if path.startswith("/hubfs"):
        if re.search(r"fee-schedule|rate-sheet|truth-in-savings|disclosure", path):
            return "schedule", 0.85      # official fee/rate schedule published as a PDF
        return "pdf", 0.4
    if path.startswith("/newsroom"):
        return "news", 0.55
    if path.startswith("/learn") or re.match(r"^/(19|20)\d\d/", path) or path.startswith("/dupacodaily") \
            or path.startswith("/business/resources") or path.startswith("/resources"):
        return "blog", 0.4
    if path.endswith("-lp") or re.search(r"/(promo|promotions?|offers?)(/|$)", path) or path in ("/switch-and-earn", "/earn-more", "/offer-expired", "/makethemost-lp",
                                        "/homebuying-guidance", "/mortgage-refi", "/money-makeover"):
        return "landing", 0.45
    if path.startswith("/personal") or path.startswith("/business") or path.startswith("/about") \
            or path in ("/contact", "/checking", "/loans", "/membership", "/personal-loans") \
            or path.startswith("/connect") or path.startswith("/online-and-mobile-banking"):
        return ("contact" if path == "/contact" else "product"), 0.85
    return "other", 0.65


# ---------------------------------------------------------------------------------------------
# Expired offers

_DATE_ANY = (r"(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|(?:January|February|March|April|May|June|July|August|"
             r"September|October|November|December)\s+\d{1,2},?\s+\d{4})")
EXPIRY_RE = re.compile(r"\b(?:offer|promotion|promo|special|rate)?\s*(?:ends?|ended|expires?|expired|valid through|"
                       r"valid until|good through|available through)\s*(?:on\s*)?" + _DATE_ANY, re.I)
ENDED_RE = re.compile(r"\b(offer|promotion|promo) (has )?(ended|expired)\b|\bno longer available\b", re.I)


def _parse_date(s: str) -> Optional[str]:
    s = s.strip().replace(",", "")
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})$", s)
    if m:
        return s
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{2,4})$", s)
    if m:
        y = int(m.group(3))
        y = y + 2000 if y < 100 else y
        return f"{y:04d}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    m = re.match(r"([A-Za-z]+)\s+(\d{1,2})\s+(\d{4})$", s)
    if m and m.group(1).lower() in MONTHS:
        return f"{m.group(3)}-{MONTHS[m.group(1).lower()]:02d}-{int(m.group(2)):02d}"
    return None


def is_expired(text: str, taken_on: str) -> bool:
    """True when the text says an offer ended, or gives an end date before the crawl date."""
    if ENDED_RE.search(text):
        return True
    for m in EXPIRY_RE.finditer(text):
        d = _parse_date(m.group(1))
        if d and d < taken_on[:10]:
            return True
    return False


# ---------------------------------------------------------------------------------------------
# Units

@dataclass
class Unit:
    url: str
    title: str
    ptype: str
    authority: float
    kind: str                  # "row" | "text"
    text: str                  # display text
    context: str               # table heading / section the unit lives in
    header: list = field(default_factory=list)
    cells: list = field(default_factory=list)
    table_id: int = -1
    pub_date: Optional[str] = None
    fetched_at: str = ""
    # derived
    terms: set = field(default_factory=set)
    myear: Optional[tuple] = None
    mrange: Optional[tuple] = None
    flags: set = field(default_factory=set)
    stale: bool = False        # superseded copy, or an offer that had already ended
    tier_rank: int = 0         # model-year tier rank inside its table (1 = newest vehicles)
    carried: bool = False      # page missing from this crawl, carried over from an earlier one
    caption: str = ""          # the first line under a table (footnotes often name the product)


def _is_heading(line: str) -> bool:
    s = line.strip()
    return bool(s) and (s.startswith("#") or (len(s) < 70 and not s.startswith("|") and not s.endswith(".")
                                              and not s.startswith("[") and not s.startswith("!")))


def split_page(url: str, title: str, md: str, fetched_at: str, table_counter: list,
               taken_on: str = "9999-12-31") -> list[Unit]:
    ptype, auth = page_type(url, title)
    md = norm_text(md)
    head = md[:600]
    dm = DATE_RE.search(head)
    pub = f"{dm.group(3)}-{MONTHS[dm.group(1).lower()]:02d}-{int(dm.group(2)):02d}" if dm else None
    units: list[Unit] = []
    lines = md.split("\n")
    recent_heads: list[str] = []      # markdown "#" headings
    labels: list[str] = []            # short label lines seen since the last long paragraph
    i = 0
    buf: list[str] = []

    def flush():
        if buf:
            txt = "\n".join(buf).strip()
            if len(txt) > 3:
                units.append(Unit(url, title, ptype, auth, "text", txt, " / ".join(recent_heads[-1:]),
                                  pub_date=pub, fetched_at=fetched_at))
            buf.clear()

    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if s.startswith("|") and i + 1 < len(lines) and re.match(r"^\|\s*:?-{2,}", lines[i + 1].strip()):
            flush()
            header = [c.strip() for c in s.strip("|").split("|")]
            table_counter[0] += 1
            tid = table_counter[0]
            ctx = " / ".join(recent_heads[-1:] + [l for l in labels[-3:] if l not in recent_heads[-1:]])
            labels.clear()
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if any(c for c in cells):
                    disp = "; ".join(f"{h}: {c}" if h and j > 0 else c
                                     for j, (h, c) in enumerate(zip(header, cells)) if c)
                    units.append(Unit(url, title, ptype, auth, "row", disp, ctx, header, cells, tid,
                                      pub, fetched_at))
                i += 1
            # footnote right under the table ("1Term Share Certificate Rates: ...") names the product
            j = i
            while j < len(lines) and j < i + 4 and not lines[j].strip():
                j += 1
            if j < len(lines) and not lines[j].strip().startswith("|"):
                cap = lines[j].strip()[:300]
                for u in units:
                    if u.table_id == tid:
                        u.caption = cap
            continue
        if not s:
            pass
        elif s.startswith("#"):
            flush()
            recent_heads.append(s.lstrip("# ").strip())
            labels.clear()
            buf.append(s)
        else:
            # a long paragraph stands alone; runs of short lines (hours, addresses, lists) are merged
            size = sum(len(b) for b in buf)
            if len(s) > 160 and size > 0 and (size > 160 or len(buf) > 2):
                flush()
            elif size + len(s) > 700:
                flush()
            if _is_heading(s) and len(s) < 50:
                labels.append(s.strip("*: "))
            buf.append(s)
            if len(s) > 160 and sum(len(b) for b in buf) > 250:
                flush()
        i += 1
    flush()
    # Prose facts: a short line carrying exactly one rate ("6-month certificate APY: 3.50%") becomes a
    # one-row table so it is answered, cited and conflict-checked exactly like a rate-sheet row.
    extra = []
    for u in units:
        if u.kind != "text":
            continue
        u.stale = is_expired(u.text, taken_on)
        for line in u.text.split("\n"):
            ln = line.strip().lstrip("-*# ").strip()
            pcts = PCT_RE.findall(ln)
            money = MONEY_RE.findall(ln)
            if ln.startswith("|") or len(ln) > 200:
                continue
            if len(pcts) == 1 and METRIC_WORD_RE.search(ln):
                pass
            elif not pcts and len(money) == 1 and len(ln) <= 160:      # "Domestic wire $25.00 each"
                m_ = MONEY_RE.search(ln)
                label = (ln[:m_.start()] + " " + ln[m_.end():]).strip(" :-*")
                table_counter[0] += 1
                r = Unit(url, title, ptype, auth, "row", ln, u.context, ["", "amount"], [label, m_.group(0)],
                         table_counter[0], pub, fetched_at)
                r.stale = u.stale
                extra.append(r)
                continue
            else:
                continue
            label = PCT_RE.sub("", ln)
            label = re.sub(r"\(.*?\)", "", label).strip(" :-")
            metric = "apy" if re.search(r"\bapy\b|yield", ln, re.I) else (
                "apr" if re.search(r"\bapr\b", ln, re.I) else "rate")
            table_counter[0] += 1
            r = Unit(url, title, ptype, auth, "row", ln, u.context, ["", metric], [label, pcts[0] + "%"],
                     table_counter[0], pub, fetched_at)
            r.stale = u.stale or is_expired(ln, taken_on)
            extra.append(r)
    units.extend(extra)
    # Model-year tiers roll forward every year ("2023-26" becomes "2024-27"): identify them by rank.
    by_table = defaultdict(list)
    for u in units:
        if u.kind == "row":
            my = model_year_range(u.text)
            if my:
                by_table[u.table_id].append((my, u))
    for lst in by_table.values():
        starts = sorted({my[1] for my, _ in lst}, reverse=True)
        for my, u in lst:
            u.tier_rank = starts.index(my[1]) + 1
    for u in units:
        basis = f"{u.context} {u.text}" if u.kind == "row" else u.text
        u.terms = terms_months(u.text if u.kind == "row" else "")
        u.myear = model_year_range(u.text) if u.kind == "row" else None
        u.mrange = money_range(u.cells[0] if u.cells else "") if u.kind == "row" else None
        if u.kind == "row" and u.mrange is None and len(u.cells) > 1:
            u.mrange = money_range(u.cells[1]) if MONEY_RE.search(u.cells[1]) and "month" not in u.cells[1].lower() \
                else None
        u.flags = {k for k, rx in FLAGS.items() if rx.search(basis)}
    return units


# ---------------------------------------------------------------------------------------------
# BM25 index per snapshot

class Index:
    def __init__(self, units: list[Unit]):
        self.units = units
        self.docs = []
        df = Counter()
        for u in units:
            t = tokens(f"{u.title} {u.context} {u.caption} {u.text}")
            tf = Counter(t)
            self.docs.append((tf, len(t)))
            df.update(tf.keys())
        n = len(units) or 1
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.avgdl = sum(l for _, l in self.docs) / n
        self.post = defaultdict(list)
        for i, (tf, _) in enumerate(self.docs):
            for t in tf:
                self.post[t].append(i)

    def search(self, q_terms: list[str], k1=1.2, b=0.6) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        for t in set(q_terms):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.post[t]:
                tf, dl = self.docs[i]
                f = tf[t]
                scores[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / self.avgdl))
        return scores

    def coverage(self, i: int, q_terms: list[str]) -> float:
        tf, _ = self.docs[i]
        # words the website never uses (a car brand, small talk) cannot identify a product: skip them
        q_terms = [t for t in q_terms if t in self.idf]
        tot = sum(self.idf[t] for t in q_terms) or 1.0
        hit = sum(self.idf[t] for t in q_terms if t in tf or (SYN.get(t, set()) & tf.keys()))
        return hit / tot if q_terms else 1.0


# ---------------------------------------------------------------------------------------------
# Question analysis

@dataclass
class QInfo:
    text: str
    terms: list
    content: list
    months: set
    model_year: Optional[int]
    amounts: list
    flags: set
    is_rate: bool
    is_list: bool
    specific: bool


def analyse(q: str) -> QInfo:
    qn = norm_text(norm_question(q))
    months = terms_months(qn)
    my = None
    if re.search(r"model|vehicle|car|auto|truck|motorcycle|loan|apr", qn, re.I):
        ym = re.search(r"\b(19[89]\d|20[0-3]\d)\b", qn)
        my = int(ym.group(1)) if ym else None
    amounts = [money_value(m) for m in MONEY_RE.finditer(qn)]
    negated = re.sub(r"\b(not|non|no|excluding|without)[\s-]+(a\s+|the\s+)?(jumbo|ira|business|special|intro)\b", " ", qn,
                     flags=re.I)
    flags = {k for k, rx in FLAGS.items() if rx.search(negated)}
    is_rate = bool(RATE_Q_RE.search(qn)) or bool(
        re.search(r"\b(pay|pays|earn|earns)\b", qn, re.I) and
        re.search(r"\b(savings?|certificates?|cds?|checking|money market|ira|club)\b", qn, re.I))
    specific = bool(months or my or amounts or re.search(r"\b(highest|lowest|best|standard)\b", qn, re.I))
    is_list = bool(LIST_Q_RE.search(qn)) and not specific
    terms = tokens(negated)
    content = [t for t in content_terms(negated) if t not in GENERIC_S and t not in ("month", "year")]
    return QInfo(qn, terms, content, months, my, amounts, flags, is_rate, is_list, specific)


# ---------------------------------------------------------------------------------------------
# The system

PCT_VALUE_RE = re.compile(r"\d+(?:\.\d+)?%")


class TemporalKnowledge:
    def __init__(self):
        self.snaps: list[tuple[str, str, Index]] = []          # (snapshot_id, taken_on, index)
        self.rows_by_snap: list[list[Unit]] = []
        self._cache: dict = {}
        self._conflicts = None
        self._known_pages: dict = {}
        self._llm_ctx = None
        self._claims_cache: dict = {}

    # ---------------------------------------------------------------- ingest
    def ingest(self, snapshot) -> None:
        counter = [0]
        units: list[Unit] = []
        seen = set()
        current = {}
        for p in snapshot.pages:
            k = url_key(p.url) + "?" + urlparse(p.url).query
            if k in seen:
                continue
            seen.add(k)
            current[k] = p
            units.extend(split_page(p.url, p.title, p.markdown, p.fetched_at, counter, snapshot.taken_on))
        # A page absent from this crawl is not proof it was removed (crawls are incomplete): keep the
        # last copy we saw, at reduced authority, so its facts remain answerable.
        for k, p in self._known_pages.items():
            if k not in current:
                for u in split_page(p.url, p.title, p.markdown, p.fetched_at, counter, snapshot.taken_on):
                    u.authority *= 0.8
                    u.carried = True
                    units.append(u)
        self._known_pages.update(current)
        self._mark_stale(units)
        idx = Index(units)
        self.snaps.append((snapshot.snapshot_id, snapshot.taken_on, idx))
        self.snaps.sort(key=lambda s: s[1])
        self._cache.clear()
        self._conflicts = None

    def _mark_stale(self, units: list[Unit]) -> None:
        """Within one snapshot, a table row that repeats a product/tier found on a more authoritative
        page but with different numbers is a stale copy (old calculator, landing page, PDF)."""
        best = {}
        for u in units:
            if u.kind != "row":
                continue
            k = self._row_key(u)
            vals = tuple(PCT_RE.findall(u.text))
            if not k or not vals:
                continue
            cur = best.get(k)
            if cur is None or u.authority > cur[0]:
                best[k] = (u.authority, {vals})
            elif u.authority == cur[0]:
                cur[1].add(vals)
        for u in units:
            if u.kind != "row":
                continue
            k = self._row_key(u)
            vals = tuple(PCT_RE.findall(u.text))
            if k in best and vals and u.authority < best[k][0] and vals not in best[k][1]:
                u.stale = True

    # ---------------------------------------------------------------- answer
    def _pick_snapshot(self, as_of: Optional[str]):
        if not self.snaps:
            return None
        if not as_of:
            return self.snaps[-1]
        eligible = [s for s in self.snaps if s[1] <= as_of[:10]]
        return eligible[-1] if eligible else self.snaps[0]

    def _row_fit(self, u: Unit, qi: QInfo) -> Optional[float]:
        """Multiplier for how well a table row's qualifiers fit the question; None = excluded."""
        m = 1.0
        if qi.months:
            if not u.terms or not (u.terms & qi.months):
                return None       # the question names a term this row doesn't have
            m *= 1.6
        if qi.model_year is not None and u.myear:
            if not (u.myear[0] <= qi.model_year <= u.myear[1]):
                return None
            m *= 1.5
        if qi.amounts and u.mrange:
            lo, hi = u.mrange
            if all(lo - 0.005 <= a <= hi + 0.005 for a in qi.amounts):
                width = (hi if hi != float("inf") else lo * 4 + 1) - lo
                m *= 1.5 + 1.0 / (1 + math.log10(1 + width))
            else:
                return None
        for f in ("jumbo", "ira", "business", "refinance", "intro", "health", "minimax"):
            if f in u.flags and f not in qi.flags:
                m *= 0.8 if qi.is_list else 0.35
            elif f in qi.flags and f not in u.flags:
                m *= 0.5
        if "special" in qi.flags and "special" in u.flags:
            m *= 1.3
        return m

    def answer(self, question: str, as_of: Optional[str] = None):
        key = (question.strip().lower(), as_of)
        if key in self._cache:
            return self._cache[key]
        self._llm_ctx = None
        ans = self._answer(question, as_of)
        if self._llm_ctx is not None:
            evidence, label, local = self._llm_ctx
            out = llm.compose(question, label, evidence)
            if out is not None:
                srcs = [Source(url=u, snapshot_id=local.sources[0].snapshot_id if local.sources else None)
                        for u in out["source_urls"]] or local.sources
                ans = Answer(text=out["answer"], sources=srcs, confident=bool(out["confident"]),
                             conflict_note=(out.get("conflict_note") or None) or local.conflict_note)
        self._cache[key] = ans
        return ans

    def _answer(self, question: str, as_of: Optional[str]):
        snap = self._pick_snapshot(as_of)
        if snap is None:
            return Answer(text=REFUSAL_TEXT, confident=False)
        sid, taken_on, idx = snap
        qi = analyse(question)
        raw = idx.search(expand(qi.terms))
        if not raw:
            return Answer(text=REFUSAL_TEXT, confident=False)
        si = self.snaps.index(snap)
        counted = self._count_answer(qi, si)
        if counted is not None:
            return counted
        place = re.search(r"\b(?:branch(?:es)?|location|office)s?\s+(?:in|near|at)\s+([A-Z][a-zA-Z]+)", question)
        if place and not any(u.ptype == "contact" and place.group(1) in u.text for u in idx.units):
            return Answer(text=f"I couldn't find a Dupaco branch in {place.group(1)} on the website's location "
                               "list. Please contact Dupaco to confirm the nearest location.", confident=False)
        how_many = re.search(r"\bhow many\s+([a-z]+)", qi.text.lower())
        count_re = None
        if how_many:
            noun = stem(how_many.group(1))[:5]
            count_re = re.compile(r"\b\d[\d,]*\+?\s+(?:[a-z-]+\s+){0,2}" + re.escape(noun), re.I)
        snap_year = int(taken_on[:4]) + int(taken_on[5:7]) / 12
        # A "value" question asks for a rate; rates are only quoted from rate tables.
        value_q = qi.is_rate and not LINK_Q_RE.search(qi.text)
        shapes = [shape for trig, shape in ANSWER_TYPES if trig.search(qi.text)
                  and not (qi.is_rate and "\\$" in shape.pattern)]     # "what APR does X charge" is a rate

        scored = []
        for i, s in raw.items():
            u = idx.units[i]
            mult = u.authority
            cov_i = idx.coverage(i, qi.content) if qi.content else 1.0
            mult *= cov_i ** 2
            if u.kind == "row":
                fit = self._row_fit(u, qi)
                if fit is None:
                    continue
                mult *= fit
                if u.stale:
                    mult *= 0.05
                if value_q and not PCT_VALUE_RE.search(u.text):
                    continue
                if not qi.is_rate and not any(sh.search(u.text) for sh in shapes):
                    mult *= 0.6
                elif not PCT_VALUE_RE.search(u.text):
                    mult *= 0.4
            else:
                if value_q:
                    continue
                if u.stale:
                    mult *= 0.1
                if u.pub_date:  # newer articles are more likely to describe the present
                    yrs = max(0.0, snap_year - (int(u.pub_date[:4]) + int(u.pub_date[5:7]) / 12))
                    mult *= 1.0 / (1 + 0.25 * yrs)
                if count_re is not None:
                    mult *= 2.0 if count_re.search(u.text) else 0.3
                    if not u.pub_date:
                        mult *= 0.5     # undated pages (bios, evergreen copy) don't timestamp a count
            for shape in shapes:
                hits = len(shape.findall(u.text))
                mult *= 2.5 if hits else 0.4
                if hits and "\\$" in shape.pattern:      # money questions (fees): one line beats a price list
                    mult *= 1.6 if u.kind == "row" else (0.5 if hits > 3 else 1.0)
            if shapes and u.ptype == "contact" and any(sh.search(u.text) for sh in shapes):
                mult *= 1.5      # the contact page is the authority for phone / address / routing
            scored.append((s * mult, i))
        if not scored:
            return Answer(text=REFUSAL_TEXT, confident=False)
        scored.sort(key=lambda x: (-x[0], x[1]))
        self._last_scored = (idx, scored[:8])      # kept for debugging / inspection
        best_s, best_i = scored[0]
        best = idx.units[best_i]

        # Honesty gate: the question's words -- above all its rare, distinguishing ones (a branch
        # name, "7/1 ARM") -- must be on the unit we answer from; otherwise say it isn't listed.
        cov = idx.coverage(best_i, qi.content) if qi.content else 1.0
        tf = idx.docs[best_i][0]
        names = {stem(w.lower()) for w in re.findall(r"(?<![.?!]\s)(?<!^)\b([A-Z][a-z]{2,})\b", question)}
        missing_rare = [t for t in qi.content if t in names and idx.idf.get(t, 0) >= 6.0 and t not in tf
                        and not (SYN.get(t, set()) & tf.keys())]
        need = 0.7 if value_q else 0.45
        if cov < need or missing_rare:
            return Answer(text=REFUSAL_TEXT, confident=False)

        if best.kind == "row" and qi.amounts and best.mrange is None:
            table = [u for u in idx.units if u.table_id == best.table_id and u.url == best.url and u.mrange]
            his = [u.mrange[1] for u in table]
            if his and max(qi.amounts) > max(his):      # e.g. a $1.5M loan when tiers stop at $832,750
                return Answer(text=REFUSAL_TEXT, confident=False)
        if best.kind == "row":
            rows = self._select_rows(idx, scored, best, qi)
            text = self._format_rows(rows)
            srcs = self._row_sources(idx, rows, sid)
        else:
            parts = [best]
            for s, i in scored[1:6]:
                u = idx.units[i]
                if u.url == best.url and u.kind == "text" and s >= 0.7 * best_s and u is not best:
                    parts.append(u)
                    break
            text = "\n\n".join(p.text[:700] for p in parts)
            srcs = [Source(url=best.url, snapshot_id=sid, quote=best.text[:200])]
            keys = {m.group(0) for sh in shapes for m in sh.finditer(best.text)}
            for s_, i in scored[1:40]:      # other pages stating the same phone / routing / address
                u = idx.units[i]
                if keys and len(srcs) < 3 and all(x.url != u.url for x in srcs) and any(k in u.text for k in keys):
                    srcs.append(Source(url=u.url, snapshot_id=sid, quote=u.text[:200]))
        note = None
        if as_of:
            note = f"Answered from the website as it stood on {taken_on} (snapshot {sid})."
        local = Answer(text=text, sources=srcs, confident=True, conflict_note=note)
        # Answers read off rate tables (single rows and lists) are already exact and fast, so they stay
        # local. The LLM only composes prose answers (hours, programs, eligibility...), where
        # extractive chunks are weakest, from the same current, non-stale evidence.
        if best.kind == "row":
            return local
        evidence = self._evidence(idx, scored, rows if best.kind == "row" else [])
        self._llm_ctx = (evidence, f"{taken_on} (snapshot {sid})", local)
        return local

    def _evidence(self, idx: Index, scored, rows: list[Unit], limit: int = 8) -> list[dict]:
        """The top non-stale units, as plain dicts for the LLM (stale copies are never shown to it)."""
        out, seen = [], set()
        if rows:
            txt = "\n".join(f"- {r.text}" for r in rows)
            ctx = rows[0].context.split(" / ")[-1] if rows[0].context else rows[0].title
            out.append({"url": rows[0].url, "ptype": rows[0].ptype, "pub_date": rows[0].pub_date,
                        "text": f"{ctx}:\n{txt}"})
            seen.update(r.text for r in rows)
        for s, i in scored:
            if len(out) >= limit:
                break
            u = idx.units[i]
            if u.stale or u.text in seen:
                continue
            seen.add(u.text)
            text = (f"{u.context.split(' / ')[-1]}: {u.text}" if u.kind == "row" and u.context else u.text)[:900]
            out.append({"url": u.url, "ptype": u.ptype, "pub_date": u.pub_date, "text": text})
        return out

    def _row_sources(self, idx: Index, rows: list[Unit], sid: str) -> list:
        """Cite the row's page plus other authoritative pages that show the same row with the same values."""
        srcs, seen = [], set()
        for r in rows:
            if r.url not in seen:
                seen.add(r.url)
                srcs.append(Source(url=r.url, snapshot_id=sid, quote=r.text[:200]))
        key = self._row_key(rows[0])
        vals = set(PCT_RE.findall(rows[0].text))
        if key:
            for u in idx.units:
                if u.kind == "row" and u.url not in seen and u.authority >= 0.8 and len(srcs) < 4 \
                        and self._row_key(u) == key and set(PCT_RE.findall(u.text)) == vals:
                    seen.add(u.url)
                    srcs.append(Source(url=u.url, snapshot_id=sid, quote=u.text[:200]))
        return srcs

    def _select_rows(self, idx: Index, scored, best: Unit, qi: QInfo) -> list[Unit]:
        same_table = [u for u in idx.units if u.table_id == best.table_id and u.url == best.url and u.kind == "row"]
        fits = [(u, self._row_fit(u, qi)) for u in same_table]
        fits = [(u, f) for u, f in fits if f is not None]
        best_fit = self._row_fit(best, qi) or 0
        sup = re.search(r"\b(highest|best|top|maximum|max|lowest|minimum|min|cheapest)\b", qi.text, re.I)
        if sup and fits:
            def last_pct(u):
                v = PCT_RE.findall(u.text)
                return float(v[-1]) if v else -1
            cand = [u for u, f in fits if f >= best_fit * 0.99 and last_pct(u) >= 0]
            if cand:
                hi = sup.group(1).lower() in ("highest", "best", "top", "maximum", "max")
                if "loan" in qi.text.lower() or "apr" in qi.text.lower():
                    hi = not hi if sup.group(1).lower() == "best" else hi
                return [max(cand, key=last_pct) if hi else min(cand, key=last_pct)]
        if qi.specific:
            rows = [u for u, f in fits if f >= best_fit * 0.99]
            if qi.amounts:
                tiered = [u for u in rows if u.mrange]
                if tiered:      # "$100,000 or less" fits every "or less" tier: keep the narrowest one
                    w = lambda u: (u.mrange[1] if u.mrange[1] != float("inf") else 1e15) - u.mrange[0]
                    rows = [u for u in tiered if w(u) == min(w(x) for x in tiered)]
            return rows[:6] or [best]
        rows = [u for u, f in fits][:12]
        if qi.is_list:
            best_s = scored[0][0]
            tables = {(best.url, best.table_id)}
            for s, i in scored[1:60]:
                u = idx.units[i]
                if u.kind == "row" and (u.url, u.table_id) not in tables and s >= 0.5 * best_s \
                        and u.authority >= 0.8 and len(tables) < 4 and PCT_VALUE_RE.search(u.text):
                    tables.add((u.url, u.table_id))
                    rows += [r for r in idx.units if r.table_id == u.table_id and r.url == u.url
                             and r.kind == "row" and self._row_fit(r, qi) is not None][:8]
        return rows or [best]

    @staticmethod
    def _format_rows(rows: list[Unit]) -> str:
        out = []
        last_ctx = None
        for r in rows:
            ctx = r.context.split(" / ")[-1] if r.context else ""
            if ctx and ctx != last_ctx:
                out.append(f"{ctx}:")
                last_ctx = ctx
            out.append(f"- {r.text}")
        return "\n".join(out)

    # ---------------------------------------------------------------- conflicts
    @staticmethod
    def _row_key(u: Unit) -> Optional[str]:
        if not u.cells:
            return None
        # the row's *name*: every cell that is not a bare value (rate, points, payment, minimum)
        label = " ".join(c for c in u.cells if not VALUE_CELL_RE.fullmatch(c.strip().lower())
                         or TIER_CELL_RE.search(c))
        label = re.sub(r"\b(19|20)\d\d\s*-\s*\d{2,4}\b|\b(19|20)\d\d\b", "", label.lower())
        if u.tier_rank:
            label = f"tier{u.tier_rank} " + label
        if u.header and u.header[0] == "" and len(u.header) == 2:      # prose fact: name lives in the heading
            label = (u.context.split(" / ")[-1].lower() + " " + label) if u.context else label
        label = re.sub(r"\d+(\.\d+)?%", "", label)
        label = re.sub(r"[^a-z0-9$,+ ]+", " ", label)
        label = re.sub(r"\s+", " ", label).strip()
        return label or None

    def conflicts(self):
        if self._conflicts is not None:
            return self._conflicts
        if not self.snaps:
            return []
        # (key, column) -> list of (value, unit, snapshot index)
        obs = defaultdict(list)
        for si, (sid, taken, idx) in enumerate(self.snaps):
            for u in idx.units:
                if u.kind != "row" or u.ptype in ("junk",):
                    continue
                k = self._row_key(u)
                if not k:
                    continue
                for h, c in zip(u.header, u.cells):
                    m = PCT_RE.search(c)
                    if not m:
                        continue
                    col = re.sub(r"[^a-z ]", "", h.lower()).strip()
                    col = "apy" if "yield" in col or "apy" in col else ("apr" if "apr" in col or "annual percentage rate"
                                                                      in col else ("rate" if "rate" in col else col))
                    obs[(k, col)].append((m.group(1) + "%", u, si))
        latest = len(self.snaps) - 1
        out = []
        for (k, col), lst in obs.items():
            vals = {v for v, _, _ in lst}
            if len(vals) < 2:
                continue
            cur = [(v, u) for v, u, si in lst if si == latest]
            if not cur:
                continue
            # winner: most authoritative page in the latest snapshot, newest fetch first
            cur.sort(key=lambda x: (-x[1].authority, x[1].fetched_at), reverse=False)
            win_v, win_u = cur[0]
            # the winning page must agree with itself and be authoritative
            if win_u.authority < 0.6:
                continue
            others = sorted({v for v, _, _ in lst if v != win_v}, key=lambda x: float(x[:-1]))
            if not others:
                continue
            # keep it a single-product conflict: skip keys that map to many distinct values (table drift)
            if len(others) > 8:
                continue
            losers = [(v, u, si) for v, u, si in lst if v != win_v]
            stale_pages = {u.url for v, u, si in losers if si == latest}
            if stale_pages:
                reason = (f"{win_u.ptype} page {win_u.url} is authoritative in the latest snapshot "
                          f"({self.snaps[latest][1]}); {', '.join(sorted(stale_pages))[:300]} still show older values")
            else:
                last_seen = max(si for _, _, si in losers)
                reason = (f"value changed over time: {', '.join(others)} in earlier snapshots "
                          f"(last {self.snaps[last_seen][0]}); latest snapshot shows {win_v}")
            topic = f"{(win_u.context.split(' / ')[-1] + ' - ') if win_u.context else ''}{k} {col}".strip()
            srcs = [Source(url=win_u.url, snapshot_id=self.snaps[latest][0], quote=win_u.text[:160])]
            for v, u, si in losers[:6]:
                srcs.append(Source(url=u.url, snapshot_id=self.snaps[si][0], quote=u.text[:160]))
            out.append(Conflict(topic=topic, values=[win_v] + others, sources=srcs, resolution=win_v, reason=reason))
        # The same rate change is often repeated across personal / IRA / business tables: report it once,
        # keeping the personal-product row (members ask about those first).
        def _rank(c):
            t = c.topic.lower()
            return (("ira" in t or "roth" in t or "traditional" in t) + ("business" in t), c.topic)
        merged, seen_vals = [], set()
        for c in sorted(out, key=_rank):
            sig = (tuple(sorted(c.values)), c.resolution)
            if sig in seen_vals:
                continue
            seen_vals.add(sig)
            merged.append(c)
        out = merged
        out.extend(self._prose_conflicts())
        out.sort(key=_rank)
        self._conflicts = out
        return out

    def _claims(self, snap_ids) -> dict:
        """Organisation-level figures stated in prose, per key: [(value, display, unit, snapshot index)]."""
        claims = defaultdict(list)
        for si in snap_ids:
            idx = self.snaps[si][2]
            for u in idx.units:
                if u.kind != "text" or u.ptype == "junk":
                    continue
                for m in COUNT_CLAIM_RE.finditer(u.text):
                    # only statements about the organisation itself ("It has ... 22 branch office
                    # locations", "Dupaco serves ...") -- not a single branch, state or program
                    a = u.text.rfind(".", 0, m.start()) + 1
                    sent = u.text[a:m.end()]
                    if not re.search(r"\bDupaco\b|^\s*It\s", sent):
                        continue
                    n = int(m.group(2).replace(",", ""))
                    noun = m.group(3).lower()
                    for key, (lo, hi) in COUNT_NOUNS.items():
                        if noun.startswith(key) and lo <= n <= hi:
                            claims[("count", key)].append((n, re.sub(r"\s+", " ", m.group(0).strip()), u, si))
                            break
                for m in ORDINAL_CLAIM_RE.finditer(u.text):
                    sent = m.group(0)
                    org = next((o for o in RECOGNIZERS if o in sent), None)
                    if org is None:
                        continue
                    what = "employer" if re.search(r"employer|workplace|place to work", sent, re.I) else "credit union"
                    claims[("ordinal", f"{org} {what}")].append(
                        (ORDINALS[m.group(1).lower()], f"{m.group(1).lower()} time", u, si))
        # a count must be repeated on at least two pages (boilerplate, not a one-off mention)
        for (kind, key), lst in list(claims.items()):
            if kind == "count":
                pages = defaultdict(set)
                for v, _, u, _ in lst:
                    pages[v].add(u.url)
                claims[(kind, key)] = [c for c in lst if len(pages[c[0]]) >= 2]
        return claims

    @staticmethod
    def _claim_winner(lst, latest_si):
        """These figures change over time: the newest *dated* statement in the latest snapshot wins;
        undated copy only counts when nothing dated exists."""
        cur = [c for c in lst if c[3] == latest_si] or lst
        pool = [c for c in cur if c[2].pub_date] or cur
        newest = max(c[2].pub_date or "" for c in pool)
        top = [c for c in pool if (c[2].pub_date or "") == newest]
        win = Counter(v for v, _, _, _ in top).most_common(1)[0][0]
        return next(c for c in top if c[0] == win)

    def _prose_conflicts(self) -> list:
        latest = len(self.snaps) - 1
        out = []
        for (kind, key), lst in self._claims(range(len(self.snaps))).items():
            if len({v for v, _, _, _ in lst}) < 2:
                continue
            win_c = self._claim_winner(lst, latest)
            win = win_c[0]
            vals, seen = [win_c[1]], {win}
            for v, disp, u, si in sorted(lst, key=lambda c: c[2].pub_date or "", reverse=True):
                if v not in seen and len(vals) < 6:
                    seen.add(v)
                    vals.append(disp)
            if kind == "count":
                topic = "number of Dupaco " + COUNT_LABELS[key]
            else:
                topic = f"how many times {key.split()[0]} recognized Dupaco ({key.split(' ', 1)[1]} ranking)"
            srcs = [Source(url=win_c[2].url, snapshot_id=self.snaps[win_c[3]][0], quote=win_c[1])]
            for v, disp, u, si in lst:
                if v != win and all(x.url != u.url for x in srcs) and len(srcs) < 6:
                    srcs.append(Source(url=u.url, snapshot_id=self.snaps[si][0], quote=disp))
            reason = (f"newest dated statement ({win_c[2].pub_date or 'undated'}, {win_c[2].url}) in the latest "
                      f"snapshot says {win_c[1]}; older articles and copy still carry earlier figures")
            out.append(Conflict(topic=topic, values=vals, sources=srcs, resolution=win_c[1], reason=reason))
        return out

    def _count_answer(self, qi: QInfo, si: int):
        """'How many branches/employees/members...' -- answered with the same rule conflicts() uses."""
        m = re.search(r"\b(?:how many|number of)\s+(?:dupaco\s+)?([a-z]+)", qi.text.lower())
        if not m:
            return None
        key = next((k for k in COUNT_NOUNS if m.group(1).startswith(k) or
                    (k == "branch" and m.group(1).startswith("location"))), None)
        if key is None:
            return None
        cache = self._claims_cache.setdefault(si, {})
        if "claims" not in cache:
            cache["claims"] = self._claims([si])
        lst = cache["claims"].get(("count", key))
        if not lst:
            return None
        c = self._claim_winner(lst, si)
        u = c[2]
        when = f" (as stated {u.pub_date})" if u.pub_date else ""
        return Answer(text=f"Dupaco has {c[1]}{when}.",
                      sources=[Source(url=u.url, snapshot_id=self.snaps[si][0], quote=c[1])],
                      confident=True,
                      conflict_note="Older pages state other figures; the newest dated statement is used.")


def build():
    return TemporalKnowledge()
