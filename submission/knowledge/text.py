"""Text normalisation, tokenising/stemming, synonyms, and parsers for terms, model years, money and tiers."""
from __future__ import annotations

from urllib.parse import urlparse
import re


TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


STOP = set("""a an the of to for and or in on at by with from is are was were be been do does did what whats
what's which who whom how when where why can could would should will shall may might i me my we our
you your it its this that these those there their they them as about any all tell give get show
provide like know need want please much many current currently right now today dupaco dupacos credit
union s vs than then also just only still going new have has had pay charge charged earn got am im
i'm looking some one here using use repeat again remind confirm say said if keep main primary
general million thousand balance balances january february march april june july august september
october november december jan feb mar apr jun jul aug sep sept oct nov dec""".split())


# Words that describe the *kind* of fact asked for, not the product; they never decide coverage.
GENERIC = set("""rate rates interest apr apy yield dividend percentage annual price cost fee fees
number info information detail details link url page website site offer offers available option
options account accounts address located location where tier tiers program between highest lowest best
top balance standard purchase regular minimum maximum over under more less much new term""".split())


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
    return re.sub(r"\\([*_$#|])", r"\1", s)


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
    "receive": "receive incoming", "send": "send outgoing domestic", "mailing": "mailing mail address box",
    "wallet": "wallet pay apple google samsung mobile", "wallets": "wallet pay apple google samsung mobile",
}


SYN = {stem(k): {stem(x) for x in v.split()} | {stem(k)} for k, v in _SYN_RAW.items()}


def expand(terms: list[str]) -> list[str]:
    out = list(terms)
    for t in terms:
        out.extend(x for x in SYN.get(t, ()) if x not in out)
    return out


STOP_S = {stem(w) for w in STOP} | STOP          # stopwords / generic words in stemmed form too
GENERIC_S = {stem(w) for w in GENERIC} | GENERIC


def content_terms(s: str) -> list[str]:
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

