"""Page typing/authority, expiry detection, and splitting pages into units (table rows, prose facts, chunks)."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from typing import Optional
from urllib.parse import urlparse
import re

from .text import DATE_RE, FLAGS, METRIC_WORD_RE, MONEY_RE, MONTHS, PCT_RE, model_year_range, money_range, norm_text, terms_months


def page_type(url: str) -> tuple[str, float]:
    """(page type, authority) from the URL path."""
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
    caption: str = ""          # footnotes / link text under a table: they often name the product


def _is_label(s: str) -> bool:
    """A short title-like line ("Share Savings", "Regular Certificates") -- not a sentence or a link."""
    return len(s) < 50 and not s.startswith(("|", "[", "!")) and not s.endswith(".")


def split_page(url: str, title: str, md: str, fetched_at: str, table_counter: list,
               taken_on: str = "9999-12-31") -> list[Unit]:
    ptype, auth = page_type(url)
    md = norm_text(md)
    head = md[:600]
    dm = DATE_RE.search(head)
    pub = f"{dm.group(3)}-{MONTHS[dm.group(1).lower()]:02d}-{int(dm.group(2)):02d}" if dm else None
    units: list[Unit] = []
    lines = md.split("\n")
    recent_heads: list[str] = []      # markdown "#" headings
    labels: list[str] = []            # short label lines seen since the last heading / table
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
            # The table's footnotes and call-to-action links ("1Term Share Certificate Rates: ...",
            # "[Open my certificate](.../certificates)") describe it, even when no heading names it.
            after, j = [], i
            while j < len(lines) and len(after) < 6 and not lines[j].strip().startswith("|"):
                t = lines[j].strip()
                if t and (t.startswith("#") or (len(t) < 50 and not t.startswith("[") and not t[0].isdigit()
                                                and not t.startswith("*"))):
                    break      # the next product's title: this table's block ends here
                if t:
                    after.append(re.sub(r"\]\([^)]*\)", "]", t)[:300])      # link text only, no URLs
                j += 1
            cap = " ".join(after)
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
            if _is_label(s):
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
