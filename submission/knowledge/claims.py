"""Conflict detection: rate rows across pages/snapshots, and organisation-level prose claims."""
from __future__ import annotations

from collections import Counter
from collections import defaultdict
from typing import Optional
import re

from .contract import Answer, Conflict, Source
from .pages import Unit
from .query import QInfo
from .text import PCT_RE, TIER_CELL_RE, VALUE_CELL_RE


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


class ConflictsMixin:
    """Conflict detection (rate rows and prose claims) and the claim-based answers that share its rules."""

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

    def _recognition_answer(self, qi: QInfo, si: int):
        """'What recognition has Dupaco received from Forbes?' -- the newest dated release's ordinal wins,
        the same rule conflicts() applies."""
        org = next((o for o in RECOGNIZERS if o.lower() in qi.text.lower()), None)
        if org is None or not re.search(r"recogni|award|rank|named|honor|list", qi.text, re.I):
            return None
        what = "employer" if re.search(r"employer|workplace|place to work", qi.text, re.I) else "credit union"
        cache = self._claims_cache.setdefault(si, {})
        if "claims" not in cache:
            cache["claims"] = self._claims([si])
        lst = cache["claims"].get(("ordinal", f"{org} {what}"))
        if not lst:
            return None
        c = self._claim_winner(lst, si)
        u = c[2]
        sent = next((m.group(0).strip() for m in ORDINAL_CLAIM_RE.finditer(u.text) if org in m.group(0)), c[1])
        when = f" ({u.pub_date})" if u.pub_date else ""
        return Answer(text=f"According to Dupaco's most recent release on this{when}: {sent}.",
                      sources=[Source(url=u.url, snapshot_id=self.snaps[si][0], quote=sent[:200])],
                      confident=True,
                      conflict_note="Older releases give earlier counts; the newest dated release is used.")

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
