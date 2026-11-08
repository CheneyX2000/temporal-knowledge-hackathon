"""The knowledge system: ingest snapshots, answer questions (local + optional LLM), list conflicts."""
from __future__ import annotations

from . import llm
from typing import Optional
from urllib.parse import urlparse
import math
import re

from .claims import ConflictsMixin
from .contract import Answer, Source
from .index import Index
from .pages import Unit, split_page
from .query import ANSWER_TYPES, LINK_Q_RE, QInfo, analyse
from .text import PCT_RE, SYN, expand, stem, url_key


_DAY_RE = re.compile(r"\b(mon|tues?|wed(nes)?|thu(rs)?|fri|sat(ur)?|sun)(day)?s?\b|\bclosed\b", re.I)


_CITY_ZIP_RE = re.compile(r"\b[A-Z][a-z]+,?\s+[A-Z]{2}\s+\d{5}(-\d{4})?\b")


REFUSAL_TEXT = ("I'm not able to confirm that from the Dupaco website - it isn't listed there. "
                "Please contact Dupaco directly for this information.")


class TemporalKnowledge(ConflictsMixin):
    def __init__(self):
        self.snaps: list[tuple[str, str, Index]] = []          # (snapshot_id, taken_on, index)
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
                    units.append(u)
        self._known_pages.update(current)
        self._mark_stale(units)
        idx = Index(units)
        self.snaps.append((snapshot.snapshot_id, snapshot.taken_on, idx))
        self.snaps.sort(key=lambda s: s[1])
        self._cache.clear()
        self._conflicts = None
        # Claim tables (counts, recognitions) are built here, not on the first question that needs them.
        # They are cached by snapshot position, so an out-of-order snapshot invalidates them all.
        if self.snaps[-1][2] is not idx:
            self._claims_cache = {}
        for si in range(len(self.snaps)):
            self._snapshot_claims(si)

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
        counted = self._count_answer(qi, si) or self._recognition_answer(qi, si)
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
            cov_i = idx.coverage(i, qi.content, qi.names) if qi.content else 1.0
            mult *= cov_i ** 2
            if u.kind == "row":
                fit = self._row_fit(u, qi)
                if fit is None:
                    continue
                mult *= fit
                if u.stale:
                    mult *= 0.05
                if value_q and not PCT_RE.search(u.text):
                    continue
                if not qi.is_rate and not any(sh.search(u.text) for sh in shapes):
                    mult *= 0.6
                elif not PCT_RE.search(u.text):
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
        best_s, best_i = scored[0]
        best = idx.units[best_i]

        # Honesty gate: the question's words -- above all its rare, distinguishing ones (a branch
        # name, "7/1 ARM") -- must be on the unit we answer from; otherwise say it isn't listed.
        cov = idx.coverage(best_i, qi.content, qi.names) if qi.content else 1.0
        tf = idx.docs[best_i][0]
        missing_rare = [t for t in qi.content if t in qi.names and idx.idf.get(t, 0) >= 6.0 and t not in tf
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
            if shapes:      # hours / phone / address / routing / fee: only the lines that carry it
                text = self._trim_to_shape(text, shapes) or text
            srcs = [Source(url=best.url, snapshot_id=sid, quote=best.text[:200])]
            keys = {m.group(0) for sh in shapes for m in sh.finditer(best.text)}
            for _, i in scored[1:40]:       # other pages stating the same phone / routing / address
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
        self._llm_ctx = (self._evidence(idx, scored), f"{taken_on} (snapshot {sid})", local)
        return local

    @staticmethod
    def _trim_to_shape(text: str, shapes) -> str:
        """Keep the lines that carry the asked-for value (plus their labels: "Lobby", "Sunday: Closed",
        "Branch hours are:"), so an extractive answer doesn't drag in unrelated numbers from the chunk."""
        lines = [l.strip() for l in text.split("\n")]
        hit = lambda l: any(sh.search(l) for sh in shapes)
        keep = set()
        for i, l in enumerate(lines):
            if not hit(l):
                continue
            if len(l) > 220:        # a paragraph: keep only the sentence(s) carrying the value
                sents = [s for s in re.split(r"(?<=[.!?])\s+", l) if hit(s)]
                lines[i] = " ".join(sents[:2])
            keep.add(i)
            j = i - 1           # walk back over labels / "closed" days directly above
            while j >= 0 and lines[j] and len(lines[j]) <= 60 and not hit(lines[j]) and (
                    _DAY_RE.search(lines[j]) or re.fullmatch(r"[*#_ ]*[^*#_:]{1,40}[*_ ]*:?", lines[j])
                    and not re.search(r"\d", lines[j])):
                keep.add(j)
                j -= 1
            if j >= 0 and lines[j].endswith(":"):       # "Branch hours are:"
                keep.add(j)
                k = lines[j].rfind(". ")
                lines[j] = lines[j][k + 2:] if k != -1 else lines[j]
            k = i + 1           # and "Sunday: Closed" style lines right after
            while k < len(lines) and lines[k] and len(lines[k]) <= 60 and not hit(lines[k]) and (
                    _DAY_RE.search(lines[k]) or _CITY_ZIP_RE.search(lines[k])):
                keep.add(k)
                k += 1
        return "\n".join(lines[i] for i in sorted(keep))

    @staticmethod
    def _evidence(idx: Index, scored, limit: int = 8) -> list[dict]:
        """The top non-stale units, as plain dicts for the LLM (stale copies are never shown to it)."""
        out, seen = [], set()
        for _, i in scored:
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
            if not (qi.months or qi.model_year is not None or qi.amounts):
                return [best]          # nothing to tier by: the single best-matching row
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
                        and u.authority >= 0.8 and len(tables) < 4 and PCT_RE.search(u.text):
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
