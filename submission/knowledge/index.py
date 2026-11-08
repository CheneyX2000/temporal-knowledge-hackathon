"""Per-snapshot BM25 index with query-coverage scoring."""
from __future__ import annotations

from collections import Counter
from collections import defaultdict
import math

from .pages import Unit
from .text import SYN, tokens


class Index:
    def __init__(self, units: list[Unit]):
        self.units = units
        self.docs = []
        df = Counter()
        for u in units:
            t = tokens(f"{u.title} {u.context} {u.text}")
            tf = Counter(t)
            self.docs.append((tf, len(t)))
            df.update(tf.keys())
        n = len(units) or 1
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        # a table's footnotes/links may vouch for what the table is (coverage), but never rank it
        self.cap = [set(tokens(u.caption)) if u.caption else set() for u in units]
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
        if self.cap[i]:
            tf = dict.fromkeys(set(tf) | self.cap[i], 1)
        # words the website never uses (a car brand, small talk) cannot identify a product: skip them
        q_terms = [t for t in q_terms if t in self.idf]
        tot = sum(self.idf[t] for t in q_terms) or 1.0
        hit = sum(self.idf[t] for t in q_terms if t in tf or (SYN.get(t, set()) & tf.keys())
                  or (len(t) >= 4 and any(k.startswith(t) for k in tf)))     # "intro" ~ "introductory"
        return hit / tot if q_terms else 1.0
