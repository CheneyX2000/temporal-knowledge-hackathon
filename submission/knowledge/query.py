"""Question analysis: qualifiers (term, model year, amount, flags), question kind and expected answer shape."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import re

from .text import FLAGS, GENERIC_S, MONEY_RE, content_terms, money_value, norm_question, norm_text, terms_months, tokens


# Answer-type detection: what the answer must *look like* for a given kind of question.
ANSWER_TYPES = [
    (re.compile(r"\b(hours?|open|close[sd]?|opening|closing)\b", re.I),
     re.compile(r"\b\d{1,2}(:\d\d)?\s*(a\.?m\.?|p\.?m\.?)|\bnoon\b", re.I)),
    (re.compile(r"\b(address|located|location of|where is)\b", re.I),
     re.compile(r"\b\d{2,6}\s+[A-Z][A-Za-z.]*(\s+[A-Z][A-Za-z.]*)*\s+(St|Ave|Rd|Ln|Dr|Lane|Street|Avenue|Road|Drive|"
                r"Blvd|Pkwy|Hwy|Way|Ct|Pl)\b|\bP\.?\s?O\.?\s+Box\s+\d+")),
    (re.compile(r"\b(phone|call|telephone)\b", re.I), re.compile(r"\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}")),
    (re.compile(r"\b(routing|aba)\b", re.I), re.compile(r"\b\d{9}\b")),
    (re.compile(r"\b(fee|fees|cost|costs|charge|charged)\b|\bhow much\b", re.I), re.compile(r"\$\s?\d")),
    (re.compile(r"\b(link|url|website|web address)\b", re.I), re.compile(r"https?://|\b[a-z]+\.dupaco\.com")),
]


LINK_Q_RE = re.compile(r"\b(link|url|website|web ?page|where can i (see|find))\b", re.I)


RATE_Q_RE = re.compile(r"\b(rates?|aprs?|apys?|yields?|dividends?|interest|percent)\b|%", re.I)


LIST_Q_RE = re.compile(r"\b(rates|what .* (do you|does dupaco) offer|all|list|options|cards|"
                       r"certificates|accounts|loans)\b", re.I)


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
    specific = bool(months or my or amounts or re.search(r"\b(highest|lowest|best|standard)\b", qn, re.I)
                    or (re.search(r"\b(fee|cost|charge|how much)\b", qn, re.I) and not RATE_Q_RE.search(qn)))
    is_list = bool(LIST_Q_RE.search(qn)) and not specific
    terms = tokens(negated)
    content = [t for t in content_terms(negated) if t not in GENERIC_S and t not in ("month", "year")]
    if "ira" in flags:      # IRA rows on the rate sheet are term certificates without saying so
        content = [t for t in content if t not in ("certif", "cd")]
    return QInfo(qn, terms, content, months, my, amounts, flags, is_rate, is_list, specific)
