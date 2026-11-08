"""
Temporal knowledge system for the credit-union snapshots (standard library only; optional LLM layer).

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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from knowledge.system import TemporalKnowledge  # noqa: E402

# Module map (knowledge/):
#   text.py     normalisation, tokens/stemming, synonyms, term/model-year/money/tier parsers
#   query.py    question analysis: qualifiers, question kind, expected answer shape
#   pages.py    page typing/authority, expiry, splitting pages into units
#   index.py    per-snapshot BM25 + query coverage
#   claims.py   conflicts() and claim-based answers (counts, recognitions)
#   system.py   ingest / answer / snapshot selection / staleness / honesty gate
#   llm.py      optional answer composer (OpenRouter or Anthropic), grounded and cached
#   contract.py harness contract types (or identical local copies)


def build():
    return TemporalKnowledge()
