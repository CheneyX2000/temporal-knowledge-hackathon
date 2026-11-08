"""
Knowledge system over five website snapshots that answers with what is true *now* (or as of a date),
cites its sources, lists the site's contradictions, and says "I don't know" when the site doesn't say.

The harness calls build() and then ingest() / answer() / conflicts(). The implementation lives in
knowledge/ (see the module map below); NOTES.md explains the rules and the trade-offs.
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
