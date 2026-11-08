# NOTES — temporal knowledge system (MVP)

**API keys: none.** Pure Python standard library, deterministic, no network calls.
`requirements.txt` is intentionally empty. Ingest of all five snapshots takes ~5 s; answers take ~20 ms.

```bash
python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl
```

Public set: **0.81** (starter: 0.09). Bundled selftest (with conflict gold): **1.00**.

## How it works

1. **Ingest keeps history, not just the latest copy.** Each snapshot gets its own index. A page that
   is missing from a later crawl is carried forward from the last crawl that had it, at 0.8×
   authority, because crawls are incomplete.
2. **Pages are typed by URL, and the type sets authority:** rate sheet (`/rates`) 1.0, product
   pages 0.85, newsroom 0.55, landing/promo pages 0.45, blog and PDF 0.4, calculators 0.25.
3. **Facts are units.** Every markdown table row is one unit. Its heading and column headers are
   kept, so `2024-27 Model Year | 60 Months | 5.39% APR` stays one fact. Short prose lines that
   carry a single rate ("6-month certificate APY: 3.50%") become one-row tables. All other prose
   is chunked; runs of short lines such as hours lists and addresses stay together.
4. **Retrieval** is BM25 with a small synonym map (mortgage↔home, CD↔certificate, car↔auto…).
   The score is multiplied by:
   - authority
   - query coverage²
   - **qualifier fit.** The question's term (60-month, 30-year), model year, dollar amount and
     flags (jumbo, IRA, business, refinance, intro) are parsed out. A row whose term, model-year
     range or balance tier contradicts them is *excluded*. A row that carries a flag the question
     didn't ask for (e.g. jumbo) is down-weighted.
   - **answer shape.** Hours questions want times, address questions want a street, routing
     questions want 9 digits, link questions want a URL.

## How it decides what is current

- **`as_of`:** use the newest snapshot taken on or before that date; otherwise use the latest.
- **Within a snapshot, the most authoritative page wins.** Rows are keyed by product name, term
  and tier. Model-year tiers are keyed by their rank in the table, because "2023-26" rolls into
  "2024-27". If a less authoritative page shows different numbers for the same key, that row is
  flagged *stale* (×0.05). This is how the calculator's 5.34% loses to the rate sheet's 5.39%.
- **Expired offers are stale too.** This covers "offer ended …" and end dates earlier than the
  crawl date.
- **Counts that change over time** (e.g. number of branches) prefer the newest dated article.
  Undated bios are down-weighted.

## When it says "I don't know"

- **Rate questions are answered only from rate rows** (table rows or rate-bearing lines). If no row
  covers the product, the answer is "isn't listed, please contact Dupaco". Example: the 7/1 ARM or
  construction-loan rate.
- **Any question is refused** when overall coverage is low or a rare, distinguishing word in the
  question is absent from the best evidence. Examples: a branch name such as "Mullan Avenue", "ARM".

## Conflicts

Rate rows that share a key but show different values are reported as conflicts in two cases: across
pages in the latest snapshot (a stale copy), and across snapshots of the same page (the value
changed). The winner is the value on the most authoritative page in the latest snapshot. The reason
field names that page and the stale pages, or says the value changed over time. 72 are reported.

## Known limitations / next steps

- Conflicts cover numeric rate facts only. Prose facts such as branch counts (22 vs 24) and Forbes
  recognitions (fourth vs fifth) are answered with recency rules but not listed as conflicts.
- Lexical retrieval misses some paraphrases. The 6-character prefix stemming occasionally collides
  ("officer" matches "office").
- Answers are extractive (row text or a page chunk), not composed. An optional LLM layer could
  phrase answers over the same retrieved, staleness-filtered evidence. It would be cached for
  consistency and fall back to this path when no key is set.
