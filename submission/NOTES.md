# NOTES — temporal knowledge system (MVP)

**API keys: optional.** The core is pure Python standard library: deterministic and offline.
An optional LLM layer composes answers from the same filtered evidence (see "LLM layer" below). It
uses a **paid API**:

- `OPENROUTER_API_KEY` (preferred). Plain HTTPS from the standard library. Default model is
  `deepseek/deepseek-v4.1-flash`, which is fast and cheap: a full public-eval run costs about $0.006.
- `ANTHROPIC_API_KEY`. Requires the `anthropic` package (in `requirements.txt`). Default model is
  `claude-opus-5-5`.

Keys are read from the environment, or from a git-ignored `.env` file. Without a key, or on any API
error, the system falls back to the local answer. Ingest of all five snapshots takes ~5 s; local answers take ~20 ms.

```bash
python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl
```

Public set (starter scores 0.09):

| Mode | Accuracy | p95 latency | Final score |
|---|---|---|---|
| Local only | 0.81 | 0.1 s | **0.81** |
| With LLM (DeepSeek V4.1 Flash) | 0.85 | 2.0 s | **0.85** |

Bundled selftest (with conflict gold): **1.00**.

LLM answers are not deterministic across separate runs. Within a run they are cached.

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

`conflicts()` reports two kinds of conflict (58 in total on the five snapshots).

1. **Rate facts.** Table rows and rate-bearing lines are keyed by product name, term and tier.
   Model-year tiers are keyed by their rank. A conflict is reported when the same key shows
   different values, either across pages of the latest snapshot (a stale copy) or across snapshots
   of the same page (the value changed).
   - Winner: the value on the most authoritative page in the latest snapshot.
   - A change repeated across the personal, IRA and business tables is reported once.
2. **Prose facts about the organisation.** These are figures that drift over time:
   - branch office locations, employees, members and charter-area counties, when stated about
     Dupaco itself and repeated on at least two pages;
   - "the Nth time Forbes recognized Dupaco", kept separate per ranking (credit union vs.
     employer).
   - Winner: the newest dated statement in the latest snapshot. Example: 22 branch office locations
     (May 2026) beats 23, 24, 20 and 19 from older releases.

Each conflict carries the competing values, their source pages, the resolution and a reason.

## LLM layer (optional, `llm.py`)

The local pipeline still decides *what is true now*: it picks the snapshot, drops stale copies and
expired offers, and refuses when the honesty gate fails. The LLM only *phrases* the answer, under
these rules:

- **Routing.** Answers read from rate tables stay local, both single rows and lists, because they
  are already exact. Only prose answers (hours, programs, eligibility…) go to the LLM, with the top 8
  **non-stale** evidence units: URL, page type and publish date, ordered by relevance.
  - Routing every non-specific question to the LLM was tried first. It was slower and lost a
    tier-disambiguation question.
- **Grounding guard.** Every number in the LLM answer must appear in the evidence or in the
  question. Otherwise the local answer is used. The LLM never sees stale values, so it cannot
  repeat them.
- **Output.** Structured JSON output: `answer`, `confident`, `source_urls`, `conflict_note`.
  Answers are cached per (question, as_of) for repeat consistency.
- **Deadline.** Each LLM call has a hard **1.9 s** wall-clock deadline (`KS_LLM_TIMEOUT`). A slower
  call falls back to the local answer, which keeps p95 latency at about 2 s.
  - OpenRouter: `response_format` json_schema (strict), `reasoning.effort` with `exclude: true`,
    and `provider.require_parameters` so the request only goes to providers that honour the schema.
  - Anthropic SDK: one retry, plus server-side refusal fallbacks (`fallbacks: "default"`) when the
    account supports them.
- **Settings.** `KS_LLM=0` disables the layer, `KS_LLM_MODEL` changes the model and
  `KS_LLM_EFFORT` changes the effort (default `low`). A faster model lowers latency for the speed
  score.

## Known limitations / next steps

- Prose conflicts cover a fixed set of organisation-level figures (branches, employees, members,
  counties, recognition counts). Other prose facts that change, such as hours or program amounts,
  are handled by recency when answering but are not listed as conflicts.
- No official conflict gold is available locally. Precision against the hidden gold is unknown.
- Lexical retrieval misses some paraphrases. The 6-character prefix stemming occasionally collides
  ("officer" matches "office").
- Without the LLM layer, answers are extractive: the row text or a page chunk.
