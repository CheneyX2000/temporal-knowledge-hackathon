# NOTES — a knowledge system that knows what's true *now*

## TL;DR

- **API keys: optional.** The core is pure Python standard library: deterministic, offline, and
  needs no packages.
- An optional LLM layer only *phrases* prose answers from evidence the local pipeline has already
  vetted. It is a **paid API**:
  - `OPENROUTER_API_KEY` (preferred): default model `deepseek/deepseek-v4.1-flash`. A whole
    public-eval run costs about $0.002–0.006.
  - Or `ANTHROPIC_API_KEY`, with `anthropic` from `requirements.txt`.
  - Without a key, or on any error, the system uses the local answer.
- Ingest of the five snapshots takes about 5 s. Local answers take about 20 ms. Checked on
  Python 3.11 from a fresh clone.

```bash
python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl   # score
python -m unittest discover -s submission/tests                                     # 15 unit tests
```

| Question set | Local only | With LLM (DeepSeek) | Notes |
|---|---|---|---|
| Public (40) | 0.836 | **0.851** (p95 2.0 s) | The starter scores 0.09. |
| Selftest (with conflict gold) | 1.00 | — | Conflicts F1 1.0. |
| `dev/blind2_eval.jsonl` (22) | **0.864** | 0.864 | Written last and run once, with no tuning afterwards. |
| `dev/blind_eval.jsonl` (25) | 0.88 on its first run | 1.00 | Its misses were then fixed, so it is no longer blind. |
| `dev/holdout_eval.jsonl` (60) | 0.68 → 0.98 | 0.98 | Used for tuning, so this number is optimistic. |

**Honest generalization estimate: about 0.86–0.88** on unseen questions. That is the two blind sets
on their first runs, with **zero stale-value trap hits** in all sets. I wrote the dev sets myself
and checked every expected value by hand against the snapshots. They cover paraphrases, other
products and tiers, `as_of` dates, refusals, fees and branches.

## Architecture (`knowledge/`)

| Module | Responsibility |
|---|---|
| `pages.py` | Types pages by URL, which sets authority. Detects expired offers. Splits pages into units: table rows (with heading, column headers and footnotes), one-line prose facts, and prose chunks. |
| `text.py` | Normalisation and tokens (light stemming, number words, synonyms). Parsers for terms, model-year ranges, dollar tiers and money. |
| `query.py` | Question analysis: term, model year, amounts, flags (jumbo, IRA, business, refinance, intro, with negation), rate/list/specific question kind, expected answer shape. |
| `index.py` | One BM25 index per snapshot, plus *coverage*: the share of the question's distinguishing words found on a unit. |
| `system.py` | `ingest` / `answer`: snapshot choice, staleness marking, scoring, the honesty gate, row selection, sources. |
| `claims.py` | `conflicts()`, plus answers that share its rules (organisation counts, recognitions). |
| `llm.py` | Optional answer composer with the grounding guard, cache and deadline. |

## How it decides what is current

1. **History, not just the latest copy.** Every snapshot keeps its own index.
   - `as_of` uses the newest snapshot taken on or before that date; no date means the latest.
   - A page missing from a later crawl is carried forward at 0.8× authority. Crawls are
     incomplete, and the selftest relies on this.
2. **Authority by page type.**
   - Order: rate sheets (`/rates`, `/business-rates`) and official fee/rate schedule PDFs (1.0 /
     0.85) > product pages (0.85) > news (0.55) > landing/promo pages (0.45) > blog and other PDFs
     (0.4) > calculators (0.25).
   - The contact page is the authority for phone, address and routing number.
3. **Stale copies are detected, not just out-ranked.** Rows are keyed by product name, term and
   tier.
   - If a less authoritative page in the same snapshot shows different numbers for the same key,
     its row is marked stale (×0.05).
   - Model-year tiers are keyed by their *rank* in the table, because "2023-26" rolls into
     "2024-27" each year.
   - This is why the calculator's 5.34% loses to the rate sheet's 5.39%.
4. **Expired offers are stale**: "offer ended", or an end date before the crawl date.
5. **Figures that drift** (branches, employees, members, counties, "the Nth time Forbes…"): the
   newest *dated* statement wins, and undated copy such as bios only counts as a fallback.
   `answer()` and `conflicts()` use the same rule, so they never disagree.

## How it picks the right tier

- **Qualifiers are hard constraints.** A row whose term (60-month, 30-year), model-year range or
  balance/loan tier contradicts the question is excluded, not merely down-weighted.
- **Narrowest tier wins.** "$100,000 or less" fits every "or less" tier, so the narrowest is kept.
- **Out of range.** An amount above every tier ($1.5M against tiers that stop at $832,750) is
  refused.
- **Flags.** A row carrying a flag the question didn't ask for (jumbo, IRA, business, intro) is
  down-weighted. "not jumbo" is understood as a negation.
- **Selection.** Specific questions get the matching row(s); a fee question gets one line, not the
  whole price list; general questions ("mortgage rates?") get the best table.

## When it says "I don't know"

- **Rates come from rate rows only** (table rows or single-rate lines), and the row must cover the
  question's product words. "7/1 ARM", "construction loan", "9-month CD" and "40-year mortgage"
  are therefore "isn't listed, please contact Dupaco", not a guess.
- **Prose.** Prose answers are refused when coverage is low, or when a capitalised name in the
  question (e.g. "Mullan Avenue") is missing from the evidence. "Branch in Chicago?" checks the
  location list.
- **Unknown words.** A word the site never uses counts as missing ("crypto loan" is refused).
  The exception is a capitalised brand name: "2026 Honda" still gets the model-year rate.

## Conflicts

`conflicts()` returns 56 conflicts. Each carries the competing values, their source pages, the
resolution and a reason.

1. **Rate facts.** The same row key with different values is reported, either across pages of the
   latest snapshot (a stale copy) or across snapshots of the same page (a change).
   - Winner: the most authoritative page in the latest snapshot.
   - A change repeated across the personal, IRA and business tables is reported once, which keeps
     precision up.
2. **Organisation figures in prose**, under the newest-dated rule above. Example: 22 branch office
   locations (May 2026) beats 23, 24, 20 and 19 from older releases.

## LLM layer (optional)

- **Routing.** Answers read from rate tables stay local because they are exact and take
  milliseconds. Only prose answers go to the LLM, with the top 8 *non-stale* evidence units (URL,
  page type, publish date).
  - Sending every general question to the LLM was tried first. It was slower and lost a tier
    question.
- **Grounding guard.** Every number in the LLM answer must appear in the evidence or the question;
  otherwise the local answer is used. The LLM never sees stale values, so it cannot repeat them.
  The guard is covered by a unit test.
- **Consistency and speed.**
  - Structured JSON output, cached per (question, as_of), so repeats are identical.
  - Hard 1.9 s wall-clock deadline (`KS_LLM_TIMEOUT`); a slower call falls back to the local
    answer, which keeps p95 at about 2 s.
  - Answers can differ between separate runs.
- **Settings.** `KS_LLM=0` disables the layer. `KS_LLM_MODEL` and `KS_LLM_EFFORT` (default `low`)
  tune it.
  - OpenRouter requests use a strict json_schema, `reasoning.exclude`, and
    `provider.require_parameters`.
  - The Anthropic backend uses one retry and server-side refusal fallbacks when available.

## Trade-offs I chose

- **Rules + BM25, not embeddings.** The hard part here is *which* number is current and which tier
  applies, not semantic recall. Explicit rules are inspectable and deterministic, which earns the
  consistency points, and they are fast. Embeddings would add a dependency and still need all of
  these rules. The cost: paraphrases need a small synonym map, and recall on unusual wording is
  weaker.
- **LLM as a phraser, not a judge.** Letting an LLM decide currency would make the 55% accuracy
  pillar depend on its reading of stale pages. Instead, staleness is decided before the LLM sees
  anything. Cost: the LLM cannot rescue a retrieval miss.
- **Refuse rather than guess on rates.** A wrong rate scores zero, as does a stale one listed
  alongside the right one. A refusal is correct whenever the site truly doesn't list the product.
  Cost: a few over-refusals when an old page's table has no heading naming the product. These are
  mitigated by reading the table's own footnotes and links.
- **Cheap model plus a deadline.** Opus-class models were measured at p95 ≈ 7 s, which cost more in
  the speed pillar than they gained in accuracy. DeepSeek V4.1 Flash with a 1.9 s deadline keeps
  speed at full marks.
- **Overfitting control.** Every fix had to be a general rule (no question-specific strings), and
  was checked on the public set, the holdout and the selftest together. The final blind set was
  never tuned on.

## Known limitations

- Ambiguous product words without context (e.g. "SUV" counts as an unknown word, and is refused).
- Prose answers in local mode are extractive chunks. The LLM layer fixes most of that.
- Conflict precision against the hidden gold is unknown, since no gold is available locally.
- Questions whose answer is spread across several pages (e.g. "do all branches keep the same
  hours?") are only partially answered.
