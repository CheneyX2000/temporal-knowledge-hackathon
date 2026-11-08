"""Unit tests (standard library only; synthetic data, no network).

    python -m unittest discover -s submission/tests
"""
import os
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

os.environ["KS_LLM"] = "0"          # never call an LLM from tests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge import llm  # noqa: E402
from knowledge.pages import is_expired, page_type, split_page  # noqa: E402
from knowledge.query import analyse  # noqa: E402
from knowledge.system import TemporalKnowledge  # noqa: E402
from knowledge.text import model_year_range, money_range, terms_months  # noqa: E402


@dataclass
class Page:
    url: str
    title: str
    markdown: str
    fetched_at: str


@dataclass
class Snap:
    snapshot_id: str
    taken_on: str
    pages: list


RATES_TABLE = """# Rates

Auto Loans

| Loan Program | Term | Annual Percentage Rate (As Low As) |
| --- | --- | --- |
| 2024-27 Model Year | 60 Months | {new60} APR1 |
| 2021-23 Model Year | 60 Months | 5.64% APR1 |

1Auto Loan Rates: based on credit qualification.

Regular Certificates

| Certificate | Dividend Rate | Annual Percentage Yield (APY) | Min. to Earn |
| --- | --- | --- | --- |
| 12 Month | 3.37% | {cd12} | $1,000.00 |
| 6 Month | 3.37% | 3.40% | $1,000.00 |

Share Savings

| Account Type | Balance Tier | Dividend Rate | Annual Percentage Yield (APY) |
| --- | --- | --- | --- |
| Share Savings | $25,000+ | 0.20% | 0.20% |
| Share Savings | $10,000 – $24,999.99 | 0.15% | 0.15% |
| Share Savings | $0 – $9,999.99 | 0.00% | 0.00% |
"""

CALCULATOR = """# Vehicle Affordability Calculator

| Loan Program | Term | Annual Percentage Rate (As Low As) |
| --- | --- | --- |
| 2023–26 Model Year | 60 Months | 5.34% APR |
"""


def snapshot(sid, day, new60, cd12, extra=()):
    pages = [Page("https://www.example-cu.test/rates", "Rates", RATES_TABLE.format(new60=new60, cd12=cd12), day),
             Page("https://www.example-cu.test/calculators/vehicle-affordability", "Calculator", CALCULATOR, day)]
    return Snap(sid, day, pages + list(extra))


class ParserTests(unittest.TestCase):
    def test_terms(self):
        self.assertEqual(terms_months("60 Months"), {60})
        self.assertEqual(terms_months("30-Year Fixed"), {360})

    def test_model_years_both_orders(self):
        self.assertEqual(model_year_range("2024-27 Model Year"), (2024, 2027))
        self.assertEqual(model_year_range("Model Year 2026-23"), (2023, 2026))
        self.assertEqual(model_year_range("2016 Model Year (and older)"), (1900, 2016))

    def test_money_tiers(self):
        self.assertEqual(money_range("$25,000+"), (25000.0, float("inf")))
        self.assertEqual(money_range("$10,000 – $24,999.99".replace("–", "-")), (10000.0, 24999.99))
        self.assertEqual(money_range("$300,000 or less"), (0.0, 300000.0))

    def test_question_qualifiers(self):
        qi = analyse("What does the regular 10 month certificate special pay (not jumbo)?")
        self.assertIn(10, qi.months)
        self.assertNotIn("jumbo", qi.flags)
        self.assertEqual(analyse("APY on a six month certificate?").months, {6})
        self.assertEqual(analyse("APR for a 2025 car, 60 months").model_year, 2025)
        self.assertEqual(analyse("High-yield savings over $1 million?").amounts, [1_000_000.0])

    def test_expiry(self):
        self.assertTrue(is_expired("Summer special 2.95% (offer ended 2025-09-30)", "2026-03-01"))
        self.assertTrue(is_expired("Offer valid through 6/30/2026.", "2026-09-28"))
        self.assertFalse(is_expired("Offer valid through 12/31/2026.", "2026-09-28"))

    def test_page_authority_order(self):
        rates = page_type("https://www.dupaco.com/rates", "")[1]
        product = page_type("https://www.dupaco.com/personal/loans/auto-loans", "")[1]
        calc = page_type("https://www.dupaco.com/calculators/vehicle-payment", "")[1]
        blog = page_type("https://www.dupaco.com/learn/blog/x", "")[1]
        self.assertGreater(rates, product)
        self.assertGreater(product, blog)
        self.assertGreater(blog, calc)

    def test_split_page_rows_and_prose_facts(self):
        md = "# Fees\n\nDomestic wire $25.00 each\n\n6-month certificate APY: 3.50%\n"
        units = split_page("https://x.test/fees", "Fees", md, "2026-01-01", [0], "2026-01-01")
        rows = [u.text for u in units if u.kind == "row"]
        self.assertIn("Domestic wire $25.00 each", rows)
        self.assertIn("6-month certificate APY: 3.50%", rows)


class SystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ks = TemporalKnowledge()
        cls.ks.ingest(snapshot("01_2026-05-12", "2026-05-12", "5.19%", "2.90%"))
        cls.ks.ingest(snapshot("02_2026-09-28", "2026-09-28", "5.39%", "3.40%"))

    def test_rate_sheet_beats_stale_calculator(self):
        a = self.ks.answer("What's the APR for a 60-month auto loan on a 2025 model-year vehicle?")
        self.assertIn("5.39%", a.text)
        self.assertNotIn("5.34%", a.text)
        self.assertTrue(any(s.url.endswith("/rates") for s in a.sources))

    def test_as_of_uses_older_snapshot(self):
        a = self.ks.answer("What was the 12-month certificate APY?", as_of="2026-06-01")
        self.assertIn("2.90%", a.text)
        self.assertNotIn("3.40%", a.text)

    def test_balance_tier(self):
        a = self.ks.answer("Share Savings APY with a balance between $10,000 and $24,999?")
        self.assertIn("0.15%", a.text)
        self.assertNotIn("0.20%", a.text)

    def test_refuses_unlisted_product(self):
        a = self.ks.answer("What is your rate on a 7/1 ARM mortgage?")
        self.assertFalse(a.confident)

    def test_repeat_is_identical(self):
        q = "What is the 6-month certificate APY?"
        self.assertEqual(self.ks.answer(q).text, self.ks.answer(q).text)

    def test_conflicts_name_winner(self):
        found = [c for c in self.ks.conflicts() if "5.39%" in c.values and "5.34%" in c.values]
        self.assertTrue(found)
        self.assertEqual(found[0].resolution, "5.39%")


class LLMGuardTests(unittest.TestCase):
    def test_disabled_without_credentials(self):
        self.assertIsNone(llm.compose("q", "2026-09-28", [{"url": "u", "ptype": "rates", "text": "x"}]))

    def test_numbers_must_be_grounded(self):
        ev = [{"url": "https://x.test/rates", "ptype": "rates", "pub_date": None, "text": "12 Month APY 3.40%"}]
        orig = (llm.available, llm.BACKEND, llm._call_openrouter)
        try:
            llm.available = lambda: True
            llm.BACKEND = "openrouter"
            llm._call_openrouter = lambda user: ('{"answer": "It pays 3.40% APY (was 2.90%).", "confident": true, '
                                                 '"source_urls": ["https://x.test/rates"], "conflict_note": ""}')
            self.assertIsNone(llm.compose("12 month APY?", "2026-09-28", ev))      # 2.90 is not in evidence
            llm._call_openrouter = lambda user: ('{"answer": "It pays 3.40% APY.", "confident": true, '
                                                 '"source_urls": ["https://x.test/rates"], "conflict_note": ""}')
            self.assertEqual(llm.compose("12 month APY?", "2026-09-28", ev)["answer"], "It pays 3.40% APY.")
        finally:
            llm.available, llm.BACKEND, llm._call_openrouter = orig


if __name__ == "__main__":
    unittest.main()
