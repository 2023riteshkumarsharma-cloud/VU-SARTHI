"""Deterministic tool behaviour against the built knowledge base (run `python -m advisor.ingest.build` first)."""
import datetime as dt

from advisor import tools
from advisor.retrieval import search


def test_lookup_by_batch_and_code():
    r = tools.lookup_courses(batch=2023, course="COMP206")["result"]["rows"]
    assert [(x["semester"], x["L-T-P"], x["credits"]) for x in r] == [(5, "3-0-2", 4)]


def test_month_resolves_to_semester(monkeypatch):
    monkeypatch.setattr(tools, "today", lambda: dt.date(2026, 9, 26))
    rows = tools.lookup_courses(batch=2024, month=1, year=2027)["result"]["rows"]
    assert {x["semester"] for x in rows} == {6}
    pos = tools.academic_calendar()["result"]["positions"]
    assert pos[2025].startswith("semester 3") and pos[2026].startswith("semester 1")


def test_semester_total_and_credit_structure():
    assert tools.semester_plan(2024, 1)["result"]["total_credits"] == 21
    cs = tools.credit_structure(2026)["result"]
    assert cs["total_credits_for_degree"] == 180
    uc = next(b for b in cs["baskets"] if b["basket"] == "University Core")
    assert uc["credits"] == 18 and uc["sources_disagree"]["semester-spread Fixed Min"] == 24
    assert all(not b["basket"].startswith("B") for b in cs["baskets"])


def test_course_tool_outputs_use_descriptive_basket_names_only():
    rows = tools.lookup_courses(batch=2022, semester=1)["result"]["rows"]
    assert any(row["basket"] == "Program Core" for row in rows)
    assert all(not row["basket"].startswith("B") for row in rows)
    assert all(not row["basket"].startswith("B") for row in tools.semester_plan(2022, 1)["result"]["slots"])
    assert all(not row["basket"].startswith("B") for row in tools.course_details("DATA130")["result"]["programme_plan"])


def test_eligibility_decisions():
    assert tools.check_eligibility("Deep Learning")["result"]["decision"] == "needs_info"  # no batch
    r = tools.check_eligibility("Deep Learning", batch=2024)["result"]
    assert r["decision"] == "needs_info" and any("DATA301" in m for m in r["missing_information"])
    assert tools.check_eligibility("DATA302", batch=2024, completed=["DATA301"])["result"]["decision"] == "eligible"
    assert tools.check_eligibility("DATA302", batch=2024, completed=[], failed=["DATA301"])["result"]["decision"] == "not_eligible"
    assert tools.check_eligibility("Machine Learning", batch=2025, completed=["MATH301"], cgpa=7)["result"]["decision"] == "cannot_determine"  # DATA206 unknown
    assert tools.check_eligibility("DATA209", batch=2025, completed=["DATA132", "MATH203"], cgpa=3.8)["result"]["decision"] == "not_eligible"  # CGPA < 4 for S3


def test_minor_tba_and_issue_listing():
    s = tools.minor_info("Law", 2025)["result"]["summary"][0]
    assert (s["total_credits"], s["tba_credits"]) == (24, 17)
    assert any("80%" in i["issue"] for i in tools.list_data_issues(category="policy")["result"]["issues"])


def test_sql_is_read_only():
    assert "error" in tools.run_sql("DELETE FROM options")["result"]
    assert tools.run_sql("SELECT count(*) n FROM offerings")["result"]["rows"][0]["n"] == 262


def test_retrieval_finds_policy_passages():
    hits = search("minimum attendance for end semester exam", scope="policy", k=3)
    assert any("7.1" in h["citation"] or "Attendance" in h["citation"] for h in hits)
    assert "8.10" in search("grade point for A+", scope="policy", k=1)[0]["citation"]
