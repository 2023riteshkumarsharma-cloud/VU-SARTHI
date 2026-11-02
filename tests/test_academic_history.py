import re

import pymupdf
from fastapi.testclient import TestClient

from advisor import academic_records, document_generation, graph, server


client = TestClient(server.app)


def _chat(message, profile_id=None):
    return client.post(
        "/api/chat",
        json={"message": message, "profile_id": profile_id},
    )


def _pdf_text(response):
    metadata = response.json()["documents"][0]
    downloaded = client.get(metadata["download_url"])
    assert downloaded.status_code == 200
    assert downloaded.headers["content-type"] == "application/pdf"
    payload = downloaded.content
    assert payload.startswith(b"%PDF")
    return metadata, "\n".join(
        page.get_text()
        for page in pymupdf.open(stream=payload, filetype="pdf")
    )


def test_frozen_records_cover_all_profiles_and_only_completed_semesters():
    assert len(academic_records.PROFILES) == 10
    for profile_id, profile in academic_records.PROFILES.items():
        history = academic_records.history_for_profile(profile_id)
        expected = set(academic_records.eligible_completed_semesters(profile))
        actual = {int(number) for number in history["semesters"]}
        assert actual == expected
        assert history["is_synthetic"] is True
        assert "not verified university regulations" in history["policy"]
        for semester_number, semester in history["semesters"].items():
            assert semester["semester"] == int(semester_number)
            assert semester["academic_year"]
            assert semester["courses"]
            credits = sum(course["credits"] for course in semester["courses"])
            points = sum(course["credits"] * course["final_grade_points"] for course in semester["courses"])
            assert round(points / credits, 2) == semester["sgpa"]
            for course in semester["courses"]:
                assert course["course_code"] and course["course_name"]
                assert course["marks_obtained"] == course["total_marks_obtained"]
                assert course["maximum_marks"] == 100
                assert course["minimum_required_attendance_percentage"] == 75
                assert course["minimum_passing_marks"] == 40
                if course["failure_reason"]:
                    assert course["initial_result_status"] == "FAIL"
                    assert course["grade"] == "F" or course["course_result_status"] == "CLEARED AFTER REATTEMPT"
                    assert course["failure_reason"] == (
                        "Attendance shortage and insufficient examination marks"
                        if course["attendance_percentage"] < 75 and course["marks_obtained"] < 40
                        else "Attendance shortage"
                        if course["attendance_percentage"] < 75
                        else "Insufficient examination marks"
                    )
                if course["course_result_status"] == "BACKLOG":
                    assert course["final_grade_points"] == 0
                    assert course["credits"] > 0
                if course["course_result_status"] == "CLEARED AFTER REATTEMPT":
                    assert len(course["attempts"]) == 2
                    assert course["attempts"][0]["course_result_status"] == "FAIL"
                    assert course["attempts"][1]["course_result_status"] == "PASS"
                    assert course["reattempt_marks"] >= 40


def test_synthetic_dataset_has_varied_failure_and_clearance_scenarios():
    records = {
        profile_id: [
            course
            for semester in academic_records.history_for_profile(profile_id)["semesters"].values()
            for course in semester["courses"]
        ]
        for profile_id in academic_records.PROFILES
    }
    assert not any(course["failure_reason"] for course in records["student_b"])
    assert any(course["failure_reason"] == "Attendance shortage" for course in records["student_d"])
    assert any(course["failure_reason"] == "Insufficient examination marks" for course in records["student_h"])
    assert any(
        course["failure_reason"] == "Attendance shortage and insufficient examination marks"
        for course in records["student_c"]
    )
    assert any(course["course_result_status"] == "CLEARED AFTER REATTEMPT" for course in records["student_a"])
    assert any(course["course_result_status"] == "BACKLOG" for course in records["student_g"])
    assert sum(bool(course["failure_reason"]) for course in records["student_c"]) >= 2


def test_academic_questions_for_each_profile_answer_in_chat_without_documents(monkeypatch):
    monkeypatch.setattr(
        graph,
        "ask",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("history questions must use stored records")),
    )
    for profile_id in academic_records.PROFILES:
        history = academic_records.history_for_profile(profile_id)
        if not history["semesters"]:
            continue
        semester = min(map(int, history["semesters"]))
        item = history["semesters"][str(semester)]
        course = item["courses"][0]
        course_question = _chat(
            f"What courses did Student {profile_id[-1].upper()} study in Semester {semester}?",
            profile_id,
        )
        assert course_question.status_code == 200
        response = course_question.json()
        assert "documents" not in response
        assert course["course_code"] in response["answer"]
        assert response["flags"]["synthetic_data"] is True

        grade_question = _chat(
            f"What grade did Student {profile_id[-1].upper()} get in {course['course_code']} "
            f"in Semester {semester}?",
            profile_id,
        )
        assert "documents" not in grade_question.json()
        assert course["final_course_grade"] in grade_question.json()["answer"]

        sgpa_question = _chat(
            f"What was Student {profile_id[-1].upper()}'s SGPA in Semester {semester}?",
            profile_id,
        )
        assert "documents" not in sgpa_question.json()
        assert f"{item['sgpa']:.2f}" in sgpa_question.json()["answer"]


def test_semester_metrics_highest_marks_backlog_and_current_future_responses():
    profile_id = "student_a"
    history = academic_records.history_for_profile(profile_id)
    sem = history["semesters"]["2"]
    response = _chat("Compare Student A's SGPA across Semesters 1 to 5")
    assert response.status_code == 200
    assert "Semester | SGPA | CGPA" in response.json()["answer"]
    assert "Current/future or unavailable semesters omitted: 5" in response.json()["answer"]
    assert "documents" not in response.json()

    top = max(sem["courses"], key=lambda course: course["marks_obtained"])
    highest = _chat("Which subject did Student A score the highest marks in during Semester 2?")
    assert top["course_name"] in highest.json()["answer"]
    assert str(top["marks_obtained"]) in highest.json()["answer"]

    current = academic_records.PROFILES["student_a"]["current_semester"]
    for requested in (current, current + 1):
        response = _chat(f"What was Student A's SGPA in Semester {requested}?")
        assert "not available" in response.json()["answer"].casefold()
        assert "documents" not in response.json()

    backlog = _chat("Which subjects does Student G have backlogs in?")
    assert "DATA301" in backlog.json()["answer"]
    assert "BACKLOG" in backlog.json()["answer"]
    cleared = _chat("Did Student H clear the backlog in Semester 2?")
    assert "CLEARED AFTER REATTEMPT" in cleared.json()["answer"]


def test_nonexistent_student_does_not_fall_back_to_selected_profile():
    response = _chat("What grade did Ritesh Sharma get in MATH203 in Semester 2?", "student_a")
    assert response.status_code == 200
    assert "not one of the available synthetic profiles" in response.json()["answer"]
    assert "documents" not in response.json()


def test_explicit_marksheet_download_uses_same_stored_rows_as_chat():
    profile_id = "student_a"
    semester = 2
    stored = academic_records.semester_result(profile_id, semester)
    course = next(course for course in stored["courses"] if course["course_code"] == "COMP201")
    chat_response = _chat(
        "What grade did Student A get in COMP201 in Semester 2?",
        profile_id,
    )
    assert course["final_course_grade"] in chat_response.json()["answer"]

    pdf_response = _chat(
        "Give me Student A's Semester 2 marksheet.",
        profile_id,
    )
    metadata, contents = _pdf_text(pdf_response)
    assert metadata["format"] == "PDF"
    assert "COMP201" in contents
    assert str(course["marks_obtained"]) in contents
    assert str(course["reattempt_marks"]) in contents
    assert course["failure_reason"] in contents
    assert "CLEARED AFTER REATTEMPT" in contents
    assert "AI GENERATED" in contents
    assert document_generation.FOOTER_TEXT in contents
    assert "Synthetic demonstration data. Not an official university record." in contents


def test_marksheets_for_all_profiles_use_only_persisted_completed_semesters():
    for profile_id, profile in academic_records.PROFILES.items():
        semesters = academic_records.eligible_completed_semesters(profile)
        if not semesters:
            continue
        semester = semesters[-1]
        response = _chat(
            f"Generate Student {profile_id[-1].upper()}'s Semester {semester} gradesheet in PDF.",
            profile_id,
        )
        assert response.status_code == 200
        metadata, contents = _pdf_text(response)
        assert metadata["format"] == "PDF"
        assert f"Semester {semester}" in contents
        assert "Synthetic grading assumptions" in contents
        assert "Current and future semesters are excluded" in contents
        assert "AI GENERATED" in contents


def test_current_and_future_result_questions_are_not_documents():
    for profile_id in ("student_e", "student_j"):
        response = _chat(f"What was Student {profile_id[-1].upper()}'s SGPA in Semester 1?", profile_id)
        assert response.status_code == 200
        assert "no completed-result records" in response.json()["answer"]
        assert "documents" not in response.json()
