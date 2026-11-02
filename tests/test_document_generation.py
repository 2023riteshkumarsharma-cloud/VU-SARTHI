import base64
from io import BytesIO
import re
import zipfile

import pymupdf
from docx import Document
from fastapi.testclient import TestClient

from advisor import academic_records, document_generation, document_intent, server


client = TestClient(server.app)


def _request(message, thread_id, profile_id=None):
    return client.post(
        "/api/chat",
        json={"message": message, "thread_id": thread_id, "profile_id": profile_id},
    )


def _download(response, index=0):
    metadata = response.json()["documents"][index]
    downloaded = client.get(metadata["download_url"])
    assert downloaded.status_code == 200
    assert downloaded.headers["content-type"] == metadata["content_type"]
    assert f'filename="{metadata["name"]}"' in downloaded.headers["content-disposition"]
    return metadata, downloaded.content


def _pdf_text(payload):
    document = pymupdf.open(stream=payload, filetype="pdf")
    return [page.get_text() for page in document]


def test_document_intent_recognizes_supported_documents_and_formats():
    cases = [
        ("Give me the Semester 3 marksheet in PDF.", "marksheet", "pdf"),
        ("Generate an academic performance report in Word.", "performance_report", "docx"),
        ("Give me the course structure for 2025 in PDF format.", "course_structure", "pdf"),
        ("Create a consolidated academic report for Student A in DOCX.", "consolidated_report", "docx"),
    ]
    for message, expected_type, expected_format in cases:
        intent = document_intent.parse_intent(message)
        assert intent.document_type == expected_type
        assert intent.format == expected_format


def test_missing_format_is_clarified_and_follow_up_generates_pdf():
    first = _request("Generate my academic profile", "doc-format-followup", "student_a")
    assert first.status_code == 200
    assert first.json()["flags"]["document_format_needed"] is True
    assert "PDF" in first.json()["answer"] and "DOCX" in first.json()["answer"]

    second = _request("PDF", "doc-format-followup", "student_a")
    assert second.status_code == 200
    assert second.json()["documents"][0]["format"] == "PDF"


def test_missing_student_and_semester_are_resolved_by_follow_up_questions():
    first = _request("Generate a marksheet in PDF.", "student-followup")
    assert first.json()["flags"]["asked_follow_up"] is True
    assert "Which synthetic student" in first.json()["answer"]

    second = _request("Student A", "student-followup")
    assert "Which semester" in second.json()["answer"]

    third = _request("Semester 3", "student-followup")
    assert third.status_code == 200
    assert third.json()["documents"][0]["format"] == "PDF"


def test_semester_marksheet_pdf_uses_recorded_data_and_watermark():
    response = _request(
        "Give me the Semester 3 marksheet in PDF.",
        "semester-3-pdf",
        "student_a",
    )
    assert response.status_code == 200
    metadata, payload = _download(response)
    assert metadata["format"] == "PDF"
    assert payload.startswith(b"%PDF")
    pages = _pdf_text(payload)
    assert len(pages) >= 1
    assert all("AI GENERATED" in page for page in pages)
    assert all(document_generation.FOOTER_TEXT in page for page in pages)
    text = "\n".join(pages)
    normalized_text = " ".join(text.split())
    semester_record = academic_records.semester_result("student_a", 3)
    assert "Aarav Sharma" in text
    assert "Semester 3" in text
    assert "Synthetic grading assumptions" in text
    assert "SGPA" in text and "CGPA" in text
    assert semester_record["courses"][0]["course_code"] in text
    assert str(semester_record["courses"][0]["marks_obtained"]) in text
    assert "Current and future semesters are excluded" in normalized_text
    assert "6.4" in text


def test_semester_marksheet_docx_has_editable_tables_and_watermark():
    response = _request(
        "Generate a Semester 3 marksheet in DOCX.",
        "semester-3-docx",
        "student_a",
    )
    assert response.status_code == 200
    metadata, payload = _download(response)
    assert metadata["format"] == "DOCX"
    doc = Document(BytesIO(payload))
    all_text = "\n".join([p.text for p in doc.paragraphs] +
                         [cell.text for table in doc.tables for row in table.rows for cell in row.cells])
    assert "Semester 3" in all_text
    semester_record = academic_records.semester_result("student_a", 3)
    assert semester_record["courses"][0]["course_code"] in all_text
    assert str(semester_record["courses"][0]["marks_obtained"]) in all_text
    assert "AI GENERATED" in doc.sections[0].header._element.xml
    assert document_generation.FOOTER_TEXT in doc.sections[0].footer._element.xml
    assert doc.tables


def test_all_completed_semester_marksheets_exclude_current_and_future():
    response = _request(
        "Generate marksheets for all completed semesters as PDF.",
        "all-completed-semesters",
        "student_a",
    )
    assert response.status_code == 200
    docs = response.json()["documents"]
    assert len(docs) == 4
    names = [entry["name"] for entry in docs]
    assert all(f"Semester_{semester}" in name for semester, name in enumerate(names, 1))
    assert not any("Semester_5" in name or "Semester_6" in name for name in names)
    for index in range(len(docs)):
        _, payload = _download(response, index)
        assert f"Semester {index + 1}" in "\n".join(_pdf_text(payload))


def test_all_ten_students_get_individual_completed_semester_eligibility():
    response = _request(
        "Generate marksheets for all completed semesters for all 10 students in PDF.",
        "ten-student-semesters",
    )
    assert response.status_code == 200
    expected = {
        (
            profile["label"].split(" – ", 1)[0].replace(" ", "_"),
            semester,
        )
        for profile in academic_records.PROFILES.values()
        for semester in academic_records.eligible_completed_semesters(profile)
    }
    documents = response.json()["documents"]
    actual = set()
    for metadata in documents:
        match = re.search(r"^(.*)_Semester_(\d+)_Marksheet\.pdf$", metadata["name"])
        assert match
        actual.add((match.group(1), int(match.group(2))))
    assert actual == expected


def test_current_and_future_semesters_are_never_generated():
    current = _request("Create a Semester 5 marksheet in PDF.", "current-semester", "student_a")
    assert current.status_code == 200
    assert "documents" not in current.json()
    assert "No eligible completed-semester" in current.json()["answer"]

    future = _request("Create a Semester 6 marksheet in PDF.", "future-semester", "student_a")
    assert future.status_code == 200
    assert "documents" not in future.json()


def test_consolidated_report_uses_the_same_persisted_history_as_chat():
    response = _request(
        "Create a consolidated academic performance report in Word.",
        "consolidated-data",
        "student_a",
    )
    metadata, payload = _download(response)
    assert metadata["format"] == "DOCX"
    doc = Document(BytesIO(payload))
    contents = "\n".join(
        [p.text for p in doc.paragraphs] +
        [cell.text for table in doc.tables for row in table.rows for cell in row.cells]
    )
    assert "6.4" in contents
    for semester in academic_records.history_for_profile("student_a")["semesters"].values():
        for course in semester["courses"]:
            assert course["course_code"] in contents
            assert str(course["marks_obtained"]) in contents
    assert "Synthetic grading assumptions" in contents
    assert "Synthetic demonstration data. Not an official university record." in contents


def test_student_profile_pdf_for_all_ten_synthetic_profiles():
    response = _request(
        "Generate an academic profile for all 10 students in PDF.",
        "all-ten-profiles",
    )
    assert response.status_code == 200
    docs = response.json()["documents"]
    assert len(docs) == 10 == len(academic_records.PROFILES)
    for index, profile in enumerate(academic_records.PROFILES.values()):
        _, payload = _download(response, index)
        text = "\n".join(_pdf_text(payload))
        assert "Synthetic demonstration data" in text
        assert "AI GENERATED" in text
        assert profile["label"].split(" – ", 1)[0] in text


def test_course_structure_docx_uses_curriculum_rows():
    response = _request(
        "Give me the course structure for 2025 in DOCX format.",
        "course-structure-docx",
    )
    assert response.status_code == 200
    metadata, payload = _download(response)
    assert metadata["format"] == "DOCX"
    doc = Document(BytesIO(payload))
    contents = "\n".join(
        [p.text for p in doc.paragraphs] +
        [cell.text for table in doc.tables for row in table.rows for cell in row.cells]
    )
    assert "2025" in contents and "57 curriculum row(s)" in contents
    assert "UCOR103" in contents
    assert "AI GENERATED" in doc.sections[0].header._element.xml


def test_course_details_document_uses_central_curriculum_records():
    response = _request(
        "Create course details for DATA103 in PDF.",
        "course-details",
    )
    assert response.status_code == 200
    _, payload = _download(response)
    text = "\n".join(_pdf_text(payload))
    assert "DATA103" in text
    assert "Programming in Python" in " ".join(text.split())
    assert "AI GENERATED" in text


def test_course_comparison_pdf_is_grounded_in_course_details():
    response = _request(
        "Compare COMP132 and COMP201 for 2025 and create a course comparison report in PDF.",
        "course-comparison",
    )
    assert response.status_code == 200
    metadata, payload = _download(response)
    assert metadata["format"] == "PDF"
    text = "\n".join(_pdf_text(payload))
    assert "COMP132" in text and "COMP201" in text
    assert "Course Comparison Report" in text
    assert "AI GENERATED" in text


def test_follow_up_converts_previous_document_to_another_format():
    first = _request(
        "Generate a Semester 3 marksheet in PDF.",
        "convert-same-document",
        "student_a",
    )
    assert first.status_code == 200
    assert first.json()["documents"][0]["format"] == "PDF"

    converted = _request("Convert this into DOCX.", "convert-same-document", "student_a")
    assert converted.status_code == 200
    metadata, payload = _download(converted)
    assert metadata["format"] == "DOCX"
    assert Document(BytesIO(payload)).sections[0].header._element.xml.count("AI GENERATED") == 1


def test_follow_up_same_report_in_word_changes_format():
    first = _request(
        "Generate an academic profile in PDF.",
        "same-report-format",
        "student_a",
    )
    assert first.json()["documents"][0]["format"] == "PDF"
    second = _request("Give me the same report in Word.", "same-report-format", "student_a")
    assert second.status_code == 200
    assert second.json()["documents"][0]["format"] == "DOCX"


def test_every_pdf_page_has_watermark_footer_and_docx_has_word_watermark():
    content = document_generation.DocumentContent(
        title="Multipage Academic Report",
        sections=[document_generation.Section(
            "Long data",
            headers=["Course", "Grade"],
            rows=[[f"COURSE{index:03}", "Not provided"] for index in range(180)],
        )],
        synthetic=True,
    )
    _, mime, pdf_payload = document_generation.render(content, "pdf")
    assert mime == "application/pdf"
    pages = _pdf_text(pdf_payload)
    assert len(pages) > 1
    assert all("AI GENERATED" in page for page in pages)
    assert all(document_generation.FOOTER_TEXT in page for page in pages)

    _, mime, docx_payload = document_generation.render(content, "docx")
    assert mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    with zipfile.ZipFile(BytesIO(docx_payload)) as archive:
        header_xml = archive.read("word/header1.xml").decode("utf-8")
        footer_xml = archive.read("word/footer1.xml").decode("utf-8")
    assert 'rotation:315' in header_xml
    assert "AI GENERATED" in header_xml
    assert document_generation.FOOTER_TEXT in footer_xml


def test_generated_downloads_are_actual_pdf_and_docx_mime_types():
    pdf_response = _request(
        "Generate a student profile in PDF.",
        "mime-pdf",
        "student_b",
    )
    pdf_metadata, pdf_payload = _download(pdf_response)
    assert pdf_metadata["content_type"] == "application/pdf"
    assert pdf_payload.startswith(b"%PDF")
    _, encoded = pdf_metadata["data_url"].split(",", 1)
    assert base64.b64decode(encoded) == pdf_payload

    docx_response = _request(
        "Generate a student profile in Word.",
        "mime-docx",
        "student_b",
    )
    docx_metadata, docx_payload = _download(docx_response)
    assert docx_metadata["content_type"] == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert zipfile.is_zipfile(BytesIO(docx_payload))


def test_visual_requests_are_not_misrouted_to_document_generation():
    assert server._document_clarification(
        "Generate an image of the 2025 course structure.",
        "visual-not-document",
        None,
    ) is None


def test_unrelated_follow_up_does_not_consume_document_clarification_or_repeat_files():
    pending_thread = "cancel-document-clarification"
    server.PENDING_DOCUMENT_INTENTS[pending_thread] = document_intent.intent_dict(
        document_intent.parse_intent("Generate a marksheet in PDF.")
    )
    assert server._document_clarification(
        "What is the attendance policy?",
        pending_thread,
        None,
    ) is None
    assert pending_thread not in server.PENDING_DOCUMENT_INTENTS

    previous_thread = "do-not-repeat-document"
    server.LAST_DOCUMENT_INTENTS[previous_thread] = document_intent.intent_dict(
        document_intent.parse_intent("Generate a Student A profile in PDF.")
    )
    assert server._document_clarification(
        "What does this mean?",
        previous_thread,
        None,
    ) is None


def test_student_record_service_reads_persisted_marks_sgpa_and_attendance():
    record = academic_records.academic_record("student_a")
    assert record["cgpa"] == 6.4
    assert record["marks"]
    assert record["sgpa"] == academic_records.semester_result("student_a", 4)["sgpa"]
    assert record["attendance"]
    assert record["academic_history"]["is_synthetic"] is True
    assert record["profile_cgpa"] == 6.4
    assert record["eligible_completed_semesters"] == [1, 2, 3, 4]
