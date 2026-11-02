"""FastAPI app: chat + voice transcription + synthetic profiles + static UI.

    uvicorn advisor.server:app --reload        # then open http://127.0.0.1:8000
"""
import base64
import json
import os
import re
import secrets
import time
import uuid
from collections import OrderedDict

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import academic_records, config, document_generation, document_intent, tools, visuals

app = FastAPI(title="VU SARTHI")
STATIC = config.ROOT / "advisor" / "static"
PROFILES = academic_records.PROFILES
ACCESS_PASSWORD = os.getenv("ADVISOR_ACCESS_PASSWORD", "")
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # Azure OpenAI transcription limit
if os.getenv("VERCEL"):
    MAX_AUDIO_BYTES = 4 * 1024 * 1024  # Leave room for multipart overhead under Vercel's request limit.
SUPPORTED_BATCHES = {2022, 2023, 2024, 2025, 2026}
PENDING_BATCH_QUESTIONS = {}
PENDING_DOCUMENT_INTENTS = OrderedDict()
LAST_DOCUMENT_INTENTS = OrderedDict()
DOCUMENT_FILES = OrderedDict()
MAX_DOCUMENT_FILES = 100
MAX_DOCUMENT_THREADS = 500
SEMESTER_ONE_RE = re.compile(r"\b(?:semester|semster|sem)\s*(?:1|one|first)(?:st)?\b|\b(?:first|1st)\s+semester\b", re.I)
@app.middleware("http")
async def protect_hosted_demo(request: Request, call_next):
    """Optional shared-password gate for hosted demos; local runs stay unchanged by default."""
    if not ACCESS_PASSWORD or request.url.path == "/api/health":
        return await call_next(request)
    scheme, _, encoded = request.headers.get("authorization", "").partition(" ")
    try:
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8") if scheme.lower() == "basic" else ""
        username, separator, password = decoded.partition(":")
        authorized = bool(separator and secrets.compare_digest(username.encode("utf-8"), b"advisor") and
                          secrets.compare_digest(password.encode("utf-8"), ACCESS_PASSWORD.encode("utf-8")))
    except (ValueError, UnicodeDecodeError):
        authorized = False
    if not authorized:
        return Response("Sign in to use this academic advisor demo.", status_code=401,
                        headers={"WWW-Authenticate": 'Basic realm="VU Academic Advisor demo"'})
    return await call_next(request)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = None
    profile_id: str | None = None


class ResponseImageIn(BaseModel):
    answer: str = Field(min_length=1, max_length=20000)
    request: str = Field(default="", max_length=2000)


def _requests_response_image(message: str) -> bool:
    return visuals.image_requested(message)


def _response_infographic(answer: str, request: str = "") -> str:
    return visuals.render(answer, request)


def _course_table_scope(message: str) -> tuple[int, int | None] | None:
    if not _requests_response_image(message):
        return None
    years = {int(year) for year in re.findall(r"\b(202[2-6])\b", message)}
    if len(years) != 1:
        return None
    semester_match = re.search(r"\b(?:semester|sem)\s*(\d{1,2})\b", message, re.I)
    semester = int(semester_match.group(1)) if semester_match else None
    is_course_list = bool(re.search(r"\bcourses?\b|\ball\s+baskets?\b|\bfull\s+list\b", message, re.I))
    if not is_course_list or visuals._visual_type(message, []) != "table":
        return None
    return next(iter(years)), semester


def _course_table_batch(message: str) -> int | None:
    """Compatibility helper used by existing callers/tests for all-batch course images."""
    scope = _course_table_scope(message)
    return scope[0] if scope else None


def _svg_image_payload(svg: str, filename: str, title: str, row_count: int) -> dict:
    data = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return {
        "content_type": "image/svg+xml",
        "format": "svg",
        "file_name": filename,
        "title": title,
        "row_count": row_count,
        "data_url": f"data:image/svg+xml;base64,{data}",
    }


def _course_table_image(batch: int, semester: int | None = None, request: str = "") -> dict:
    from .tools import lookup_courses

    result = lookup_courses(batch=batch, semester=semester)
    course_rows = result["result"].get("rows", [])
    if not course_rows:
        raise HTTPException(404, f"No course rows were found for batch {batch}.")
    headers = ["Semester", "Course Code", "Course Title", "Basket", "L-T-P", "Credits", "Prerequisites"]
    rows = [
        [
            "" if value is None else str(value)
            for value in (
                course["semester"], course["code"], course["title"], course["basket"],
                course["L-T-P"], course["credits"], course["prerequisites"],
            )
        ]
        for course in course_rows
    ]
    title = f"Courses Offered – {batch}"
    if semester is not None:
        title += f" · Semester {semester}"
    heading = visuals.image_heading(request or title, "", "table", {"title": title, "headers": headers})
    try:
        svg = visuals.render_table_data(title, headers, rows, request=request)
    except ValueError as exc:
        raise HTTPException(422, f"Could not render the course data: {exc}") from exc
    return _svg_image_payload(svg, f"courses-offered-{batch}.svg", heading, len(rows))


def _unavailable(value) -> str:
    return "Not available in the synthetic source record" if value is None else str(value)


def _student_document_content(profile_id: str, document_type: str, semester: int | None,
                              detailed: bool = False, include_sgpa_cgpa: bool = False):
    record = academic_records.academic_record(profile_id)
    student_name = record["label"].split(" – ", 1)[0]
    title_map = {
        "marksheet": "Marksheet",
        "performance_report": "Academic Performance Report",
        "consolidated_report": "Consolidated Academic Report",
        "sgpa_cgpa_report": "SGPA and CGPA Report",
        "semester_summary": "Semester-wise Academic Summary",
        "student_profile": "Student Academic Profile",
        "subject_grade_report": "Subject-wise Grade Report",
        "backlog_report": "Backlog Report",
        "attendance_report": "Attendance Report",
    }
    heading = title_map.get(document_type, "Academic Information")
    if semester is not None:
        title = f"{student_name} Semester {semester} {heading}"
    else:
        title = f"{student_name} {heading}"
    history = record["academic_history"]
    semester_records = [
        history["semesters"][str(number)]
        for number in record["eligible_completed_semesters"]
        if str(number) in history["semesters"]
    ]
    if semester is not None:
        academic_records.validate_completed_semester(record, semester)
        semester_records = [academic_records.semester_result(profile_id, semester)]
    sections = [
        document_generation.Section(
            "Student information",
            headers=["Field", "Stored value"],
            rows=[
                ["Student", student_name],
                ["Admission batch", _unavailable(record["batch"])],
                ["Current semester", _unavailable(record["current_semester"])],
                ["Current CGPA", _unavailable(record["cgpa"])],
                ["Credits earned in persisted history", _unavailable(record["credits_earned"])],
                ["Minor", _unavailable(record["minor"])],
            ],
        ),
        document_generation.Section(
            "Synthetic grading assumptions",
            paragraphs=[
                history["disclaimer"],
                record["academic_policy"],
                "Current and future semesters are excluded. Course rows show the original semester attempt; "
                "where a supplementary attempt exists, it is listed separately and the final grade is identified.",
            ],
        ),
    ]

    if document_type == "sgpa_cgpa_report":
        sections.append(document_generation.Section(
            "Semester grade-point averages",
            headers=["Semester", "Academic year", "SGPA", "CGPA", "Result status"],
            rows=[
                [
                    str(item["semester"]), item["academic_year"], f"{item['sgpa']:.2f}",
                    f"{item['cgpa']:.2f}", item["result_status"],
                ]
                for item in semester_records
            ],
        ))
    elif document_type == "attendance_report":
        sections.append(document_generation.Section(
            "Course attendance",
            headers=["Semester", "Course code", "Course", "Attendance", "Minimum required", "Result"],
            rows=[
                [
                    str(item["semester"]), course["course_code"], course["course_name"],
                    f"{course['attendance_percentage']}%",
                    f"{course['minimum_required_attendance_percentage']}%",
                    course["course_result_status"],
                ]
                for item in semester_records for course in item["courses"]
            ],
        ))
    elif document_type == "backlog_report":
        failed_courses = [
            (item, course)
            for item in semester_records for course in item["courses"]
            if course["failure_reason"]
        ]
        sections.append(document_generation.Section(
            "Failed attempts and current clearance",
            paragraphs=["No failed attempts are recorded in the completed semesters."]
            if not failed_courses else [],
            headers=[
                "Semester", "Course code", "Course", "Attendance", "Original marks",
                "Minimum pass", "Failure reason", "Current status", "Reattempt marks",
                "Final grade",
            ] if failed_courses else [],
            rows=[
                [
                    str(item["semester"]), course["course_code"], course["course_name"],
                    f"{course['attendance_percentage']}%",
                    f"{course['marks_obtained']}/{course['maximum_marks']}",
                    str(course["minimum_passing_marks"]), course["failure_reason"],
                    course["course_result_status"],
                    str(course["reattempt_marks"]) if course["reattempt_marks"] is not None else "None",
                    course["final_course_grade"],
                ]
                for item, course in failed_courses
            ],
        ))
    else:
        for item in semester_records:
            courses = item["courses"]
            if document_type == "subject_grade_report":
                courses = [course for course in courses if course["failure_reason"] or course["grade"]]
            sections.append(document_generation.Section(
                f"Semester {item['semester']} · {item['academic_year']}",
                paragraphs=[
                    f"SGPA: {item['sgpa']:.2f} · CGPA after semester: {item['cgpa']:.2f} · "
                    f"Result: {item['result_status']} · Credits attempted: {item['attempted_credits']} · "
                    f"Credits earned to date: {item['credits_earned_to_date']}."
                ],
                headers=[
                    "Course code", "Course", "Credits", "Internal / end-semester",
                    "Total marks", "Attendance", "Grade", "Grade points", "Result status",
                ],
                rows=[
                    [
                        course["course_code"], course["course_name"], str(course["credits"]),
                        f"{course['internal_assessment_marks']}/40 · "
                        f"{course['end_semester_examination_marks']}/60",
                        f"{course['marks_obtained']}/{course['maximum_marks']}",
                        f"{course['attendance_percentage']}%",
                        course["final_course_grade"], str(course["final_grade_points"]),
                        course["course_result_status"],
                    ]
                    for course in courses
                ],
            ))
            failed_courses = [course for course in item["courses"] if course["failure_reason"]]
            if failed_courses:
                sections.append(document_generation.Section(
                    f"Semester {item['semester']} failed attempts and reattempts",
                    paragraphs=[
                        (
                            f"{course['course_code']} — {course['course_name']}: original attempt "
                            f"{course['marks_obtained']}/{course['maximum_marks']} marks "
                            f"(minimum passing marks {course['minimum_passing_marks']}), attendance "
                            f"{course['attendance_percentage']}% (minimum required "
                            f"{course['minimum_required_attendance_percentage']}%), initial result "
                            f"{course['initial_result_status']}; failure reason: "
                            f"{course['failure_reason']}. Supplementary status: "
                            f"{course['supplementary_examination_status']}; reattempt period: "
                            f"{course['reattempt_academic_period'] or 'not applicable'}; reattempt marks: "
                            f"{course['reattempt_marks'] if course['reattempt_marks'] is not None else 'none'}; "
                            f"final result: {course['course_result_status']}, grade "
                            f"{course['final_course_grade']} ({course['final_grade_points']} points)."
                        )
                        for course in failed_courses
                    ],
                ))
        if include_sgpa_cgpa and document_type != "sgpa_cgpa_report":
            sections.append(document_generation.Section(
                "Semester grade-point averages",
                headers=["Semester", "Academic year", "SGPA", "CGPA"],
                rows=[
                    [str(item["semester"]), item["academic_year"], f"{item['sgpa']:.2f}", f"{item['cgpa']:.2f}"]
                    for item in semester_records
                ],
            ))
        if not semester_records:
            sections.append(document_generation.Section(
                "Semester results",
                paragraphs=[
                    "No completed semesters are available. Current and future semesters do not have result records."
                ],
            ))
    return document_generation.DocumentContent(
        title=title,
        subtitle=f"Batch {record['batch']} · Synthetic student record",
        sections=sections,
        synthetic=True,
    )


def _course_document_content(intent: document_intent.DocumentIntent, profile: dict | None):
    batch = intent.batch or (profile or {}).get("batch")
    if batch is None:
        return None, "Which admission batch should I use? The course structure varies by batch (2022–2026)."
    semester = intent.semester
    lookup = tools.lookup_courses(batch=batch, semester=semester)
    result = lookup["result"]
    rows = result.get("rows", [])
    if not rows:
        return None, f"No course-structure rows were found for batch {batch}."
    table_rows = [
        [
            str(row.get(key) if row.get(key) is not None else "Not specified")
            for key in ("semester", "code", "title", "basket", "L-T-P", "credits", "prerequisites")
        ]
        for row in rows
    ]
    subtitle = f"Batch {batch}" + (f" · Semester {semester}" if semester is not None else "")
    sections = [document_generation.Section(
        "Course structure",
        paragraphs=[
            f"{result['count']} curriculum row(s) retrieved from the university curriculum database."
        ],
        headers=["Semester", "Code", "Course title", "Basket", "L-T-P", "Credits", "Prerequisites"],
        rows=table_rows,
    )]
    if result.get("truncated"):
        sections.append(document_generation.Section("Coverage note", paragraphs=[result["truncated"]]))
    if result.get("data_notes"):
        sections.append(document_generation.Section(
            "Source data notes",
            paragraphs=[note.get("detail", note.get("issue", "")) for note in result["data_notes"]],
        ))
    return document_generation.DocumentContent(
        title=f"Vidyashilp Course Structure {batch}" + (f" Semester {semester}" if semester else ""),
        subtitle=subtitle,
        sections=sections,
        sources=list(dict.fromkeys(lookup.get("sources", []))),
    ), None


def _course_comparison_content(message: str, intent: document_intent.DocumentIntent, profile: dict | None):
    mentions = tools.course_mentions(message)
    codes = list(dict.fromkeys(code for resolved in mentions.values() for code in resolved))
    if len(codes) < 2:
        return None, "Which two or more curriculum courses should I compare? Include their course codes or names."
    batch = intent.batch or (profile or {}).get("batch")
    rows, sources = [], []
    for code in codes:
        details = tools.course_details(code)
        sources.extend(details.get("sources", []))
        groups = details.get("result", [])
        if isinstance(groups, dict):
            groups = [groups]
        for item in groups:
            for offering in item.get("programme_plan", []):
                if batch is not None and offering.get("batch") != batch:
                    continue
                rows.append([
                    str(item.get("code", code)),
                    str(item.get("canonical_title", "Not specified")),
                    str(offering.get("batch", "Not specified")),
                    str(offering.get("semester", "Not specified")),
                    str(offering.get("L-T-P") or "Not specified"),
                    str(offering.get("credits") if offering.get("credits") is not None else "Not specified"),
                    str(offering.get("prerequisites") or "Not specified"),
                ])
    if not rows:
        return None, "No comparable course offerings were found in the curriculum data for that batch."
    return document_generation.DocumentContent(
        title="Course Comparison Report",
        subtitle=f"Batch {batch}" if batch is not None else "Curriculum comparison across available batches",
        sections=[document_generation.Section(
            "Retrieved course offerings",
            headers=["Code", "Course", "Batch", "Semester", "L-T-P", "Credits", "Prerequisites"],
            rows=rows,
        )],
        sources=list(dict.fromkeys(sources)),
    ), None


def _course_details_content(message: str, intent: document_intent.DocumentIntent):
    mentions = tools.course_mentions(message)
    codes = list(dict.fromkeys(code for resolved in mentions.values() for code in resolved))
    if not codes:
        return None, "Which course code or title should I use for the course-details document?"
    rows, sources = [], []
    for code in codes:
        details = tools.course_details(code)
        sources.extend(details.get("sources", []))
        groups = details.get("result", [])
        if isinstance(groups, dict):
            groups = [groups]
        for item in groups:
            for offering in item.get("programme_plan", []):
                if intent.batch is not None and offering.get("batch") != intent.batch:
                    continue
                rows.append([
                    str(item.get("code", code)),
                    str(item.get("canonical_title", "Not specified")),
                    str(offering.get("batch", "Not specified")),
                    str(offering.get("semester", "Not specified")),
                    str(offering.get("basket") or "Not specified"),
                    str(offering.get("L-T-P") or "Not specified"),
                    str(offering.get("credits") if offering.get("credits") is not None else "Not specified"),
                    str(offering.get("prerequisites") or "Not specified"),
                ])
    if not rows:
        return None, "No course details were found in the curriculum data for that batch."
    return document_generation.DocumentContent(
        title="Course Details",
        subtitle=f"Batch {intent.batch}" if intent.batch else "All available curriculum batches",
        sections=[document_generation.Section(
            "Retrieved course offerings",
            headers=["Code", "Course", "Batch", "Semester", "Basket", "L-T-P", "Credits", "Prerequisites"],
            rows=rows,
        )],
        sources=list(dict.fromkeys(sources)),
    ), None


def _retrieval_document_content(message: str, intent: document_intent.DocumentIntent):
    scope = "policy" if intent.document_type in {
        "admission_information", "fee_structure", "academic_calendar", "academic_regulations",
    } else "all"
    result = tools.search_documents(message, scope=scope, k=8)["result"]
    passages = result.get("passages", [])
    if not passages:
        return None, "I could not find supporting university information for that document request."
    sections = [
        document_generation.Section(
            title=passage.get("citation", "Retrieved university information"),
            paragraphs=[passage.get("text", "")],
        )
        for passage in passages
    ]
    if result.get("warning"):
        sections.insert(0, document_generation.Section(
            "Evidence note", paragraphs=[result["warning"]]
        ))
    title = {
        "fee_structure": "University Fee Structure",
        "academic_calendar": "Academic Calendar Information",
        "admission_information": "Admission and Eligibility Information",
        "academic_regulations": "Academic Regulations",
    }.get(intent.document_type, "University Academic Information")
    return document_generation.DocumentContent(
        title=title, sections=sections,
        sources=list(dict.fromkeys(passage.get("citation", "") for passage in passages if passage.get("citation"))),
    ), None


def _remember_document(file_name: str, media_type: str, payload: bytes) -> dict:
    document_id = secrets.token_urlsafe(24)
    DOCUMENT_FILES[document_id] = (file_name, media_type, payload)
    while len(DOCUMENT_FILES) > MAX_DOCUMENT_FILES:
        DOCUMENT_FILES.popitem(last=False)
    return {
        "name": file_name,
        "format": file_name.rsplit(".", 1)[-1].upper(),
        "content_type": media_type,
        "data_url": f"data:{media_type};base64,{base64.b64encode(payload).decode('ascii')}",
        "download_url": f"/api/documents/{document_id}",
    }


def _remember_document_intent(storage: OrderedDict, thread_key: str, intent: document_intent.DocumentIntent) -> None:
    storage.pop(thread_key, None)
    storage[thread_key] = document_intent.intent_dict(intent)
    while len(storage) > MAX_DOCUMENT_THREADS:
        storage.popitem(last=False)


def _generate_documents(message: str, intent: document_intent.DocumentIntent, profile: dict | None):
    generated = []
    sources = []
    personal_types = {
        "marksheet", "performance_report", "consolidated_report", "sgpa_cgpa_report",
        "semester_summary", "student_profile", "subject_grade_report", "backlog_report", "attendance_report",
    }
    if intent.document_type in personal_types:
        if not intent.profile_ids:
            return None, (
                "Which synthetic student profile should I use? Select a profile in the sidebar or name a student: "
                + document_intent.listed_profiles()
            )
        for profile_id in intent.profile_ids:
            selected = PROFILES[profile_id]
            semesters = (
                intent.semester if intent.semester is not None else None,
            ) if intent.semester is not None else (
                academic_records.eligible_completed_semesters(selected)
                if intent.all_semesters else (None,)
            )
            for semester in semesters:
                if semester is None and intent.document_type == "marksheet":
                    return None, "Which semester's marksheet do you need, or should I generate one for every completed semester?"
                if semester is not None:
                    record = academic_records.academic_record(profile_id)
                    if semester not in record["eligible_completed_semesters"]:
                        continue
                content = _student_document_content(
                    profile_id, intent.document_type, semester,
                    detailed=intent.detailed,
                    include_sgpa_cgpa=intent.include_sgpa_cgpa,
                )
                file_name, media_type, payload = document_generation.render(content, intent.format)
                generated.append(_remember_document(file_name, media_type, payload))
        if not generated:
            return None, "No eligible completed-semester documents are available for the selected student profile(s)."
    elif intent.document_type == "course_structure":
        content, error = _course_document_content(intent, profile)
        if error:
            return None, error
        file_name, media_type, payload = document_generation.render(content, intent.format)
        generated.append(_remember_document(file_name, media_type, payload))
        sources = content.sources
    elif intent.document_type == "course_comparison":
        content, error = _course_comparison_content(message, intent, profile)
        if error:
            return None, error
        file_name, media_type, payload = document_generation.render(content, intent.format)
        generated.append(_remember_document(file_name, media_type, payload))
        sources = content.sources
    elif intent.document_type == "course_details":
        content, error = _course_details_content(message, intent)
        if error:
            return None, error
        file_name, media_type, payload = document_generation.render(content, intent.format)
        generated.append(_remember_document(file_name, media_type, payload))
        sources = content.sources
    else:
        content, error = _retrieval_document_content(message, intent)
        if error:
            return None, error
        file_name, media_type, payload = document_generation.render(content, intent.format)
        generated.append(_remember_document(file_name, media_type, payload))
        sources = content.sources
    return {"documents": generated, "sources": [{"id": f"S{i}", "citation": cite, "tool": "document_retrieval", "snippet": ""} for i, cite in enumerate(sources, 1)]}, None


def _document_clarification(message: str, thread_key: str, profile: dict | None):
    if _requests_response_image(message) and not document_intent.FORMAT_RE.search(message):
        PENDING_DOCUMENT_INTENTS.pop(thread_key, None)
        return None
    pending = PENDING_DOCUMENT_INTENTS.get(thread_key)
    previous = LAST_DOCUMENT_INTENTS.get(thread_key)
    file_format_match = document_intent.FORMAT_RE.fullmatch(message.strip().rstrip("."))
    continuation = bool(re.search(
        r"\b(?:convert|turn|change|export)\s+(?:this|that|it)\b|"
        r"\b(?:same|this|that|above|previous)\s+(?:document|report|profile|marksheet|"
        r"course structure|result|content|information)\b|"
        r"\b(?:make|generate)\s+(?:this|that|it)\b|\bmore\s+detailed\b|"
        r"\binclude\s+(?:the\s+)?(?:sgpa|cgpa)\b|"
        r"\ball\s+completed\s+semesters?\b", message, re.I,
    ))
    if file_format_match and pending:
        intent = document_intent.intent_from_dict(pending)
        intent.format = "pdf" if file_format_match.group(1).casefold() == "pdf" else "docx"
    elif file_format_match and previous:
        intent = document_intent.intent_from_dict(previous)
        intent.format = "pdf" if file_format_match.group(1).casefold() == "pdf" else "docx"
    elif pending and not document_intent.is_document_request(message):
        intent = document_intent.intent_from_dict(pending)
        resolved_profiles = academic_records.resolve_profile_ids(message)
        semester_match = re.search(r"\b(?:semester|sem)\s*(\d)\b", message, re.I)
        year_match = re.search(r"\b(202[2-6])\b", message)
        all_semesters_reply = bool(re.search(r"\ball\s+completed\s+semesters?\b", message, re.I))
        if not (resolved_profiles or semester_match or year_match or all_semesters_reply):
            PENDING_DOCUMENT_INTENTS.pop(thread_key, None)
            return None
        if resolved_profiles:
            intent.profile_ids = resolved_profiles
        if semester_match:
            intent.semester = int(semester_match.group(1))
            intent.all_semesters = False
        if year_match:
            intent.batch = int(year_match.group(1))
        if all_semesters_reply:
            intent.all_semesters = True
            intent.semester = None
    elif document_intent.is_document_request(message):
        parsed = document_intent.parse_intent(message, profile.get("id") if profile else None)
        if continuation and previous:
            old = document_intent.intent_from_dict(previous)
            if parsed.document_type == "academic_information":
                parsed.document_type = old.document_type
            if not parsed.profile_ids:
                parsed.profile_ids = old.profile_ids
            parsed.semester = parsed.semester if parsed.semester is not None else old.semester
            parsed.batch = parsed.batch if parsed.batch is not None else old.batch
            parsed.all_semesters = parsed.all_semesters or old.all_semesters
            parsed.detailed = parsed.detailed or old.detailed
            parsed.include_sgpa_cgpa = parsed.include_sgpa_cgpa or old.include_sgpa_cgpa
            parsed.format = parsed.format or old.format
        elif parsed.format is None and previous:
            parsed.format = document_intent.intent_from_dict(previous).format
        intent = parsed
    elif continuation and previous:
        intent = document_intent.intent_from_dict(previous)
        if re.search(r"\bmore\s+detailed\b|\bdetail(?:ed)?\b", message, re.I):
            intent.detailed = True
        if re.search(r"\bsgpa\b|\bcgpa\b", message, re.I):
            intent.include_sgpa_cgpa = True
        if re.search(r"\ball\s+completed\s+semesters?\b", message, re.I):
            intent.all_semesters = True
            intent.semester = None
        format_match = document_intent.FORMAT_RE.search(message)
        if format_match:
            intent.format = "pdf" if format_match.group(1).casefold() == "pdf" else "docx"
    else:
        return None

    if profile and not intent.profile_ids and intent.document_type in {
        "marksheet", "performance_report", "consolidated_report", "sgpa_cgpa_report",
        "semester_summary", "student_profile", "subject_grade_report", "backlog_report", "attendance_report",
    }:
        intent.profile_ids = [profile["id"]]
    if intent.format is None and intent.document_type in {
        "marksheet", "performance_report", "consolidated_report", "sgpa_cgpa_report",
        "semester_summary", "subject_grade_report", "backlog_report", "attendance_report",
    }:
        intent.format = "pdf"
    if intent.format is None:
        _remember_document_intent(PENDING_DOCUMENT_INTENTS, thread_key, intent)
        return {
            "answer": "Which file format do you prefer: PDF or Microsoft Word (DOCX)?",
            "sources": [], "latency_ms": 0,
            "flags": {"tools": [], "cited": [], "asked_follow_up": True, "document_format_needed": True},
        }

    if intent.document_type == "course_structure" and intent.batch is None and profile:
        intent.batch = profile.get("batch")
    personal_types = {
        "marksheet", "performance_report", "consolidated_report", "sgpa_cgpa_report",
        "semester_summary", "student_profile", "subject_grade_report", "backlog_report", "attendance_report",
    }
    if intent.document_type in personal_types and not intent.profile_ids:
        _remember_document_intent(PENDING_DOCUMENT_INTENTS, thread_key, intent)
        answer = (
            "Which synthetic student profile should I use? Select one in the sidebar or name a student: "
            + document_intent.listed_profiles()
        )
        return {"answer": answer, "sources": [], "latency_ms": 0,
                "flags": {"tools": [], "cited": [], "asked_follow_up": True}}

    document_result, error = _generate_documents(message, intent, profile)
    if error:
        if "Which semester" in error or "Which admission batch" in error or "Which two or more" in error:
            _remember_document_intent(PENDING_DOCUMENT_INTENTS, thread_key, intent)
        return {"answer": error, "sources": [], "latency_ms": 0,
                "flags": {"tools": [], "cited": [], "asked_follow_up": True}}
    PENDING_DOCUMENT_INTENTS.pop(thread_key, None)
    _remember_document_intent(LAST_DOCUMENT_INTENTS, thread_key, intent)
    source_entries = [
        {"id": source["id"], "citation": source["citation"], "tool": source["tool"], "snippet": source["snippet"]}
        for source in document_result["sources"]
    ]
    return {
        "answer": f"Generated {len(document_result['documents'])} downloadable synthetic demonstration document(s).",
        "documents": document_result["documents"],
        "sources": with_source_links(source_entries),
        "latency_ms": 0,
        "flags": {"tools": ["academic_records", "document_generation"], "cited": [s["id"] for s in source_entries],
                  "generated_documents": True},
    }


def _visual_chat_message(message: str) -> str:
    """Ask the existing advisor for content, not for image generation."""
    cleaned = re.sub(
        r"\b(?:please|can you|could you|i want|i need|generate|create|make|draw|render|show|give|"
        r"visuali[sz]e|convert|turn|image|picture|infographic|visualization|visualisation|visual|poster|"
        r"flow\s*chart|chart|graph|mind\s*map|diagram|as an image|as a picture|into an image|"
        r"into a picture|into an infographic|into a visual|as a table|into a table)\b",
        " ",
        message,
        flags=re.I,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\r\n.,!?;:-")
    visual_type = visuals._visual_type(message, [])
    if visual_type in {"flowchart", "process_diagram", "architecture_diagram", "org_chart", "mind_map"}:
        instruction = "Return only the relevant information as concise, ordered steps or explicit hierarchy for a diagram."
    else:
        instruction = "Return the relevant factual data as a Markdown table with every distinct row and value preserved."
    subject = cleaned or "Use the relevant previous conversation context."
    return f"{subject}\n\n{instruction} Do not describe the rendering process."


def _clarify_semester_batch(message: str, thread_key: str, profile: dict | None):
    """Ask for admission batch before answering batch-dependent first-semester questions."""
    pending = PENDING_BATCH_QUESTIONS.get(thread_key)
    years = re.findall(r"\b(\d{4})\b", message)
    selected = next((int(year) for year in years if int(year) in SUPPORTED_BATCHES), None)
    if pending and selected is not None:
        PENDING_BATCH_QUESTIONS.pop(thread_key, None)
        return (f"{pending}\n\nThe student clarified their admission batch is {selected}. Answer the original question specifically for batch {selected}.", None)

    if pending:
        batch_reply = (len(message.strip()) <= 60 and
                       (bool(re.search(r"\b(?:batch|year|admission)\b|\b\d{4}\b", message, re.I)) or
                        bool(re.search(r"\b(?:not sure|don't know|do not know|unsure)\b", message, re.I))))
        if batch_reply:
            available = ", ".join(map(str, sorted(SUPPORTED_BATCHES)))
            return (None, f"The curriculum data covers batches {available}. Which admission batch should I use?")
        PENDING_BATCH_QUESTIONS.pop(thread_key, None)

    if profile or not SEMESTER_ONE_RE.search(message):
        return message, None
    if selected is not None:
        return message, None

    PENDING_BATCH_QUESTIONS[thread_key] = message
    if len(PENDING_BATCH_QUESTIONS) > 500:
        PENDING_BATCH_QUESTIONS.pop(next(iter(PENDING_BATCH_QUESTIONS)))
    available = ", ".join(map(str, sorted(SUPPORTED_BATCHES)))
    return (None, f"Which batch/admission year should I use? Semester 1 courses vary by batch. Available batches: {available}.")


@app.on_event("startup")
def warm_up():
    """Load local data and construct clients without spending API calls at startup."""
    from .retrieval import index
    index()
    if config.chat_ready():
        from . import graph
        graph._GRAPH = graph.build()
        graph.llm(True)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"chat": config.chat_ready(), "chat_provider": config.chat_provider(), "embeddings": config.EMB_PATH.exists(),
            "transcription": config.transcription_ready(), "transcription_provider": config.transcription_provider(),
            "knowledge_base": config.DB_PATH.exists()}


@app.get("/api/documents/{document_id}")
def download_document(document_id: str):
    stored = DOCUMENT_FILES.get(document_id)
    if stored is None:
        raise HTTPException(404, "Generated document not found or expired.")
    file_name, media_type, payload = stored
    return Response(
        content=payload,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{file_name}"'},
    )


def source_links(citation: str, snippet: str = ""):
    """Map visible citations to the local source files included with this project."""
    text = str(citation)
    lower = text.lower()
    links = []
    page_match = re.search(r"\bp{1,2}\.\s*(\d+)", lower)
    fragment = f"#page={page_match.group(1)}" if page_match else ""
    if "handbook" in lower:
        links.append({"label": "Open Student Handbook", "url": f"/api/source/handbook{fragment}"})
    if "sop" in lower:
        links.append({"label": "Open Student SOP", "url": f"/api/source/sop{fragment}"})
    if "minorcoursesforbtech_students.xlsx" in lower or ("minor" in lower and "workbook" in lower):
        links.append({"label": "Open minor courses workbook", "url": "/api/source/minors"})
    if "semester_spread_structures_sept_2026.xlsx" in lower or "semester spread" in lower or "struct_" in lower:
        links.append({"label": "Open semester spread workbook", "url": "/api/source/semester_spread"})
    if snippet and ("semester" in lower or "minor" in lower or "workbook" in lower or "excel" in lower or "sql over knowledge base" in lower):
        from urllib.parse import urlencode
        links.append({"label": "Open the data chunk used", "url": "/api/source/chunk?" + urlencode({"citation": text, "text": snippet})})
    return links


def with_source_links(sources):
    return [{**source, "links": source_links(source.get("citation", ""), source.get("snippet", ""))} for source in sources]


@app.get("/api/source/chunk", response_class=HTMLResponse)
def source_chunk(citation: str, text: str):
    """Show the exact structured row/passage returned as evidence for a citation."""
    from html import escape
    return HTMLResponse(f"<!doctype html><meta charset='utf-8'><title>Source data chunk</title>"
                        f"<main style='font:16px/1.6 system-ui;max-width:900px;margin:40px auto;padding:0 20px'>"
                        f"<h1>Data chunk used by the advisor</h1><p>{escape(citation)}</p>"
                        f"<pre style='white-space:pre-wrap;background:#f4f6fa;padding:20px;border-radius:8px'>{escape(text)}</pre></main>")


@app.get("/api/source/{source_key}")
def open_source(source_key: str):
    path = config.SOURCES.get(source_key)
    if path is None or not path.is_file():
        raise HTTPException(404, "Source document not found")
    media_type = {
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".pdf": "application/pdf",
    }.get(path.suffix.lower(), "application/octet-stream")
    return Response(
        content=path.read_bytes(),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{path.name}"'},
    )


@app.get("/api/profiles")
def profiles():
    return [{"id": p["id"], "label": p["label"], "batch": p["batch"], "current_semester": p["current_semester"],
             "cgpa": p["cgpa"], "minor": p["minor"], "notes": p["notes"],
             "credits_earned": p.get("credits_earned"), "failed_courses": p.get("failed_courses") or []} for p in PROFILES.values()]


@app.get("/api/profiles/{profile_id}/transcript", response_class=HTMLResponse)
def sample_transcript(profile_id: str):
    """Downloadable synthetic transcript sample, derived only from the demo profile."""
    from html import escape
    p = PROFILES.get(profile_id)
    if not p:
        raise HTTPException(404, "Unknown synthetic profile")
    courses = list(dict.fromkeys((p.get("completed_courses") or []) + (p.get("failed_courses") or [])))
    rows = "".join(f"<tr><td>{escape(code)}</td><td>{escape((p.get('grades') or {}).get(code, 'Not specified'))}</td>"
                   f"<td>{'Failed' if code in (p.get('failed_courses') or []) else 'Completed'}</td></tr>" for code in courses)
    html = ("<!doctype html><meta charset='utf-8'><title>Synthetic academic record</title>"
            "<main style='font:16px/1.6 system-ui;max-width:900px;margin:40px auto;padding:0 20px'>"
            "<h1>Synthetic transcript sample</h1><p>Demonstration data only. Not an official university record.</p>"
            f"<p><b>{escape(p['label'])}</b> · Batch {p['batch']} · Semester {p['current_semester']} · "
            f"CGPA {escape(str(p.get('cgpa') if p.get('cgpa') is not None else 'Not specified'))} · "
            f"Credits {escape(str(p.get('credits_earned', 'Not specified')))}</p>"
            f"<table border='1' cellpadding='8' cellspacing='0'><thead><tr><th>Course</th><th>Grade</th><th>Status</th></tr></thead><tbody>{rows}</tbody></table></main>")
    return HTMLResponse(html, headers={"Content-Disposition": f'attachment; filename="synthetic-{profile_id}-transcript.html"'})


@app.get("/api/profiles/{profile_id}/registration", response_class=HTMLResponse)
def sample_registration(profile_id: str):
    """Generate a clearly labeled proposed registration record from the synthetic profile."""
    from html import escape
    from . import tools
    p = PROFILES.get(profile_id)
    if not p:
        raise HTTPException(404, "Unknown synthetic profile")
    semester = int(p.get("current_semester") or 0) + 1
    plan = tools.semester_plan(p["batch"], semester)["result"] if semester <= 8 else {"slots": [], "total_credits": 0}
    rows = []
    for slot in plan.get("slots", []):
        for option in slot.get("options", []):
            code = option.get("code", "")
            check = tools.check_eligibility(code, batch=p["batch"], completed=p.get("completed_courses"),
                                           failed=p.get("failed_courses"), cgpa=p.get("cgpa"), target_semester=semester) if code and not code.startswith("(") else {"result": {"decision": "cannot_determine"}}
            rows.append(f"<tr><td>{escape(str(slot.get('slot','')))}</td><td>{escape(str(code))}</td>"
                        f"<td>{escape(str(option.get('title','')))}</td><td>{escape(check['result'].get('decision','cannot_determine').replace('_',' '))}</td></tr>")
    html = ("<!doctype html><meta charset='utf-8'><title>Synthetic draft registration</title>"
            "<main style='font:16px/1.6 system-ui;max-width:1000px;margin:40px auto;padding:0 20px'>"
            "<h1>Draft course registration sample</h1><p><b>Demonstration only.</b> This is not an official registration and no enrollment has been submitted.</p>"
            f"<p>{escape(p['label'])} · Batch {p['batch']} · Proposed semester {semester} · "
            f"{plan.get('total_credits',0)} planned credits</p>"
            f"<table border='1' cellpadding='8' cellspacing='0'><thead><tr><th>Slot</th><th>Code</th><th>Course</th><th>Data-based check</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
            "<p>Verify prerequisites, source conflicts, and registration dates with the Program Chair or Registrar.</p></main>")
    return HTMLResponse(html, headers={"Content-Disposition": f'attachment; filename="synthetic-{profile_id}-draft-registration.html"'})


@app.get("/api/plan/{profile_id}")
def agentic_plan(profile_id: str):
    """Deterministic advising pipeline: plan -> eligibility -> credit summary -> verification."""
    p = PROFILES.get(profile_id)
    if not p:
        raise HTTPException(404, "Unknown synthetic profile")
    next_semester = int(p.get("current_semester") or 0) + 1
    if next_semester > 8:
        return {"answer": "This synthetic profile is already at the final listed semester; no next-semester plan is available.", "sources": [], "flags": {"pipeline": ["verification"]}}
    from . import tools
    regulation_retrieval = tools.search_documents("registration eligibility prerequisites progression academic rules", scope="policy", k=2)
    plan = tools.semester_plan(p["batch"], next_semester)
    result = plan["result"]
    if not result.get("slots"):
        return {"answer": result.get("note", "No plan data is available for this semester."), "sources": [], "flags": {"pipeline": ["plan", "verification"]}}
    source_map = {}
    source_list = []
    for passage in regulation_retrieval.get("result", {}).get("passages", []):
        citation = passage.get("citation", "")
        if citation and citation not in source_map:
            source_map[citation] = f"S{len(source_list) + 1}"
            source_list.append({"id": source_map[citation], "citation": citation, "tool": "search_documents",
                                "snippet": passage.get("text", "")[:600]})
    eligible = []
    needs_review = []
    rows = []
    for slot in result["slots"]:
        options = []
        for option in slot["options"]:
            code = option.get("code")
            if not code or code.startswith("("):
                options.append(f"{option.get('title', 'Unresolved course')} (course code needs verification)")
                needs_review.append(option.get("title", "unresolved slot"))
                continue
            check = tools.check_eligibility(code, batch=p["batch"], completed=p.get("completed_courses"),
                                           failed=p.get("failed_courses"), cgpa=p.get("cgpa"), target_semester=next_semester)
            decision = check["result"].get("decision", "cannot_determine")
            label = f"{code} {option.get('title', '')}: {decision.replace('_', ' ')}"
            options.append(label)
            (eligible if decision == "eligible" else needs_review).append(label)
            for cite in check.get("sources", []):
                if cite not in source_map:
                    source_map[cite] = f"S{len(source_list) + 1}"
                    source_list.append({"id": source_map[cite], "citation": cite, "tool": "check_eligibility",
                                        "snippet": "; ".join(check["result"].get("reasons", []))[:600]})
        src_idx = slot.get("src")
        slot_source_id = ""
        if src_idx and src_idx <= len(plan.get("sources", [])):
            cite = plan["sources"][src_idx - 1]
            if cite not in source_map:
                source_map[cite] = f"S{len(source_list) + 1}"
                source_list.append({"id": source_map[cite], "citation": cite, "tool": "semester_plan",
                                    "snippet": f"{slot['basket']} · {slot['slot']}: {json.dumps(slot['options'], ensure_ascii=False)}; {slot['credits']} credits"})
            slot_source_id = source_map[cite]
        rows.append(f"| {slot['basket']} · {slot['slot']} | {'; '.join(options)} [{slot_source_id}] | {slot['credits']} |")
    for idx, source in enumerate(source_list, 1):
        source["id"] = f"S{idx}"
    answer = (f"## Draft plan for semester {next_semester} · batch {p['batch']}\n\n"
              f"The plan lists {result['total_credits']} credits for {result.get('term', 'this semester')}. Eligibility checks use the selected synthetic profile.\n\n"
              "| Basket / slot | Courses and deterministic eligibility | Slot credits |\n|---|---|---:|\n" + "\n".join(rows) +
              "\n\n**Pipeline:** regulation/curriculum data retrieved → eligibility checked → credits summarized → unresolved or non-eligible results flagged.\n\n")
    if needs_review:
        answer += "**Needs review:** " + "; ".join(needs_review[:12]) + ". Confirm prerequisites, unresolved source issues, and actual registration with the university before acting.\n\n"
    if eligible:
        answer += "**Eligible per available data:** " + "; ".join(eligible[:12]) + ". This is an advisory check, not registration approval."
    return {"answer": answer, "sources": with_source_links(source_list),
            "flags": {"pipeline": ["regulation_retrieval", "course_eligibility", "credit_calculation", "recommendation", "verification"],
                      "tools": ["semester_plan", "check_eligibility"], "human_confirmation_required": True}}


@app.get("/api/plan/{profile_id}/infographic.svg")
def plan_infographic(profile_id: str):
    """Locally render a lightweight course structure graphic; no image service/API is used."""
    from html import escape
    from fastapi.responses import Response
    p = PROFILES.get(profile_id)
    if not p:
        raise HTTPException(404, "Unknown synthetic profile")
    semester = int(p.get("current_semester") or 0) + 1
    if semester > 8:
        return Response("<svg xmlns='http://www.w3.org/2000/svg' width='720' height='140'><text x='24' y='60'>No next semester is available for this profile.</text><text x='700' y='126' text-anchor='end' font-family='system-ui' font-size='12' fill='#687080'>AI GENERATED IMAGE</text></svg>", media_type="image/svg+xml")
    from .tools import semester_plan
    result = semester_plan(p["batch"], semester)
    data = result["result"]
    slots = data.get("slots", [])
    height = 174 + max(1, len(slots)) * 66
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='1000' height='{height}' viewBox='0 0 1000 {height}'>",
             "<rect width='100%' height='100%' rx='20' fill='#f6f7fa'/>",
             f"<text x='36' y='48' font-family='system-ui' font-size='27' font-weight='700' fill='#092f87'>Draft course structure · Batch {p['batch']} · Semester {semester}</text>",
             f"<text x='36' y='80' font-family='system-ui' font-size='16' fill='#4a5468'>{escape(str(data.get('term', '')))} · {data.get('total_credits', 0)} total credits · Synthetic profile</text>"]
    for i, slot in enumerate(slots):
        y = 104 + i * 66
        names = " / ".join((f"{o.get('code', '')} {o.get('title', '')}").strip() for o in slot.get("options", []))
        parts.extend([f"<rect x='28' y='{y}' width='944' height='54' rx='10' fill='white' stroke='#d8dfeb'/>",
                      f"<text x='46' y='{y+22}' font-family='system-ui' font-size='14' font-weight='650' fill='#ac162c'>{escape(str(slot.get('basket','')))} · {escape(str(slot.get('slot','')))}</text>",
                      f"<text x='300' y='{y+22}' font-family='system-ui' font-size='14' fill='#131b2e'>{escape(names[:88])}</text>",
                      f"<text x='910' y='{y+22}' text-anchor='end' font-family='system-ui' font-size='14' fill='#092f87'>{escape(str(slot.get('credits','')))} cr</text>"])
    source = result.get("sources", ["No curriculum source"])[0]
    parts.append(f"<text x='36' y='{height-40}' font-family='system-ui' font-size='12' fill='#4a5468'>Source: {escape(str(source))}. Confirm registration with the university.</text>")
    parts.append(f"<text x='976' y='{height-12}' text-anchor='end' font-family='system-ui' font-size='12' font-weight='600' fill='#687080'>AI GENERATED IMAGE</text></svg>")
    return Response("".join(parts), media_type="image/svg+xml")


@app.post("/api/response-image.svg")
def response_image(body: ResponseImageIn):
    try:
        image = _response_infographic(body.answer, body.request)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return Response(image, media_type="image/svg+xml")


@app.post("/api/chat")
def chat(body: ChatIn):
    if body.profile_id and body.profile_id not in PROFILES:
        raise HTTPException(400, "Unknown profile")
    thread = body.thread_id or uuid.uuid4().hex
    profile = academic_records.academic_record(body.profile_id) if body.profile_id else None
    course_scope = _course_table_scope(body.message)
    if course_scope is not None:
        batch, semester = course_scope
        visual = _course_table_image(batch, semester, body.message)
        return {
            "answer": "",
            "sources": [],
            "latency_ms": 0,
            "flags": {"tools": ["lookup_courses"], "cited": [], "generate_image": True, "visual_only": True},
            "visual": visual,
            "thread_id": thread,
        }
    if (
        academic_records.is_academic_history_query(body.message)
        and not document_intent.is_document_request(body.message)
        and not visuals.image_requested(body.message)
    ):
        answer = academic_records.answer_academic_question(body.message, body.profile_id)
        if answer is not None:
            return {
                "answer": answer,
                "sources": [],
                "latency_ms": 0,
                "flags": {"tools": ["academic_records"], "cited": [], "synthetic_data": True},
                "thread_id": thread,
            }
    thread_key = f"{thread}:{body.profile_id or 'none'}"
    document_result = _document_clarification(body.message, thread_key, profile)
    if document_result is not None:
        return {**document_result, "thread_id": thread}
    effective_message, clarification = _clarify_semester_batch(body.message, thread_key, profile)
    if clarification:
        return {"answer": clarification, "sources": [], "latency_ms": 0,
                "flags": {"tools": [], "cited": [], "asked_follow_up": True, "needs_batch": True},
                "thread_id": thread}
    if not config.chat_ready():
        preview_message = _visual_chat_message(effective_message) if _requests_response_image(body.message) else effective_message
        result = {**preview(preview_message), "thread_id": thread}
        if _requests_response_image(body.message):
            result["flags"]["generate_image"] = True
        return result
    from .graph import ask
    # A profile switch starts a fresh conversation so one student's record never leaks into another's thread.
    try:
        request_message = _visual_chat_message(effective_message) if _requests_response_image(body.message) else effective_message
        out = ask(request_message, thread_id=thread_key, profile=profile)
    except Exception as exc:
        provider = config.chat_provider() or "configured"
        raise HTTPException(502, f"The {provider} chat request failed ({type(exc).__name__}). Check the provider key, model/deployment, endpoint, and quota.") from exc
    out["sources"] = with_source_links(out.get("sources", []))
    if _requests_response_image(body.message):
        out.setdefault("flags", {})["generate_image"] = True
    return {**out, "thread_id": thread}


def preview(message):
    """No LLM configured yet: show the retrieved evidence so the pipeline can still be checked end to end."""
    from .retrieval import search
    from .tools import remove_basket_codes

    t0 = time.perf_counter()
    hits = search(message, k=4)
    sources = [{"id": f"S{i}", "citation": h["citation"], "tool": "search_documents", "snippet": remove_basket_codes(h["text"][:600])} for i, h in enumerate(hits, 1)]
    sources = with_source_links(sources)
    answer = ("**Preview mode** – no chat API provider is configured yet (fill in `.env`), so no answer is generated. "
              "These are the passages the advisor would ground its answer in:\n\n" + "\n".join(f"- [{s['id']}] {s['citation']}" for s in sources))
    return {"answer": answer, "sources": sources, "latency_ms": round((time.perf_counter() - t0) * 1000),
            "flags": {"tools": ["search_documents"], "cited": [s["id"] for s in sources], "preview": True}}


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    provider = config.transcription_provider()
    if provider is None:
        raise HTTPException(503, "Voice input needs an Azure transcription deployment or OPENAI_API_KEY in .env")
    data = await audio.read()
    if not data or len(data) > MAX_AUDIO_BYTES:
        max_audio_mb = MAX_AUDIO_BYTES // (1024 * 1024)
        raise HTTPException(400, f"Audio must be between 1 byte and {max_audio_mb} MB")
    if provider == "azure":
        from openai import AzureOpenAI
        client = AzureOpenAI(azure_endpoint=config.AZURE_ENDPOINT, api_key=config.AZURE_API_KEY, api_version=config.TRANSCRIBE_API_VERSION)
        model = config.TRANSCRIBE_DEPLOYMENT
    else:
        from openai import OpenAI
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        model = config.OPENAI_TRANSCRIBE_MODEL
    try:
        r = client.audio.transcriptions.create(model=model, file=(audio.filename or "speech.webm", data, audio.content_type or "audio/webm"),
                                               prompt="Vidyashilp University academic advising. Course codes like DATA301, COMP201, MATH203; CGPA, L-T-P, semester, minor.")
    except Exception as exc:
        raise HTTPException(502, f"The {provider} transcription request failed ({type(exc).__name__}). Check the API key, model/deployment, and quota.") from exc
    return {"text": r.text}


app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/assets", StaticFiles(directory=config.ROOT / "assets"), name="assets")  # official logo + campus photo
