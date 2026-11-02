"""Centralized access to the synthetic academic records used by chat and documents."""
from copy import deepcopy
from functools import lru_cache
import json
import re

from . import config

HISTORY_PATH = config.ROOT / "advisor" / "data" / "academic_history.json"
SYNTHETIC_POLICY = (
    "Synthetic demonstration assumptions, not verified university regulations: attendance "
    "minimum 75%; minimum total pass mark 40/100; internal assessment maximum 40 and "
    "end-semester examination maximum 60. Grade points: A+=10 (90+), A=9 (85–89), "
    "B+=8 (80–84), B=7 (75–79), C+=6 (70–74), C=5 (65–69), D=4 (60–64), "
    "E=3 (40–59), F=0 (below 40 or failed attendance). SGPA is credit-weighted within "
    "a semester and CGPA is credit-weighted across completed semesters, using the final "
    "grade after any reattempt. Failed courses contribute zero grade points and no earned "
    "credits until cleared. A reattempt replaces the failed grade in the GPA calculation, "
    "while the original marks, attendance, and failure reason remain in the attempt history."
)

PROFILES = {
    profile["id"]: profile
    for profile in json.loads(
        (config.ROOT / "eval" / "profiles.json").read_text(encoding="utf-8")
    )
}


@lru_cache(maxsize=1)
def _history_data() -> dict:
    if not HISTORY_PATH.is_file():
        raise FileNotFoundError(
            f"Synthetic academic history has not been generated: {HISTORY_PATH}"
        )
    return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))


def history_for_profile(profile_id: str) -> dict:
    if profile_id not in PROFILES:
        raise KeyError(f"Unknown synthetic student profile: {profile_id}")
    try:
        dataset = _history_data()
        history = deepcopy(dataset["students"][profile_id])
        history["policy"] = dataset["policy"]
        history["grade_scale"] = deepcopy(dataset["grade_scale"])
        return history
    except KeyError as exc:
        raise ValueError(f"No persisted synthetic academic history for {profile_id}.") from exc


def semester_result(profile_id: str, semester: int) -> dict | None:
    history = history_for_profile(profile_id)
    return deepcopy(history["semesters"].get(str(semester)))


def eligible_completed_semesters(profile: dict) -> list[int]:
    """Return semester numbers before the profile's current semester."""
    current = profile.get("current_semester")
    if not isinstance(current, int):
        return []
    return list(range(1, max(1, min(current, 9))))


def resolve_profile_ids(message: str, selected_profile_id: str | None = None) -> list[str]:
    """Resolve explicit synthetic student references, or use the selected profile."""
    lowered = message.casefold()
    if re.search(r"\b(?:all|each|every)\s+(?:10\s+)?students?\b|\ball\s+10\b", lowered):
        return list(PROFILES)

    matches = []
    letter = re.search(r"\bstudent\s+([a-j])\b", lowered)
    if letter:
        candidate = f"student_{letter.group(1)}"
        if candidate in PROFILES:
            matches.append(candidate)

    for profile_id, profile in PROFILES.items():
        name = profile["label"].split(" – ", 1)[0].casefold()
        if name in lowered and profile_id not in matches:
            matches.append(profile_id)

    if matches:
        return matches
    if re.search(r"\bstudent\s+[A-Za-z]\b", message, re.I) or re.search(
        r"\b(?:for|of|did|does|is|was|me)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b",
        message,
    ):
        return []
    return [selected_profile_id] if selected_profile_id in PROFILES else []


def academic_record(profile_id: str) -> dict:
    """Return a detached, authoritative snapshot without inventing absent fields."""
    profile = PROFILES.get(profile_id)
    if profile is None:
        raise KeyError(f"Unknown synthetic student profile: {profile_id}")
    completed_courses = profile.get("completed_courses")
    failed_courses = profile.get("failed_courses")
    history = history_for_profile(profile_id)
    semesters = history["semesters"]
    latest = semesters.get(str(max(map(int, semesters)))) if semesters else None
    return {
        "id": profile["id"],
        "label": profile["label"],
        "batch": profile.get("batch"),
        "current_semester": profile.get("current_semester"),
        "eligible_completed_semesters": eligible_completed_semesters(profile),
        "cgpa": latest["cgpa"] if latest else profile.get("cgpa"),
        "profile_cgpa": profile.get("cgpa"),
        "credits_earned": latest["credits_earned_to_date"] if latest else profile.get("credits_earned"),
        "completed_courses": list(completed_courses) if completed_courses is not None else None,
        "failed_courses": list(failed_courses) if failed_courses is not None else None,
        "grades": dict(profile.get("grades") or {}),
        "minor": profile.get("minor"),
        "notes": profile.get("notes"),
        "semester_course_results": history["semesters"],
        "marks": [
            course["marks_obtained"]
            for semester in semesters.values()
            for course in semester["courses"]
        ],
        "sgpa": latest["sgpa"] if latest else None,
        "attendance": [
            course["attendance_percentage"]
            for semester in semesters.values()
            for course in semester["courses"]
        ],
        "academic_history": history,
        "academic_policy": history["policy"],
        "is_synthetic": True,
    }


def validate_completed_semester(record: dict, semester: int) -> None:
    if semester not in record["eligible_completed_semesters"]:
        current = record.get("current_semester")
        raise ValueError(
            f"Semester {semester} is not an eligible completed semester for "
            f"this profile (current semester: {current})."
        )


def is_academic_history_query(message: str) -> bool:
    """Identify questions about stored semester results, not general curriculum queries."""
    return bool(re.search(
        r"\b(?:sgpa|cgpa|marks?|grades?|attendance|backlogs?|fail(?:ed|ure|ing)?|"
        r"failed courses?|"
        r"semester results?|semester[-\s]?wise|studied|study|highest[-\s]?scor(?:e|ing)|"
        r"clear(?:ed)? (?:the )?backlog|course results?)\b|"
        r"\b(?:courses?|subjects?)\b.{0,45}\b(?:took|taken|completed|complete)\b",
        message,
        re.I,
    ))


def _semester_number(message: str) -> int | None:
    match = re.search(r"\b(?:semester|sem)\s*(\d{1,2})\b", message, re.I)
    return int(match.group(1)) if match else None


def _format_result_table(rows: list[list[str]], include_attendance: bool = False) -> str:
    headers = ["Course Code", "Course", "Credits", "Marks", "Grade", "Grade Points", "Result"]
    if include_attendance:
        headers.append("Attendance")
    return (
        "| " + " | ".join(headers) + " |\n"
        "| " + " | ".join("---" for _ in headers) + " |\n"
        + "\n".join("| " + " | ".join(row) + " |" for row in rows)
    )


def _course_rows(semesters: list[dict], query: str) -> list[dict]:
    records = [course for semester in semesters for course in semester["courses"]]
    code = re.search(r"\b[A-Z]{2,6}\d{2,4}\b", query, re.I)
    if code:
        return [course for course in records if course["course_code"].casefold() == code.group(0).casefold()]

    lowered = query.casefold()
    exact = [
        course for course in records
        if course["course_name"].casefold() in lowered
    ]
    if exact:
        return exact
    if re.search(r"\bmath(?:ematics)?\b", lowered):
        return [course for course in records if course["course_code"].casefold().startswith("math")]
    return []


def answer_academic_question(
    message: str,
    selected_profile_id: str | None = None,
) -> str | None:
    """Answer result questions from persisted synthetic records without a model-generated file."""
    if not is_academic_history_query(message):
        return None

    ids = resolve_profile_ids(message, selected_profile_id)
    if not ids:
        explicit_student = re.search(r"\bstudent\s+([A-Za-z])\b", message, re.I)
        if explicit_student:
            candidate = explicit_student.group(1).upper()
            return f"Student {candidate} is not one of the available synthetic profiles."
        explicit_name = re.search(
            r"\b(?:for|of|did|does|is|was|me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b",
            message,
        )
        if explicit_name:
            return f"{explicit_name.group(1)} is not one of the available synthetic profiles."
        return (
            "I couldn't identify a matching synthetic student. Select a profile or name one of these students: "
            + ", ".join(
                f"Student {profile_id[-1].upper()} ({profile['label'].split(' – ', 1)[0]})"
                for profile_id, profile in PROFILES.items()
            )
        )

    semester_number = _semester_number(message)
    range_match = re.search(
        r"\bsemesters?\s*(\d{1,2})\s*(?:to|through|[-–])\s*(\d{1,2})\b",
        message,
        re.I,
    )
    multiple_students = len(ids) > 1
    replies = []
    for profile_id in ids:
        profile = PROFILES[profile_id]
        history = history_for_profile(profile_id)
        name = profile["label"].split(" – ", 1)[0]
        semesters_by_number = history["semesters"]
        if semester_number is not None and str(semester_number) not in semesters_by_number:
            current = profile.get("current_semester")
            replies.append(
                f"Completed results for Semester {semester_number} are not available for {name}. "
                f"The current semester is {current}; current and future semesters have no completed-result records."
            )
            continue

        if range_match:
            first, last = map(int, range_match.groups())
            selected_semesters = [
                semesters_by_number[str(number)]
                for number in range(first, last + 1)
                if str(number) in semesters_by_number
            ]
            omitted = [
                number for number in range(first, last + 1)
                if str(number) not in semesters_by_number
            ]
        elif semester_number is not None:
            selected_semesters = [semesters_by_number[str(semester_number)]]
            omitted = []
        else:
            selected_semesters = list(semesters_by_number.values())
            omitted = []

        if not selected_semesters:
            replies.append(
                f"No completed semester-result records are available for {name}. "
                "The profile is in Semester 1, so no results have been generated yet."
            )
            continue

        prefix = f"**{name} · Synthetic demonstration data**\n\n"
        lower = message.casefold()
        if re.search(r"\b(?:sgpa|cgpa)\b", lower) and (
            range_match or re.search(r"\bcompare\b|\bsemester[-\s]?wise\b", lower)
        ):
            table_rows = [
                [str(item["semester"]), f"{item['sgpa']:.2f}", f"{item['cgpa']:.2f}"]
                for item in selected_semesters
            ]
            table = (
                "| Semester | SGPA | CGPA |\n| --- | ---: | ---: |\n"
                + "\n".join("| " + " | ".join(row) + " |" for row in table_rows)
            )
            reply = prefix + table
            if omitted:
                reply += (
                    "\n\nCurrent/future or unavailable semesters omitted: "
                    + ", ".join(map(str, omitted))
                    + "."
                )
            replies.append(reply)
            continue

        if re.search(r"\bcgpa\b", lower):
            item = selected_semesters[-1]
            replies.append(prefix + f"CGPA after Semester {item['semester']}: **{item['cgpa']:.2f}**.")
            continue
        if re.search(r"\bsgpa\b", lower):
            item = selected_semesters[-1]
            replies.append(prefix + f"SGPA for Semester {item['semester']}: **{item['sgpa']:.2f}**.")
            continue

        courses = _course_rows(selected_semesters, message)
        if re.search(r"\bwhy\b|\b(?:failure reason|reason for failing)\b", lower):
            failed = [
                course for course in courses
                if course["failure_reason"]
            ]
            if not failed:
                replies.append(prefix + "No failed attempt matching that subject is recorded.")
                continue
            explanations = []
            for course in failed:
                status = course["course_result_status"]
                details = (
                    f"{course['course_name']} ({course['course_code']}): "
                    f"{course['failure_reason']} "
                    f"({course['attendance_percentage']}% attendance; "
                    f"{course['marks_obtained']}/{course['maximum_marks']} marks). "
                    f"Current status: {status}."
                )
                if course["reattempt_marks"] is not None:
                    details += (
                        f" Cleared after reattempt with {course['reattempt_marks']}/"
                        f"{course['maximum_marks']} marks and final grade "
                        f"{course['final_course_grade']}."
                    )
                explanations.append(details)
            replies.append(prefix + " ".join(explanations))
            continue

        if re.search(r"\b(?:clear(?:ed)?|pass(?:ed)?)\b.{0,30}\bbacklog\b", lower):
            selected = courses or [
                course
                for item in selected_semesters
                for course in item["courses"]
                if course["failure_reason"]
            ]
            if selected:
                details = [
                    f"{course['course_name']} ({course['course_code']}): "
                    f"{course['course_result_status']}."
                    for course in selected
                ]
                replies.append(prefix + " ".join(details))
            else:
                replies.append(prefix + "No failed course or backlog is recorded in that semester.")
            continue

        if re.search(
            r"\b(?:backlogs?|failed courses?|subjects? .*failed|failed .*subjects?|"
            r"fail(?:ed)?\b.{0,35}\bsubjects?)\b",
            lower,
        ):
            failed = [
                course
                for semester in selected_semesters
                for course in semester["courses"]
                if course["failure_reason"]
            ]
            if re.search(r"\bbacklogs?\b", lower):
                failed = [course for course in failed if course["course_result_status"] == "BACKLOG"]
            if not failed:
                replies.append(prefix + "No matching failed courses or uncleared backlogs are recorded.")
                continue
            table_rows = [
                [
                    course["course_code"],
                    course["course_name"],
                    course["failure_reason"],
                    course["course_result_status"],
                    str(course["marks_obtained"]),
                    f"{course['attendance_percentage']}%",
                ]
                for course in failed
            ]
            replies.append(
                prefix + "| Code | Course | Original failure reason | Current status | Original marks | Attendance |\n"
                "| --- | --- | --- | --- | ---: | ---: |\n"
                + "\n".join("| " + " | ".join(row) + " |" for row in table_rows)
            )
            continue

        if re.search(r"\bhighest[-\s]?scor(?:e|ing)\b|\bhighest marks?\b", lower):
            if not courses:
                courses = [course for item in selected_semesters for course in item["courses"]]
            if courses:
                top = max(courses, key=lambda course: course["marks_obtained"])
                replies.append(
                    prefix + f"Highest original-attempt mark: **{top['course_name']} "
                    f"({top['course_code']})**, {top['marks_obtained']}/{top['maximum_marks']}."
                )
            else:
                replies.append(prefix + "No course matches that subject or semester.")
            continue

        if courses and re.search(r"\b(?:grade|marks?|attendance|course|subject)\b", lower):
            include_attendance = "attendance" in lower or "fail" in lower or "backlog" in lower
            table_rows = [
                [
                    course["course_code"],
                    course["course_name"],
                    str(course["credits"]),
                    f"{course['marks_obtained']}/{course['maximum_marks']}",
                    course["final_course_grade"],
                    str(course["final_grade_points"]),
                    course["course_result_status"],
                ] + ([f"{course['attendance_percentage']}%"] if include_attendance else [])
                for course in courses
            ]
            replies.append(prefix + _format_result_table(table_rows, include_attendance))
            continue

        table_rows = [
            [
                course["course_code"],
                course["course_name"],
                str(course["credits"]),
                f"{course['marks_obtained']}/{course['maximum_marks']}",
                course["final_course_grade"],
                str(course["final_grade_points"]),
                course["course_result_status"],
            ]
            for item in selected_semesters for course in item["courses"]
        ]
        replies.append(
            prefix + _format_result_table(table_rows)
            + "\n\n**SGPA:** "
            + ", ".join(f"Semester {item['semester']}: {item['sgpa']:.2f}" for item in selected_semesters)
        )
    return "\n\n".join(replies)
