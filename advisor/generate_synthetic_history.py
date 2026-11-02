"""Create the frozen synthetic transcript dataset from existing profiles and curriculum rows."""
import argparse
import hashlib
import json
import random

from . import academic_records, tools


_GRADE_SCALE = (
    (90, "A+", 10),
    (85, "A", 9),
    (80, "B+", 8),
    (75, "B", 7),
    (70, "C+", 6),
    (65, "C", 5),
    (60, "D", 4),
    (40, "E", 3),
    (0, "F", 0),
)
_PERFORMANCE_BASELINES = {
    "student_a": 72,
    "student_b": 81,
    "student_c": 48,
    "student_d": 76,
    "student_f": 88,
    "student_g": 74,
    "student_h": 65,
    "student_i": 74,
}
_FAILURES = {
    ("student_a", 2, "COMP201"): {
        "attendance": 86,
        "marks": 34,
        "reattempt_marks": 78,
        "reattempt_period": "Supplementary assessment after Semester 2, 2025",
    },
    ("student_c", 2, "MATH203"): {"attendance": 68, "marks": 58},
    ("student_c", 2, "COMP201"): {"attendance": 67, "marks": 32},
    ("student_d", 1, "DATA103"): {"attendance": 63, "marks": 76},
    ("student_g", 5, "DATA301"): {"attendance": 71, "marks": 53},
    ("student_h", 2, "MATH203"): {
        "attendance": 87,
        "marks": 35,
        "reattempt_marks": 74,
        "reattempt_period": "Supplementary assessment after Semester 2, 2026",
    },
    ("student_i", 3, "COMP203"): {"attendance": 84, "marks": 29},
}


def _grade(marks: int) -> tuple[str, int]:
    for minimum, grade, points in _GRADE_SCALE:
        if marks >= minimum:
            return grade, points
    raise AssertionError("The grade scale must cover every possible mark.")


def _seed(*values: object) -> int:
    source = "|".join(map(str, values)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(source).digest()[:8], "big")


def _split_marks(total: int, rng: random.Random) -> tuple[int, int]:
    lower = max(0, total - 60)
    upper = min(40, total)
    internal = rng.randint(lower, upper)
    return internal, total - internal


def _selected_curriculum_rows(profile_id: str, profile: dict, semester: int) -> list[dict]:
    result = tools.lookup_courses(batch=profile["batch"], semester=semester)["result"]
    rows = result.get("rows", [])
    failures = {
        course_code
        for (student_id, course_semester, course_code) in _FAILURES
        if student_id == profile_id and course_semester == semester
    }
    known_courses = set(profile.get("completed_courses") or [])
    known_courses.update(profile.get("failed_courses") or [])

    required = []
    choices: dict[str, list[dict]] = {}
    for row in rows:
        code = str(row.get("code") or "")
        if not code or code.startswith("("):
            continue
        slot = row.get("slot")
        if slot:
            choices.setdefault(str(slot), []).append(row)
        else:
            required.append(row)

    selected = list(required)
    for slot, options in sorted(choices.items()):
        preferred = [row for row in options if row["code"] in failures]
        if not preferred:
            preferred = [row for row in options if row["code"] in known_courses]
        rng = random.Random(_seed(profile_id, semester, slot))
        selected.append((preferred or options)[rng.randrange(len(preferred or options))])
    return selected


def _course_attempt(profile_id: str, semester: int, row: dict, baseline: int) -> dict:
    code = str(row["code"])
    rng = random.Random(_seed(profile_id, semester, code))
    spec = _FAILURES.get((profile_id, semester, code))
    marks = spec["marks"] if spec else max(40, min(98, round(baseline + rng.randint(-12, 12))))
    attendance = spec["attendance"] if spec else rng.randint(78, 99)
    internal, examination = _split_marks(marks, rng)
    attendance_failed = attendance < 75
    marks_failed = marks < 40
    failed = attendance_failed or marks_failed
    if attendance_failed and marks_failed:
        reason = "Attendance shortage and insufficient examination marks"
    elif attendance_failed:
        reason = "Attendance shortage"
    elif marks_failed:
        reason = "Insufficient examination marks"
    else:
        reason = None

    initial_grade, initial_points = _grade(marks)
    if failed:
        initial_grade, initial_points = "F", 0
    attempts = [{
        "attempt_number": 1,
        "academic_period": f"Semester {semester}",
        "attendance_percentage": attendance,
        "internal_assessment_marks": internal,
        "end_semester_examination_marks": examination,
        "marks_obtained": marks,
        "maximum_marks": 100,
        "grade": initial_grade,
        "grade_points": initial_points,
        "course_result_status": "FAIL" if failed else "PASS",
        "failure_reason": reason,
    }]

    reattempt_marks = spec.get("reattempt_marks") if spec else None
    if failed and reattempt_marks is not None:
        final_grade, final_points = _grade(reattempt_marks)
        if reattempt_marks < 40:
            raise ValueError(f"Reattempt for {profile_id} {code} must pass the synthetic mark threshold.")
        final_status = "CLEARED AFTER REATTEMPT"
        supplementary_status = "CLEARED"
        reattempt_period = spec["reattempt_period"]
        attempts.append({
            "attempt_number": 2,
            "academic_period": reattempt_period,
            "attendance_percentage": None,
            "internal_assessment_marks": None,
            "end_semester_examination_marks": reattempt_marks,
            "marks_obtained": reattempt_marks,
            "maximum_marks": 100,
            "grade": final_grade,
            "grade_points": final_points,
            "course_result_status": "PASS",
            "failure_reason": None,
        })
    elif failed:
        final_grade, final_points = "F", 0
        final_status = "BACKLOG"
        supplementary_status = "NOT CLEARED"
        reattempt_period = None
    else:
        final_grade, final_points = initial_grade, initial_points
        final_status = "PASS"
        supplementary_status = "NOT APPLICABLE"
        reattempt_period = None

    return {
        "course_code": code,
        "course_name": str(row["title"]),
        "credits": int(row["credits"]),
        "marks_obtained": marks,
        "maximum_marks": 100,
        "grade": final_grade,
        "grade_points": final_points,
        "attendance_percentage": attendance,
        "minimum_required_attendance_percentage": 75,
        "internal_assessment_marks": internal,
        "internal_assessment_maximum": 40,
        "end_semester_examination_marks": examination,
        "end_semester_examination_maximum": 60,
        "total_marks_obtained": marks,
        "minimum_passing_marks": 40,
        "initial_grade": initial_grade,
        "initial_grade_points": initial_points,
        "initial_result_status": "FAIL" if failed else "PASS",
        "course_result_status": final_status,
        "result_status": final_status,
        "failure_reason": reason,
        "supplementary_examination_status": supplementary_status,
        "reattempt_marks": reattempt_marks,
        "reattempt_academic_period": reattempt_period,
        "final_course_grade": final_grade,
        "final_grade_points": final_points,
        "attempts": attempts,
    }


def _calibrate_to_existing_cgpa(semesters: dict, target: float | None) -> None:
    if target is None:
        return
    courses = [
        course
        for semester in semesters.values()
        for course in semester["courses"]
    ]
    denominator = sum(course["credits"] for course in courses)
    if not denominator:
        return
    mark_for_points = {3: 50, 4: 62, 5: 67, 6: 72, 7: 77, 8: 82, 9: 87, 10: 94}
    states: dict[int, tuple[int, tuple[int, ...]]] = {0: (0, ())}
    for course in courses:
        if course["course_result_status"] == "BACKLOG":
            options = (0,)
        else:
            options = tuple(range(3, 11))
        next_states = {}
        for total, (cost, selected) in states.items():
            for points in options:
                preferred_mark = (
                    course["reattempt_marks"]
                    if course["course_result_status"] == "CLEARED AFTER REATTEMPT"
                    else course["marks_obtained"]
                )
                difference = mark_for_points.get(points, 0) - preferred_mark if points else 0
                new_total = total + course["credits"] * points
                new_cost = cost + difference * difference
                previous = next_states.get(new_total)
                if previous is None or new_cost < previous[0]:
                    next_states[new_total] = (new_cost, selected + (points,))
        states = next_states

    candidates = [
        state
        for points, state in states.items()
        if round(points / denominator, 2) == target
    ]
    if candidates:
        _, selected_points = min(candidates, key=lambda state: state[0])
    else:
        _, (_, selected_points) = min(
            states.items(),
            key=lambda item: (
                abs(round(item[0] / denominator, 2) - target),
                item[1][0],
            ),
        )

    for course, points in zip(courses, selected_points):
        if course["course_result_status"] == "BACKLOG":
            continue
        marks = mark_for_points[points]
        rng = random.Random(_seed(course["course_code"], course["semester"] if "semester" in course else 0, points))
        internal, examination = _split_marks(marks, rng)
        grade, grade_points = _grade(marks)
        course["grade"] = course["final_course_grade"] = grade
        course["grade_points"] = course["final_grade_points"] = grade_points
        course["result_status"] = course["course_result_status"]
        if course["course_result_status"] == "CLEARED AFTER REATTEMPT":
            course["reattempt_marks"] = marks
            course["attempts"][1]["end_semester_examination_marks"] = marks
            course["attempts"][1]["marks_obtained"] = marks
            course["attempts"][1]["grade"] = grade
            course["attempts"][1]["grade_points"] = grade_points
        else:
            course["marks_obtained"] = course["total_marks_obtained"] = marks
            course["initial_grade"] = grade
            course["initial_grade_points"] = grade_points
            course["internal_assessment_marks"] = internal
            course["end_semester_examination_marks"] = examination
            course["attempts"][0]["internal_assessment_marks"] = internal
            course["attempts"][0]["end_semester_examination_marks"] = examination
            course["attempts"][0]["marks_obtained"] = marks
            course["attempts"][0]["grade"] = grade
            course["attempts"][0]["grade_points"] = grade_points


def generate_dataset() -> dict:
    students = {}
    for profile_id, profile in academic_records.PROFILES.items():
        semester_records = {}
        weighted_points = 0
        attempted_credits = 0
        earned_credits = 0
        for semester in academic_records.eligible_completed_semesters(profile):
            courses = [
                _course_attempt(
                    profile_id,
                    semester,
                    row,
                    _PERFORMANCE_BASELINES.get(profile_id, 73),
                )
                for row in _selected_curriculum_rows(profile_id, profile, semester)
            ]
            if not courses:
                raise ValueError(f"No curriculum courses found for {profile_id}, semester {semester}.")
            term_credits = sum(course["credits"] for course in courses)
            term_points = sum(course["credits"] * course["final_grade_points"] for course in courses)
            weighted_points += term_points
            attempted_credits += term_credits
            earned_credits += sum(
                course["credits"]
                for course in courses
                if course["course_result_status"] in {"PASS", "CLEARED AFTER REATTEMPT"}
            )
            sgpa = round(term_points / term_credits, 2)
            cgpa = round(weighted_points / attempted_credits, 2)
            semester_records[str(semester)] = {
                "semester": semester,
                "academic_year": f"{profile['batch'] + (semester - 1) // 2}-{profile['batch'] + (semester - 1) // 2 + 1}",
                "sgpa": sgpa,
                "cgpa": cgpa,
                "attempted_credits": term_credits,
                "credits_earned_to_date": earned_credits,
                "result_status": (
                    "BACKLOG"
                    if any(course["course_result_status"] == "BACKLOG" for course in courses)
                    else "PASS"
                ),
                "courses": courses,
            }
        _calibrate_to_existing_cgpa(semester_records, profile.get("cgpa"))
        weighted_points = 0
        attempted_credits = 0
        earned_credits = 0
        for item in semester_records.values():
            term_credits = sum(course["credits"] for course in item["courses"])
            term_points = sum(course["credits"] * course["final_grade_points"] for course in item["courses"])
            item["sgpa"] = round(term_points / term_credits, 2)
            weighted_points += term_points
            attempted_credits += term_credits
            item["cgpa"] = round(weighted_points / attempted_credits, 2)
            earned_credits += sum(
                course["credits"]
                for course in item["courses"]
                if course["course_result_status"] in {"PASS", "CLEARED AFTER REATTEMPT"}
            )
            item["credits_earned_to_date"] = earned_credits
        students[profile_id] = {
            "student_id": profile_id,
            "student_name": profile["label"].split(" – ", 1)[0],
            "batch": profile["batch"],
            "current_semester": profile["current_semester"],
            "is_synthetic": True,
            "disclaimer": "Synthetic demonstration data. Not actual student results or an official university record.",
            "semesters": semester_records,
        }
    return {
        "version": 1,
        "is_synthetic": True,
        "disclaimer": "Synthetic demonstration data. Not actual student results or official university regulations.",
        "policy": academic_records.SYNTHETIC_POLICY,
        "grade_scale": [
            {"minimum_marks": minimum, "grade": grade, "grade_points": points}
            for minimum, grade, points in _GRADE_SCALE
        ],
        "students": students,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace the existing frozen dataset.")
    args = parser.parse_args()
    path = academic_records.HISTORY_PATH
    if path.exists() and not args.force:
        raise SystemExit(f"{path} already exists. Use --force only to intentionally replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(generate_dataset(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote one frozen synthetic transcript dataset for {len(academic_records.PROFILES)} profiles to {path}")


if __name__ == "__main__":
    main()
