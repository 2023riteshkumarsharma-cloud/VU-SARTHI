"""Natural-language intent detection for generated academic documents."""
from dataclasses import asdict, dataclass
import re

from .academic_records import PROFILES, is_academic_history_query, resolve_profile_ids

FORMAT_RE = re.compile(r"\b(pdf|docx|doc|word)\b", re.I)
ACTION_RE = re.compile(r"\b(generate|create|make|give|provide|get|download|export|prepare|produce|convert)\b", re.I)
DOCUMENT_RE = re.compile(
    r"\b(document|marks?\s*sheets?|grades?\s*sheets?|transcripts?|academic\s+(?:performance|profile|summary|record|reports?)|"
    r"completed\s+semester\s+report|subject[-\s]wise\s+grade\s+report|"
    r"sgpa|cgpa|backlog|attendance|course\s+(?:details?|lists?|structure|comparison|credits?)|"
    r"programme?\s+structure|curricul(?:um|a)|academic\s+regulations?|admission\s+information|"
    r"eligibility\s+criteria|fee\s+structure|academic\s+calendar)\b",
    re.I,
)


@dataclass
class DocumentIntent:
    document_type: str
    format: str | None
    profile_ids: list[str]
    semester: int | None
    all_semesters: bool
    batch: int | None
    detailed: bool = False
    include_sgpa_cgpa: bool = False


def is_document_request(message: str) -> bool:
    has_format = bool(FORMAT_RE.search(message))
    has_action = bool(ACTION_RE.search(message))
    has_document = bool(DOCUMENT_RE.search(message))
    if is_academic_history_query(message):
        explicit_document = bool(re.search(
            r"\b(?:document|marks?\s*sheets?|grades?\s*sheets?|transcripts?|reports?|"
            r"pdf|docx|word|download|export)\b",
            message,
            re.I,
        ))
        return explicit_document and (has_action or has_format)
    return (has_format and (has_action or has_document)) or (has_action and has_document)


def _document_type(message: str) -> str:
    text = message.casefold()
    if "consolidated" in text:
        return "consolidated_report"
    if re.search(r"\bmarks?\s*sheets?\b|\bgrades?\s*sheets?\b|\btranscripts?\b", text):
        return "marksheet"
    if re.search(r"\b(?:grade|result|academic)\s+reports?\b|\bexport\s+(?:these\s+)?results?\b", text):
        return "performance_report"
    if "attendance" in text:
        return "attendance_report"
    if "subject-wise grade" in text or "subject wise grade" in text:
        return "subject_grade_report"
    if "backlog" in text or "failed course" in text:
        return "backlog_report"
    if "student profile" in text or "academic profile" in text:
        return "student_profile"
    if "sgpa" in text or "cgpa" in text:
        return "sgpa_cgpa_report"
    if "performance" in text:
        return "performance_report"
    if "academic report" in text:
        return "performance_report"
    if "summary" in text or "completed semester" in text:
        return "semester_summary"
    if "comparison" in text or re.search(r"\bcompare\b", text):
        return "course_comparison"
    if re.search(r"\bcourse\s+details?\b", text):
        return "course_details"
    if re.search(r"\b(course|curricul|programme structure|program structure)\b", text):
        return "course_structure"
    if "fee" in text:
        return "fee_structure"
    if "calendar" in text:
        return "academic_calendar"
    if "eligibility" in text or "admission" in text:
        return "admission_information"
    if "regulation" in text or "handbook" in text:
        return "academic_regulations"
    return "academic_information"


def parse_intent(message: str, selected_profile_id: str | None = None) -> DocumentIntent:
    format_match = FORMAT_RE.search(message)
    file_format = None
    if format_match:
        file_format = "pdf" if format_match.group(1).casefold() == "pdf" else "docx"
    semester_match = re.search(r"\b(?:semester|sem)\s*(\d{1,2})\b", message, re.I)
    year_match = re.search(r"\b(202[2-6])\b", message)
    all_semesters = bool(re.search(r"\ball\s+(?:completed\s+)?semesters?\b|\bevery\s+completed\s+semester\b", message, re.I))
    return DocumentIntent(
        document_type=_document_type(message),
        format=file_format,
        profile_ids=resolve_profile_ids(message, selected_profile_id),
        semester=int(semester_match.group(1)) if semester_match else None,
        all_semesters=all_semesters,
        batch=int(year_match.group(1)) if year_match else None,
        detailed=bool(re.search(r"\bmore\s+detailed\b|\bdetail(?:ed)?\b", message, re.I)),
        include_sgpa_cgpa=bool(re.search(r"\binclude\s+(?:the\s+)?(?:sgpa|cgpa)\b|\bsgpa\s+and\s+cgpa\b", message, re.I)),
    )


def intent_dict(intent: DocumentIntent) -> dict:
    return asdict(intent)


def intent_from_dict(value: dict) -> DocumentIntent:
    return DocumentIntent(**value)


def listed_profiles() -> str:
    return "; ".join(
        f"Student {profile_id.rsplit('_', 1)[-1].upper()}: {profile['label'].split(' – ', 1)[0]}"
        for profile_id, profile in PROFILES.items()
    )
