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

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config

app = FastAPI(title="VU AI Academic Advisor")
STATIC = config.ROOT / "advisor" / "static"
PROFILES = {p["id"]: p for p in json.loads((config.ROOT / "eval" / "profiles.json").read_text(encoding="utf-8"))}
ACCESS_PASSWORD = os.getenv("ADVISOR_ACCESS_PASSWORD", "")
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # Azure OpenAI transcription limit
if os.getenv("VERCEL"):
    MAX_AUDIO_BYTES = 4 * 1024 * 1024  # Leave room for multipart overhead under Vercel's request limit.
SUPPORTED_BATCHES = {2022, 2023, 2024, 2025, 2026}
PENDING_BATCH_QUESTIONS = {}
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
        return Response("<svg xmlns='http://www.w3.org/2000/svg' width='720' height='140'><text x='24' y='60'>No next semester is available for this profile.</text></svg>", media_type="image/svg+xml")
    from .tools import semester_plan
    result = semester_plan(p["batch"], semester)
    data = result["result"]
    slots = data.get("slots", [])
    height = 150 + max(1, len(slots)) * 66
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
    parts.append(f"<text x='36' y='{height-16}' font-family='system-ui' font-size='12' fill='#4a5468'>Source: {escape(str(source))}. Confirm registration with the university.</text></svg>")
    return Response("".join(parts), media_type="image/svg+xml")


@app.post("/api/chat")
def chat(body: ChatIn):
    if body.profile_id and body.profile_id not in PROFILES:
        raise HTTPException(400, "Unknown profile")
    thread = body.thread_id or uuid.uuid4().hex
    profile = PROFILES.get(body.profile_id)
    thread_key = f"{thread}:{body.profile_id or 'none'}"
    effective_message, clarification = _clarify_semester_batch(body.message, thread_key, profile)
    if clarification:
        return {"answer": clarification, "sources": [], "latency_ms": 0,
                "flags": {"tools": [], "cited": [], "asked_follow_up": True, "needs_batch": True},
                "thread_id": thread}
    if not config.chat_ready():
        return {**preview(effective_message), "thread_id": thread}
    from .graph import ask
    # A profile switch starts a fresh conversation so one student's record never leaks into another's thread.
    try:
        out = ask(effective_message, thread_id=thread_key, profile=profile)
    except Exception as exc:
        provider = config.chat_provider() or "configured"
        raise HTTPException(502, f"The {provider} chat request failed ({type(exc).__name__}). Check the provider key, model/deployment, endpoint, and quota.") from exc
    out["sources"] = with_source_links(out.get("sources", []))
    return {**out, "thread_id": thread}


def preview(message):
    """No LLM configured yet: show the retrieved evidence so the pipeline can still be checked end to end."""
    from .retrieval import search
    t0 = time.perf_counter()
    hits = search(message, k=4)
    sources = [{"id": f"S{i}", "citation": h["citation"], "tool": "search_documents", "snippet": h["text"][:600]} for i, h in enumerate(hits, 1)]
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
