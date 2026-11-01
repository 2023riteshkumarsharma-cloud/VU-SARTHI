# VU AI Academic Advisor (DATA308)

A grounded academic advisor for Vidyashilp University's B.Tech (DS) programme, batches 2022–2026. It answers exact
curriculum questions (by batch, semester, month, credits, L-T-P, prerequisites, baskets, minors) and policy questions
(Student Handbook, SOP). Every answer cites the exact sheet cell or handbook page. It asks follow-up questions when
information is missing and flags conflicting sources. You can type or speak.

## Quick start

```bash
pip install -r requirements.txt
python -m advisor.ingest.build          # rebuild knowledge_base/ from the 4 source files (fails if validation fails)
python -m pytest tests                  # 10 tests, no credentials needed
cp .env.example .env                    # fill in the API provider credentials you have
python -m advisor.ingest.build          # re-run once .env is filled: adds dense embeddings (Bedrock, Azure, or OpenAI; cached)
uvicorn advisor.server:app              # open http://127.0.0.1:8000
python eval/run_eval.py --judge         # 4-stage evaluation -> eval/results/
```

Chat supports Azure OpenAI, OpenAI, Groq, and Gemini through the credentials in `.env`; `LLM_PROVIDER=auto` chooses a configured provider (Azure first). AWS Bedrock is optional for embeddings. Without chat credentials the web app runs in **preview mode**: it shows the passages retrieval would ground an answer in.

## Data: what is converted into what

| Source | Converted to | Used by |
|---|---|---|
| `Semester_Spread_Structures_Sept_2026.xlsx`: 5 semester-spread grids | `offerings` (262 plan slots) and `options` (282 courses; paired cells such as `MATH401/COMP401` split, with per-course L-T-P taken from the title text) in SQLite | deterministic tools, course cards |
| same workbook: 5 `Struct_*` sheets | `baskets` (credit requirements), `structure_courses` (title-linked to codes) | credit structure, electives |
| `MinorCoursesforBTech_Students.xlsx`: 7 minors | `minors` (incl. TBA-credit placeholders), `minor_batches` (24-credit totals) | minor tool, cards |
| Student Handbook PDF (92 pp) | `knowledge_base/curated/handbook.md`: PyMuPDF plus reviewed patches (grade table image transcribed, formulas, mangled tables); 128 clause/page-aware chunks | hybrid retrieval |
| SOP PDF (6 pp, two-column) | `knowledge_base/curated/sop.md`, verbatim hand transcription with page anchors; 16 chunks | hybrid retrieval |
| derived | `terms` (batch × semester → months; batch year = admission year, Handbook §1.2), `issues` (135 detected conflicts/inconsistencies + 3 curated policy conflicts) | calendar tool, conflict flags |

Every record keeps provenance (`sheet` + cell range, or PDF/printed page). `knowledge_base/VALIDATION_REPORT.md` lists
the 42 build checks, including:

- every grid cell consumed
- each batch totals 180 credits
- basket sums vs Struct sheet
- each minor totals 24 credits
- prerequisite resolution
- ~20 hand-verified golden facts

`knowledge_base/LEGACY_PACKAGE_AUDIT.md` documents why the earlier `Knowledge_package/` was replaced. Its 2023 batch
had semesters 4–6 wrong or missing, and its records carry no batch field.

## Architecture

```
Excel ─┐ adapters → SourceUnits (IR + provenance) → validate → SQLite (exact facts) ─┐
PDF ───┘                                  └→ curated Markdown → chunks ┐          │
                                  SQLite → per-batch cards ────────────┴→ BM25 + dense + RRF
question (text | voice → Azure transcription) → LangGraph: agent (LLM + 11 tools) ⇄ tools → verify → answer + [S#] sources
```

### Optional extensions (no additional API credentials)

- **Synthetic records:** `eval/profiles.json` remains the source of demonstration student profiles. The profile panel can open a generated HTML transcript sample for each profile. It is explicitly marked synthetic and is not an official student record.
- **Speech output:** each answer can be read aloud using the browser's built-in Speech Synthesis API. This runs locally in the browser and uses no AI service. Browser voices and pronunciation vary; verify course codes and regulation details in the written answer.
- **Course infographic:** a selected synthetic profile can open an SVG course structure for its next semester. The SVG is rendered from the existing deterministic semester-plan data; it does not use an image model. The plan view also offers a deterministic draft pipeline: curriculum retrieval → eligibility checks → credit summary → flagged recommendation → verification. It does not register students. Confirm eligibility, conflicts, and registration with the university.
- **Source chunks:** source cards continue to link to the original files. Excel-backed cards also offer “Open the data chunk used,” showing the exact row or passage returned as evidence. The chunk link is generated from the tool's returned evidence snippet.
- **Existing voice input:** audio is transcribed using a configured Azure/OpenAI transcription service, or browser speech recognition when available. This is an input modality and still depends on browser support, microphone permission, and (for server transcription) the configured provider's quota.

**Models/tools and limits:** chat generation uses the configured existing provider (Azure OpenAI, OpenAI, Groq, or Gemini); retrieval uses BM25 and, when an existing compatible embedding index is available, dense embeddings. The deterministic planner uses Python tools over the validated SQLite curriculum and profile JSON. Browser speech uses the browser's speech synthesis engine. The course map uses generated SVG. No new API keys or model services are required for speech output, the infographic, synthetic records, or the optional planning pipeline. The planner can surface bad/missing source data but cannot decide policy exceptions; people remain responsible for official eligibility decisions and registration.

The planner is intended to be compared with the existing `full` RAG + tools evaluation stage using the same synthetic profiles and cases. It is a transparent sequence of deterministic tool calls rather than autonomous agents, so its main benefit is a repeatable per-student draft plan; its limitations include the current coverage and quality of the source data and the profile's completeness.

- **Tools** (`advisor/tools.py`): pure Python over read-only SQLite:
  - `lookup_courses`, `course_details`, `semester_plan`, `credit_structure`, `minor_info`
  - `prerequisite_chain`, `academic_calendar`
  - `check_eligibility`: offered to the batch? prerequisites passed or failed? progression CGPA?
  - `list_data_issues`, `search_documents` (hybrid retrieval)
  - `run_sql`: SELECT-only, row-limited
- **Grounding**: the tools node numbers every source S1, S2, …. The `verify` node strips any citation no tool
  produced and computes flags for follow-up, conflict and insufficient information.
- **Adding a format** (DOCX/PPTX/HTML/CSV/OCR): write `extract(path) -> list[SourceUnit]` and register it in
  `advisor/ingest/__init__.py`. Re-ingestion is hash-tracked (`sources.json`), and embeddings are cached per chunk text.
- **Embeddings**: AWS Bedrock `amazon.titan-embed-text-v2:0` (1024-d) when `BEDROCK_EMBED_MODEL` + AWS keys are set, else Azure OpenAI or OpenAI when configured. Without an embedding provider, retrieval falls back to BM25. The model id is stored with the vectors so queries always use the same model; the weak-evidence cut-off (cosine < 0.30) was calibrated on in-scope vs off-topic probes.
- **Deliberately not used**: Neo4j/GraphRAG/RAPTOR, a reranker, multi-agent orchestration. The corpus is about
  700 chunks, so an in-memory index and a prerequisite dict answer every query in the eval set.

## Evaluation

`eval/test_cases.json` has 30 cases:

- 8 factual
- 6 policy
- 6 eligibility with synthetic profiles
- 4 missing-information
- 4 conflict
- 1 ambiguous
- 1 out-of-scope

`eval/run_eval.py` runs each case through 4 stages: Basic LLM → structured prompt → RAG → RAG + tools + student data.

- **Rule-based labels**: correct / partial / hallucinated / incorrect, from required, forbidden and behaviour patterns per case. `--judge` adds an LLM grade against the expected outcome.
- **Metrics per stage**: accuracy, hallucination rate, eligibility correctness, incorrect recommendations, source correctness, missing-info and conflict handling, average and maximum latency.

Synthetic students live in `eval/profiles.json`. They contain no real names or IDs.
