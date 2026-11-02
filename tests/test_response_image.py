import base64
import xml.etree.ElementTree as ET

from fastapi.testclient import TestClient

from advisor import graph, server


def test_response_infographic_renders_only_the_requested_table_and_watermark():
    response = TestClient(server.app).post(
        "/api/response-image.svg",
        json={"answer": "## Course plan\n\n| Course | Basket | Credits |\n|---|---|---:|\n| DATA103 | B3 Program Core | 3 |\n\n<script>alert(1)</script>"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
    assert ">Course</text>" in response.text and ">Credits</text>" in response.text
    assert ">DATA103</text>" in response.text and ">3</text>" in response.text
    assert ">Program Core</text>" in response.text
    assert "B3 Program Core" not in response.text
    assert "Here are the courses" not in response.text
    assert "alert(1)" not in response.text
    assert "<script>" not in response.text
    assert "AI GENERATED IMAGE" in response.text


def test_image_request_returns_image_generation_flag(monkeypatch):
    seen = {}

    def fake_ask(message, thread_id, profile):
        seen["message"] = message
        return {
            "answer": "A grounded answer [S1].",
            "sources": [],
            "latency_ms": 1,
            "flags": {"tools": [], "cited": []},
        }

    monkeypatch.setattr(server.config, "chat_ready", lambda: True)
    monkeypatch.setattr(graph, "ask", fake_ask)

    response = TestClient(server.app).post(
        "/api/chat",
        json={"message": "Make an image of your answer."},
    )

    assert response.status_code == 200
    assert response.json()["flags"]["generate_image"] is True
    assert "image" not in seen["message"].lower()
    assert "Markdown table" in seen["message"]


def test_2025_course_image_uses_exact_database_rows_and_returns_svg_data(monkeypatch):
    monkeypatch.setattr(server.config, "chat_ready", lambda: True)
    monkeypatch.setattr(graph, "ask", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("image request must not be sent to chat model")))

    response = TestClient(server.app).post(
        "/api/chat",
        json={"message": "Generate an image of the 2025 course table only."},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["answer"] == ""
    assert result["flags"]["visual_only"] is True
    image = result["visual"]
    assert image["content_type"] == "image/svg+xml"
    assert image["format"] == "svg"
    assert image["title"] == "Courses Offered at Vidyashilp University – 2025"
    assert image["row_count"] == 57
    svg_bytes = base64.b64decode(image["data_url"].split(",", 1)[1])
    assert svg_bytes.startswith(b"<svg ")
    root = ET.fromstring(svg_bytes)
    assert root.attrib["width"].isdigit() and root.attrib["height"].isdigit()
    visible_text = " ".join(node.text or "" for node in root.iter("{http://www.w3.org/2000/svg}text"))
    assert "Courses Offered at Vidyashilp University – 2025" in visible_text
    assert "AI GENERATED IMAGE" in visible_text
    assert "Here is the course table" not in visible_text
    assert "UCOR103" in visible_text and "Communication Skills" in visible_text
    assert "PSCY102" in visible_text and "ECON105" in visible_text
    text_nodes = [node.text for node in root.iter("{http://www.w3.org/2000/svg}text")]
    assert text_nodes.count("PSCY102") == 1
    assert text_nodes.count("ECON105") == 1


def test_2025_course_chart_request_does_not_get_mistaken_for_table():
    assert server._course_table_batch("Generate a bar chart of 2025 courses.") is None


def test_full_2022_semester_one_course_list_image_is_structured_and_uses_names():
    response = TestClient(server.app).post(
        "/api/chat",
        json={"message": "provide full list of Semester 1 including all baskets for 2022 in image format"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["flags"]["visual_only"] is True
    assert result["visual"]["row_count"] == 10
    assert result["visual"]["title"] == "Courses Offered at Vidyashilp University – 2022 · Semester 1"
    svg = base64.b64decode(result["visual"]["data_url"].split(",", 1)[1]).decode("utf-8")
    root = ET.fromstring(svg)
    text_nodes = [node.text for node in root.iter("{http://www.w3.org/2000/svg}text")]
    assert "Program Core" in text_nodes
    assert not any(value and value.startswith("B") and "Program Core" in value for value in text_nodes)
    assert "DATA130" in text_nodes
    assert "AI GENERATED IMAGE" in text_nodes


def test_ordinary_question_does_not_return_visual(monkeypatch):
    monkeypatch.setattr(server.config, "chat_ready", lambda: True)
    monkeypatch.setattr(
        graph,
        "ask",
        lambda message, thread_id, profile: {
            "answer": "Eligibility is determined by the handbook.",
            "sources": [],
            "latency_ms": 1,
            "flags": {"tools": [], "cited": []},
        },
    )

    response = TestClient(server.app).post("/api/chat", json={"message": "What is the eligibility criteria?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Eligibility is determined by the handbook."
    assert "visual" not in response.json()
    assert not response.json()["flags"].get("generate_image")


def test_image_intent_detects_common_requests():
    assert server._requests_response_image("Can you create an infographic from this answer?")
    assert server._requests_response_image("Please show the response as an image.")
    assert server._requests_response_image("Generate image of that explanation.")
    assert server._requests_response_image("Visualize this information.")
    assert server._requests_response_image("Give me a visual representation.")
    assert server._requests_response_image("Create a flowchart.")
    assert server._requests_response_image("Make a comparison table as an image.")
    assert server._requests_response_image("provide full list of Semester 1 including all baskets for 2022 in image format")
    assert not server._requests_response_image("What images are required for admission?")
    assert not server._requests_response_image("What is the eligibility criteria?")


def test_table_visual_preserves_values_without_conversational_content():
    response = TestClient(server.app).post(
        "/api/response-image.svg",
        json={"answer": "Here is the complete course information.\n\n## Courses Offered – 2025\n| Code | Course | Credits |\n|---|---|---:|\n| DATA103 | Data Science | 3 |\n\nYou can ask about other semesters.", "request": "Generate an image of the course table."},
    )

    assert response.status_code == 200
    assert "Courses Offered &#8211; 2025" in response.text or "Courses Offered" in response.text
    assert "DATA103" in response.text and "Data Science" in response.text and ">3</text>" in response.text
    assert "complete course information" not in response.text
    assert "other semesters" not in response.text
    assert "AI GENERATED IMAGE" in response.text


def test_generated_image_uses_question_heading_and_omits_source_column():
    response = TestClient(server.app).post(
        "/api/response-image.svg",
        json={
            "answer": (
                "## Academic results\n\n"
                "| Code | Course | Grade | Source |\n|---|---|---|---|\n"
                "| MATH201 | Calculus | B+ | [S1] |"
            ),
            "request": "Generate an image of Student A's Semester 3 grades.",
        },
    )

    assert response.status_code == 200
    assert "Student A – Semester 3 Academic Results" in response.text
    assert ">Source</text>" not in response.text
    assert "MATH201" in response.text and "Calculus" in response.text
    assert "AI GENERATED IMAGE" in response.text


def test_comparison_image_contains_only_comparison_rows():
    response = TestClient(server.app).post(
        "/api/response-image.svg",
        json={"answer": "Here is the comparison.\n\n## Jio vs Airtel\n| Feature | Jio | Airtel |\n|---|---|---|\n| Network | 5G | 5G |\n| Plans | Value | Premium |\n\nLet me know if you want another comparison.", "request": "Generate an image of this Jio vs Airtel comparison."},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
    assert "Jio vs Airtel" in response.text
    assert "Network" in response.text and "Plans" in response.text
    assert "Here is the comparison" not in response.text
    assert "Let me know" not in response.text
    assert "AI GENERATED IMAGE" in response.text


def test_chart_uses_source_numeric_values_and_has_watermark():
    response = TestClient(server.app).post(
        "/api/response-image.svg",
        json={"answer": "A summary of the actual data.\n\n## Enrollment\n| Year | Students |\n|---|---:|\n| 2024 | 12 |\n| 2025 | 18 |", "request": "Show me a bar chart of the data."},
    )

    assert response.status_code == 200
    assert "Students" in response.text
    assert "12" in response.text and "18" in response.text
    assert "actual data" not in response.text
    assert "AI GENERATED IMAGE" in response.text


def test_line_and_pie_charts_render_source_values():
    client = TestClient(server.app)
    answer = "## Enrolment\n| Year | Students |\n|---|---:|\n| 2024 | 12 |\n| 2025 | 18 |"

    line = client.post("/api/response-image.svg", json={"answer": answer, "request": "Create a line chart."})
    pie = client.post("/api/response-image.svg", json={"answer": answer, "request": "Create a pie chart."})

    assert line.status_code == pie.status_code == 200
    assert "<polyline" in line.text and "12" in line.text and "18" in line.text
    assert "<path" in pie.text and "12" in pie.text and "18" in pie.text
    assert "AI GENERATED IMAGE" in line.text and "AI GENERATED IMAGE" in pie.text


def test_chart_without_source_numeric_values_returns_clear_error():
    response = TestClient(server.app).post(
        "/api/response-image.svg",
        json={"answer": "## Courses\n| Code | Title |\n|---|---|\n| DATA103 | Data Science |", "request": "Create a bar chart."},
    )

    assert response.status_code == 422
    assert "numeric" in response.json()["detail"]


def test_unsupported_chart_format_returns_clear_error():
    response = TestClient(server.app).post(
        "/api/response-image.svg",
        json={"answer": "## Enrollment\n| Year | Students |\n|---|---:|\n| 2025 | 18 |", "request": "Create a scatter chart."},
    )

    assert response.status_code == 422
    assert "Unsupported visual format" in response.json()["detail"]


def test_flowchart_and_full_explanation_are_distinct_visuals():
    answer = "## Admission process\nApplicants submit the application. The university reviews the documents.\n\n## Requirements\n- Completed application\n- Required documents\n\n| Stage | Owner |\n|---|---|\n| Review | Admissions |"
    client = TestClient(server.app)

    flowchart = client.post("/api/response-image.svg", json={"answer": answer, "request": "Generate a flowchart."})
    full_answer = client.post("/api/response-image.svg", json={"answer": answer, "request": "Generate an image of the entire explanation."})

    assert flowchart.status_code == full_answer.status_code == 200
    assert "Vidyashilp University – Admission Process" in flowchart.text
    assert "Applicants submit the application." in flowchart.text
    assert "AI GENERATED IMAGE" in flowchart.text
    assert "Applicants submit the application." in full_answer.text
    assert "Required documents" in full_answer.text
    assert "Review" in full_answer.text and "Admissions" in full_answer.text
    assert "AI GENERATED IMAGE" in full_answer.text


def test_context_aware_headings_cover_comparison_revenue_and_student_visuals():
    client = TestClient(server.app)

    comparison = client.post(
        "/api/response-image.svg",
        json={
            "answer": "## Jio vs Airtel\n| Provider | ARPU |\n|---|---:|\n| Jio | 200 |\n| Airtel | 250 |",
            "request": "Compare Jio and Airtel based on ARPU.",
        },
    )
    revenue = client.post(
        "/api/response-image.svg",
        json={
            "answer": "## Revenue\n| Year | Revenue |\n|---|---:|\n| 2020 | 10 |\n| 2025 | 20 |",
            "request": "Generate a chart showing the annual revenue from 2020 to 2025.",
        },
    )
    student = client.post(
        "/api/response-image.svg",
        json={
            "answer": "## Marks\n| Semester | Marks |\n|---|---:|\n| 1 | 85 |",
            "request": "Show me the semester-wise marks of Student A.",
        },
    )

    assert comparison.status_code == revenue.status_code == student.status_code == 200
    assert "Jio vs Airtel – ARPU Comparison" in comparison.text
    assert "Annual Revenue Trend (2020–2025)" in revenue.text
    assert "Student A – Semester-wise Academic Performance" in student.text
    assert "ARPU" in comparison.text and ">200</text>" in comparison.text
    assert ">10</text>" in revenue.text and ">20</text>" in revenue.text
    assert ">85</text>" in student.text


def test_explicit_table_request_without_table_returns_validation_error():
    response = TestClient(server.app).post(
        "/api/response-image.svg",
        json={"answer": "No structured data was returned.", "request": "Convert the table into an image."},
    )

    assert response.status_code == 422
    assert "No table" in response.json()["detail"]


def test_text_only_image_refusal_is_not_rendered_as_an_image():
    response = TestClient(server.app).post(
        "/api/response-image.svg",
        json={"answer": "I can only provide text-based responses with tables for course information.", "request": "Generate an image of the course table."},
    )

    assert response.status_code == 422
    assert "text-only response" in response.json()["detail"]


def test_org_chart_requires_and_renders_explicit_hierarchy():
    client = TestClient(server.app)
    response = client.post(
        "/api/response-image.svg",
        json={"answer": "## Organization\n- Dean\n  - Faculty\n  - Registrar", "request": "Create an organizational chart."},
    )
    missing_hierarchy = client.post(
        "/api/response-image.svg",
        json={"answer": "## Organization\nDean works with faculty.", "request": "Create an organizational chart."},
    )

    assert response.status_code == 200
    assert "Dean" in response.text and "Faculty" in response.text and "Registrar" in response.text
    assert "AI GENERATED IMAGE" in response.text
    assert missing_hierarchy.status_code == 422
    assert "hierarchy" in missing_hierarchy.json()["detail"]
