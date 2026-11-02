"""Deterministic SVG rendering for answer content explicitly requested as a visual."""
import math
import re
import textwrap
from html import escape


_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
_CITATION = re.compile(r"\[(?:S\d+)(?:\s*[,;]\s*S\d+)*\]")
_NUMBER = re.compile(r"^\s*[-+]?\$?\d[\d,]*(?:\.\d+)?\s*%?\s*$")
_COLORS = ("#092f87", "#ac162c", "#43866a", "#c17a22", "#6855a3", "#32808a")
_WATERMARK = "AI GENERATED IMAGE"


def image_requested(message: str) -> bool:
    """Return true only for an explicit visual-generation request."""
    return bool(re.search(
        r"\b(?:generate|create|make|draw|render|turn|convert|show|give|provide|want|need|visuali[sz]e)\b"
        r".{0,80}\b(?:image|picture|infographic|visual(?:ization|isation)?|poster|chart|"
        r"flowchart|mind\s*map|diagram)\b|"
        r"\bvisuali[sz]e\b|"
        r"\bimage\s+format\b|"
        r"\b(?:as|into)\s+(?:an?\s+)?(?:image|picture|infographic|visual|poster|chart|diagram)\b|"
        r"\b(?:image|picture|infographic|visual|chart|flowchart|mind\s*map|diagram)\b"
        r".{0,50}\b(?:of|for|from|based on|showing)\b",
        message, re.I | re.S,
    ))


def _visual_type(request: str, tables: list[dict]) -> str:
    text = request.lower()
    if re.search(r"\b(?:entire|whole|full)\s+(?:explanation|answer|response)\b", text):
        return "overview"
    if re.search(r"\b(?:scatter|heatmap|heat map)\b", text):
        return "unsupported"
    if "organizational chart" in text or "organisation chart" in text or "organization chart" in text or re.search(r"\borg\s+chart\b", text):
        return "org_chart"
    if "pie chart" in text:
        return "pie_chart"
    if "line chart" in text or "line graph" in text:
        return "line_chart"
    if "bar chart" in text or "bar graph" in text:
        return "bar_chart"
    if re.search(r"\b(?:chart|graph)\b", text):
        return "bar_chart"
    if "flowchart" in text or "flow chart" in text:
        return "flowchart"
    if "process diagram" in text:
        return "process_diagram"
    if "mind map" in text or "mindmap" in text:
        return "mind_map"
    if "architecture diagram" in text or "architecture" in text:
        return "architecture_diagram"
    if "diagram" in text:
        return "process_diagram"
    if "table" in text or "comparison" in text or "all baskets" in text or "full list" in text:
        return "table"
    if tables:
        return "table"
    return "overview"


def _split_table_row(line: str) -> list[str]:
    value = line.strip()
    if value.startswith("|"):
        value = value[1:]
    if value.endswith("|") and not value.endswith(r"\|"):
        value = value[:-1]
    return [cell.strip().replace(r"\|", "|") for cell in re.split(r"(?<!\\)\|", value)]


def _tables(answer: str) -> list[dict]:
    lines = answer.splitlines()
    found = []
    index = 0
    while index + 1 < len(lines):
        if "|" not in lines[index] or not _TABLE_SEPARATOR.match(lines[index + 1]):
            index += 1
            continue
        start = index
        headers = _split_table_row(lines[index])
        rows = []
        index += 2
        while index < len(lines) and "|" in lines[index] and lines[index].strip():
            row = _split_table_row(lines[index])
            if len(row) == len(headers):
                rows.append(row)
            index += 1
        title = ""
        for preceding in reversed(lines[max(0, start - 4):start]):
            if preceding.strip().startswith("#"):
                title = re.sub(r"^#+\s*", "", preceding.strip())
                break
        found.append({"title": title, "headers": headers, "rows": rows})
    return found


def _select_table(tables: list[dict], request: str) -> dict:
    if not tables:
        raise ValueError("No table was found in the advisor's response.")
    if len(tables) == 1:
        return tables[0]
    words = [w for w in re.findall(r"[a-z0-9]+", request.lower())
             if w not in {"image", "table", "make", "create", "generate", "show", "visual", "this", "that", "the", "as", "an", "a", "of", "for", "from", "into", "previous"}]
    ranked = sorted(
        tables,
        key=lambda table: sum(
            1 for word in words
            if word in (" ".join([table["title"], *table["headers"]])).lower()
        ),
        reverse=True,
    )
    return ranked[0]


def _clean(value: str) -> str:
    from .tools import remove_basket_codes

    return remove_basket_codes(_CITATION.sub("", value)).strip()


def _wrap(value: str, width: int) -> list[str]:
    return textwrap.wrap(value, width=max(6, width), break_long_words=True, break_on_hyphens=False) or [""]


def _without_source_columns(table: dict) -> dict:
    source_columns = {
        index for index, value in enumerate(table["headers"])
        if re.search(r"\bsource\b", str(value), re.I)
    }
    if not source_columns:
        return table
    columns = [
        index for index in range(len(table["headers"]))
        if index not in source_columns
    ]
    if not columns:
        raise ValueError("The table has no visible columns after removing source metadata.")
    return {
        **table,
        "headers": [table["headers"][index] for index in columns],
        "rows": [[row[index] for index in columns] for row in table["rows"]],
    }


def _svg_start(width: int, height: int) -> list[str]:
    return [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='#ffffff'/>",
    ]


def _title(parts: list[str], title: str) -> None:
    if title:
        parts.append(
            f"<text x='32' y='46' font-family='system-ui,sans-serif' font-size='24' "
            f"font-weight='700' fill='#092f87'>{escape(title)}</text>"
        )


def image_heading(request: str, answer: str, visual_type: str, table: dict | None = None) -> str:
    """Derive a grounded image heading from the request and its visual content."""
    context_title = table.get("title", "") if table else ""
    if not context_title:
        context_title = next(
            (
                re.sub(r"^#+\s*", "", line.strip())
                for line in answer.splitlines()
                if line.strip().startswith("#")
            ),
            "",
        )

    query = " ".join(request.split())
    query_year = re.search(r"\b(20\d{2})\b", query)
    if re.search(r"\bcourses?\b", query, re.I) and query_year:
        semester = re.search(r"\bsemester\s*(\d{1,2})\b", query, re.I)
        heading = f"Courses Offered at Vidyashilp University – {query_year.group(1)}"
        return f"{heading} · Semester {semester.group(1)}" if semester else heading

    course_context_year = re.search(r"\bcourses?\s+offered\s*[–—-]\s*(20\d{2})\b", context_title, re.I)
    if course_context_year:
        heading = f"Courses Offered at Vidyashilp University – {course_context_year.group(1)}"
        semester = re.search(
            r"\bsemester\s*(\d{1,2})\b",
            query + " " + context_title,
            re.I,
        )
        return f"{heading} · Semester {semester.group(1)}" if semester else heading

    if re.search(r"\bannual\s+revenue\b", query, re.I):
        year_range = re.search(r"\b(20\d{2})\s*(?:to|[-–—])\s*(20\d{2})\b", query, re.I)
        if year_range:
            return f"Annual Revenue Trend ({year_range.group(1)}–{year_range.group(2)})"
        return "Annual Revenue Trend"

    if re.search(r"\badmission\b", query, re.I) or re.search(r"\badmission\b", context_title, re.I):
        if visual_type in {"flowchart", "process_diagram", "overview"}:
            return "Vidyashilp University – Admission Process"

    student = re.search(r"\bStudent\s+([A-Z][\w-]*)\b", query, re.I)
    if student and re.search(r"\bmarksheet\b", query, re.I):
        semester = re.search(r"\b(?:semester|sem)\s*(\d{1,2})\b", query, re.I)
        if semester:
            return f"Student {student.group(1)} – Semester {semester.group(1)} Marksheet"
    if student and re.search(r"\bsemester[-\s]?wise\s+marks\b", query, re.I):
        return f"Student {student.group(1)} – Semester-wise Academic Performance"
    if student and (semester := re.search(r"\bsemester\s*(\d{1,2})\b", query, re.I)):
        return f"Student {student.group(1)} – Semester {semester.group(1)} Academic Results"

    if re.search(r"\bcompare\b|\bcomparison\b", query, re.I):
        pair = re.search(r"\b(Jio)\s+(?:vs\.?|and)\s+(Airtel)\b", query, re.I)
        metric = re.search(r"\b(?:based\s+on|by|for|using)\s+([A-Za-z][\w -]*?)(?:[?.!,]|$)", query, re.I)
        if pair and metric:
            metric_words = metric.group(1).strip().split()
            metric_title = " ".join(
                word.upper() if word.isupper() else word.title() for word in metric_words
            )
            return f"{pair.group(1).title()} vs {pair.group(2).title()} – {metric_title} Comparison"
        if pair:
            metric = re.search(r"\bARPU\b", context_title, re.I)
            if metric:
                return "Jio vs Airtel – ARPU Comparison"

    if context_title and context_title.casefold() not in {
        "your answer", "advisor response", "information", "generated image",
    }:
        return context_title

    if table:
        headers = [
            header.strip()
            for header in table.get("headers", [])
            if header.strip() and not re.search(r"\bsource\b", header, re.I)
        ]
        if any(re.search(r"\bcourse\b", header, re.I) for header in headers):
            return "Course Information"
        if headers:
            return f"{headers[0]} Overview"
    if visual_type == "flowchart":
        return "Process Flowchart"
    if visual_type == "bar_chart":
        return "Data Comparison"
    if visual_type == "line_chart":
        return "Trend Over Time"
    if visual_type == "pie_chart":
        return "Data Distribution"
    return ""


def _watermark(parts: list[str], width: int, height: int) -> None:
    parts.append(
        f"<text x='{width - 20}' y='{height - 12}' text-anchor='end' "
        "font-family='system-ui,sans-serif' font-size='12' font-weight='600' "
        f"letter-spacing='.4' fill='#687080'>{_WATERMARK}</text>"
    )


def _render_table(table: dict) -> str:
    table = _without_source_columns(table)
    headers = [_clean(cell) for cell in table["headers"]]
    rows = [[_clean(cell) for cell in row] for row in table["rows"]]
    if not rows:
        raise ValueError("The response table has no data rows.")
    if len(headers) > 20 or len(rows) > 300:
        raise ValueError("This table is too large to render reliably (maximum 20 columns and 300 rows).")
    widths = [
        min(360, max(150, max(len(headers[column]), *(len(row[column]) for row in rows)) * 7 + 34))
        for column in range(len(headers))
    ]
    width = max(760, sum(widths) + 64)
    if width > 6000:
        raise ValueError("This table is too wide to render reliably.")
    top = 74
    header_lines = [_wrap(value, int((widths[i] - 24) / 8)) for i, value in enumerate(headers)]
    header_height = max(46, max(map(len, header_lines)) * 20 + 20)
    wrapped_rows = [[_wrap(value, int((widths[i] - 24) / 8)) for i, value in enumerate(row)] for row in rows]
    row_heights = [max(42, max(map(len, row)) * 19 + 18) for row in wrapped_rows]
    height = top + header_height + sum(row_heights) + 48
    parts = _svg_start(width, height)
    if table["title"]:
        _title(parts, table["title"])
    x = 32
    for i, lines in enumerate(header_lines):
        parts.append(f"<rect x='{x}' y='{top}' width='{widths[i]}' height='{header_height}' fill='#e8eef9' stroke='#ccd6e8'/>")
        for line_num, line in enumerate(lines):
            parts.append(
                f"<text x='{x + 12}' y='{top + 25 + line_num * 19}' font-family='system-ui,sans-serif' "
                f"font-size='14' font-weight='700' fill='#092f87'>{escape(line)}</text>"
            )
        x += widths[i]
    y = top + header_height
    for row_index, row in enumerate(wrapped_rows):
        x = 32
        for column_index, lines in enumerate(row):
            parts.append(
                f"<rect x='{x}' y='{y}' width='{widths[column_index]}' height='{row_heights[row_index]}' "
                f"fill='{'#ffffff' if row_index % 2 == 0 else '#f9fafc'}' stroke='#d8dfeb'/>"
            )
            for line_num, line in enumerate(lines):
                parts.append(
                    f"<text x='{x + 12}' y='{y + 23 + line_num * 19}' font-family='system-ui,sans-serif' "
                    f"font-size='13' fill='#131b2e'>{escape(line)}</text>"
                )
            x += widths[column_index]
        y += row_heights[row_index]
    _watermark(parts, width, height)
    parts.append("</svg>")
    return "".join(parts)


def render_table_data(
    title: str,
    headers: list[str],
    rows: list[list[str]],
    request: str = "",
) -> str:
    """Render validated structured rows without passing them through Markdown parsing."""
    if not headers or any(len(row) != len(headers) for row in rows):
        raise ValueError("Table rows must contain exactly one value for each column.")
    table = {"title": title, "headers": headers, "rows": rows}
    heading = image_heading(request or title, "", "table", table)
    return _render_table({**table, "title": heading})


def _numeric_table(table: dict) -> tuple[str, str, list[str], list[float], int]:
    table = _without_source_columns(table)
    if not table["rows"] or len(table["headers"]) < 2:
        raise ValueError("A chart needs categories and numeric values in a response table.")
    numeric_column = None
    values = []
    for column in range(1, len(table["headers"])):
        parsed = []
        for row in table["rows"]:
            raw_value = _clean(row[column])
            value = raw_value.replace(",", "").replace("$", "").replace("%", "").strip()
            if not _NUMBER.fullmatch(raw_value) or not value:
                parsed = []
                break
            parsed.append(float(value))
        if parsed:
            numeric_column, values = column, parsed
            break
    if numeric_column is None:
        raise ValueError("No numeric series was found; a chart cannot be generated without source values.")
    return (_clean(table["headers"][0]), _clean(table["headers"][numeric_column]),
            [_clean(row[0]) for row in table["rows"]], values, numeric_column)


def _render_chart(table: dict, chart_type: str) -> str:
    category_label, value_label, labels, values, numeric_column = _numeric_table(table)
    if len(labels) > 40:
        raise ValueError("This chart has too many data points to label clearly (maximum 40).")
    if chart_type == "pie_chart" and any(value < 0 for value in values):
        raise ValueError("A pie chart requires non-negative source values.")
    width, height = 960, max(520, 500 + len(labels) * 24)
    title = table["title"] or f"{value_label} by {category_label}"
    parts = _svg_start(width, height)
    _title(parts, title)
    parts.append(f"<text x='32' y='98' font-family='system-ui,sans-serif' font-size='14' fill='#4a5468'>{escape(category_label)} · {escape(value_label)}</text>")
    if chart_type == "pie_chart":
        total = sum(values)
        if total <= 0:
            raise ValueError("A pie chart needs a positive total.")
        cx, cy, radius = width // 2, 285, 155
        start = -math.pi / 2
        for index, (label, value) in enumerate(zip(labels, values)):
            end = start + 2 * math.pi * value / total
            x1, y1 = cx + radius * math.cos(start), cy + radius * math.sin(start)
            x2, y2 = cx + radius * math.cos(end), cy + radius * math.sin(end)
            large = 1 if end - start > math.pi else 0
            path = f"M {cx} {cy} L {x1:.2f} {y1:.2f} A {radius} {radius} 0 {large} 1 {x2:.2f} {y2:.2f} Z"
            parts.append(f"<path d='{path}' fill='{_COLORS[index % len(_COLORS)]}' stroke='white' stroke-width='2'/>")
            legend_y = 470 + index * 22
            parts.extend([
                f"<rect x='90' y='{legend_y - 12}' width='12' height='12' fill='{_COLORS[index % len(_COLORS)]}'/>",
                f"<text x='110' y='{legend_y}' font-family='system-ui,sans-serif' font-size='13' fill='#131b2e'>{escape(label)}: {escape(_clean(table['rows'][index][numeric_column]))}</text>",
            ])
            start = end
    else:
        left, right, top, bottom = 90, 32, 136, height - 90
        maximum = max(values) or 1
        parts.extend([
            f"<path d='M {left} {top} V {bottom} H {width - right}' fill='none' stroke='#7c8798'/>",
            f"<text x='24' y='{(top + bottom) // 2}' transform='rotate(-90 24 {(top + bottom) // 2})' "
            f"font-family='system-ui,sans-serif' font-size='13' fill='#4a5468'>{escape(value_label)}</text>",
        ])
        slot = (width - left - right) / len(values)
        points = []
        for i, (label, value) in enumerate(zip(labels, values)):
            x = left + slot * (i + 0.5)
            y = bottom - (bottom - top) * value / maximum
            points.append((x, y))
            if chart_type == "bar_chart":
                bar_width = min(58, slot * .64)
                parts.append(f"<rect x='{x - bar_width / 2:.2f}' y='{y:.2f}' width='{bar_width:.2f}' height='{bottom-y:.2f}' rx='3' fill='#092f87'/>")
            else:
                parts.append(f"<circle cx='{x:.2f}' cy='{y:.2f}' r='5' fill='#ac162c'/>")
            label_lines = _wrap(label, max(8, int(slot / 7)))[:2]
            for line_index, line in enumerate(label_lines):
                parts.append(f"<text x='{x:.2f}' y='{bottom + 20 + line_index * 16}' text-anchor='middle' font-family='system-ui,sans-serif' font-size='11' fill='#4a5468'>{escape(line)}</text>")
            parts.append(f"<text x='{x:.2f}' y='{y - 8:.2f}' text-anchor='middle' font-family='system-ui,sans-serif' font-size='11' fill='#131b2e'>{escape(_clean(table['rows'][i][numeric_column]))}</text>")
        if chart_type == "line_chart":
            joined = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
            parts.append(f"<polyline points='{joined}' fill='none' stroke='#ac162c' stroke-width='3'/>")
    _watermark(parts, width, height)
    parts.append("</svg>")
    return "".join(parts)


def _content_lines(answer: str, include_tables: bool = False) -> tuple[str, list[str]]:
    title = ""
    content = []
    in_table = False
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            in_table = False
            continue
        if "|" in stripped and (_TABLE_SEPARATOR.match(stripped) or (in_table or stripped.startswith("|"))):
            in_table = True
            if include_tables and not _TABLE_SEPARATOR.match(stripped):
                cells = [_clean(cell) for cell in _split_table_row(stripped)]
                content.append(" · ".join(cell for cell in cells if cell))
            continue
        in_table = False
        if stripped.startswith("#") and not title:
            title = re.sub(r"^#+\s*", "", stripped)
            continue
        line = re.sub(r"^#{1,6}\s*", "", stripped)
        line = re.sub(r"[*_`]+", "", line)
        line = re.sub(r"^\s*(?:[-+*]|\d+[.)])\s+", "", line)
        line = _clean(line)
        if line:
            content.extend(re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", line))
    return title, content


def _render_org_chart(answer: str, heading: str = "") -> str:
    title = ""
    nodes = []
    for line in answer.splitlines():
        heading = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        bullet = re.match(r"^(\s*)(?:[-+*]|\d+[.)])\s+(.+)$", line)
        if heading:
            depth, label = len(heading.group(1)) - 1, heading.group(2).strip()
            if not title:
                title = label
                continue
        elif bullet:
            depth, label = len(bullet.group(1).expandtabs(2)) // 2, bullet.group(2).strip()
        else:
            continue
        nodes.append((depth, _clean(re.sub(r"[*_`]+", "", label))))
    if not nodes or max(depth for depth, _ in nodes) == 0:
        raise ValueError("An organizational chart needs explicit parent-child hierarchy in the response.")
    if len(nodes) > 60:
        raise ValueError("This hierarchy is too large to render clearly (maximum 60 items).")

    width, row_height, top = 1200, 76, 96
    height = top + row_height * len(nodes) + 52
    parents = {}
    parts = _svg_start(width, height)
    _title(parts, heading or title or "Organizational chart")
    for index, (depth, label) in enumerate(nodes):
        x = 56 + min(depth, 3) * 280
        y = top + index * row_height
        if depth:
            parent = parents.get(depth - 1)
            if parent is None:
                raise ValueError("The organizational chart contains a child without an explicit parent.")
            parent_x, parent_y = parent
            parts.append(
                f"<path d='M{parent_x + 236} {parent_y + 30} H{x - 12} V{y + 30} H{x}' "
                "fill='none' stroke='#7c8798' stroke-width='2'/>"
            )
        parents[depth] = (x, y)
        parts.append(f"<rect x='{x}' y='{y}' width='236' height='58' rx='9' fill='white' stroke='#cbd5e5'/>")
        for line_index, text in enumerate(_wrap(label, 30)[:2]):
            parts.append(f"<text x='{x + 12}' y='{y + 25 + line_index * 17}' font-family='system-ui,sans-serif' font-size='13' fill='#131b2e'>{escape(text)}</text>")
    _watermark(parts, width, height)
    parts.append("</svg>")
    return "".join(parts)


def _render_text_visual(answer: str, visual_type: str, heading: str = "") -> str:
    if visual_type == "org_chart":
        return _render_org_chart(answer, heading)
    title, lines = _content_lines(answer, include_tables=visual_type == "overview")
    source_title = title
    title = heading or title
    if visual_type == "architecture_diagram":
        lines = [
            _clean(part.strip())
            for source_line in answer.splitlines()
            if re.search(r"(?:→|->|-->)", source_line)
            for part in re.split(r"(?:→|->|-->)", source_line)
            if part.strip()
        ]
        if len(lines) < 2:
            raise ValueError("An architecture diagram needs explicit component relationships in the response.")
    if not lines:
        raise ValueError("No relevant text was found to visualize.")
    if len(lines) > 60:
        raise ValueError("This response is too long to render clearly as a single visual.")
    if visual_type in {"flowchart", "process_diagram", "architecture_diagram", "org_chart"}:
        width, gap = 960, 24
        wrapped = [_wrap(line, 72) for line in lines]
        box_heights = [max(70, len(value) * 20 + 30) for value in wrapped]
        height = 135 + sum(box_heights) + max(0, len(lines) - 1) * gap + 50
        parts = _svg_start(width, height)
        _title(parts, title or visual_type.replace("_", " ").title())
        y = 96
        for i, line in enumerate(wrapped):
            box_height = box_heights[i]
            parts.append(f"<rect x='100' y='{y}' width='760' height='{box_height}' rx='10' fill='white' stroke='#cbd5e5' stroke-width='1.5'/>")
            for line_index, text in enumerate(line):
                parts.append(f"<text x='480' y='{y + 28 + line_index * 20}' text-anchor='middle' font-family='system-ui,sans-serif' font-size='14' fill='#131b2e'>{escape(text)}</text>")
            if i < len(lines) - 1:
                parts.append(f"<path d='M480 {y + box_height} V{y + box_height + gap - 5}' stroke='#092f87' stroke-width='2' marker-end='url(#arrow)'/>")
            y += box_height + gap
        parts.insert(2, "<defs><marker id='arrow' markerWidth='8' markerHeight='8' refX='4' refY='4' orient='auto'><path d='M0 0L8 4L0 8Z' fill='#092f87'/></marker></defs>")
    elif visual_type == "mind_map":
        width, height = 1100, max(500, 220 + len(lines) * 76)
        parts = _svg_start(width, height)
        _title(parts, title or "Mind map")
        cx, cy = 550, 125
        parts.append(f"<rect x='{cx-145}' y='{cy-28}' width='290' height='56' rx='24' fill='#092f87'/>")
        parts.append(f"<text x='{cx}' y='{cy+6}' text-anchor='middle' font-family='system-ui,sans-serif' font-size='16' fill='white'>{escape(source_title or 'Overview')}</text>")
        for index, line in enumerate(lines):
            y = 220 + index * 72
            side = -1 if index % 2 == 0 else 1
            node_x = 245 if side < 0 else 855
            start_x = cx + side * 145
            edge_x = node_x - side * 150
            wrapped = _wrap(line, 40)[:2]
            node_height = max(48, len(wrapped) * 17 + 20)
            parts.append(f"<path d='M{start_x} {cy} Q{edge_x} {cy} {edge_x} {y + node_height // 2}' fill='none' stroke='{_COLORS[index % len(_COLORS)]}' stroke-width='2'/>")
            parts.append(f"<rect x='{node_x - 150}' y='{y}' width='300' height='{node_height}' rx='9' fill='white' stroke='{_COLORS[index % len(_COLORS)]}'/>")
            for line_index, text in enumerate(wrapped):
                parts.append(f"<text x='{node_x}' y='{y + 25 + line_index * 17}' text-anchor='middle' font-family='system-ui,sans-serif' font-size='12' fill='#131b2e'>{escape(text)}</text>")
    else:
        wrapped_lines = [wrapped for line in lines for wrapped in _wrap(line, 100)]
        if len(wrapped_lines) > 160:
            raise ValueError("This response is too long to render clearly as a single visual.")
        width, height = 960, 160 + len(wrapped_lines) * 29
        parts = _svg_start(width, height)
        _title(parts, title or "Advisor response")
        parts.append(f"<rect x='32' y='86' width='896' height='{max(52, len(wrapped_lines) * 27 + 28)}' rx='12' fill='white' stroke='#d8dfeb'/>")
        for index, line in enumerate(wrapped_lines):
            parts.append(f"<text x='52' y='{121 + index * 27}' font-family='system-ui,sans-serif' font-size='15' fill='#131b2e'>{escape(line)}</text>")
    _watermark(parts, width, height)
    parts.append("</svg>")
    return "".join(parts)


def render(answer: str, request: str = "") -> str:
    """Render only the requested table/chart/diagram or the answer's relevant text."""
    if re.search(
        r"\b(?:i can only provide text|cannot generate images?|can't generate images?|"
        r"unable to generate images?|use external tools to convert)\b",
        answer,
        re.I,
    ):
        raise ValueError("The advisor returned a text-only response, not visual source content.")
    tables = _tables(answer)
    visual_type = _visual_type(request, tables)
    if visual_type in {"table", "bar_chart", "line_chart", "pie_chart"}:
        selected = _select_table(tables, request)
        selected = {**selected, "title": image_heading(request, answer, visual_type, selected)}
        return _render_table(selected) if visual_type == "table" else _render_chart(selected, visual_type)
    if visual_type in {"flowchart", "process_diagram", "mind_map", "org_chart", "architecture_diagram", "overview"}:
        return _render_text_visual(
            answer, visual_type, image_heading(request, answer, visual_type),
        )
    raise ValueError(f"Unsupported visual format requested: {visual_type}. Try a table, bar/line/pie chart, or supported diagram.")
