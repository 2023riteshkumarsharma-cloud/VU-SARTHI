"""Professional PDF and DOCX rendering for source-grounded academic documents."""
from dataclasses import dataclass, field
from html import escape
from io import BytesIO
import re
import unicodedata


PDF_MIME = "application/pdf"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
FOOTER_TEXT = "AI-generated document | Synthetic data – demonstration purposes only"


@dataclass
class Section:
    title: str
    paragraphs: list[str] = field(default_factory=list)
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)


@dataclass
class DocumentContent:
    title: str
    subtitle: str = ""
    sections: list[Section] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    synthetic: bool = False


def _safe_filename(value: str, extension: str) -> str:
    base = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    base = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_")[:110] or "academic_document"
    return f"{base}.{extension}"


def filename_for(content: DocumentContent, extension: str) -> str:
    return _safe_filename(content.title, extension)


def _pdf_text(value) -> str:
    text = str(value).replace("–", "-").replace("—", "-").replace("’", "'").replace("“", '"').replace("”", '"')
    return escape(text, quote=False)


def _register_pdf_fonts():
    from pathlib import Path

    import reportlab
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    fonts = Path(reportlab.__file__).parent / "fonts"
    if "VUSans" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("VUSans", str(fonts / "Vera.ttf")))
        pdfmetrics.registerFont(TTFont("VUSans-Bold", str(fonts / "VeraBd.ttf")))


def _watermark(canvas, doc):
    canvas.saveState()
    canvas.setFillColorRGB(0.82, 0.84, 0.87)
    canvas.setFillAlpha(0.18)
    canvas.translate(297.5, 420)
    canvas.rotate(35)
    canvas.setFont("VUSans-Bold", 48)
    canvas.drawCentredString(0, 0, "AI GENERATED")
    canvas.restoreState()
    canvas.saveState()
    canvas.setFillAlpha(1)
    canvas.setFillColorRGB(0.35, 0.38, 0.42)
    canvas.setFont("VUSans", 8)
    canvas.drawString(40, 24, FOOTER_TEXT)
    canvas.drawRightString(555, 24, f"Page {doc.page}")
    canvas.restoreState()


def render_pdf(content: DocumentContent) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    _register_pdf_fonts()
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=22 * mm, bottomMargin=18 * mm, title=content.title,
    )
    base = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocumentTitle", parent=base["Title"], fontName="VUSans-Bold",
        fontSize=19, leading=24, textColor=colors.HexColor("#18324A"),
        alignment=TA_CENTER, spaceAfter=8,
    )
    subtitle_style = ParagraphStyle(
        "DocumentSubtitle", parent=base["Normal"], fontSize=9, leading=13,
        textColor=colors.HexColor("#596776"), alignment=TA_CENTER, spaceAfter=10,
    )
    heading_style = ParagraphStyle(
        "DocumentHeading", parent=base["Heading2"], fontName="VUSans-Bold",
        fontSize=12, leading=16, textColor=colors.HexColor("#18324A"),
        spaceBefore=12, spaceAfter=6, keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "DocumentBody", parent=base["BodyText"], fontName="VUSans",
        fontSize=9, leading=13, spaceAfter=6,
    )
    cell_style = ParagraphStyle("Cell", parent=body_style, fontSize=7.5, leading=10, spaceAfter=0)

    story = [Paragraph(_pdf_text(content.title), title_style)]
    if content.subtitle:
        story.append(Paragraph(_pdf_text(content.subtitle), subtitle_style))
    if content.synthetic:
        story.append(Paragraph("<b>Synthetic demonstration data. Not an official university record.</b>", subtitle_style))
    for section in content.sections:
        if section.title:
            story.append(Paragraph(_pdf_text(section.title), heading_style))
        for paragraph in section.paragraphs:
            story.append(Paragraph(_pdf_text(paragraph).replace("\n", "<br/>"), body_style))
        if section.headers:
            data = [[Paragraph(f"<b>{_pdf_text(cell)}</b>", cell_style) for cell in section.headers]]
            data.extend(
                [[Paragraph(_pdf_text(cell), cell_style) for cell in row] for row in section.rows]
            )
            available_width = A4[0] - doc.leftMargin - doc.rightMargin
            weights = []
            for header in section.headers:
                normalized = header.casefold()
                if "prerequisite" in normalized:
                    weights.append(1.6)
                elif "course" in normalized or "title" in normalized:
                    weights.append(1.7)
                elif "basket" in normalized:
                    weights.append(1.15)
                elif "code" in normalized:
                    weights.append(0.9)
                elif any(key in normalized for key in ("semester", "batch", "credit", "l-t-p")):
                    weights.append(0.7)
                else:
                    weights.append(1.0)
            weight_total = sum(weights)
            table = Table(
                data, colWidths=[available_width * weight / weight_total for weight in weights],
                repeatRows=1, hAlign="LEFT", splitByRow=1,
            )
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#18324A")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CAD3DC")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.extend([table, Spacer(1, 5)])
    if content.sources:
        story.append(Paragraph("Sources", heading_style))
        for source in content.sources:
            story.append(Paragraph(_pdf_text(source), body_style))
    doc.build(story, onFirstPage=_watermark, onLaterPages=_watermark)
    return buffer.getvalue()


def _add_word_watermark(header):
    from docx.oxml import parse_xml
    from docx.oxml.ns import nsdecls

    watermark = parse_xml(
        f'<w:pict {nsdecls("w")} xmlns:v="urn:schemas-microsoft-com:vml" '
        'xmlns:o="urn:schemas-microsoft-com:office:office">'
        '<v:shapetype id="_x0000_t136" coordsize="1600,21600" '
        'o:spt="136" adj="10800" path="m@7,0l@8,0m@5,21600l@6,21600e">'
        '<v:formulas><v:f eqn="sum #0 0 10800"/><v:f eqn="prod #0 2 1"/>'
        '<v:f eqn="sum 21600 0 @1"/><v:f eqn="sum 0 0 @2"/>'
        '<v:f eqn="sum 21600 0 @3"/><v:f eqn="if @0 @3 0"/>'
        '<v:f eqn="if @0 21600 @1"/><v:f eqn="if @0 0 @2"/>'
        '<v:f eqn="if @0 @4 21600"/></v:formulas>'
        '<v:path textpathok="t" o:connecttype="custom" o:connectlocs="@9,0;@10,10800;@11,21600;@12,10800" o:connectangles="270,180,90,0"/>'
        '<v:textpath on="t" fitshape="t"/>'
        '<v:handles><v:h position="#0,bottomRight" xrange="6629,14971" /></v:handles>'
        '<o:lock v:ext="edit" text="t" shapetype="t"/>'
        '</v:shapetype>'
        '<v:shape id="VUAIWatermark" type="#_x0000_t136" '
        'style="position:absolute;margin-left:0;margin-top:0;width:468pt;height:117pt;'
        'rotation:315;z-index:-251654144;mso-position-horizontal:center;'
        'mso-position-horizontal-relative:margin;mso-position-vertical:center;'
        'mso-position-vertical-relative:margin" fillcolor="#D1D5DB" stroked="f">'
        '<v:fill opacity=".22"/><v:textpath style="font-family:\'Arial\';font-size:1pt" '
        'string="AI GENERATED"/></v:shape></w:pict>'
    )
    header.paragraphs[0]._p.append(watermark)


def render_docx(content: DocumentContent) -> bytes:
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt, RGBColor

    document = Document()
    section = document.sections[0]
    section.page_width = Inches(8.27)
    section.page_height = Inches(11.69)
    section.top_margin = Inches(0.85)
    section.bottom_margin = Inches(0.75)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)
    section.header_distance = Inches(0.3)
    section.footer_distance = Inches(0.35)
    _add_word_watermark(section.header)

    normal = document.styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(9)
    title = document.add_heading(content.title, 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.style.font.name = "Arial"
    title.style.font.color.rgb = RGBColor(24, 50, 74)
    if content.subtitle:
        paragraph = document.add_paragraph(content.subtitle)
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if content.synthetic:
        paragraph = document.add_paragraph("Synthetic demonstration data. Not an official university record.")
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.runs[0].bold = True
    for item in content.sections:
        if item.title:
            document.add_heading(item.title, level=1)
        for text in item.paragraphs:
            document.add_paragraph(text)
        if item.headers:
            table = document.add_table(rows=1, cols=len(item.headers))
            table.style = "Light Shading Accent 1"
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            table.autofit = True
            header_cells = table.rows[0].cells
            for idx, header_text in enumerate(item.headers):
                header_cells[idx].text = str(header_text)
                for run in header_cells[idx].paragraphs[0].runs:
                    run.bold = True
            for row in item.rows:
                cells = table.add_row().cells
                for idx, value in enumerate(row):
                    cells[idx].text = str(value)
    if content.sources:
        document.add_heading("Sources", level=1)
        for source in content.sources:
            document.add_paragraph(source, style="List Bullet")

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run(FOOTER_TEXT)
    footer.add_run("    Page ")
    page_field = OxmlElement("w:fldSimple")
    page_field.set(qn("w:instr"), "PAGE")
    footer._p.append(page_field)
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def render(content: DocumentContent, file_format: str) -> tuple[str, str, bytes]:
    if file_format == "pdf":
        return filename_for(content, "pdf"), PDF_MIME, render_pdf(content)
    if file_format == "docx":
        return filename_for(content, "docx"), DOCX_MIME, render_docx(content)
    raise ValueError(f"Unsupported document format: {file_format}")
