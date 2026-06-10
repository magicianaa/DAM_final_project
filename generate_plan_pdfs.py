from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
BODY_FONT_PATH = Path(r"C:\Windows\Fonts\simsun.ttc")
HEADING_FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")
BODY_FONT = "SimSunLocal"
HEADING_FONT = "SimHeiLocal"


def register_font() -> None:
    if not BODY_FONT_PATH.exists():
        raise FileNotFoundError(f"Chinese font not found: {BODY_FONT_PATH}")
    if not HEADING_FONT_PATH.exists():
        raise FileNotFoundError(f"Chinese font not found: {HEADING_FONT_PATH}")
    pdfmetrics.registerFont(TTFont(BODY_FONT, str(BODY_FONT_PATH)))
    pdfmetrics.registerFont(TTFont(HEADING_FONT, str(HEADING_FONT_PATH)))


def escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("  ", "&nbsp;&nbsp;")
    )


def split_table_row(line: str) -> list[str]:
    cells = line.strip().strip("|").split("|")
    return [cell.strip() for cell in cells]


def is_separator_row(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return False
    chars = set(stripped.replace("|", "").replace(" ", ""))
    return bool(chars) and chars <= {"-", ":"}


def parse_markdown(md: str, styles: dict[str, ParagraphStyle]) -> list:
    story = []
    lines = md.splitlines()
    i = 0
    pending_list: list[str] = []

    def flush_list() -> None:
        nonlocal pending_list
        if pending_list:
            items = [
                ListItem(Paragraph(escape(item), styles["body"]), leftIndent=12)
                for item in pending_list
            ]
            story.append(ListFlowable(items, bulletType="bullet", leftIndent=18))
            story.append(Spacer(1, 0.12 * cm))
            pending_list = []

    while i < len(lines):
        line = lines[i].rstrip()

        if not line.strip():
            flush_list()
            i += 1
            continue

        if line.startswith("# "):
            flush_list()
            story.append(Paragraph(escape(line[2:].strip()), styles["title"]))
            story.append(Spacer(1, 0.35 * cm))
            i += 1
            continue

        if line.startswith("## "):
            flush_list()
            story.append(Spacer(1, 0.12 * cm))
            story.append(Paragraph(escape(line[3:].strip()), styles["h2"]))
            story.append(Spacer(1, 0.12 * cm))
            i += 1
            continue

        if line.startswith("### "):
            flush_list()
            story.append(Paragraph(escape(line[4:].strip()), styles["h3"]))
            story.append(Spacer(1, 0.08 * cm))
            i += 1
            continue

        if line.startswith("- "):
            pending_list.append(line[2:].strip())
            i += 1
            continue

        if line and line[0].isdigit() and ". " in line[:5]:
            flush_list()
            story.append(Paragraph(escape(line), styles["body"]))
            story.append(Spacer(1, 0.05 * cm))
            i += 1
            continue

        if line.startswith("|") and line.endswith("|"):
            flush_list()
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                if not is_separator_row(lines[i]):
                    table_lines.append(lines[i])
                i += 1

            data = []
            for row in table_lines:
                data.append([Paragraph(escape(cell), styles["table"]) for cell in split_table_row(row)])
            if data:
                col_count = max(len(row) for row in data)
                for row in data:
                    while len(row) < col_count:
                        row.append(Paragraph("", styles["table"]))
                table = Table(data, repeatRows=1, hAlign="LEFT")
                table.setStyle(
                    TableStyle(
                        [
                            ("FONTNAME", (0, 0), (-1, -1), BODY_FONT),
                            ("FONTNAME", (0, 0), (-1, 0), HEADING_FONT),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F0F0F0")),
                            ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                            ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#9CA3AF")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 4),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                            ("TOPPADDING", (0, 0), (-1, -1), 4),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ]
                    )
                )
                story.append(table)
                story.append(Spacer(1, 0.18 * cm))
            continue

        flush_list()
        story.append(Paragraph(escape(line), styles["body"]))
        story.append(Spacer(1, 0.06 * cm))
        i += 1

    flush_list()
    return story


def build_pdf(md_path: Path, pdf_path: Path) -> None:
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName=HEADING_FONT,
            fontSize=20,
            leading=28,
            alignment=1,
            textColor=colors.black,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "Heading2",
            parent=base["Heading2"],
            fontName=HEADING_FONT,
            fontSize=14,
            leading=21,
            textColor=colors.black,
            spaceBefore=8,
            spaceAfter=4,
        ),
        "h3": ParagraphStyle(
            "Heading3",
            parent=base["Heading3"],
            fontName=HEADING_FONT,
            fontSize=11.5,
            leading=17,
            textColor=colors.black,
            spaceBefore=4,
            spaceAfter=2,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName=BODY_FONT,
            fontSize=10.0,
            leading=15.5,
            firstLineIndent=0,
            textColor=colors.black,
        ),
        "table": ParagraphStyle(
            "Table",
            parent=base["BodyText"],
            fontName=BODY_FONT,
            fontSize=7.6,
            leading=10.6,
            textColor=colors.black,
        ),
    }
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=1.55 * cm,
        leftMargin=1.55 * cm,
        topMargin=1.45 * cm,
        bottomMargin=1.45 * cm,
        title=md_path.stem,
    )
    story = parse_markdown(md_path.read_text(encoding="utf-8"), styles)
    doc.build(story)


def main() -> None:
    register_font()
    files = [
        "城市空气质量预测与异常污染检测系统_项目计划书.md",
        "融合协同过滤与内容特征的电影推荐系统_项目计划书.md",
        "融合协同过滤与内容特征的电影推荐系统_开发计划.md",
    ]
    for filename in files:
        md_path = ROOT / filename
        pdf_path = md_path.with_suffix(".pdf")
        build_pdf(md_path, pdf_path)
        print(f"generated {pdf_path.name}")


if __name__ == "__main__":
    main()
