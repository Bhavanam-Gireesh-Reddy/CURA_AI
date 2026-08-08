"""Dependency-free PDF reports for model predictions and RAG guidance."""

from __future__ import annotations

from datetime import datetime, timezone
import os
import textwrap
import uuid

DISCLAIMER = "This AI-generated report is informational only and is not medical advice. Please consult a qualified healthcare professional."


def _pdf_text(value: str) -> str:
    return value.encode("latin-1", "replace").decode("latin-1").replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def create_prediction_report(report_directory: str, title: str, prediction: str, confidence: float, rag_answer: str) -> str:
    """Write a simple, spacious A4 text report without external dependencies."""
    os.makedirs(report_directory, exist_ok=True)
    # The reference report uses a deliberately minimal document treatment: plain
    # black text, visible Markdown markers, wide readable line spacing, and no
    # decorative header. Preserve that approach for the medical reports.
    lines = [
        title,
        "",
        "**Prediction Details:**",
        "",
        f"* **Prediction:** {prediction}",
        f"* **Model Confidence:** {confidence:.2f}%",
        "",
        "**Retrieved Medical Information:**",
        "",
    ]
    for paragraph in rag_answer.splitlines() or [rag_answer]:
        lines.extend(textwrap.wrap(paragraph.strip(), width=100) or [""])
    lines += ["", "**Medical Disclaimer:**"] + textwrap.wrap(DISCLAIMER, width=100)
    lines += ["", f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"]
    # 27 lines at 27 pt fit within the reference's airy A4 layout.
    pages = [lines[index:index + 27] for index in range(0, len(lines), 27)]

    objects: list[bytes] = [b"", b""]
    page_numbers: list[int] = []
    for page_lines in pages:
        commands = ["BT", "/F1 11 Tf", "32 800 Td", "27 TL"]
        for line in page_lines:
            commands += [f"({_pdf_text(line)}) Tj", "T*"]
        stream = "\n".join(commands + ["ET"]).encode("latin-1")
        page_number, content_number = len(objects) + 1, len(objects) + 2
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595.28 841.89] /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> /Contents {content_number} 0 R >>".encode())
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
        page_numbers.append(page_number)
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(f'{number} 0 R' for number in page_numbers)}] /Count {len(page_numbers)} >>".encode()

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(pdf)); pdf.extend(f"{number} 0 obj\n".encode()); pdf.extend(obj); pdf.extend(b"\nendobj\n")
    xref = len(pdf); pdf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]: pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    filename = f"{uuid.uuid4()}_cura_ai_report.pdf"
    with open(os.path.join(report_directory, filename), "wb") as report: report.write(pdf)
    return filename
