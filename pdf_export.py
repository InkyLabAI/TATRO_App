"""Paginated Unicode PDF output, including Arabic and Hebrew layout."""
import os
from pathlib import Path


def save_pdf(text, path, title, language):
    import arabic_reshaper
    from bidi.algorithm import get_display
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4

    if language == "Chinese":
        font = "STSong-Light"
        pdfmetrics.registerFont(UnicodeCIDFont(font))
    else:
        candidates = [os.environ.get("TATRO_PDF_FONT", ""),
                      "C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                      str(Path.home() / ".fonts/DejaVuSans.ttf"), "/Library/Fonts/Arial Unicode.ttf"]
        font_path = next((p for p in candidates if p and Path(p).is_file()), None)
        if not font_path:
            raise FileNotFoundError("Install DejaVu Sans or set TATRO_PDF_FONT to a Unicode .ttf font.")
        font = "TatroUnicode"
        pdfmetrics.registerFont(TTFont(font, font_path))
    rtl = language in {"Arabic", "Hebrew", "Persian (Farsi)"}

    def visual(line):
        return get_display(arabic_reshaper.reshape(line), base_dir="R") if rtl else line

    path = Path(path)
    temporary = path.with_suffix(".pdf.tmp")
    pdf = canvas.Canvas(str(temporary), pagesize=A4)
    pdf.setTitle(title)
    width, height = A4
    margin, size, leading = 48, 11, 17
    y = height - margin
    pdf.setFont(font, size)

    def draw(line):
        nonlocal y
        if y < margin:
            pdf.showPage()
            pdf.setFont(font, size)
            y = height - margin
        if rtl:
            pdf.drawRightString(width - margin, y, visual(line))
        else:
            pdf.drawString(margin, y, line)
        y -= leading

    try:
        # Wrap in logical order before bidi conversion; split long words and CJK too.
        for paragraph in (title + "\n\n" + text).splitlines():
            line = ""
            units = list(paragraph) if language == "Chinese" else paragraph.split(" ")
            separator = "" if language == "Chinese" else " "
            for unit in units:
                candidate = line + (separator if line else "") + unit
                if pdfmetrics.stringWidth(visual(candidate), font, size) <= width - 2 * margin:
                    line = candidate
                    continue
                if line:
                    draw(line)
                line = ""
                for character in unit:
                    if pdfmetrics.stringWidth(visual(line + character), font, size) > width - 2 * margin:
                        draw(line)
                        line = ""
                    line += character
            draw(line)
        pdf.save()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
