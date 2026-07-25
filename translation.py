# ============================================================
# translation.py — Script 2
# Qwen translation + PDF export: translate_dataframe_with_qwen(...),
# save_final_translation_pdf(...), run_translation(...).
#
# Works both run locally and pasted into its own Colab cell. Imports
# qwen_generate, load_qwen_model, and save_json from transcription.py,
# which needs to be in the same folder (or already run, in Colab).
# ============================================================
from pathlib import Path
from html import escape
import re
from turtle import pd
import pandas
import torch
import arabic_reshaper
from bidi.algorithm import get_display

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_RIGHT, TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import numpy as np
from transcription import qwen_generate, load_qwen_model, save_json

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import simpleSplit
from reportlab.lib import colors

import arabic_reshaper
from bidi.algorithm import get_display
from html import escape

# DejaVu Sans is what the PDF is built with (it covers Hebrew/Arabic/Latin
# glyphs). Colab/most Linux boxes have it after `apt-get install
# fonts-dejavu-core`; this checks a few common locations so the script
# doesn't only work on one OS.
_FONT_SEARCH_DIRS = [
    "/usr/share/fonts/truetype/dejavu",   # Debian/Ubuntu, Colab
    "/usr/share/fonts/dejavu",            # Fedora/RHEL
    "/usr/local/share/fonts",
    str(Path.home() / ".fonts"),
    str(Path.home() / "Library/Fonts"),   # macOS
]

def _find_font_path(filename):
    for directory in _FONT_SEARCH_DIRS:
        candidate = Path(directory) / filename
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f"Could not find {filename}. Install the DejaVu fonts — "
        "`apt-get install fonts-dejavu-core` on Linux/Colab, or download "
        "DejaVu Sans from dejavu-fonts.github.io on Windows/Mac and drop it "
        "in one of: " + ", ".join(_FONT_SEARCH_DIRS)
    )

def translate_chunks_with_qwen(tokenizer, model, chunks, source_lang_name, target_lang_name, max_new_tokens=2048):
    """
    chunks:
        len == 1:
            [{'speaker':..., 'transcription':...}]

        len == 3:
            [previous_chunk, current_chunk, next_chunk]

    Returns:
        Translation of ONLY the current chunk.
    """

    # --------------------------------------------------
    # Single chunk (first or last segment)
    # --------------------------------------------------
    if len(chunks) == 1:

        text = chunks[0]["transcription"].strip()

        if not text:
            return ""

        system_prompt = (
            f"You are an expert translator specialized in {source_lang_name} speech transcripts. "
            f"You translate spoken {source_lang_name} into fluent Modern Standard {target_lang_name}."
        )

        user_prompt = f"""
                        Translate the following {source_lang_name} speech transcript into Modern Standard {target_lang_name}.

                        Rules:

                        1. Preserve the original meaning.
                        2. Produce natural fluent {target_lang_name}.
                        3. Take into account the grammar and spelling errors in the transcription text.
                        4. Do NOT invent missing content.
                        5. If a short portion is impossible to understand, write:
                        [غير واضح]
                        6. Output ONLY the translation.
                        7. Output must be in {target_lang_name}.

                        Transcript:

                        {text}
                        """

        return qwen_generate(tokenizer, model, system_prompt,user_prompt,max_new_tokens=max_new_tokens,)
    # --------------------------------------------------
    # Three chunks:
    # previous / current / next
    # Translate ONLY current
    # --------------------------------------------------

    prev_chunk = chunks[0]
    cur_chunk = chunks[1]
    next_chunk = chunks[2]

    system_prompt = (
        f"You are an expert translator specialized in noisy {source_lang_name} speech transcripts. "
        f"You translate spoken {source_lang_name} into fluent Modern Standard {target_lang_name}. "
    )

    user_prompt = f"""
                    You are given three consecutive transcript chunks from the same conversation.

                    The first and third chunks are CONTEXT ONLY.

                    Translate ONLY the SECOND chunk.

                    Read all three chunks before translating.

                    Rules:

                    1. Translate ONLY the second chunk.
                    2. Take into account during translation the grammar and spelling errors in transcription text. 
                    3. Take into account the boudaries of the second chunk, which may be incomplete sentences.
                    4. Use the first and third chunks only to understand context and resolve ambiguities.
                    5. Never translate the first or third chunk.
                    6. Preserve the meaning.
                    7. Produce fluent Modern Standard {target_lang_name}.
                    8. Do not add any word in a different language than {target_lang_name}.
                    9. Do NOT output speaker names.
                    10. Output contains ONLY the translation of the second chunk in {target_lang_name}.
                    

                    Context:

                    Previous chunk
                    Speaker: {prev_chunk["speaker"]}
                    Text:
                    {prev_chunk["transcription"]}


                    Current chunk (TRANSLATE THIS)
                    Speaker: {cur_chunk["speaker"]}
                    Text:
                    {cur_chunk["transcription"]}


                    Next chunk
                    Speaker: {next_chunk["speaker"]}
                    Text:
                    {next_chunk["transcription"]}
                    """

    return qwen_generate(tokenizer, model, system_prompt,user_prompt,max_new_tokens=max_new_tokens)


def parse_translated_chunks(text):

    result = {}

    blocks = re.findall(
        r"\[Chunk\s+(\d+)\]\s*(.*?)(?=\[Chunk\s+\d+\]|\Z)",
        text,
        flags=re.S
    )

    for idx, content in blocks:
        chunktext = content.strip()
            # Remove Qwen accidentally returning metadata prefixes
        chunktext = re.sub(
            r"^\s*Speaker\s*:\s*[^\n]+\n?",
            "",
            chunktext,
            flags=re.IGNORECASE
        )

        chunktext = re.sub(
            r"^\s*Text\s*:\s*",
            "",
            chunktext,
            flags=re.IGNORECASE
        )

        # Remove possible markdown/code formatting
        chunktext = chunktext.replace("```text", "")
        chunktext = chunktext.replace("```", "")
        result[int(idx)] = chunktext

    return result

def translate_dataframe_with_qwen(df, tokenizer, model, source_lang_name, target_lang_name, source_column, target_column, max_new_tokens=2048):

    assert source_column in df.columns, \
        f"Column not found: {source_column}"

    df = df.reset_index(drop=True).copy()

    chunks = []

    for idx, row in df.iterrows():
        chunks.append(
            {
                "id": idx + 1,
                "speaker": row.get("speaker", "unknown"),
                "transcription": str(row[source_column])
            }
        )

    total = len(chunks)

    translated = []

    # ---------------------------------------------------------
    # Translate chunk by chunk using context window
    # ---------------------------------------------------------

    for i in range(total):

        # First chunk: no previous context
        if i == 0:

            context_chunks = [
                chunks[i]
            ]

        # Last chunk: no future context
        elif i == total - 1:

            context_chunks = [
                chunks[i]
            ]

        # Middle chunks: previous + current + next
        else:

            context_chunks = [
                chunks[i - 1],
                chunks[i],
                chunks[i + 1]
            ]

        translation = translate_chunks_with_qwen(tokenizer, model, context_chunks, source_lang_name, target_lang_name, max_new_tokens=max_new_tokens)

        translated.append(translation.strip())

        print(
            f"Translated {i + 1}/{total}",
            end="\r"
        )

    print()

    df[target_column] = translated

    return df




from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import simpleSplit

import arabic_reshaper
from bidi.algorithm import get_display
import unicodedata


def clean_rtl_text(text):
    """
    Remove hidden Unicode direction control characters
    that can reverse Arabic paragraphs.
    """
    if not text:
        return ""

    bad_chars = {
        "\u200e",  # LRM
        "\u200f",  # RLM
        "\u202a",  # LRE
        "\u202b",  # RLE
        "\u202c",  # PDF
        "\u202d",  # LRO
        "\u202e",  # RLO
        "\u2066",  # LRI
        "\u2067",  # RLI
        "\u2069",  # PDI
    }

    return "".join(c for c in str(text) if c not in bad_chars)


def arabic_visual(text):
    """
    Convert logical Arabic text into ReportLab display text.
    """
    if not text:
        return ""

    text = clean_rtl_text(text)

    try:
        text = arabic_reshaper.reshape(text)
    except Exception:
        pass

    return get_display(text, base_dir="R")


def wrap_arabic(text, font, size, width):
    """
    Wrap Arabic before bidi conversion.
    """
    words = clean_rtl_text(text).split()

    lines = []
    current = ""

    for word in words:

        test = word if not current else current + " " + word

        visual = arabic_visual(test)

        if pdfmetrics.stringWidth(visual, font, size) <= width:
            current = test
        else:
            if current:
                lines.append(current)

            current = word

    if current:
        lines.append(current)

    return lines

def save_final_translation_pdf(df, path, title, target_column, target_is_rtl, target_lang_name):

    pdfmetrics.registerFont(
        TTFont(
            "DejaVuSans",
            _find_font_path("DejaVuSans.ttf")
        )
    )

    pdfmetrics.registerFont(
        TTFont(
            "DejaVuSans-Bold",
            _find_font_path("DejaVuSans-Bold.ttf")
        )
    )


    c = canvas.Canvas(str(path), pagesize=letter)


    page_width, page_height = letter


    margin = 2 * cm

    right_x = page_width - margin

    usable_width = page_width - (2 * margin)


    y = page_height - margin
    # -------- TITLE --------

    c.setFont("DejaVuSans-Bold", 16)

    title_text = (f"{target_lang_name} Translation - {title}")

    if target_is_rtl:
        c.drawRightString(right_x, y, arabic_visual(title_text))
    else:
        c.drawString(margin, y, title_text)
    y -= 40
    # -------- CONTENT --------

    for _, row in df.iterrows():
        speaker_number = row.get("speaker_number","")

        speaker = (
            f"Speaker {speaker_number}"
            if speaker_number != ""
            else row.get("speaker_label","")
        )


        c.setFont("DejaVuSans-Bold", 10)
        if target_is_rtl:

            c.drawRightString(right_x, y, arabic_visual(str(speaker)))
        else:
            c.drawString(margin, y, str(speaker))

        y -= 18
        text = row.get(target_column,"")

        c.setFont("DejaVuSans",10)
        if target_is_rtl:

            lines = wrap_arabic(text, "DejaVuSans", 10, usable_width)

            for line in lines:
                if y < margin:

                    c.showPage()
                    y = page_height - margin
                    c.setFont("DejaVuSans", 10)

                c.drawRightString(right_x, y, arabic_visual(line))
                y -= 15
        else:
            lines = simpleSplit(text, "DejaVuSans", 10, usable_width)
            for line in lines:
                if y < margin:
                    c.showPage()
                    y = page_height - margin

                c.drawString(margin, y, line)
                y -= 15
        y -= 15
        if y < margin:

            c.showPage()
            y = page_height - margin
    c.save()

    print("Final PDF saved ->", path)

# ============================================================
# FUNCTION 2 — TRANSLATION
# dataset -> Qwen translate -> save JSON -> final PDF
# ============================================================
def run_translation(dataset_df, input_path, output_dir, source_selected, target_selected, config):
    input_stem = input_path.stem
    translated_json_out = output_dir / f"{input_stem}_translated.json"
    final_pdf_out        = output_dir / f"{input_stem}_{target_selected['name'].lower()}_final.pdf"
    source_lang_name = source_selected["name"]
    target_lang_name = target_selected["name"]
    target_is_rtl = target_selected["is_rtl"]

    print("\n--- Translating with Qwen ---")
    qwen_tokenizer, qwen_model = load_qwen_model(config["qwen_model"])
    translated_df = translate_dataframe_with_qwen(
        dataset_df, qwen_tokenizer, qwen_model, source_lang_name, target_lang_name,
        source_column="source_text", target_column="target_text", 
        max_new_tokens=config["max_new_tokens"]
    )
    del qwen_tokenizer, qwen_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    save_json(translated_df.to_dict(orient="records"), translated_json_out)

    print("\n--- Building final PDF ---")
    save_final_translation_pdf(
        translated_df, final_pdf_out, title=input_stem, target_column="target_text",
        target_is_rtl=target_is_rtl, target_lang_name=target_lang_name,
    )
    return translated_df, translated_json_out, final_pdf_out

if __name__ == "__main__":
    translated_json_out = Path("/mnt/c/Users/ahhac/Desktop/Ar_Heb_ASR_Translation/ATRA_Offline/fypoutput/politic_podcast - Trim_translated.json")
    final_pdf_out = Path("/mnt/c/Users/ahhac/Desktop/Ar_Heb_ASR_Translation/ATRA_Offline/fypoutput/correct_pdf_format.pdf")
    translated_df = pandas.read_json(translated_json_out)
    language = {"name": "Arabic",              "code": "ar", "whisper_code": "ar", "is_rtl": True}
    target_is_rtl = language["is_rtl"]
    target_lang_name = language["name"]
    input_stem = "correct_pdf_format"
    save_final_translation_pdf(
            translated_df, final_pdf_out, title=input_stem, target_column="target_text",
            target_is_rtl=target_is_rtl, target_lang_name=target_lang_name,
        )