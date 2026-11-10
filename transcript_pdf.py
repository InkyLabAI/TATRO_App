"""Export saved transcript passages without running an AI model again."""
from artifacts import fingerprint, read_json, write_json
from pdf_export import save_pdf
from progress import report


def ensure_transcript_pdf(artifacts, records, force=False):
    if not records:
        raise ValueError("No transcription is available for PDF export.")
    corrected = all(row.get("corrected", False) for row in records)
    suffix = "_corrected_transcription" if corrected else "_transcription"
    pdf = artifacts.path(suffix + ".pdf")
    metadata = artifacts.path(suffix + ".pdf.meta.json")
    expected = {"transcript": fingerprint(records), "pdf_format": "speaker-text-v2"}
    if not force and pdf.exists() and metadata.exists() and read_json(metadata) == expected:
        report("Reusing saved transcription PDF")
        return pdf

    report("Creating corrected transcription PDF" if corrected else "Creating transcription PDF")
    text = "\n\n".join(
        f"Speaker {row['speaker_number']}\n{row['source_text']}"
        for row in records
    )
    artifacts.directory.mkdir(parents=True, exist_ok=True)
    title = "Corrected transcription" if corrected else "Transcription (uncorrected)"
    save_pdf(text, pdf, f"{artifacts.video.stem} - {title}", records[0]["source_language"])
    write_json(metadata, expected)
    return pdf
