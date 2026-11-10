"""Read reusable raw transcription and choose text for summarization."""
import math

from artifacts import read_json
from languages import LANGUAGES


def raw_transcript_path(artifacts):
    # Accept the original location used before video-named results folders.
    legacy = artifacts.video.with_name(artifacts.video.stem + "_transcript.json")
    return next((path for path in (artifacts.raw, legacy) if path.is_file()), None)


def load_raw_transcript(path):
    rows = read_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Saved raw transcript must contain a non-empty list of chunks.")
    required = {"id", "speaker", "start", "end", "transcription"}
    for row in rows:
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError("Saved raw transcript is missing chunk IDs, speaker, text or timestamps.")
        if not isinstance(row["transcription"], str):
            raise ValueError("Saved transcript text must be a string.")
        if not all(isinstance(row[key], (int, float)) and math.isfinite(row[key]) for key in ("start", "end")) or row["end"] < row["start"]:
            raise ValueError("Saved raw transcript contains invalid timestamps.")
    return rows


def summary_records(artifacts):
    if artifacts.transcript.is_file():
        rows = read_json(artifacts.transcript)
        if not isinstance(rows, list) or not rows:
            raise ValueError("Saved transcript dataset is empty or invalid.")
        return rows  # Corrected text when available; otherwise the saved uncorrected dataset.
    raw_path = raw_transcript_path(artifacts)
    if raw_path is None:
        raise ValueError("Transcribe the video before generating a summary.")
    raw = load_raw_transcript(raw_path)
    speakers, rows = {}, []
    for chunk in raw:
        if not chunk["transcription"].strip():
            continue
        language = chunk.get("source_language") or chunk.get("language")
        selected = next((item for item in LANGUAGES.values() if language in (item["name"], item["code"])), None)
        if selected is None:
            raise ValueError("The raw transcript has no recognized source language. Run processing to set its language.")
        speakers.setdefault(chunk["speaker"], len(speakers) + 1)
        rows.append({"id": chunk["id"], "speaker_number": speakers[chunk["speaker"]],
                     "start": chunk["start"], "end": chunk["end"],
                     "source_language": selected["name"], "source_text": chunk["transcription"], "corrected": False})
    if not rows:
        raise ValueError("The raw transcript contains no speech text.")
    return rows
