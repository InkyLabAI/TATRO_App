#!/usr/bin/env python3
"""
Caspi + pyannote diarization transcription script

This script does:
1. MP4 -> mono 16 kHz WAV
2. Speaker diarization with pyannote
3. Split audio by speaker chunks
4. Transcribe each chunk with Caspi, not Whisper
5. Save JSON and PDF outputs

Install first in Colab:
    !pip install -q qwen-asr "pyannote.audio>=3.1" torch torchaudio moviepy reportlab python-bidi soundfile numpy accelerate
    !apt-get -y install ffmpeg fonts-dejavu

Example in Colab:
    !python caspi_diarization_transcript.py \
        --mp4 "/content/drive/MyDrive/hebrew/piece_of_hebrew.mp4" \
        --output-dir "/content/drive/MyDrive/output"
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import soundfile as sf
import torchaudio
import torch

try:
    from moviepy.editor import VideoFileClip
except Exception:
    from moviepy import VideoFileClip

from pyannote.audio import Pipeline
from qwen_asr import Qwen3ASRModel

from bidi.algorithm import get_display
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_RIGHT


DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
CASPI_MODEL = "OzLabs/Caspi-1.7B"


def get_hf_token(token_arg: str | None) -> str:
    """
    Get Hugging Face token.

    Priority:
    1. --hf-token argument
    2. HF_TOKEN environment variable
    3. Colab Secret named HF_TOKEN
    """
    if token_arg:
        return token_arg

    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token

    try:
        from google.colab import userdata
        colab_token = userdata.get("HF_TOKEN")
        if colab_token:
            return colab_token
    except Exception:
        pass

    raise ValueError(
        "HF token not found. Use --hf-token, set HF_TOKEN environment variable, "
        "or add HF_TOKEN in Colab Secrets."
    )


def extract_audio(video_path: str, audio_path: str) -> None:
    """
    Extract mono 16 kHz WAV audio from MP4.
    """
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(
        audio_path,
        fps=16000,
        nbytes=2,
        codec="pcm_s16le",
        ffmpeg_params=["-ac", "1"],
        logger=None,
    )
    clip.close()
    print(f"Audio extracted -> {audio_path}")


def run_diarization(audio_path: str, hf_token: str):
    """
    Run speaker diarization using pyannote.
    """
    print(f"Loading diarization model: {DIARIZATION_MODEL}")

    pipeline = Pipeline.from_pretrained(
        DIARIZATION_MODEL,
        token=hf_token,
    )

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    print("Running diarization...")
    waveform, sr = torchaudio.load(audio_path)
    diarization = pipeline({"waveform": waveform, "sample_rate": sr})

    print("Diarization complete.")
    return diarization


def get_annotation_from_diarization(diarization):
    """
    Support both old and new pyannote outputs.

    Old:
        diarization.itertracks(...)

    New:
        diarization.speaker_diarization.itertracks(...)
    """
    if hasattr(diarization, "itertracks"):
        return diarization

    if hasattr(diarization, "speaker_diarization"):
        return diarization.speaker_diarization

    if hasattr(diarization, "exclusive_speaker_diarization"):
        return diarization.exclusive_speaker_diarization

    raise TypeError("Unsupported diarization output format.")


def diarization_to_chunks(diarization, max_duration: float = 30.0) -> list[dict]:
    """
    Convert diarization result into chunks.

    Each chunk contains:
    - start
    - end
    - speaker
    """
    annotation = get_annotation_from_diarization(diarization)

    raw_turns = []

    for seg, _, spk in annotation.itertracks(yield_label=True):
        raw_turns.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "speaker": str(spk),
        })

    if not raw_turns:
        return []

    raw_turns = sorted(raw_turns, key=lambda x: x["start"])

    # Merge consecutive turns from the same speaker
    merged = [raw_turns[0].copy()]

    for turn in raw_turns[1:]:
        if turn["speaker"] == merged[-1]["speaker"]:
            merged[-1]["end"] = max(merged[-1]["end"], turn["end"])
        else:
            merged.append(turn.copy())

    # Split long turns
    chunks = []

    for turn in merged:
        start = turn["start"]
        final_end = turn["end"]

        while start < final_end:
            end = min(start + max_duration, final_end)

            if end - start > 1.0:  # Only keep chunks longer than 1 second
                chunks.append({
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "speaker": turn["speaker"],
                })

            start = end

    return chunks


def load_caspi_model():
    """
    Load Caspi ASR model.
    """
    if torch.cuda.is_available():
        dtype = torch.bfloat16
        device_map = "cuda:0"
        batch_size = 1
    else:
        dtype = torch.float32
        device_map = "cpu"
        batch_size = 1

    print(f"Loading Caspi model: {CASPI_MODEL}")
    print(f"Device: {device_map}")
    print(f"dtype : {dtype}")

    model = Qwen3ASRModel.from_pretrained(
        CASPI_MODEL,
        dtype=dtype,
        device_map=device_map,
        max_inference_batch_size=batch_size,
        max_new_tokens=256,
    )

    print("Caspi model loaded.")
    return model


def extract_caspi_text(result) -> str:
    """
    Safely extract text from Caspi output.
    """
    if result is None:
        return ""

    if isinstance(result, list):
        if len(result) == 0:
            return ""
        result = result[0]

    if hasattr(result, "text"):
        return str(result.text).strip()

    if isinstance(result, dict):
        return str(result.get("text", "")).strip()

    return str(result).strip()


def cut_audio_chunk(audio_path: str, chunk_path: str, start: float, end: float) -> None:
    """
    Cut one chunk using ffmpeg.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "quiet",
        "-i",
        audio_path,
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-ar",
        "16000",
        "-ac",
        "1",
        chunk_path,
    ]

    subprocess.run(cmd, check=True)


def transcribe_chunks_caspi(
    audio_path: str,
    chunks: list[dict],
    model,
    language: str | None = None,
) -> list[dict]:
    """
    Transcribe diarized chunks with Caspi.
    """
    results = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for idx, chunk in enumerate(chunks):
            start = float(chunk["start"])
            end = float(chunk["end"])
            duration = round(end - start, 3)

            chunk_path = os.path.join(tmp_dir, f"chunk_{idx:04d}.wav")

            cut_audio_chunk(audio_path, chunk_path, start, end)

            audio_array, sr = sf.read(chunk_path)
            audio_array = np.array(audio_array, dtype=np.float32)

            caspi_result = model.transcribe(
                audio=(audio_array, sr),
                language=language,
            )

            text = extract_caspi_text(caspi_result)

            results.append({
                "id": idx + 1,
                "speaker": chunk["speaker"],
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": duration,
                "transcription": text,
            })

            if (idx + 1) % 5 == 0 or (idx + 1) == len(chunks):
                print(f"Transcribed {idx + 1}/{len(chunks)} chunks...")

    return results


def save_json(transcriptions: list[dict], path: str) -> None:
    """
    Save transcript as JSON.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(transcriptions, f, ensure_ascii=False, indent=2)

    print(f"JSON saved -> {path}")


def register_pdf_fonts() -> None:
    """
    Register DejaVu fonts for Hebrew support.
    """
    regular = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    pdfmetrics.registerFont(TTFont("DejaVuSans", regular))
    pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", bold))


def prepare_pdf_text(text: str) -> str:
    """
    Prepare RTL text for PDF.
    """
    if text is None or str(text).strip() == "":
        return "&lt;silence&gt;"

    return escape(get_display(str(text)))


def save_pdf(transcriptions: list[dict], path: str, title: str) -> None:
    """
    Save transcript as PDF.
    """
    register_pdf_fonts()

    doc = SimpleDocTemplate(
        path,
        pagesize=letter,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="DejaVuSans-Bold",
        fontSize=16,
        spaceAfter=16,
        alignment=TA_RIGHT,
    )

    speaker_style = ParagraphStyle(
        "Speaker",
        parent=styles["Normal"],
        fontName="DejaVuSans-Bold",
        fontSize=10,
        leading=15,
        spaceBefore=10,
        spaceAfter=2,
        alignment=TA_RIGHT,
    )

    text_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="DejaVuSans",
        fontSize=10,
        leading=15,
        spaceAfter=6,
        alignment=TA_RIGHT,
    )

    story = [
        Paragraph(prepare_pdf_text(f"Transcription — {title}"), title_style),
        Spacer(1, 0.3 * cm),
    ]

    for chunk in transcriptions:
        speaker = prepare_pdf_text(f"{chunk['speaker']}:")
        text = prepare_pdf_text(chunk.get("transcription", ""))

        story.append(Paragraph(speaker, speaker_style))
        story.append(Paragraph(text, text_style))

    doc.build(story)
    print(f"PDF saved -> {path}")


def preview_chunks(chunks: list[dict], limit: int = 5) -> None:
    """
    Print a small preview of diarization chunks.
    """
    print(f"{'#':>4}  {'Speaker':<14}  {'Start':>7}  {'End':>7}  {'Dur':>6}")
    print("-" * 46)

    for i, c in enumerate(chunks[:limit]):
        dur = c["end"] - c["start"]
        print(
            f"{i + 1:>4}  {c['speaker']:<14}  "
            f"{c['start']:>7.2f}  {c['end']:>7.2f}  {dur:>6.2f}s"
        )

    if len(chunks) > limit:
        print(f"... and {len(chunks) - limit} more")


def preview_transcriptions(transcriptions: list[dict], limit: int = 10) -> None:
    """
    Print a small preview of transcription results.
    """
    print(f"{'ID':>4}  {'Speaker':<14}  {'Start':>7}  {'End':>7}  Text")
    print("-" * 80)

    for t in transcriptions[:limit]:
        text = t["transcription"]
        text_preview = text[:50] + ("..." if len(text) > 50 else "")

        print(
            f"{t['id']:>4}  {t['speaker']:<14}  "
            f"{t['start']:>7.2f}  {t['end']:>7.2f}  {text_preview}"
        )

    if len(transcriptions) > limit:
        print(f"... and {len(transcriptions) - limit} more records")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe an MP4 using pyannote diarization + Caspi ASR."
    )

    parser.add_argument(
        "--mp4",
        required=True,
        help="Path to input MP4 file.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Folder where JSON and PDF outputs will be saved.",
    )

    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token. Optional if HF_TOKEN is in env or Colab Secrets.",
    )

    parser.add_argument(
        "--max-chunk-duration",
        type=float,
        default=30.0,
        help="Maximum chunk duration in seconds.",
    )

    parser.add_argument(
        "--language",
        default=None,
        help="Language passed to Caspi. Default None. You can try Hebrew if needed.",
    )

    return parser.parse_args()


def main(mp4_path: Path, output_dir: Path):
    # args = parse_args()

    # mp4_path = Path(args.mp4).expanduser().resolve()
    # output_dir = Path(args.output_dir).expanduser().resolve()

    # assert mp4_path.exists(), f"MP4 not found: {mp4_path}"
    # assert mp4_path.suffix.lower() == ".mp4", "Input must be an .mp4 file"
    max_chunk_duration = 30
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_token = get_hf_token(None) #args.hf_token)

    video_stem = mp4_path.stem

    json_out = output_dir / f"{video_stem}_caspi_transcript.json"
    pdf_out = output_dir / f"{video_stem}_caspi_transcript.pdf"

    print("Input video:", mp4_path)
    print("Output dir :", output_dir)
    print("JSON out   :", json_out)
    print("PDF out    :", pdf_out)

    audioDir = output_dir / "temp_audio"
    audioDir.mkdir(parents=True, exist_ok=True)
    audio_path = os.path.join(audioDir, "audio.wav")

    print(f"\nExtracting audio from '{mp4_path.name}'...")
    extract_audio(str(mp4_path), audio_path)

    print("\nRunning speaker diarization...")
    diarization = run_diarization(audio_path, hf_token)

    print(f"\nBuilding chunks, max {max_chunk_duration}s...")
    chunks = diarization_to_chunks(
        diarization,
        max_duration=max_chunk_duration,
    )

    if not chunks:
        raise RuntimeError("Diarization returned no speaker turns.")

    print(f"{len(chunks)} chunks created.")
    preview_chunks(chunks)

    print("\nLoading Caspi...")
    caspi_model = load_caspi_model()

    print("\nTranscribing chunks with Caspi...")
    transcriptions = transcribe_chunks_caspi(
        audio_path=audio_path,
        chunks=chunks,
        model=caspi_model,
        language=None,
    )

    print(f"\nTranscription complete: {len(transcriptions)} records.")
    preview_transcriptions(transcriptions)

    print("\nSaving outputs...")
    save_json(transcriptions, str(json_out))
    save_pdf(transcriptions, str(pdf_out), video_stem)

    print("\nDone.")
    print(f"JSON: {json_out}")
    print(f"PDF : {pdf_out}")


if __name__ == "__main__":
    mp4_path = Path("/mnt/c/Users/ahhac/Desktop/Ar_Heb_ASR_Translation/video_demo.mp4")
    output_dir = Path("/mnt/c/Users/ahhac/Desktop/Ar_Heb_ASR_Translation/DemoResults")
    main(mp4_path, output_dir)
