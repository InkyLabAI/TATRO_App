# ============================================================
# transcription.py — Script 1
# Whisper transcription + speaker diarization: language dictionary,
# audio/diarization helpers, run_transcription(...).
#
# Self-contained module — works both run locally (`python main.py` /
# `python app.py`) and pasted into its own Colab cell. Needs the
# packages in requirements.txt installed, plus ffmpeg on PATH.
# ============================================================
import os
import re
import json
import subprocess
import gc
from progress import report
from pathlib import Path
import numpy as np
import pandas as pd
# ============================================================
# LANGUAGE DICTIONARY
# ============================================================
from languages import LANGUAGES

def ask_language(prompt_label, exclude_code=None):
    print(prompt_label)
    for key, lang in LANGUAGES.items():
        print(f"  {key}. {lang['name']}")
    choice = input("Enter number (1-4): ").strip()
    while choice not in LANGUAGES or LANGUAGES[choice]["code"] == exclude_code:
        if choice in LANGUAGES and LANGUAGES[choice]["code"] == exclude_code:
            print("Target must be different from the source language.")
        else:
            print("Invalid choice.")
        choice = input("Enter number (1-4): ").strip()
    return LANGUAGES[choice]

# ============================================================
# SHARED HELPERS (audio / diarization / chunks / save)
# ============================================================
MAX_SPEAKER_GAP = 0.5
SHORT_ISLAND_SECONDS = 2.0
MIN_TURN_DURATION = 1

def extract_audio(input_file, audio_path):
    report("Extracting audio from video")
    command = ["ffmpeg", "-y", "-loglevel", "quiet", "-i", input_file, "-ar", "16000", "-ac", "1", audio_path]
    subprocess.run(command, check=True)
    print(f"Audio extracted -> {audio_path}")

def run_diarization(audio_path, hf_token, diarization_model):
    import torch
    from pyannote.audio import Pipeline
    report("Loading weights for speaker detection")
    print("Loading diarization model...")
    pipeline = Pipeline.from_pretrained(diarization_model, token=hf_token)
    print("Diarization device:", "cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    report("Detecting speakers and speech boundaries")
    print("Running diarization...")
    diarization = pipeline(audio_path)
    report("Speaker detection completed")
    print("Diarization finished")
    return diarization

def get_annotation_from_diarization(diarization):
    if hasattr(diarization, "itertracks"):
        return diarization
    if hasattr(diarization, "speaker_diarization"):
        return diarization.speaker_diarization
    if hasattr(diarization, "exclusive_speaker_diarization"):
        return diarization.exclusive_speaker_diarization
    raise TypeError("Unsupported diarization output format")

def diarization_to_chunks(diarization, max_duration, merge_window=3):
    annotation = get_annotation_from_diarization(diarization)
    raw_turns = []
    for turn_id, (seg, _, spk) in enumerate(annotation.itertracks(yield_label=True), start=1):
        start, end = float(seg.start), float(seg.end)
        # if end - start < MIN_TURN_DURATION:
        #     continue
        raw_turns.append({
        "start": start,
        "end": end,
        "duration": end - start,
        "speaker": spk,
        "boundaries": [[start, end]],
    })
    if not raw_turns:
        return [], []
    
    chunks = build_chunks(raw_turns, min_duration=MIN_TURN_DURATION, max_duration=max_duration)
    print(f"total number of created chunks: {len(chunks)}")
    return chunks, []

def build_chunks(raw_turns, min_duration, max_duration):
    """Keep every speaker boundary and split long turns before audio extraction.

    Short utterances are retained: dropping them can incorrectly join two runs
    separated by another speaker. min_duration is retained for caller compatibility.
    """
    if max_duration <= 0:
        raise ValueError("Maximum audio chunk duration must be positive.")
    chunks = []
    for turn in raw_turns:
        start = float(turn["start"])
        while start < turn["end"]:
            end = min(start + max_duration, turn["end"])
            part = {"start": start, "end": end, "duration": end - start,
                    "speaker": turn["speaker"], "boundaries": [[start, end]]}
            if (chunks and chunks[-1]["speaker"] == part["speaker"]
                    and chunks[-1]["duration"] + part["duration"] <= max_duration):
                chunks[-1]["end"] = end
                chunks[-1]["duration"] += part["duration"]
                chunks[-1]["boundaries"].extend(part["boundaries"])
            else:
                chunks.append(part)
            start = end
    return chunks


def load_whisper_model(whisper_model_name):
    import torch
    from faster_whisper import WhisperModel
    if torch.cuda.is_available():
        device, compute_type = "cuda", "float16"
    else:
        device, compute_type = "cpu", "int8"
    report(f"Loading transcription weights: {whisper_model_name} on {device}")
    print("Loading Whisper model...", whisper_model_name, "| Device:", device)
    model = WhisperModel(whisper_model_name, device=device, compute_type=compute_type)
    print("Whisper model loaded")
    return model

def transcribe_chunks_with_whisper(audio_path, chunks, model, whisper_language):
    import librosa
    results = []
    #audio = load_audio(audio_path)
    audio, sr = librosa.load(audio_path,sr=16000,mono=True)
    report("Transcribing audio chunks", 0, len(chunks))
    for idx, chunk in enumerate(chunks):
        start = float(chunk["start"])
        end = float(chunk["end"])
        duration = round(chunk["duration"], 3)   # use speech duration, not end-start

        #print(f"start={start:.3f}s, end={end:.3f}s")
        #print(f"Speaker: {chunk['speaker']}, Chunk {idx + 1}/{len(chunks)}")
        #print(f"Boundaries: {chunk['boundaries']}")
        #print("-----------------------------------------------")

        # Concatenate all speech segments belonging to this chunk
        audio_segments = []

        for seg_start, seg_end in chunk["boundaries"]:
            s = int(seg_start * sr)
            e = int(seg_end * sr)
            audio_segments.append(audio[s:e])

        if not audio_segments:
            continue

        audio_chunk = np.concatenate(audio_segments)

        segments, info = model.transcribe(
            audio_chunk,
            language=whisper_language,
            beam_size=1,
            best_of=1,
            condition_on_previous_text=False,
            word_timestamps=False,
            vad_filter=False,
        )

        text = " ".join(seg.text.strip() for seg in segments).strip()

        results.append({
            "id": idx + 1,
            "speaker": chunk["speaker"],
            "start": round(start, 3),        # original timeline
            "end": round(end, 3),            # original timeline
            "duration": duration,            # actual speech duration
            "boundaries": chunk["boundaries"],
            "language": whisper_language,
            "transcription": text,
        })

        report("Chunks transcribed", idx + 1, len(chunks))
        if (idx + 1) % 5 == 0 or (idx + 1) == len(chunks):
            print(f"Transcribed {idx + 1}/{len(chunks)} chunks")

    return results

def run_transcription(input_path, output_dir, hf_token, source_selected, config):
    """Transcribe -> merge 3/2 -> correct -> save -> index in the video's results folder.

    Reuse saved raw Whisper chunks, then regenerate downstream results.
    Loading an existing video in the UI does not call this function.
    """
    import torch
    from artifacts import Artifacts, write_json, atomic_write
    from chunks import merge_chunks
    from local_llm import session, correct_chunks
    from retrieval import ensure_index, EMBEDDING_MODELS
    from transcript_pdf import ensure_transcript_pdf
    from transcript_data import raw_transcript_path, load_raw_transcript

    artifacts = Artifacts(input_path)
    artifacts.directory.mkdir(parents=True, exist_ok=True)
    cached_raw = raw_transcript_path(artifacts)
    if cached_raw is not None:
        report("Reusing saved raw transcription; skipping audio extraction, speaker detection and Whisper")
        raw = load_raw_transcript(cached_raw)
        languages = {row.get("language") for row in raw if row.get("language")}
        if languages and languages != {source_selected["whisper_code"]}:
            raise ValueError("The saved raw transcript language differs from the selected video language. Select its original language.")
        if cached_raw != artifacts.raw:
            write_json(artifacts.raw, raw)
    else:
        audio_out = artifacts.path("_audio.wav")
        extract_audio(str(artifacts.video), str(audio_out))
        diarization = run_diarization(str(audio_out), hf_token, config["diarization_model"])
        report("Creating audio chunks")
        chunks, _ = diarization_to_chunks(diarization, max_duration=config["max_chunk_duration"])
        if not chunks:
            raise ValueError("Diarization returned no speech.")
        del diarization
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        whisper = load_whisper_model(config["whisper_model"])
        try:
            raw = transcribe_chunks_with_whisper(str(audio_out), chunks, whisper, source_selected["whisper_code"])
        finally:
            del whisper
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        report("Saving original transcription")
        write_json(artifacts.raw, raw)
    report("Merging same-speaker chunks: window 3, overlap 2")
    merged = merge_chunks(raw)
    write_json(artifacts.merged, merged)
    corrected = bool(config.get("use_qwen_correction", True))
    if corrected:
        report("Loading weights for correction")
        with session(config["qwen_model"]) as generator:
            merged = correct_chunks(merged, source_selected["name"], generator)
    rows = [{
        "id": row["id"], "speaker_number": row["speaker_number"], "speaker_label": row["speaker"],
        "start": row["start"], "end": row["end"], "duration": row["duration"],
        "initial_chunk_ids": row["initial_chunk_ids"], "boundaries": row["boundaries"],
        "source_language": source_selected["name"], "source_text": row["transcription"],
        "corrected": corrected, "correction_model": config["qwen_model"] if corrected else None,
    } for row in merged if row["transcription"].strip()]
    if not rows:
        raise ValueError("No transcribed speech was found.")
    report("Saving processed transcription")
    write_json(artifacts.transcript, rows)
    atomic_write(artifacts.path("_corrected_transcription.txt" if corrected else "_transcription.txt"),
                 "\n\n".join(row["source_text"] for row in rows))
    ensure_transcript_pdf(artifacts, rows, force=True)
    # Keep raw windows immutable; the index uses their corrected text and the same timestamps.
    ensure_index(artifacts, rows, config.get("embedding_model", EMBEDDING_MODELS[0]), force=True)
    return pd.DataFrame(rows), artifacts.transcript
