# ============================================================
# transcription.py — Script 1
# Whisper transcription + speaker diarization: language dictionary,
# audio/diarization helpers, run_transcription(...).
#
# Self-contained module — works both run locally (`python main.py` /
# `python app.py`) and pasted into its own Colab cell. Needs the
# packages in requirements.txt installed, plus ffmpeg on PATH.
# ============================================================
from ast import Return
import os
import re
import json
import subprocess
import gc
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import librosa
# ============================================================
# LANGUAGE DICTIONARY
# ============================================================
LANGUAGES = {
    "1":  {"name": "Hebrew",              "code": "he", "whisper_code": "he", "is_rtl": True},
    "2":  {"name": "Arabic",              "code": "ar", "whisper_code": "ar", "is_rtl": True},
    "3":  {"name": "English",             "code": "en", "whisper_code": "en", "is_rtl": False},
    "4":  {"name": "French",              "code": "fr", "whisper_code": "fr", "is_rtl": False},
    "5":  {"name": "Spanish",             "code": "es", "whisper_code": "es", "is_rtl": False},
    "6":  {"name": "Chinese",             "code": "zh", "whisper_code": "zh", "is_rtl": False},
    "7":  {"name": "Turkish",             "code": "tr", "whisper_code": "tr", "is_rtl": False},
    "8":  {"name": "German",              "code": "de", "whisper_code": "de", "is_rtl": False},
    "9":  {"name": "Italian",             "code": "it", "whisper_code": "it", "is_rtl": False},
    "10": {"name": "Russian",             "code": "ru", "whisper_code": "ru", "is_rtl": False},
    "11": {"name": "Persian (Farsi)",     "code": "fa", "whisper_code": "fa", "is_rtl": True},
}

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
    command = ["ffmpeg", "-y", "-loglevel", "quiet", "-i", input_file, "-ar", "16000", "-ac", "1", audio_path]
    subprocess.run(command, check=True)
    print(f"Audio extracted -> {audio_path}")

def run_diarization(audio_path, hf_token, diarization_model):
    print("Loading diarization model...")
    pipeline = Pipeline.from_pretrained(diarization_model, token=hf_token)
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    print("Running diarization...")
    diarization = pipeline(audio_path)
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

def smooth_short_speaker_islands(turns, merge_window=3, max_island_duration=SHORT_ISLAND_SECONDS, max_gap=MAX_SPEAKER_GAP):
    """Absorb short "islands" of a different speaker sandwiched inside a run
    of the same speaker."""
    if len(turns) < merge_window:
        return turns, []
    smoothed, audit = [], []
    i = 0
    while i < len(turns):
        if i + merge_window - 1 < len(turns):
            window = turns[i:i + merge_window]
            first_turn, last_turn = window[0], window[-1]
            middle_turns = window[1:-1]
            first_spk, last_spk = first_turn["speaker"], last_turn["speaker"]
            gaps_ok = True
            prev_end = first_turn["end"]
            for t in window[1:]:
                if t["start"] - prev_end > max_gap:
                    gaps_ok = False
                    break
                prev_end = t["end"]
            middle_durations_ok = all((t["end"] - t["start"]) <= max_island_duration for t in middle_turns)
            middle_diff_speaker = bool(middle_turns) and all(t["speaker"] != first_spk for t in middle_turns)
            is_short_island = (
                first_spk == last_spk
                and middle_diff_speaker
                and middle_durations_ok
                and gaps_ok
            )
            if is_short_island:
                merged_from = []
                for t in window:
                    merged_from.extend(t["merged_from"])
                new_turn = {
                    "start": first_turn["start"], "end": last_turn["end"], "speaker": first_spk,
                    "merged_from": merged_from,
                }
                audit.append({
                    "action": "absorbed_short_speaker_island",
                    "window_size": merge_window,
                    "previous_speaker": first_spk,
                    "middle_speakers": [t["speaker"] for t in middle_turns],
                    "next_speaker": last_spk,
                    "new_start": round(new_turn["start"], 3), "new_end": round(new_turn["end"], 3),
                    "merged_original_turns": new_turn["merged_from"],
                })
                smoothed.append(new_turn)
                i += merge_window
                continue
        smoothed.append(turns[i].copy())
        i += 1
    return smoothed, audit

def split_long_turns(turns, max_duration):
    chunks = []
    for turn in turns:
        start, final_end, speaker = turn["start"], turn["end"], turn["speaker"]
        while start < final_end:
            end = min(start + max_duration, final_end)
            if end - start >= MIN_TURN_DURATION:
                chunks.append({
                    "start": round(start, 3), "end": round(end, 3), "speaker": speaker,
                    "duration": round(end - start, 3), "merged_from": turn["merged_from"],
                })
            start = end
    return chunks

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
    """
    Returns a list of chunks with keys:
        start
        end
        duration      # sum of speech durations only
        speaker
        boundaries    # list of [start, end] speech segments
    """

    if not raw_turns:
        return []

    # ------------------------------------------------------------------
    # Pass 1: Merge consecutive turns from the same speaker
    # ------------------------------------------------------------------
    chunks = []

    current = raw_turns[0].copy()

    for turn in raw_turns[1:]:

        # same speaker and merged speech duration stays below max_duration
        if (
            turn["speaker"] == current["speaker"]
            and current["duration"] + turn["duration"] <= max_duration
        ):
            current["end"] = turn["end"]
            current["duration"] += turn["duration"]
            current["boundaries"].extend(turn["boundaries"])

        else:
            chunks.append(current)
            current = turn.copy()

    chunks.append(current)

    # ------------------------------------------------------------------
    # Pass 2: Remove short chunks and merge neighbors when possible
    # ------------------------------------------------------------------
    i = 0
    while i < len(chunks):

        if chunks[i]["duration"] >= min_duration:
            i += 1
            continue

        # remove short chunk
        removed = chunks.pop(i)

        # try merging neighbors
        if (
            i > 0
            and i < len(chunks)
            and chunks[i - 1]["speaker"] == chunks[i]["speaker"]
            and chunks[i - 1]["duration"] + chunks[i]["duration"] <= max_duration
        ):
            left = chunks[i - 1]
            right = chunks.pop(i)

            left["end"] = right["end"]
            left["duration"] += right["duration"]
            left["boundaries"].extend(right["boundaries"])

            # recheck merged chunk if needed
            i = max(i - 1, 0)

    return chunks

def load_whisper_model(whisper_model_name):
    if torch.cuda.is_available():
        device, compute_type = "cuda", "float16"
    else:
        device, compute_type = "cpu", "int8"
    print("Loading Whisper model...", whisper_model_name, "| Device:", device)
    model = WhisperModel(whisper_model_name, device=device, compute_type=compute_type)
    print("Whisper model loaded")
    return model

def transcribe_chunks_with_whisper(audio_path, chunks, model, whisper_language):
    results = []
    #audio = load_audio(audio_path)
    audio, sr = librosa.load(audio_path,sr=16000,mono=True)
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

        if (idx + 1) % 5 == 0 or (idx + 1) == len(chunks):
            print(f"Transcribed {idx + 1}/{len(chunks)} chunks")

    return results

def load_qwen_model(qwen_model_name):
    print("Loading Qwen model...", qwen_model_name)
    tokenizer = AutoTokenizer.from_pretrained(qwen_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(qwen_model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    model.eval()
    print("Qwen loaded successfully.")
    return tokenizer, model

# Codepoint ranges Qwen occasionally leaks into Hebrew/Arabic/English/French
# output (mostly CJK). DejaVu Sans has no glyphs for these, so they'd render
# as empty "tofu" boxes in the final PDF — always safe to strip.

def clean_qwen_output(text):
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"<think>.*?</think>","",text,flags=re.S)
    text = re.sub(r"```(?:\w+)?","",text)

    return text.strip()

def parse_qwen_chunks(text):

    result = {}

    blocks = re.findall(
        r"\[Chunk\s+(\d+)\]\s*(.*?)(?=\[Chunk|\Z)",
        text,
        flags=re.S
    )

    for idx, content in blocks:
         
        chunktext = content.strip()
            # Remove Qwen accidentally returning metadata prefixes
        chunktext = re.sub(
            r"^.*speaker.*\n?",
            "",
            chunktext,
            flags=re.IGNORECASE
        )

        # 2. Deletes everything from the start of the text up to "text:" (case-insensitive)
        chunktext = re.sub(
            r"^[\s\S]*?text\s*:\s*",
            "",
            chunktext,
            flags=re.IGNORECASE
        )   

        # Remove possible markdown/code formatting
        #chunktext = chunktext.replace("```text", "")
        #chunktext = chunktext.replace("```", "")
        result[int(idx)] = chunktext

    return result

def qwen_generate(tokenizer, model, system_prompt, user_prompt, max_new_tokens=2048):
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    # enable_thinking=False: Qwen3.5 thinks by default (emits <think>...</think>
    # before the real answer) — not wanted for a translation/correction task.
    # Harmless no-op for older templates (e.g. Qwen2.5) that ignore it.
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.05,use_cache=True,eos_token_id=tokenizer.eos_token_id)
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return clean_qwen_output(tokenizer.decode(generated_ids, skip_special_tokens=True))

def correct_transcriptions(transcriptions, tokenizer, model, source_lang_name, max_new_tokens=1024):

    total = len(transcriptions)

    if total == 0:
        return transcriptions

    for i in range(total):

        # ------------------------------------------
        # First chunk
        # ------------------------------------------
        if i == 0:

            context = [transcriptions[i]]

        # ------------------------------------------
        # Last chunk
        # ------------------------------------------
        elif i == total - 1:

            context = [transcriptions[i]]
        # ------------------------------------------
        # Middle chunks
        # ------------------------------------------
        else:

            context = [
                transcriptions[i - 1],
                transcriptions[i],
                transcriptions[i + 1]
            ]

        corrected = correct_transcription_with_qwen(tokenizer,model, context, source_lang_name, max_new_tokens=max_new_tokens)

        transcriptions[i]["transcription"] = corrected.strip()

        print(f"Corrected {i + 1}/{total}", end="\r")

    print()

    return transcriptions

def correct_transcription_with_qwen(tokenizer, model, chunks, language_name, max_new_tokens=1024):

    """
    chunks:
        len == 1:
            direct correction

        len == 3:
            [previous_chunk, current_chunk, next_chunk]
            Correct ONLY current_chunk using context.
    """

    # --------------------------------------------------
    # Single chunk correction
    # --------------------------------------------------
    if len(chunks) == 1:

        text = str(chunks[0]["transcription"]).strip()

        if text == "":
            return ""

        system_prompt = (
            "You are an expert ASR transcript corrector. "
            "Your task is to correct speech recognition errors "
            "while preserving the original meaning "
        )

        user_prompt = f"""
                        Correct the following {language_name} ASR transcription.

                        Rules:

                        1. Keep the same language.
                        2. Preserve the original meaning.
                        3. Correct spelling mistakes, missing punctuation, and incorrect word segmentation.
                        4. Output text must be in the same language ({language_name}).
                        5. Do not output in a different language.
                        6. Do not add information that was not spoken.
                        7. Do not remove information.
                        8. Do not summarize.
                        9. Do not translate.
                        11. If a small part is impossible to understand, keep it unchanged.
                        12. Output ONLY the corrected transcript.

                        Transcript:

                        {text}
                        """

        return qwen_generate(tokenizer, model, system_prompt, user_prompt, max_new_tokens=max_new_tokens)


    # --------------------------------------------------
    # Three chunk context correction
    # --------------------------------------------------

    previous_chunk = chunks[0]
    current_chunk = chunks[1]
    next_chunk = chunks[2]


    system_prompt = (
        "You are an expert speech transcript post-processing system."

"Your task is to recover the intended spoken text from noisy automatic speech recognition transcripts."

"The transcript may contain recognition errors, missing words, incorrect words, broken word boundaries and missing punctuation."
)


    user_prompt = f"""
                    You are given three consecutive chunks from the same conversation.

                    The first and third chunks are CONTEXT ONLY.

                    Correct ONLY the SECOND chunk.

                    Read all three chunks before making corrections.

                    Rules:

                    1. Correct only the second chunk.
                    2. Use the first and third chunks only to understand context and resolve ambiguities to correct the text of the second chunk.
                    3. Do not modify, translate, or output the first and third chunks which are for context only.
                    4. Preserve the original meaning.
                    5. Correct wrong words, spelling mistakes, missing punctuation, and incorrect word boundaries in the second chunk.
                    6. Do not add any word in a different language.
                    7. Do not remove information.
                    8. Do not summarize.
                    9. Do not translate.
                    10. Do NOT explain your corrections.
                    11. Do NOT add comments.
                    12. Do NOT add markdown.
                    14. Output ONLY the corrected version of the second chunk.
                    15. Output text must be entierely in ({language_name}).

                    Previous chunk (context only):

                    Speaker: {previous_chunk["speaker"]}
                    Text:
                    {previous_chunk["transcription"]}


                    Current chunk (CORRECT THIS):

                    Speaker: {current_chunk["speaker"]}
                    Text:
                    {current_chunk["transcription"]}


                    Next chunk (context only):

                    Speaker: {next_chunk["speaker"]}
                    Text:
                    {next_chunk["transcription"]}
                    """

    return qwen_generate(tokenizer, model, system_prompt, user_prompt, max_new_tokens=max_new_tokens)

def prepare_context_chunks(chunks):

    merged = []

    current = None

    for c in chunks:

        if current is None:
            current = {
                "ids": [c["id"]],
                "speaker": c["speaker"],
                "transcription": c["transcription"],
                 "start": c["start"],        # original timeline
                 "end": c["end"],            # original timeline
                 "duration": c["duration"],            # actual speech duration
                "boundaries": c["boundaries"],

            }

        elif current["speaker"] == c["speaker"]:

            current["ids"].append(c["id"])
            current["transcription"] += " " + c["transcription"]
            current["end"] = c["end"]
            current["duration"] += c["duration"]
            current["boundaries"].append(c["boundaries"])

        else:

            merged.append(current)

            current = {
                "ids": [c["id"]],
                "speaker": c["speaker"],
                "transcription": c["transcription"],
                "start": c["start"],        # original timeline
                "end": c["end"],            # original timeline
                "duration": c["duration"],            # actual speech duration
                "boundaries": c["boundaries"],
            }

    if current:
        merged.append(current)

    return merged
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    return " ".join(text.split())

def save_json(records, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print("JSON saved ->", path)

def save_csv(records_or_df, path):
    df = records_or_df if isinstance(records_or_df, pd.DataFrame) else pd.DataFrame(records_or_df)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print("CSV saved ->", path)


def merge_consecutive_same_speaker(transcriptions, maxNb=4):
    if not transcriptions:
        return []

    merged = []
    current = transcriptions[0].copy()
    chunk_count = 1
    nbChunk = 1
    for t in transcriptions[1:]:
        if t["speaker"] == current["speaker"] and chunk_count < maxNb:
            current["transcription"] += " " + t["transcription"]
            current["end"] = max(current["end"], t["end"])
            current["duration"] += t["duration"]
            current["boundaries"].extend(t["boundaries"])
            chunk_count += 1
        else:
            current["id"] = nbChunk
            merged.append(current)
            current = t.copy()
            chunk_count = 1
            nbChunk += 1

    merged.append(current)
    print(f"Merged consecutive same-speaker chunks: {len(transcriptions)} -> {len(merged)}")
    return merged


# ============================================================
# FUNCTION 1 — TRANSCRIPTION
# audio -> diarize -> Whisper -> (optional Qwen cleanup) -> dataset
# ============================================================
def run_transcription(input_path, output_dir, hf_token, source_selected, config):
    input_stem = input_path.stem
    audio_out        = output_dir / f"{input_stem}_audio.wav"
    raw_json_out     = output_dir / f"{input_stem}_transcript.json"
    dataset_json_out = output_dir / f"{input_stem}_transcript_dataset.json"
    whisper_language = source_selected["whisper_code"]
    source_lang_name = source_selected["name"]

    print("\n--- Extracting audio ---")
    extract_audio(str(input_path), str(audio_out))

    print("\n--- Running diarization ---")
    diarization = run_diarization(str(audio_out), hf_token, config["diarization_model"])

    print("\n--- Building clean chunks ---")
    chunks, chunk_audit = diarization_to_chunks(diarization, max_duration=config["max_chunk_duration"], merge_window=config["merge_window"])
    assert chunks, "ERROR: Diarization returned no speaker turns."

    # Free the diarization model's GPU memory before Whisper loads — without
    # this, both models can be resident at once and blow past a T4's VRAM.
    del diarization
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n--- Transcribing with Whisper ---")
    whisper_model = load_whisper_model(config["whisper_model"])
    transcriptions = transcribe_chunks_with_whisper(str(audio_out), chunks, whisper_model, whisper_language)
    transcriptions = merge_consecutive_same_speaker(transcriptions,maxNb=4)
    save_json(transcriptions, raw_json_out)
    del whisper_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # with open(raw_json_out, "r", encoding="utf-8") as f:
    #    transcriptions = json.load(f)
    
    if config["use_qwen_correction"]:
        print("\n--- Correcting transcript with Qwen ---")
        qwen_tokenizer, qwen_model = load_qwen_model(config["qwen_model"])
        transcriptions = correct_transcriptions(transcriptions, qwen_tokenizer, qwen_model, source_lang_name,
                                                 max_new_tokens=config["max_new_tokens"]) 
                                                 #N=config["NbChunksToMerge"], M=config["NbChunksOverlap"])
        del qwen_tokenizer, qwen_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n--- Building dataset ---")
    speaker_labels = []
    for t in transcriptions:
        if t["speaker"] not in speaker_labels:
            speaker_labels.append(t["speaker"])
    speaker_map = {speaker: i + 1 for i, speaker in enumerate(speaker_labels)}

    rows = []
    for t in transcriptions:
        rows.append({
            "id": t["id"], "speaker_number": speaker_map[t["speaker"]], "speaker_label": t["speaker"],
            "start": t["start"], "end": t["end"], "duration": t["duration"],
            "source_language": source_lang_name, "source_text": t["transcription"], "target_text": "",
        })
    dataset_df = pd.DataFrame(rows)
    dataset_df["source_text"] = dataset_df["source_text"].astype(str).str.strip()
    dataset_df = dataset_df[dataset_df["source_text"] != ""].reset_index(drop=True)
    dataset_df["id"] = range(1, len(dataset_df) + 1)

    save_json(dataset_df.to_dict(orient="records"), dataset_json_out)
    return dataset_df, dataset_json_out


if __name__ == "__main__":
    # Example usage
    input_path = Path("/mnt/c/Users/ahhac/Desktop/Ar_Heb_ASR_Translation/video_demo.mp4")
    output_dir = Path("mnt/c/Users/ahhac/Desktop/Ar_Heb_ASR_Translation")
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_token = "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"
    source_selected = LANGUAGES["1"]  # Hebrew
    config = {
            "whisper_model": "large-v3",
            "diarization_model": "pyannote/speaker-diarization-3.1",
            "qwen_model": "Qwen/Qwen3-4B-Instruct-2507-FP8",
            "max_chunk_duration": 30.0,
            "merge_window": 3,
            "max_new_tokens": 2048,
            "use_qwen_correction": True,
            "NbChunksToMerge": 6,
            "NbChunksOverlap": 2
        }
    dataset_df, dataset_json_out = run_transcription(input_path, output_dir, hf_token, source_selected, config)