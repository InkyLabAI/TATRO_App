"""Video operations, reusable artifacts, and explicit regeneration boundaries."""
from pathlib import Path
from threading import RLock
from progress import report

from artifacts import Artifacts, atomic_write, fingerprint, read_json, write_json
from local_llm import session, summarize, translate_text
from pdf_export import save_pdf
from retrieval import ensure_index, embed_texts, search
from transcript_pdf import ensure_transcript_pdf
from transcript_data import summary_records

# Models are loaded per operation and released afterwards. Serialize GPU operations.
MODEL_LOCK = RLock()


def records_for(video):
    records = read_json(Artifacts(video).transcript)
    if not records or not isinstance(records, list):
        raise ValueError("No transcript records found.")
    for row in records:
        if not str(row.get("source_text", "")).strip() or row["end"] < row["start"]:
            raise ValueError("Transcript contains invalid text or timestamps.")
    return records


def saved_pdf(video, target, kind):
    """Preview only a PDF whose source is still current; never run a model here."""
    artifacts = Artifacts(video)
    if kind == "summary":
        pdf = artifacts.summary(target["code"])
        meta = artifacts.summary(target["code"], "meta.json")
    else:
        pdf = artifacts.path(f"_{target['code']}_translation.pdf")
        meta = artifacts.path(f"_translated_{target['code']}.meta.json")
    if not pdf.exists():
        return None
    if meta.exists():
        records = summary_records(artifacts) if kind == "summary" else records_for(video)
        expected = {"transcript": fingerprint(records), "target": target["name"]}
        if read_json(meta) != expected:
            return None
    return pdf


def transcribe(video, language, hf_token, config):
    from transcription import run_transcription
    report("Waiting for the local model worker")
    with MODEL_LOCK:
        return run_transcription(Path(video), Artifacts(video).directory, hf_token, language, config)


def document_preview(video, target, translation_requested):
    if not Artifacts(video).transcript.exists():
        return None
    records = records_for(video)
    if not translation_requested or records[0]["source_language"] == target["name"]:
        return ensure_transcript_pdf(Artifacts(video), records)
    return saved_pdf(video, target, "translation")


def index_video(video, embedding_model, force=False):
    report("Waiting for the local model worker")
    with MODEL_LOCK:
        return ensure_index(Artifacts(video), records_for(video), embedding_model, force=force)


def query_video(video, question, user_language, embedding_model, llm_model):
    if not question.strip():
        raise ValueError("Enter a question first.")
    records = records_for(video)
    source = records[0]["source_language"]
    report("Waiting for the local model worker")
    with MODEL_LOCK:
        if user_language != source:
            report("Loading weights for question translation")
            with session(llm_model) as generator:
                report("Translating question into the video language")
                question = translate_text(question, user_language, source, generator)
        else:
            report("Question language matches video; skipping translation")
        vectors = ensure_index(Artifacts(video), records, embedding_model)
        query = embed_texts([question], embedding_model, query=True)[0]
    report("Finding the best answer segment")
    return search(records, vectors, query)


def translate_video(video, target, llm_model, force=False):
    artifacts = Artifacts(video)
    records = records_for(video)
    source = records[0]["source_language"]
    if source == target["name"]:
        report("Source and target languages match; using transcription PDF without translation")
        return ensure_transcript_pdf(artifacts, records, force=force)
    output = artifacts.path(f"_translated_{target['code']}.json")
    pdf = artifacts.path(f"_{target['code']}_translation.pdf")
    meta_path = artifacts.path(f"_translated_{target['code']}.meta.json")
    expected = {"transcript": fingerprint(records), "target": target["name"]}
    # Older files without metadata remain usable; new files validate transcript identity.
    reusable = (output.exists() and (not meta_path.exists() or read_json(meta_path) == expected))
    if reusable and not force:
        report("Reusing saved translation")
        translated = read_json(output)
    else:
        report("Waiting for the local model worker / loading weights for translation")
        with MODEL_LOCK, session(llm_model) as generator:
            report("Starting translation", 0, len(records))
            translated = []
            for index, row in enumerate(records, start=1):
                translated.append({**row, "target_language": target["name"],
                                   "target_text": translate_text(row["source_text"], source, target["name"], generator)})
                report("Chunks translated", index, len(records))
        report("Saving translated transcript")
        write_json(output, translated)
        write_json(meta_path, expected)
    if force or not reusable or not pdf.exists():
        text = "\n\n".join(f"Speaker {row['speaker_number']}\n{row['target_text']}" for row in translated)
        report("Creating translation PDF")
        save_pdf(text, pdf, f"{artifacts.video.stem} - {target['name']}", target["name"])
    return pdf


def summary_video(video, target, llm_model, force=False):
    artifacts = Artifacts(video)
    records = summary_records(artifacts)
    pdf = artifacts.summary(target["code"])
    text_path = artifacts.summary(target["code"], "txt")
    meta_path = artifacts.summary(target["code"], "meta.json")
    expected = {"transcript": fingerprint(records), "target": target["name"]}
    reusable = not meta_path.exists() or read_json(meta_path) == expected
    if not force and reusable and pdf.exists():
        report("Reusing saved summary PDF")
        return pdf
    if not force and reusable and text_path.exists():
        report("Reusing saved summary text")
        text = text_path.read_text(encoding="utf-8")
    else:
        report("Summarizing corrected transcription" if all(row.get("corrected") for row in records)
               else "Corrected transcription unavailable; summarizing uncorrected transcription")
        report("Waiting for the local model worker / loading weights for summary")
        with MODEL_LOCK, session(llm_model) as generator:
            text = summarize(records, records[0]["source_language"], target["name"], generator)
        report("Saving summary text")
        atomic_write(text_path, text)
        write_json(meta_path, expected)
    report("Creating summary PDF")
    save_pdf(text, pdf, f"{artifacts.video.stem} - Summary", target["name"])
    return pdf
